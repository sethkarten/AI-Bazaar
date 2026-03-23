"""SellerAgent for the Lemon Market (Environment B).

A fixed-policy honest seller with no LLM, no overhead, no supply/production
machinery.  Description and price reflect the car's true quality.

Sybil sellers are implemented as SybilIdentity(SellerAgent) in sybil.py;
all sybil listing logic is coordinated by DeceptivePrincipal.

This class satisfies the firm interface expected by bazaar_env step() loops:
  in_business, reputation, cash, name, listings, sales_info,
  update_reputation(), pay_overhead_costs(), pay_taxes(), reflect(),
  mark_out_of_business(), create_listings()
"""
import json
from collections import defaultdict
from typing import Any, Dict, List, Optional
import logging

from ai_bazaar.market_core.market_core import Ledger, Market
from ai_bazaar.agents.firm import BaseFirmAgent
from ai_bazaar.utils.common import V_MAX, SELLER_PERSONAS, SELLER_PERSONA_DESCRIPTIONS
from ai_bazaar.utils.agent_prompt_log import maybe_append_lemon_agent_prompt


class SellerAgent(BaseFirmAgent):
    """Fixed-policy honest seller for LEMON_MARKET.

    Key design choices:
    - No LLM calls — paper §5.2 specifies a fixed "straightforward description policy"
    - No overhead costs or taxes — C2C resale, not a production firm
    - Always honest: description and price reflect true quality
    - Listing stores true quality_value for correct EMA reputation updates
    - price = V_MAX * quality_value
    """

    def __init__(
        self,
        name: str,
        goods: List[str],
        initial_cash: float,
        ledger: Ledger,
        market: Market,
        persona: Optional[str] = None,
        args=None,
    ) -> None:
        BaseFirmAgent.__init__(self)
        self.logger = logging.getLogger("main")
        self.name = name
        self.goods = goods
        self.ledger = ledger
        self.market = market
        self.args = args
        self.sybil = False  # honest seller; SybilIdentity overrides this to True
        self.persona = persona  # selects description template from SELLER_PERSONAS
        self.listings: List[Dict[str, Any]] = []

        self.ledger.credit(self.name, initial_cash)
        self.ledger.agent_inventories[self.name] = {}

        # Required by _build_firms_state() in bazaar_env
        self.total_quantity_sold_by_good: Dict[str, float] = {"car": 0.0}
        self.total_quantity_sold_by_good_this_timestep: Dict = defaultdict(
            lambda: defaultdict(float)
        )
        self.listings_posted_this_step: int = 0

    # ------------------------------------------------------------------ #
    # Core listing method                                                   #
    # ------------------------------------------------------------------ #

    def create_listings(self, timestep: int = None) -> List[Dict[str, Any]]:
        """Create listings for all unposted cars with truthful description and price."""
        if timestep is None:
            timestep = 0
        new_listings = []
        self.listings_posted_this_step = 0

        for item in self.listings:
            if item.get("posted"):
                continue
            quality = item.get("quality", "unknown")
            quality_value = item.get("quality_value", 0.5)

            templates = SELLER_PERSONAS.get(self.persona or "standard", SELLER_PERSONAS["standard"])
            description = templates.get(quality, f"Used car. Condition: {quality}.")
            price = V_MAX * quality_value

            listing = {
                "id": f"{self.name}_listing_{len(new_listings)}",
                "firm_id": self.name,
                "description": description,
                "price": price,
                "reputation": self.reputation,
                "quality": quality,          # true quality label
                "quality_value": quality_value,  # true quality value
            }
            new_listings.append(listing)
            item["posted"] = True

        self.listings_posted_this_step = len(new_listings)
        return new_listings

    # ------------------------------------------------------------------ #
    # No-op overrides                                                       #
    # ------------------------------------------------------------------ #

    def pay_overhead_costs(self, timestep: int) -> float:
        """No overhead costs — C2C resale, not a production firm."""
        return 0.0

    def pay_taxes(self, timestep: int, tax_rate: float) -> float:
        """No taxes — buyers pay, sellers in this model do not."""
        return 0.0

    # ------------------------------------------------------------------ #
    # Properties                                                            #
    # ------------------------------------------------------------------ #

    @property
    def inventory(self) -> dict:
        return self.ledger.agent_inventories.get(self.name, {})


class LLMSellerAgent(SellerAgent):
    """LLM-driven honest seller for LEMON_MARKET.

    Inherits the full SellerAgent interface. Overrides create_listings() to call
    llm.send_msg directly (no shared message_history — thread-safe for parallel
    lemon_market_firm_phases). Falls back to SELLER_PERSONAS templates on LLM failure.

    Price is always V_MAX * quality_value regardless of persona.
    """

    def __init__(
        self,
        name: str,
        goods: List[str],
        initial_cash: float,
        ledger: Ledger,
        market: Market,
        persona: Optional[str] = None,
        args=None,
        llm_instance=None,
    ) -> None:
        super().__init__(name, goods, initial_cash, ledger, market, persona, args)
        self.llm = llm_instance
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        style = SELLER_PERSONA_DESCRIPTIONS.get(
            self.persona or "standard",
            SELLER_PERSONA_DESCRIPTIONS["standard"],
        )
        return (
            "You are an honest used-car seller in a marketplace. "
            "You always describe the car's true condition accurately. "
            f"Description style: {style}"
        )

    def create_listings(self, timestep: int = None) -> List[Dict[str, Any]]:
        """LLM-generated honest descriptions; falls back to templates if LLM unavailable."""
        new_listings = []
        self.listings_posted_this_step = 0
        for item in self.listings:
            if item.get("posted"):
                continue
            quality = item.get("quality", "unknown")
            quality_value = item.get("quality_value", 0.5)
            description = self._llm_description(quality, quality_value, timestep or 0)
            price = V_MAX * quality_value
            new_listings.append({
                "id": f"{self.name}_listing_{len(new_listings)}",
                "firm_id": self.name,
                "description": description,
                "price": price,
                "reputation": self.reputation,
                "quality": quality,
                "quality_value": quality_value,
            })
            item["posted"] = True
        self.listings_posted_this_step = len(new_listings)
        return new_listings

    def _llm_description(self, quality: str, quality_value: float, timestep: int = 0) -> str:
        """One-shot llm.send_msg call — no message_history, safe for parallel threads."""
        templates = SELLER_PERSONAS.get(self.persona or "standard", SELLER_PERSONAS["standard"])
        fallback = templates.get(quality, f"Used car. Condition: {quality}.")
        if self.llm is None:
            return fallback
        prompt = (
            f"The car's true condition is '{quality}' (quality value {quality_value:.2f}). "
            "Write a short honest listing description. "
            "Do not mention numeric quality values. "
            'Respond with a single JSON object: {"description": "<text>"}'
        )
        try:
            raw, _ = self.llm.send_msg(self.system_prompt, prompt)
            maybe_append_lemon_agent_prompt(
                self.args,
                "seller",
                self.name,
                "listing",
                timestep,
                self.system_prompt,
                prompt,
                raw,
                depth=0,
            )
            parsed = json.loads(raw) if isinstance(raw, str) else raw
            desc = str(parsed.get("description", "")).strip()
            return desc if desc else fallback
        except Exception:
            return fallback
