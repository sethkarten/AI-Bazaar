"""BuyerAgent for the Lemon Market (Environment B).

A pure LLM-driven buyer with no CES/WTP machinery. The agent receives a
structured observation of visible listings (description, price, and optionally
seller reputation) plus its own transaction history and the market mean quality,
then decides whether to bid on one listing or pass.
"""
import json
import logging
import random
from typing import List, Optional

from ai_bazaar.agents.llm_agent import LLMAgent
from ai_bazaar.market_core.market_core import Ledger, Market, Order
from ai_bazaar.utils.common import Message, V_MAX

BUYER_TRANSACTION_HISTORY_LEN = 10


class BuyerAgent(LLMAgent):
    """LLM-driven buyer for the Lemon Market.

    Key design choices:
    - Infinite cash (1e12 credited at construction) — never budget-gated
    - Single LLM call per timestep over all discovery_limit visible listings
    - Transaction history (last N) passed as structured context
    - Seller reputation EMA withheld when args.no_buyer_rep is True (ablation)
    """

    def __init__(
        self,
        llm: str,
        port: int,
        name: str,
        ledger: Ledger,
        market: Market,
        persona: str,
        args=None,
        llm_instance=None,
        prompt_algo: str = "io",
        history_len: int = 5,
        timeout: int = 10,
    ) -> None:
        super().__init__(
            llm,
            port,
            name,
            prompt_algo,
            history_len,
            timeout,
            args=args,
            llm_instance=llm_instance,
        )
        self.lemon_agent_role = "buyer"
        self.logger = logging.getLogger("main")
        self.ledger = ledger
        self.market = market
        self.persona = persona

        self.transaction_history: list = []
        self.discovered_listings_this_step: list = []  # Listing objects seen this step
        self.sybil_seen_total: int = 0
        self.sybil_passed_total: int = 0
        self.sybil_pass_rate_this_step: Optional[float] = None
        # Infinite cash — buyer is never budget-constrained
        self.ledger.credit(name, 1e12)
        self.ledger.agent_inventories[name] = {}

        self.system_prompt = self._create_system_prompt()

    # System prompt
    def _create_system_prompt(self) -> str:
        return (
            f"You are {self.name}, a buyer in a used-car peer-to-peer market. "
            f"Your persona: {self.persona}. "
            "Your goal is to purchase good-value cars and avoid paying more than a car "
            "is worth. "
            "Fair market values by quality tier: "
            f"mint ≈ ${V_MAX * 1.0:,.0f}, "
            f"good ≈ ${V_MAX * 0.7:,.0f}, "
            f"fair ≈ ${V_MAX * 0.4:,.0f}, "
            f"poor ≈ ${V_MAX * 0.1:,.0f}. "
            "A listing priced above its claimed quality tier's fair value is likely overpriced. "
            "Be aware: some sellers misrepresent car quality in their descriptions. "
            "Evaluate each listing by weighing the written description, the seller's "
            "reputation score (if shown), and your own past transaction history. "
            "You may buy at most one car per round. "
            "If no listing offers good value, pass and wait for a better opportunity."
        )

    # Core decision method
    def make_orders(
        self,
        timestep: int,
        listings: list,
        discovery_limit: int = 5,
        include_reputation: bool = True,
    ) -> List[Order]:
        """Sample up to discovery_limit listings, prompt the LLM, return ≤1 Order."""
        if not listings:
            return []

        visible = random.sample(listings, min(discovery_limit, len(listings)))
        self.discovered_listings_this_step = list(visible)

        obs = self._build_observation(
            timestep, visible, include_reputation
        )

        self.add_message(
            timestep,
            Message.UPDATE_BID,
            observation=obs,
        )

        fallback = ["pass", None]
        try:
            result = self.act_llm(
                timestep,
                ["decision", "listing_id"],
                self._parse_bid,
                on_parse_failure_return=fallback,
            )
            decision = result[0]
            listing_id = result[1]
        except Exception:
            decision, listing_id = "pass", None

        self.add_message(
            timestep,
            Message.ACTION_BID,
            decision=decision,
            listing_id=listing_id,
        )

        if decision != "bid" or not listing_id:
            return []

        # Resolve listing_id to a Listing object in the visible set
        matched = next((L for L in visible if L.id == listing_id), None)
        if matched is None:
            return []

        return [
            Order(
                consumer_id=self.name,
                firm_id=matched.firm_id,
                good="car",
                quantity=1,
                max_price=matched.price,
                listing_id=matched.id,
            )
        ]

    # Observation builder
    def _build_observation(
        self,
        timestep: int,
        visible_listings: list,
        include_reputation: bool,
    ) -> dict:
        listing_dicts = []
        for L in visible_listings:
            entry = {
                "listing_id": L.id,
                "listed_price": L.price,
                "description": L.description,
            }
            if include_reputation:
                entry["seller_reputation"] = round(
                    getattr(L, "reputation", 0.5), 4
                )
            listing_dicts.append(entry)

        personal_history = self.transaction_history[-BUYER_TRANSACTION_HISTORY_LEN:]
        personal_mean_quality = (
            sum(r["quality_received"] for r in personal_history) / len(personal_history)
            if personal_history else None
        )
        return {
            "timestep": timestep,
            "persona": self.persona,
            "your_mean_quality_received": personal_mean_quality,
            "your_transaction_history": personal_history,
            "listings_visible": listing_dicts,
        }

    # Post-purchase review (second LLM call after market clearing)
    def review_transaction(
        self,
        seller_id: str,
        description: str,
        quality_received: float,
        quality_label: str,
        timestep: int,
    ) -> Optional[bool]:
        """Compare seller description to true quality; return True=upvote, False=downvote, None=abstain.

        Abstain casts no vote and does not affect seller reputation.

        If no LLM is configured, uses a rule-based fallback: upvote when the quality
        label appears in the description (honest seller), downvote otherwise.
        """
        if self.llm is None:
            return quality_label.lower() in description.lower()

        # Format mirrors other agent actions: JSON with optional "thought" for CoT/SC
        vote_format = '{"vote": "upvote"|"downvote"|"abstain"}'
        if self.prompt_algo in ("cot", "sc"):
            vote_format = '{"thought": "<brief reasoning>", "vote": "upvote"|"downvote"|"abstain"}'

        review_system = (
            f"You are {self.name}, a buyer who just received a used car you purchased. "
            "Your task is to review whether the seller's listing description accurately "
            "represented the car's true quality."
        )
        user_prompt = (
            f"Seller '{seller_id}' listed the car with this description:\n"
            f"  \"{description}\"\n\n"
            f"The actual quality you received: {quality_label} "
            f"(quality score {quality_received:.2f} on a 0–1 scale where 1.0=mint, 0.7=good, 0.4=fair, 0.1=poor).\n\n"
            "Review the seller's description against the quality you received:\n"
            "  - 'upvote'   if the description was accurate or undersold the quality\n"
            "  - 'downvote' if the description was misleading or oversold the quality\n"
            "  - 'abstain'  if you cannot make a determination\n\n"
            f"Respond with a single JSON object. Format: {vote_format}"
        )

        try:
            response, _ = self.llm.send_msg(review_system, user_prompt)
            self._maybe_log_lemon_prompt(
                "review", timestep, review_system, user_prompt, response, depth=0
            )
            return self._parse_review(response)
        except Exception:
            return None  # abstain on error

    @staticmethod
    def _parse_review(response: str) -> Optional[bool]:
        """Parse vote JSON. Returns True=upvote, False=downvote, None=abstain."""
        try:
            # Strip markdown fences if present
            text = response.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            data = json.loads(text.strip())
            vote = str(data.get("vote", "abstain")).strip().lower()
        except Exception:
            # Fallback: scan raw text for the vote keyword
            text = response.strip().lower()
            if "upvote" in text:
                vote = "upvote"
            elif "downvote" in text:
                vote = "downvote"
            else:
                vote = "abstain"
        if vote == "upvote":
            return True
        if vote == "downvote":
            return False
        return None  # abstain

    # Post-transaction callback
    def record_transaction(
        self,
        seller_id: str,
        price_paid: float,
        quality_received: float,
        quality_label: str,
        timestep: int,
    ) -> None:
        self.transaction_history.append(
            {
                "timestep": timestep,
                "seller_id": seller_id,
                "price_paid": price_paid,
                "quality_received": quality_received,
                "quality_label": quality_label,
                "consumer_surplus": quality_received * V_MAX - price_paid,
            }
        )

    # LLMAgent abstract override
    def add_message(self, timestep: int, m_type: Message, **kwargs) -> None:
        self.add_message_history_timestep(timestep)

        if m_type == Message.UPDATE_BID:
            obs = kwargs.get("observation", {})
            bid_format = '{"decision": "bid"|"pass", "listing_id": "<id or null>"}'
            if self.prompt_algo in ("cot", "sc"):
                bid_format = (
                    '{"thought": "<brief reasoning>", '
                    '"decision": "bid"|"pass", "listing_id": "<id or null>"}'
                )
            self.message_history[timestep]["user_prompt"] = (
                f"Your current observation:\n{json.dumps(obs, indent=2)}\n\n"
                "Decide whether to bid on one listing or pass this round. "
                "If you bid, specify the listing_id of the car you want to buy. "
                f"Respond with a single JSON object. Format: {bid_format}"
            )
            self.message_history[timestep]["expected_format"] = bid_format

        elif m_type == Message.ACTION_BID:
            decision = kwargs.get("decision", "pass")
            listing_id = kwargs.get("listing_id", None)
            self.message_history[timestep]["historical"] += (
                f"Bid decision: {decision}"
                + (f" on listing {listing_id}" if listing_id else "")
                + "\n"
            )
            self.message_history[timestep]["action"] += (
                f"decision={decision} listing_id={listing_id}\n"
            )

        elif m_type == Message.UPDATE_REVIEW:
            seller_id   = kwargs.get("seller_id", "")
            description = kwargs.get("description", "")
            quality_label = kwargs.get("quality_label", "")
            self.message_history[timestep]["historical"] += (
                f"Review prompt: seller={seller_id} quality_received={quality_label} "
                f"description=\"{description[:80]}\"\n"
            )

        elif m_type == Message.ACTION_REVIEW:
            vote = kwargs.get("vote", "upvote")
            self.message_history[timestep]["historical"] += f"Review vote: {vote}\n"
            self.message_history[timestep]["action"] += f"review_vote={vote}\n"

        elif m_type == Message.REFLECTION:
            self.message_history[timestep]["historical"] += (
                "Reflection: " + kwargs.get("reflection_msg", "") + "\n"
            )

    # Parser
    @staticmethod
    def _parse_bid(items: list) -> list:
        """Parse [decision, listing_id] from the list of extracted key values."""
        if not items or len(items) < 2:
            return ["pass", None]
        decision = str(items[0] if items[0] is not None else "pass").lower().strip()
        if decision not in ("bid", "pass"):
            decision = "pass"
        listing_id = items[1]
        if listing_id is not None:
            listing_id = str(listing_id).strip()
            if listing_id.lower() in ("null", "none", ""):
                listing_id = None
        return [decision, listing_id]

    # Interface methods (satisfy downstream loops)
    def reflect(self, timestep: int, **kwargs) -> None:
        """Optional diary entry — no ledger needed for buyers."""
        if self.llm is None or getattr(self.args, "no_diaries", False):
            return
        recent = self.transaction_history[-3:]
        if not recent:
            return
        summary = "; ".join(
            f"t={r['timestep']} {r['quality_label']} ${r['price_paid']:.0f} CS={r['consumer_surplus']:.0f}"
            for r in recent
        )
        diary_prompt = (
            f"Recent purchases: {summary}\n"
            "Write a 1-2 sentence diary entry reflecting on your buying experience. "
            "Have you been getting good value? Are you suspicious of any sellers?"
        )
        entry, _ = self.llm.send_msg(self.system_prompt, diary_prompt)
        self._maybe_log_lemon_prompt(
            "diary", timestep, self.system_prompt, diary_prompt, entry, depth=0
        )
        self.write_diary_entry(timestep, entry)

    def consume_inventory(self) -> None:
        """Clear car inventory from ledger after each timestep."""
        inv = self.ledger.agent_inventories.get(self.name, {})
        inv.pop("car", None)

    def pay_taxes(self, *args, **kwargs) -> None:
        """No-op — buyers pay no taxes."""
        pass

    def receive_income(self, *args, **kwargs) -> None:
        """No-op — buyers have infinite cash; no income phase needed."""
        pass

    @property
    def cash(self) -> float:
        return self.ledger.agent_money.get(self.name, 0.0)

    @property
    def utility(self) -> float:
        """Proxy: cumulative consumer surplus across all transactions."""
        return sum(r["consumer_surplus"] for r in self.transaction_history)

    @property
    def inventory(self) -> dict:
        return self.ledger.agent_inventories.get(self.name, {})
