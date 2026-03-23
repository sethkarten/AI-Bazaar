"""
Unit tests for BuyerAgent (Lemon Market Environment B).

Covers: construction, observation building (with/without reputation),
add_message (UPDATE_BID / ACTION_BID), _parse_bid, record_transaction,
consume_inventory, utility/cash/inventory properties, make_orders with
mock LLM (bid and pass), empty-listings guard, and --no-buyer-rep ablation.

Run from project root:
  python -m pytest tests/test_buyer_agent.py -v
"""
import json
import pytest
from types import SimpleNamespace

from ai_bazaar.market_core.market_core import Ledger, Market, Listing, Order
from ai_bazaar.agents.buyer import BuyerAgent, BUYER_TRANSACTION_HISTORY_LEN
from ai_bazaar.utils.common import Message, V_MAX


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_args(**kwargs):
    defaults = dict(
        bracket_setting="three",
        no_diaries=True,
        no_buyer_rep=False,
        use_parsing_agent=False,
        max_tokens=512,
        service="google-ai",
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _make_buyer(name="buyer_0", persona="cautious", llm_instance=None, **kwargs):
    ledger = Ledger()
    market = Market()
    args = _make_args(**kwargs)
    buyer = BuyerAgent(
        llm="None",
        port=0,
        name=name,
        ledger=ledger,
        market=market,
        persona=persona,
        args=args,
        llm_instance=llm_instance,
    )
    return buyer, ledger, market


def _make_listing(lid="l1", firm_id="firm_0", price=20000.0, reputation=0.8,
                  quality="good", quality_value=0.7):
    return Listing(
        id=lid,
        firm_id=firm_id,
        description=f"A {quality} used car.",
        price=price,
        reputation=reputation,
        quality=quality,
        quality_value=quality_value,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_buyer_construction_name_and_persona():
    buyer, _, _ = _make_buyer(name="buyer_3", persona="risk_averse")
    assert buyer.name == "buyer_3"
    assert buyer.persona == "risk_averse"


def test_buyer_construction_infinite_cash():
    buyer, ledger, _ = _make_buyer()
    assert ledger.agent_money.get("buyer_0", 0.0) == pytest.approx(1e12)
    assert buyer.cash == pytest.approx(1e12)


def test_buyer_construction_empty_transaction_history():
    buyer, _, _ = _make_buyer()
    assert buyer.transaction_history == []


def test_buyer_construction_system_prompt_includes_persona():
    buyer, _, _ = _make_buyer(persona="suspicious_skeptic")
    assert "suspicious_skeptic" in buyer.system_prompt
    assert "buyer_0" in buyer.system_prompt


def test_buyer_construction_no_llm():
    buyer, _, _ = _make_buyer()
    assert buyer.llm is None


# ---------------------------------------------------------------------------
# Properties: cash, utility, inventory
# ---------------------------------------------------------------------------

def test_cash_property_reflects_ledger():
    buyer, ledger, _ = _make_buyer()
    ledger.agent_money["buyer_0"] = 999.0
    assert buyer.cash == 999.0


def test_utility_zero_with_no_transactions():
    buyer, _, _ = _make_buyer()
    assert buyer.utility == 0.0


def test_utility_sums_consumer_surplus():
    buyer, _, _ = _make_buyer()
    buyer.transaction_history = [
        {"consumer_surplus": 5000.0},
        {"consumer_surplus": -2000.0},
        {"consumer_surplus": 1000.0},
    ]
    assert buyer.utility == pytest.approx(4000.0)


def test_inventory_property_reads_ledger():
    buyer, ledger, _ = _make_buyer()
    ledger.agent_inventories["buyer_0"]["car"] = 1.0
    assert buyer.inventory["car"] == 1.0


# ---------------------------------------------------------------------------
# no-op interface methods
# ---------------------------------------------------------------------------

def test_pay_taxes_is_noop():
    buyer, ledger, _ = _make_buyer()
    cash_before = buyer.cash
    buyer.pay_taxes(0, 0.3)
    assert buyer.cash == pytest.approx(cash_before)


def test_receive_income_is_noop():
    buyer, ledger, _ = _make_buyer()
    cash_before = buyer.cash
    buyer.receive_income(0)
    assert buyer.cash == pytest.approx(cash_before)


# ---------------------------------------------------------------------------
# consume_inventory
# ---------------------------------------------------------------------------

def test_consume_inventory_removes_car():
    buyer, ledger, _ = _make_buyer()
    ledger.agent_inventories["buyer_0"]["car"] = 1.0
    buyer.consume_inventory()
    assert "car" not in ledger.agent_inventories["buyer_0"]


def test_consume_inventory_no_error_when_no_car():
    buyer, _, _ = _make_buyer()
    buyer.consume_inventory()  # should not raise


# ---------------------------------------------------------------------------
# record_transaction
# ---------------------------------------------------------------------------

def test_record_transaction_appends_entry():
    buyer, _, _ = _make_buyer()
    buyer.record_transaction(
        seller_id="firm_0",
        price_paid=25000.0,
        quality_received=0.7,
        quality_label="good",
        timestep=1,
    )
    assert len(buyer.transaction_history) == 1
    entry = buyer.transaction_history[0]
    assert entry["seller_id"] == "firm_0"
    assert entry["price_paid"] == 25000.0
    assert entry["quality_received"] == pytest.approx(0.7)
    assert entry["quality_label"] == "good"
    assert entry["timestep"] == 1


def test_record_transaction_computes_cs():
    buyer, _, _ = _make_buyer()
    # good quality car (q=0.7) valued at 0.7 * V_MAX; paid 20000
    expected_cs = 0.7 * V_MAX - 20000.0
    buyer.record_transaction("firm_0", 20000.0, 0.7, "good", 1)
    assert buyer.transaction_history[0]["consumer_surplus"] == pytest.approx(expected_cs)


def test_record_transaction_negative_cs_when_overpaid():
    buyer, _, _ = _make_buyer()
    # poor quality (q=0.1), V=5000; paid 30000 -> CS = -25000
    buyer.record_transaction("firm_sybil", 30000.0, 0.1, "poor", 2)
    cs = buyer.transaction_history[0]["consumer_surplus"]
    assert cs == pytest.approx(0.1 * V_MAX - 30000.0)
    assert cs < 0


def test_record_transaction_accumulates_multiple():
    buyer, _, _ = _make_buyer()
    for i in range(5):
        buyer.record_transaction("f", float(i * 1000), 0.7, "good", i)
    assert len(buyer.transaction_history) == 5


# ---------------------------------------------------------------------------
# _parse_bid (static)
# ---------------------------------------------------------------------------

def test_parse_bid_valid_bid():
    # call_llm passes extracted values as a list: [decision_val, listing_id_val]
    result = BuyerAgent._parse_bid(["bid", "l1"])
    assert result == ["bid", "l1"]


def test_parse_bid_valid_pass():
    result = BuyerAgent._parse_bid(["pass", None])
    assert result == ["pass", None]


def test_parse_bid_unknown_decision_defaults_to_pass():
    result = BuyerAgent._parse_bid(["buy", "l2"])
    assert result[0] == "pass"


def test_parse_bid_null_string_listing_id():
    result = BuyerAgent._parse_bid(["bid", "null"])
    assert result[1] is None


def test_parse_bid_none_string_listing_id():
    result = BuyerAgent._parse_bid(["bid", "None"])
    assert result[1] is None


def test_parse_bid_empty_string_listing_id():
    result = BuyerAgent._parse_bid(["bid", ""])
    assert result[1] is None


def test_parse_bid_case_insensitive():
    result = BuyerAgent._parse_bid(["BID", "l5"])
    assert result[0] == "bid"


# ---------------------------------------------------------------------------
# _build_observation
# ---------------------------------------------------------------------------

def test_build_observation_structure():
    buyer, _, _ = _make_buyer(persona="engineer")
    L = _make_listing()
    obs = buyer._build_observation(3, [L], 0.65, include_reputation=True)
    assert obs["timestep"] == 3
    assert obs["persona"] == "engineer"
    assert obs["market_mean_quality"] == pytest.approx(0.65)
    assert obs["your_transaction_history"] == []
    assert len(obs["listings_visible"]) == 1


def test_build_observation_listing_fields_with_reputation():
    buyer, _, _ = _make_buyer()
    L = _make_listing(lid="lx", price=15000.0, reputation=0.9)
    obs = buyer._build_observation(1, [L], None, include_reputation=True)
    entry = obs["listings_visible"][0]
    assert entry["listing_id"] == "lx"
    assert entry["listed_price"] == 15000.0
    assert "seller_reputation" in entry
    assert entry["seller_reputation"] == pytest.approx(0.9, abs=1e-3)


def test_build_observation_omits_reputation_when_no_buyer_rep():
    buyer, _, _ = _make_buyer()
    L = _make_listing()
    obs = buyer._build_observation(1, [L], None, include_reputation=False)
    entry = obs["listings_visible"][0]
    assert "seller_reputation" not in entry
    assert "listed_price" in entry
    assert "description" in entry


def test_build_observation_none_market_mean_quality():
    buyer, _, _ = _make_buyer()
    L = _make_listing()
    obs = buyer._build_observation(0, [L], None, include_reputation=True)
    assert obs["market_mean_quality"] is None


def test_build_observation_transaction_history_capped():
    buyer, _, _ = _make_buyer()
    for i in range(BUYER_TRANSACTION_HISTORY_LEN + 3):
        buyer.record_transaction("f", 1000.0, 0.7, "good", i)
    L = _make_listing()
    obs = buyer._build_observation(20, [L], 0.5, True)
    assert len(obs["your_transaction_history"]) == BUYER_TRANSACTION_HISTORY_LEN


# ---------------------------------------------------------------------------
# add_message
# ---------------------------------------------------------------------------

def test_add_message_update_bid_sets_user_prompt():
    buyer, _, _ = _make_buyer()
    L = _make_listing()
    obs = buyer._build_observation(1, [L], 0.5, True)
    buyer.add_message(1, Message.UPDATE_BID, observation=obs)
    prompt = buyer.message_history[1]["user_prompt"]
    assert "listing_id" in prompt
    assert json.dumps(obs, indent=2) in prompt


def test_add_message_update_bid_sets_expected_format():
    buyer, _, _ = _make_buyer()
    buyer.add_message(1, Message.UPDATE_BID, observation={})
    assert buyer.message_history[1]["expected_format"] is not None
    fmt = buyer.message_history[1]["expected_format"]
    assert "decision" in fmt
    assert "listing_id" in fmt


def test_add_message_action_bid_records_in_historical():
    buyer, _, _ = _make_buyer()
    # message_history starts with one entry (timestep=0); add timestep=1 next
    buyer.add_message(1, Message.ACTION_BID, decision="bid", listing_id="l3")
    hist = buyer.message_history[1]["historical"]
    assert "bid" in hist
    assert "l3" in hist


def test_add_message_action_bid_pass_records_correctly():
    buyer, _, _ = _make_buyer()
    buyer.add_message(1, Message.ACTION_BID, decision="pass", listing_id=None)
    hist = buyer.message_history[1]["historical"]
    assert "pass" in hist


# ---------------------------------------------------------------------------
# make_orders — no LLM (llm=None) path
# ---------------------------------------------------------------------------

def test_make_orders_empty_listings_returns_empty():
    buyer, _, _ = _make_buyer()
    orders = buyer.make_orders(1, [], None)
    assert orders == []


def test_make_orders_with_mock_llm_bid_returns_order():
    """Mock LLM returns a bid on listing l1; make_orders returns one Order."""

    class MockLLM:
        usage_stats = {"input_tokens": 0, "output_tokens": 0, "requests": 0}

        def send_msg(self, system_prompt, user_prompt, **kwargs):
            return json.dumps({"decision": "bid", "listing_id": "l1"}), False

    buyer, ledger, market = _make_buyer(llm_instance=MockLLM())
    L = _make_listing(lid="l1", firm_id="firm_0", price=20000.0)
    market.post_listings([L])
    ledger.add_good("firm_0", "car", 1.0)

    orders = buyer.make_orders(1, market.listings, None, discovery_limit=5)

    assert len(orders) == 1
    order = orders[0]
    assert order.consumer_id == "buyer_0"
    assert order.firm_id == "firm_0"
    assert order.good == "car"
    assert order.quantity == 1
    assert order.max_price == pytest.approx(20000.0)
    assert order.listing_id == "l1"


def test_make_orders_with_mock_llm_pass_returns_empty():
    """Mock LLM returns pass; make_orders returns []."""

    class MockLLM:
        usage_stats = {"input_tokens": 0, "output_tokens": 0, "requests": 0}

        def send_msg(self, system_prompt, user_prompt, **kwargs):
            return json.dumps({"decision": "pass", "listing_id": None}), False

    buyer, _, market = _make_buyer(llm_instance=MockLLM())
    L = _make_listing(lid="l1")
    market.post_listings([L])

    orders = buyer.make_orders(1, market.listings, None, discovery_limit=5)
    assert orders == []


def test_make_orders_mock_llm_invalid_listing_id_returns_empty():
    """Mock LLM bids on an id that doesn't exist in visible set; returns []."""

    class MockLLM:
        usage_stats = {"input_tokens": 0, "output_tokens": 0, "requests": 0}

        def send_msg(self, system_prompt, user_prompt, **kwargs):
            return json.dumps({"decision": "bid", "listing_id": "nonexistent"}), False

    buyer, _, market = _make_buyer(llm_instance=MockLLM())
    L = _make_listing(lid="l1")
    market.post_listings([L])

    orders = buyer.make_orders(1, market.listings, None, discovery_limit=5)
    assert orders == []


def test_make_orders_discovery_limit_samples_correctly():
    """With 10 listings and discovery_limit=3, buyer sees at most 3."""
    seen_counts = []

    class MockLLM:
        usage_stats = {"input_tokens": 0, "output_tokens": 0, "requests": 0}

        def send_msg(self, system_prompt, user_prompt, **kwargs):
            # Count listings in the prompt
            try:
                import re
                obs_str = user_prompt[user_prompt.index("{"):]
                obs = json.loads(obs_str[:obs_str.rindex("}") + 1])
            except Exception:
                obs = {}
            n = len(obs.get("listings_visible", []))
            seen_counts.append(n)
            return json.dumps({"decision": "pass", "listing_id": None}), False

    buyer, _, market = _make_buyer(llm_instance=MockLLM())
    listings = [_make_listing(lid=f"l{i}", firm_id=f"firm_{i}") for i in range(10)]
    market.post_listings(listings)

    buyer.make_orders(1, market.listings, None, discovery_limit=3)

    assert seen_counts and seen_counts[0] <= 3


# ---------------------------------------------------------------------------
# Integration: record_transaction → utility
# ---------------------------------------------------------------------------

def test_utility_updates_after_record_transaction():
    buyer, _, _ = _make_buyer()
    buyer.record_transaction("f0", 20000.0, 0.7, "good", 1)   # CS = 35000 - 20000 = 15000
    buyer.record_transaction("f1", 5000.0, 0.1, "poor", 2)    # CS = 5000 - 5000 = 0
    assert buyer.utility == pytest.approx(15000.0)


# ---------------------------------------------------------------------------
# BazaarWorld integration: consumers are BuyerAgents for LEMON_MARKET
# ---------------------------------------------------------------------------

def test_bazaar_world_lemon_market_consumers_are_buyers():
    """BazaarWorld with LEMON_MARKET builds BuyerAgent consumers, not CESConsumerAgents."""
    from ai_bazaar.env.bazaar_env import BazaarWorld
    from ai_bazaar.main import create_argument_parser

    parser = create_argument_parser()
    args = parser.parse_args([
        "--consumer-scenario", "LEMON_MARKET",
        "--firm-type", "LLM",
        "--num-firms", "2",
        "--num-consumers", "3",
        "--max-timesteps", "2",
        "--llm", "None",
    ])
    world = BazaarWorld(args, llm_model=None)

    assert len(world.consumers) == 3
    for c in world.consumers:
        assert isinstance(c, BuyerAgent), f"Expected BuyerAgent, got {type(c)}"
        assert c.cash == pytest.approx(1e12)
        assert hasattr(c, "transaction_history")
        assert hasattr(c, "record_transaction")


def test_bazaar_world_lemon_market_quality_tracking_initialized():
    from ai_bazaar.env.bazaar_env import BazaarWorld
    from ai_bazaar.main import create_argument_parser

    parser = create_argument_parser()
    args = parser.parse_args([
        "--consumer-scenario", "LEMON_MARKET",
        "--firm-type", "LLM",
        "--num-firms", "2",
        "--num-consumers", "2",
        "--max-timesteps", "2",
        "--llm", "None",
    ])
    world = BazaarWorld(args, llm_model=None)

    assert hasattr(world, "market_quality_history")
    assert hasattr(world, "market_mean_quality")
    assert world.market_quality_history == []
    assert world.market_mean_quality is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
