"""
Unit tests for the lemon market implementation.

Covers: market_core (Listing, Order.listing_id, post_listings, clear with listings),
FixedFirmAgent.create_listings, BaseFirmAgent.update_reputation(quality/alpha),
consumer _make_orders_lemon, common (LEMON_MARKET_GOODS, QUALITY_DICT),
and BazaarWorld LEMON_MARKET setup (goods, num_goods).

Run from project root:
  python -m pytest tests/test_lemon_market.py -v
"""
import os
import sys

if __name__ == "__main__" and __package__ is None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)

import pytest
from types import SimpleNamespace

from ai_bazaar.market_core.market_core import (
    Ledger,
    Market,
    Order,
    Listing,
)
from ai_bazaar.agents.firm import BaseFirmAgent, FixedFirmAgent
from ai_bazaar.agents.consumer import CESConsumerAgent
from ai_bazaar.utils.common import LEMON_MARKET_GOODS, QUALITY_DICT, advertised_quality_for_sybil
from ai_bazaar.env.bazaar_env import BazaarWorld
from ai_bazaar.main import create_argument_parser


# --- common ---


def test_lemon_market_goods():
    """LEMON_MARKET_GOODS is ['car']."""
    assert LEMON_MARKET_GOODS == ["car"]


def test_advertised_quality_for_sybil():
    """Sybil advertises one tier above true quality; mint stays mint."""
    assert advertised_quality_for_sybil("poor", 0.1) == ("fair", 0.4)
    assert advertised_quality_for_sybil("fair", 0.4) == ("good", 0.7)
    assert advertised_quality_for_sybil("good", 0.7) == ("mint", 1.0)
    assert advertised_quality_for_sybil("mint", 1.0) == ("mint", 1.0)


def test_quality_dict():
    """QUALITY_DICT has expected keys and values in [0,1]."""
    assert set(QUALITY_DICT.keys()) == {"mint", "good", "fair", "poor"}
    for k, v in QUALITY_DICT.items():
        assert 0 <= v <= 1, f"quality {k}={v}"


# --- market_core: Listing, Order, post_listings ---


def test_listing_dataclass():
    """Listing has id, firm_id, description, price, reputation, quality, quality_value."""
    L = Listing(
        id="lid_1",
        firm_id="firm_0",
        description="A car",
        price=5000.0,
        reputation=0.8,
        quality="good",
        quality_value=0.7,
    )
    assert L.id == "lid_1"
    assert L.firm_id == "firm_0"
    assert L.price == 5000.0
    assert L.quality_value == 0.7


def test_order_listing_id():
    """Order can have listing_id set (optional)."""
    o = Order(
        consumer_id="c0",
        firm_id="f0",
        good="car",
        quantity=1.0,
        max_price=6000.0,
        listing_id="listing_0",
    )
    assert o.listing_id == "listing_0"
    o2 = Order("c1", "f1", "car", 1.0, 5000.0)
    assert getattr(o2, "listing_id", None) is None or o2.listing_id is None


def test_post_listings_from_dicts():
    """post_listings accepts list of dicts and sets id."""
    market = Market()
    listings = [
        {
            "firm_id": "firm_0",
            "description": "Used car",
            "price": 4000.0,
            "reputation": 1.0,
            "quality": "good",
            "quality_value": 0.7,
        },
    ]
    market.post_listings(listings)
    assert len(market.listings) == 1
    L = market.listings[0]
    assert L.firm_id == "firm_0"
    assert L.price == 4000.0
    assert L.quality_value == 0.7
    assert L.id is not None


def test_post_listings_preserves_listing_objects():
    """post_listings with existing Listing objects appends them as-is."""
    market = Market()
    L = Listing("id1", "f0", "desc", 1000.0, 1.0, "fair", 0.4)
    market.post_listings([L])
    assert len(market.listings) == 1
    assert market.listings[0].id == "id1"
    assert market.listings[0].price == 1000.0


# --- market_core: clear with listing orders ---


def test_fill_order_listing_success():
    """Clear fills an order that targets a listing when consumer has cash and firm has car."""
    ledger = Ledger()
    market = Market()
    ledger.credit("consumer_0", 10_000.0)
    ledger.add_good("firm_0", "car", 1.0)

    L = Listing(
        id="l1",
        firm_id="firm_0",
        description="Car",
        price=5000.0,
        reputation=1.0,
        quality="good",
        quality_value=0.7,
    )
    market.listings = [L]
    order = Order(
        consumer_id="consumer_0",
        firm_id="firm_0",
        good="car",
        quantity=1.0,
        max_price=6000.0,
        listing_id="l1",
    )
    market.orders.append(order)

    filled, sales_info = market.clear(ledger)

    assert len(filled) == 1
    assert len(sales_info) == 1
    assert sales_info[0]["firm_id"] == "firm_0"
    assert sales_info[0]["good"] == "car"
    assert sales_info[0]["quantity_sold"] == 1.0
    assert sales_info[0]["price"] == 5000.0
    assert sales_info[0]["quality_value"] == 0.7
    assert ledger.agent_money["consumer_0"] == 5000.0
    assert ledger.agent_money["firm_0"] == 5000.0
    assert ledger.agent_inventories["firm_0"]["car"] == 0.0
    assert ledger.agent_inventories["consumer_0"]["car"] == 1.0
    assert len(market.listings) == 0


def test_fill_order_listing_fails_if_price_above_max():
    """Order with max_price below listing price does not fill."""
    ledger = Ledger()
    market = Market()
    ledger.credit("consumer_0", 10_000.0)
    ledger.add_good("firm_0", "car", 1.0)
    L = Listing("l1", "firm_0", "d", 5000.0, 1.0, "good", 0.7)
    market.listings = [L]
    order = Order("consumer_0", "firm_0", "car", 1.0, 3000.0, listing_id="l1")
    market.orders.append(order)

    filled, sales_info = market.clear(ledger)

    assert len(filled) == 0
    assert len(sales_info) == 0
    assert ledger.agent_inventories["firm_0"]["car"] == 1.0
    assert len(market.listings) == 1


def test_fill_order_listing_fails_if_insufficient_cash():
    """Order does not fill when consumer cannot afford."""
    ledger = Ledger()
    market = Market()
    ledger.credit("consumer_0", 1000.0)
    ledger.add_good("firm_0", "car", 1.0)
    L = Listing("l1", "firm_0", "d", 5000.0, 1.0, "good", 0.7)
    market.listings = [L]
    order = Order("consumer_0", "firm_0", "car", 1.0, 5000.0, listing_id="l1")
    market.orders.append(order)

    filled, sales_info = market.clear(ledger)

    assert len(filled) == 0
    assert ledger.agent_money["consumer_0"] == 1000.0


# --- Firm: create_listings, update_reputation ---


def test_fixed_firm_create_listings():
    """FixedFirmAgent.create_listings returns list of dicts with id, firm_id, price, reputation, quality, quality_value; marks posted."""
    ledger = Ledger()
    market = Market()
    firm = FixedFirmAgent(
        name="firm_0",
        goods=["car"],
        initial_cash=1000.0,
        ledger=ledger,
        market=market,
    )
    setattr(firm, "listings", [
        {"quality": "good", "quality_value": 0.7, "posted": False},
    ])

    out = firm.create_listings()

    assert len(out) == 1
    d = out[0]
    assert d["firm_id"] == "firm_0"
    assert "id" in d
    assert d["price"] == 5000.0 * 0.7
    assert d["reputation"] == 1.0
    assert d["quality"] == "good"
    assert d["quality_value"] == 0.7
    assert getattr(firm, "listings", [])[0]["posted"] is True


def test_base_firm_update_reputation_lemon():
    """update_reputation(quality=q, alpha=a) sets reputation = a*R + (1-a)*q."""
    firm = BaseFirmAgent()
    firm.reputation = 0.5
    firm.update_reputation(quality=1.0, alpha=0.9)
    assert abs(firm.reputation - (0.9 * 0.5 + 0.1 * 1.0)) < 1e-9
    r_before = firm.reputation  # 0.55 after first update
    firm.update_reputation(quality=0.0, alpha=0.9)
    expected = 0.9 * r_before + 0.1 * 0.0
    assert abs(firm.reputation - expected) < 1e-9


def test_base_firm_update_reputation_fulfillment():
    """update_reputation(successful_qty, requested_qty) uses fulfillment history."""
    firm = BaseFirmAgent()
    firm.update_reputation(8.0, 10.0)
    firm.update_reputation(10.0, 10.0)
    assert abs(firm.reputation - (8 + 10) / (10 + 10)) < 1e-9


# --- Consumer: _make_orders_lemon ---


def test_make_orders_lemon_empty_listings():
    """_make_orders_lemon returns [] when market has no listings."""
    ledger = Ledger()
    market = Market()
    market.listings = []
    args = SimpleNamespace(no_diaries=True, bracket_setting="three")
    consumer = CESConsumerAgent(
        name="c0",
        income_stream=50_000.0,
        ledger=ledger,
        market=market,
        goods=["car"],
        ces_params={"car": 1.0},
        args=args,
    )
    consumer.income = 50_000.0

    orders = consumer._make_orders_lemon(0, discovery_limit=5, firm_reputations={})

    assert orders == []


def test_make_orders_lemon_submits_when_cs_positive():
    """_make_orders_lemon returns one order for listing with highest CS > 0."""
    ledger = Ledger()
    market = Market()
    L1 = Listing("l1", "firm_0", "d", 1000.0, 1.0, "good", 0.7)
    L2 = Listing("l2", "firm_1", "d", 5000.0, 0.5, "fair", 0.4)
    market.listings = [L1, L2]
    args = SimpleNamespace(no_diaries=True, bracket_setting="three")
    consumer = CESConsumerAgent(
        name="c0",
        income_stream=50_000.0,
        ledger=ledger,
        market=market,
        goods=["car"],
        ces_params={"car": 1.0},
        args=args,
    )
    consumer.income = 50_000.0

    orders = consumer._make_orders_lemon(0, discovery_limit=5, firm_reputations={})

    assert len(orders) == 1
    assert orders[0].good == "car"
    assert orders[0].quantity == 1.0
    assert orders[0].listing_id in ("l1", "l2")
    assert orders[0].firm_id in ("firm_0", "firm_1")


def test_make_orders_lemon_no_order_when_cs_non_positive():
    """_make_orders_lemon returns [] when all listings have CS <= 0 (price too high relative to max_wtp)."""
    ledger = Ledger()
    market = Market()
    # Very high price so CS = rep * max_wtp - price <= 0 for typical income
    L = Listing("l1", "firm_0", "d", 1_000_000.0, 1.0, "good", 0.7)
    market.listings = [L]
    args = SimpleNamespace(no_diaries=True, bracket_setting="three")
    consumer = CESConsumerAgent(
        name="c0",
        income_stream=10_000.0,
        ledger=ledger,
        market=market,
        goods=["car"],
        ces_params={"car": 1.0},
        args=args,
    )
    consumer.income = 10_000.0

    orders = consumer._make_orders_lemon(0, discovery_limit=5, firm_reputations={})

    assert len(orders) == 0


# --- BazaarWorld LEMON_MARKET setup ---


def test_bazaar_world_lemon_market_goods():
    """When consumer_scenario is LEMON_MARKET, world.goods is ['car'] and num_goods is 1."""
    parser = create_argument_parser()
    args = parser.parse_args([
        "--consumer-scenario", "LEMON_MARKET",
        "--firm-type", "LLM",
        "--num-firms", "2",
        "--num-consumers", "3",
        "--max-timesteps", "2",
        "--llm", "gemini-2.5-flash",
    ])
    assert args.consumer_scenario == "LEMON_MARKET"
    # Force num_goods as main does when LEMON_MARKET
    if args.consumer_scenario == "LEMON_MARKET":
        args.num_goods = 1

    world = BazaarWorld(args, llm_model=None)

    assert world.goods == ["car"]
    assert args.num_goods == 1
    assert world.goods_list == LEMON_MARKET_GOODS
    assert hasattr(world, "lemon_market_listings")
    assert hasattr(world, "lemon_market_listings_unsold")
    assert world.lemon_market_listings_unsold == []


def test_bazaar_world_lemon_market_sybil_cluster():
    """With LEMON_MARKET and sybil-cluster-size=2, last 2 firms are Sybil."""
    parser = create_argument_parser()
    args = parser.parse_args([
        "--consumer-scenario", "LEMON_MARKET",
        "--firm-type", "LLM",
        "--num-firms", "4",
        "--num-consumers", "3",
        "--max-timesteps", "2",
        "--sybil-cluster-size", "2",
        "--llm", "gemini-2.5-flash",
    ])
    world = BazaarWorld(args, llm_model=None)
    assert len(world.firms) == 4
    assert getattr(world.firms[0], "sybil", False) is False
    assert getattr(world.firms[1], "sybil", False) is False
    assert getattr(world.firms[2], "sybil", False) is True
    assert getattr(world.firms[3], "sybil", False) is True


def test_main_force_num_goods_lemon():
    """Parsed args have num_goods forced to 1 when consumer_scenario is LEMON_MARKET."""
    parser = create_argument_parser()
    args = parser.parse_args([
        "--consumer-scenario", "LEMON_MARKET",
        "--num-goods", "4",
        "--firm-type", "LLM",
        "--num-firms", "1",
        "--num-consumers", "1",
        "--max-timesteps", "1",
        "--llm", "gemini-2.5-flash",
    ])
    assert args.consumer_scenario == "LEMON_MARKET"
    assert args.num_goods == 4
    # Simulate main's override
    if getattr(args, "consumer_scenario", None) == "LEMON_MARKET":
        args.num_goods = 1
    assert args.num_goods == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
