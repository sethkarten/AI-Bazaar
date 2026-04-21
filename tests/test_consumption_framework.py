"""
Test the consumption/transaction framework with 4 goods.

Uses FixedFirmAgent and FixedConsumerAgent only (no LLM). Verifies that:
- Market clearing records transactions on the ledger with timestep, issuer, payer, good, quantity.
- Both firm and consumer see the same transaction in their history.
- Good durability framework: consume_inventory uses transaction history and GOODS_CONSUMPTION_RATE (FIFO, rate cap).
"""
import os
import sys

if __name__ == "__main__" and __package__ is None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)

import pytest

from ai_bazaar.market_core.market_core import (
    Ledger,
    Market,
    Order,
    Quote,
    TransactionRecord,
)
from ai_bazaar.agents.firm import FixedFirmAgent
from ai_bazaar.agents.consumer import FixedConsumerAgent
from ai_bazaar.utils.common import GOODS_CONSUMPTION_RATE


GOODS_4 = ["food", "clothing", "electronics", "furniture"]


def test_consumption_framework_four_goods():
    """Run one full cycle: firm produces 4 goods, consumer buys, assert transaction records."""
    ledger = Ledger()
    market = Market()
    timestep = 0

    # One firm, 4 goods; unit_costs + markup = 12.0 per good
    firm = FixedFirmAgent(
        name="firm_0",
        goods=GOODS_4,
        initial_cash=2000.0,
        ledger=ledger,
        market=market,
        unit_costs={g: 1.0 for g in GOODS_4},
        markup=0.50,
    )

    # One consumer, 4 goods
    consumer = FixedConsumerAgent(
        name="consumer_0",
        income_stream=500.0,
        ledger=ledger,
        market=market,
        goods=GOODS_4,
        quantity_per_good=5.0,
    )

    # No transactions before any trade
    print("Assert no transactions for firm_0 before trade")
    assert ledger.agent_transactions.get("firm_0", []) == []
    print("Assert no transactions for consumer_0 before trade")
    assert ledger.agent_transactions.get("consumer_0", []) == []
    print("Assert consumer.transactions is empty before trade")
    assert consumer.transactions == []
    print("Assert firm.transactions is empty before trade")
    assert firm.transactions == []

    # Consumer receives income
    consumer.receive_income(timestep=timestep)
    print("Assert consumer.cash == 500.0 after income")
    assert consumer.cash == 500.0

    # Firm: buy supply, produce all 4 goods, post quotes
    firm.purchase_supplies(quantity_to_purchase=40.0, unit_price=10.0, timestep=timestep)
    firm.produce_goods(timestep=timestep)
    # 40 supply / 4 goods = 10 each
    for g in GOODS_4:
        print(f"Assert firm.inventory[{g!r}] == 10.0")
        assert firm.inventory[g] == 10.0

    prices = firm.set_price(timestep=timestep)
    firm.post_quotes(prices)
    print("Assert len(market.quotes) == 4")
    assert len(market.quotes) == 4

    # Consumer makes and submits orders for all 4 goods
    orders = consumer.make_orders(timestep=timestep)
    print("Assert len(orders) == 4")
    assert len(orders) == 4
    consumer.submit_orders(orders)
    print("Assert len(market.orders) == 4")
    assert len(market.orders) == 4

    # Clear market with timestep so transactions are recorded
    filled_orders, sales_info = market.clear(ledger, timestep=timestep)

    # All 4 orders should fill (consumer has 500, each order 5*12=60, total 240)
    print("Assert len(filled_orders) == 4")
    assert len(filled_orders) == 4
    print("Assert len(sales_info) == 4")
    assert len(sales_info) == 4

    # --- Transaction framework assertions ---
    print("Assert firm_0 in ledger.agent_transactions")
    assert "firm_0" in ledger.agent_transactions
    print("Assert consumer_0 in ledger.agent_transactions")
    assert "consumer_0" in ledger.agent_transactions

    firm_tx = ledger.agent_transactions["firm_0"]
    consumer_tx = ledger.agent_transactions["consumer_0"]
    print("Assert len(firm_tx) == 4")
    assert len(firm_tx) == 4
    print("Assert len(consumer_tx) == 4")
    assert len(consumer_tx) == 4

    # Same records appear for both (by design)
    for i in range(4):
        r = firm_tx[i]
        print(f"Assert firm_tx[{i}] is TransactionRecord")
        assert isinstance(r, TransactionRecord)
        print(f"Assert r.timestep == timestep")
        assert r.timestep == timestep
        print("Assert r.issuer == firm_0")
        assert r.issuer == "firm_0"
        print("Assert r.payer == consumer_0")
        assert r.payer == "consumer_0"
        print("Assert r.good in GOODS_4")
        assert r.good in GOODS_4
        print("Assert r.quantity == 5.0")
        assert r.quantity == 5.0
        print(f"Assert consumer_tx[{i}] matches firm_tx[{i}] (timestep)")
        assert consumer_tx[i].timestep == r.timestep
        print(f"Assert consumer_tx[{i}] matches (issuer)")
        assert consumer_tx[i].issuer == r.issuer
        print(f"Assert consumer_tx[{i}] matches (payer)")
        assert consumer_tx[i].payer == r.payer
        print(f"Assert consumer_tx[{i}] matches (good)")
        assert consumer_tx[i].good == r.good
        print(f"Assert consumer_tx[{i}] matches (quantity)")
        assert consumer_tx[i].quantity == r.quantity

    # Agent properties mirror ledger
    print("Assert firm.transactions == firm_tx")
    assert firm.transactions == firm_tx
    print("Assert consumer.transactions == consumer_tx")
    assert consumer.transactions == consumer_tx

    # Goods covered: one transaction per good
    goods_in_tx = {r.good for r in firm_tx}
    print("Assert goods_in_tx == set(GOODS_4)")
    assert goods_in_tx == set(GOODS_4)

    # Ledger copy preserves transactions
    ledger2 = ledger.copy()
    print("Assert ledger2 has 4 transactions for firm_0")
    assert len(ledger2.agent_transactions["firm_0"]) == 4
    print("Assert ledger2 has 4 transactions for consumer_0")
    assert len(ledger2.agent_transactions["consumer_0"]) == 4


def test_consumption_framework_two_consumers():
    """Two consumers buy from one firm; each sees only their own transactions in agent_transactions."""
    ledger = Ledger()
    market = Market()
    timestep = 0

    firm = FixedFirmAgent(
        name="firm_0",
        goods=GOODS_4,
        initial_cash=2000.0,
        ledger=ledger,
        market=market,
        unit_costs={g: 1.0 for g in GOODS_4},
        markup=0.50,
    )
    consumer_a = FixedConsumerAgent(
        name="consumer_a",
        income_stream=300.0,
        ledger=ledger,
        market=market,
        goods=GOODS_4,
        quantity_per_good=2.0,
    )
    consumer_b = FixedConsumerAgent(
        name="consumer_b",
        income_stream=300.0,
        ledger=ledger,
        market=market,
        goods=GOODS_4,
        quantity_per_good=2.0,
    )

    for c in (consumer_a, consumer_b):
        c.receive_income(timestep=timestep)

    firm.purchase_supplies(quantity_to_purchase=80.0, unit_price=10.0, timestep=timestep)
    firm.produce_goods(timestep=timestep)
    firm.post_quotes(firm.set_price(timestep=timestep))

    orders_a = consumer_a.make_orders(timestep=timestep)
    orders_b = consumer_b.make_orders(timestep=timestep)
    consumer_a.submit_orders(orders_a)
    consumer_b.submit_orders(orders_b)

    filled, _ = market.clear(ledger, timestep=timestep)

    # 4 orders per consumer = 8 filled
    print("Assert len(filled) == 8")
    assert len(filled) == 8

    # Firm sees all 8 transactions
    print("Assert firm_0 has 8 transactions")
    assert len(ledger.agent_transactions["firm_0"]) == 8
    # Each consumer sees 4 (their own)
    print("Assert consumer_a has 4 transactions")
    assert len(ledger.agent_transactions["consumer_a"]) == 4
    print("Assert consumer_b has 4 transactions")
    assert len(ledger.agent_transactions["consumer_b"]) == 4

    for r in ledger.agent_transactions["consumer_a"]:
        print("Assert consumer_a tx: payer == consumer_a")
        assert r.payer == "consumer_a"
        print("Assert consumer_a tx: issuer == firm_0")
        assert r.issuer == "firm_0"
        print("Assert consumer_a tx: good in GOODS_4")
        assert r.good in GOODS_4
        print("Assert consumer_a tx: quantity == 2.0")
        assert r.quantity == 2.0

    for r in ledger.agent_transactions["consumer_b"]:
        print("Assert consumer_b tx: payer == consumer_b")
        assert r.payer == "consumer_b"
        print("Assert consumer_b tx: issuer == firm_0")
        assert r.issuer == "firm_0"


# --- Durability framework tests ---


def test_durability_consumption_rates():
    """After a trade, consume_inventory drains food/clothing up to rate; electronics/furniture (rate 0) unchanged."""
    ledger = Ledger()
    market = Market()
    timestep = 0
    firm = FixedFirmAgent(
        name="firm_0",
        goods=GOODS_4,
        initial_cash=2000.0,
        ledger=ledger,
        market=market,
        unit_costs={g: 1.0 for g in GOODS_4},
        markup=0.50,
    )
    consumer = FixedConsumerAgent(
        name="consumer_0",
        income_stream=500.0,
        ledger=ledger,
        market=market,
        goods=GOODS_4,
        quantity_per_good=5.0,
    )
    consumer.receive_income(timestep=timestep)
    firm.purchase_supplies(quantity_to_purchase=40.0, unit_price=10.0, timestep=timestep)
    firm.produce_goods(timestep=timestep)
    firm.post_quotes(firm.set_price(timestep=timestep))
    orders = consumer.make_orders(timestep=timestep)
    consumer.submit_orders(orders)
    market.clear(ledger, timestep=timestep)
    # Consumer has 5 of each good; food rate 140, clothing 7, electronics/furniture 0
    print("Assert consumer has 5 of each good before consume_inventory")
    for g in GOODS_4:
        assert consumer.inventory[g] == 5.0
    consumer.consume_inventory(timestep)
    print("Assert food fully consumed (rate 140 >= 5)")
    assert consumer.inventory["food"] == 0.0
    print("Assert clothing fully consumed (rate 7 >= 5)")
    assert consumer.inventory["clothing"] == 0.0
    print("Assert electronics unchanged (rate 0)")
    assert consumer.inventory["electronics"] == 5.0
    print("Assert furniture unchanged (rate 0)")
    assert consumer.inventory["furniture"] == 5.0


def test_durability_capped_by_rate():
    """Consumption is capped by GOODS_CONSUMPTION_RATE; excess is carried over."""
    ledger = Ledger()
    market = Market()
    consumer = FixedConsumerAgent(
        name="consumer_0",
        income_stream=0.0,
        ledger=ledger,
        market=market,
        goods=GOODS_4,
        quantity_per_good=1.0,
    )
    # Manually add 200 food (one batch at t=0) so we can test rate cap
    ledger.record_transaction(0, "firm_0", "consumer_0", "food", 200.0)
    ledger.add_good("consumer_0", "food", 200.0)
    print("Assert consumer has 200 food before consume_inventory")
    assert consumer.inventory["food"] == 200.0
    consumer.consume_inventory(0)
    # Food rate = 140; 60 should remain
    print("Assert only 140 food consumed, 60 remain")
    assert consumer.inventory["food"] == 60.0


def test_durability_fifo_batches():
    """Consumption uses FIFO by acquisition timestep; _consumed_from_batch tracks per-batch consumption."""
    ledger = Ledger()
    market = Market()
    consumer = FixedConsumerAgent(
        name="consumer_0",
        income_stream=0.0,
        ledger=ledger,
        market=market,
        goods=GOODS_4,
        quantity_per_good=1.0,
    )
    # Two batches: t=0 80 food, t=1 80 food. Rate 140 → consume 80 from first, 60 from second
    ledger.record_transaction(0, "firm_0", "consumer_0", "food", 80.0)
    ledger.record_transaction(1, "firm_0", "consumer_0", "food", 80.0)
    ledger.add_good("consumer_0", "food", 160.0)
    print("Assert consumer has 160 food before consume_inventory")
    assert consumer.inventory["food"] == 160.0
    consumer.consume_inventory(0)
    print("Assert 20 food remain (140 consumed: 80 + 60)")
    assert consumer.inventory["food"] == 20.0
    print("Assert batch t=0 fully consumed")
    assert consumer._consumed_from_batch[(0, "food")] == 80.0
    print("Assert batch t=1 partially consumed (60)")
    assert consumer._consumed_from_batch[(1, "food")] == 60.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
