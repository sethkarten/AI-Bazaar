# tests/test_fixed_market.py
"""
Integration test for FixedConsumerAgent and FixedFirmAgent interactions.
Tests buying and selling between firms and consumers in a market.
"""

import sys
import os

# Add the project root to the path
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)

# Import directly from modules to avoid package-level imports
sys.path.insert(0, os.path.join(project_root, 'ai_bazaar', 'market_core'))
from market_core import Ledger, Market, Order, Quote

sys.path.insert(0, os.path.join(project_root, 'ai_bazaar', 'agents'))
from firm import FixedFirmAgent
from consumer import FixedConsumerAgent


def test_fixed_consumer_agent():
    """Test FixedConsumerAgent basic functionality."""
    
    print("\nTESTING FIXED CONSUMER AGENT...\n")
    
    # Create ledger and market
    ledger = Ledger()
    market = Market()
    
    # Initialize consumer
    print("\nInitializing consumer...")
    consumer = FixedConsumerAgent(
        name="test_consumer",
        income_stream=100.0,
        ledger=ledger,
        market=market,
        goods=["widget", "gadget"]
    )
    
    # Test consumer initialization
    print("consumer.name", consumer.name)
    print("consumer.income", consumer.income)
    print("consumer.cash", consumer.cash)
    print("consumer.inventory", consumer.inventory)
    print("ledger.agent_money", ledger.agent_money)
    print("ledger.agent_inventories", ledger.agent_inventories)
    
    assert consumer.name == "test_consumer"
    assert consumer.income == 100.0
    assert consumer.cash == 0.0  # Starts with 0 cash
    assert consumer.inventory == {"widget": 0.0, "gadget": 0.0}
    assert ledger.agent_money["test_consumer"] == 0.0
    assert ledger.agent_inventories["test_consumer"] == {"widget": 0.0, "gadget": 0.0}
    # Verify that consumer.inventory is the same object as ledger inventory
    assert consumer.inventory is ledger.agent_inventories["test_consumer"]
    
    # Test income reception
    print("\nTesting income reception...")
    consumer.receive_income(timestep=0)
    print("consumer.cash after income", consumer.cash)
    print("ledger.agent_money after income", ledger.agent_money)
    
    assert consumer.cash == 100.0
    assert ledger.agent_money["test_consumer"] == 100.0
    
    # Test order submission
    print("\nTesting order submission...")
    order = consumer.submit_order("firm1", "widget", 2.0, 15.0)
    print("order", order)
    print("market.orders", market.orders)
    
    assert order.consumer_id == "test_consumer"
    assert order.firm_id == "firm1"
    assert order.good == "widget"
    assert order.quantity == 2.0
    assert order.max_price == 15.0
    assert len(market.orders) == 1
    assert market.orders[0] is order
    
    print("\nFixedConsumerAgent basic tests passed!")


def test_firm_consumer_trading():
    """Test complete trading cycle between firm and consumer."""
    
    print("\nTESTING FIRM-CONSUMER TRADING...\n")
    
    # Create shared ledger and market
    ledger = Ledger()
    market = Market()
    
    # Initialize firm
    print("\nInitializing firm...")
    firm = FixedFirmAgent(
        name="test_firm",
        goods=["widget", "gadget"],
        initial_cash=1000.0,
        ledger=ledger,
        market=market,
        unit_costs={"widget": 14.5, "gadget": 14.5},
        markup=0.50,
    )
    
    # Initialize consumer
    print("\nInitializing consumer...")
    consumer = FixedConsumerAgent(
        name="test_consumer",
        income_stream=200.0,
        ledger=ledger,
        market=market,
        goods=["widget", "gadget"]
    )
    
    # Give consumer some income
    consumer.receive_income(timestep=0)
    print(f"Consumer cash after income: {consumer.cash}")
    
    # Firm purchases supplies and produces goods
    print("\nFirm purchasing supplies...")
    supplies_purchased = firm.purchase_supplies(quantity_to_purchase=20.0, unit_price=10.0, timestep=0)
    print(f"Supplies purchased: {supplies_purchased}")
    print(f"Firm cash after supply purchase: {firm.cash}")
    print(f"Firm supplies: {firm.supplies}")
    
    assert supplies_purchased == 20.0
    assert firm.cash == 800.0  # 1000 - 20*10
    assert firm.supplies == 20.0
    
    # Firm produces goods
    print("\nFirm producing goods...")
    firm.produce_goods(timestep=0)
    print(f"Firm widget inventory: {firm.inventory['widget']}")
    print(f"Firm gadget inventory: {firm.inventory['gadget']}")
    print(f"Firm supplies after production: {firm.supplies}")
    
    assert firm.inventory["widget"] == 10.0  # 20 supplies / 2 goods
    assert firm.inventory["gadget"] == 10.0
    assert firm.supplies == 0.0
    
    # Firm sets prices and posts quotes
    print("\nFirm setting prices and posting quotes...")
    prices = firm.set_price(timestep=0)
    quotes = firm.post_quotes(prices)
    print(f"Prices: {prices}")
    print(f"Quotes posted: {len(quotes)}")
    print(f"Market quotes: {len(market.quotes)}")
    
    assert prices == {"widget": 15.0, "gadget": 15.0}
    assert len(quotes) == 2
    assert len(market.quotes) == 2
    
    # Consumer makes orders
    print("\nConsumer making orders...")
    orders = consumer.make_orders(timestep=0)
    print(f"Orders made: {len(orders)}")
    print(f"Market orders: {len(market.orders)}")
    
    # Check that orders were created correctly
    for order in orders:
        print(f"Order: {order.consumer_id} -> {order.firm_id}, {order.good}, qty={order.quantity}, max_price={order.max_price}")
        assert order.consumer_id == "test_consumer"
        assert order.firm_id == "test_firm"
        assert order.good in ["widget", "gadget"]
        assert order.quantity > 0
        assert order.max_price == 15.0
    
    # Clear the market to execute trades
    print("\nClearing the market...")
    filled_orders = market.clear(ledger)
    print(f"Filled orders: {len(filled_orders)}")
    
    # Check final state
    print("\nChecking final state...")
    print(f"Consumer cash: {consumer.cash}")
    print(f"Consumer inventory: {consumer.inventory}")
    print(f"Firm cash: {firm.cash}")
    print(f"Firm inventory: {firm.inventory}")
    
    # Verify trades were executed
    assert len(filled_orders) > 0
    assert consumer.cash < 200.0  # Consumer spent money
    assert sum(consumer.inventory.values()) > 0  # Consumer has goods
    assert firm.cash > 800.0  # Firm received money
    assert sum(firm.inventory.values()) < 20.0  # Firm sold some goods
    
    print("\nFirm-consumer trading tests passed!")


def test_multiple_agents_market():
    """Test multiple firms and consumers interacting in the same market."""
    
    print("\nTESTING MULTIPLE AGENTS MARKET...\n")
    
    # Create shared ledger and market
    ledger = Ledger()
    market = Market()
    
    # Initialize multiple firms
    print("\nInitializing multiple firms...")
    firm1 = FixedFirmAgent(
        name="firm1",
        goods=["widget"],
        initial_cash=1000.0,
        ledger=ledger,
        market=market,
        unit_costs={"widget": 11.5},
        markup=0.50,
    )
    
    firm2 = FixedFirmAgent(
        name="firm2",
        goods=["gadget"],
        initial_cash=1000.0,
        ledger=ledger,
        market=market,
        unit_costs={"gadget": 17.5},
        markup=0.50,
    )
    
    # Initialize multiple consumers
    print("\nInitializing multiple consumers...")
    consumer1 = FixedConsumerAgent(
        name="consumer1",
        income_stream=150.0,
        ledger=ledger,
        market=market,
        goods=["widget", "gadget"]
    )
    
    consumer2 = FixedConsumerAgent(
        name="consumer2",
        income_stream=200.0,
        ledger=ledger,
        market=market,
        goods=["widget", "gadget"]
    )
    
    # Give consumers income
    consumer1.receive_income(timestep=0)
    consumer2.receive_income(timestep=0)
    
    print(f"Consumer1 cash: {consumer1.cash}")
    print(f"Consumer2 cash: {consumer2.cash}")
    
    # Firms purchase supplies and produce
    print("\nFirms purchasing supplies and producing...")
    firm1.purchase_supplies(quantity_to_purchase=10.0, unit_price=10.0, timestep=0)
    firm2.purchase_supplies(quantity_to_purchase=15.0, unit_price=10.0, timestep=0)
    
    firm1.produce_goods(timestep=0)
    firm2.produce_goods(timestep=0)
    
    print(f"Firm1 widget inventory: {firm1.inventory['widget']}")
    print(f"Firm2 gadget inventory: {firm2.inventory['gadget']}")
    
    # Firms post quotes
    print("\nFirms posting quotes...")
    firm1_quotes = firm1.post_quotes(firm1.set_price(timestep=0))
    firm2_quotes = firm2.post_quotes(firm2.set_price(timestep=0))
    
    print(f"Total quotes in market: {len(market.quotes)}")
    print(f"Firm1 quotes: {len(firm1_quotes)}")
    print(f"Firm2 quotes: {len(firm2_quotes)}")
    
    assert len(market.quotes) == 2
    assert len(firm1_quotes) == 1
    assert len(firm2_quotes) == 1
    
    # Consumers make orders
    print("\nConsumers making orders...")
    consumer1_orders = consumer1.make_orders(timestep=0)
    consumer2_orders = consumer2.make_orders(timestep=0)
    
    print(f"Consumer1 orders: {len(consumer1_orders)}")
    print(f"Consumer2 orders: {len(consumer2_orders)}")
    print(f"Total orders in market: {len(market.orders)}")
    
    # Clear the market
    print("\nClearing the market...")
    filled_orders = market.clear(ledger)
    print(f"Filled orders: {len(filled_orders)}")
    
    # Check final state
    print("\nChecking final state...")
    print(f"Consumer1 cash: {consumer1.cash}, inventory: {consumer1.inventory}")
    print(f"Consumer2 cash: {consumer2.cash}, inventory: {consumer2.inventory}")
    print(f"Firm1 cash: {firm1.cash}, inventory: {firm1.inventory}")
    print(f"Firm2 cash: {firm2.cash}, inventory: {firm2.inventory}")
    
    # Verify trades occurred
    assert len(filled_orders) > 0
    assert consumer1.cash < 150.0  # Consumer1 spent money
    assert consumer2.cash < 200.0  # Consumer2 spent money
    assert firm1.cash > 800.0  # Firm1 received money
    assert firm2.cash > 850.0  # Firm2 received money
    
    # Check that consumers have goods
    assert sum(consumer1.inventory.values()) > 0
    assert sum(consumer2.inventory.values()) > 0
    
    # Check that firms sold some goods
    assert firm1.inventory["widget"] < 10.0
    assert firm2.inventory["gadget"] < 15.0
    
    print("\nMultiple agents market tests passed!")


def test_consumer_budget_allocation():
    """Test consumer budget allocation across goods."""
    
    print("\nTESTING CONSUMER BUDGET ALLOCATION...\n")
    
    # Create ledger and market
    ledger = Ledger()
    market = Market()
    
    # Initialize consumer with specific income
    consumer = FixedConsumerAgent(
        name="test_consumer",
        income_stream=100.0,
        ledger=ledger,
        market=market,
        goods=["widget", "gadget"]
    )
    
    # Give consumer income
    consumer.receive_income(timestep=0)
    print(f"Consumer cash: {consumer.cash}")
    
    # Create firm (post_quotes called with custom dict below, not set_price)
    firm = FixedFirmAgent(
        name="test_firm",
        goods=["widget", "gadget"],
        initial_cash=1000.0,
        ledger=ledger,
        market=market,
    )
    
    # Give firm inventory
    firm.ledger.add_good("test_firm", "widget", 10.0)
    firm.ledger.add_good("test_firm", "gadget", 10.0)
    
    # Post quotes with different prices
    quotes = firm.post_quotes({"widget": 10.0, "gadget": 20.0})
    print(f"Quotes posted: {len(quotes)}")
    for quote in quotes:
        print(f"Quote: {quote.good} at {quote.price}")
    
    # Consumer makes orders
    orders = consumer.make_orders(timestep=0)
    print(f"Orders made: {len(orders)}")
    
    # Check budget allocation
    total_spent = 0
    for order in orders:
        print(f"Order: {order.good}, quantity: {order.quantity}, max_price: {order.max_price}")
        total_spent += order.quantity * order.max_price
    
    print(f"Total budget allocated: {total_spent}")
    print(f"Consumer cash before orders: {consumer.cash}")
    
    # Clear market
    filled_orders = market.clear(ledger)
    print(f"Filled orders: {len(filled_orders)}")
    print(f"Consumer cash after trades: {consumer.cash}")
    print(f"Consumer inventory: {consumer.inventory}")
    
    # Verify budget was allocated correctly
    # Consumer should spend roughly half budget on each good (perfect substitution)
    # Widget: 50/10 = 5 units, Gadget: 50/20 = 2.5 units
    expected_widget_quantity = 50.0 / 10.0  # 5.0
    expected_gadget_quantity = 50.0 / 20.0  # 2.5
    
    print(f"Expected widget quantity: {expected_widget_quantity}")
    print(f"Expected gadget quantity: {expected_gadget_quantity}")
    print(f"Actual widget quantity: {consumer.inventory['widget']}")
    print(f"Actual gadget quantity: {consumer.inventory['gadget']}")
    
    # Allow for some tolerance due to market clearing mechanics
    assert abs(consumer.inventory["widget"] - expected_widget_quantity) < 1.0
    assert abs(consumer.inventory["gadget"] - expected_gadget_quantity) < 1.0
    
    print("\nConsumer budget allocation tests passed!")


def test_complete_market_simulation():
    """Test a complete market simulation with multiple rounds."""
    
    print("\nTESTING COMPLETE MARKET SIMULATION...\n")
    
    # Create shared ledger and market
    ledger = Ledger()
    market = Market()
    
    # Initialize agents (unit_costs + markup = 15 per good)
    firm = FixedFirmAgent(
        name="firm",
        goods=["widget", "gadget"],
        initial_cash=2000.0,
        ledger=ledger,
        market=market,
        unit_costs={"widget": 14.5, "gadget": 14.5},
        markup=0.50,
    )
    
    consumer = FixedConsumerAgent(
        name="consumer",
        income_stream=100.0,
        ledger=ledger,
        market=market,
        goods=["widget", "gadget"]
    )
    
    print("Starting market simulation...")
    
    # Run simulation for multiple timesteps
    for timestep in range(3):
        print(f"\n--- TIMESTEP {timestep} ---")
        
        # Consumer receives income
        consumer.receive_income(timestep=timestep)
        print(f"Consumer cash: {consumer.cash}")
        
        # Firm purchases supplies
        supplies_purchased = firm.purchase_supplies(
            quantity_to_purchase=20.0, 
            unit_price=10.0, 
            timestep=timestep
        )
        print(f"Firm purchased {supplies_purchased} supplies")
        
        # Firm produces goods
        firm.produce_goods(timestep=timestep)
        print(f"Firm inventory: {firm.inventory}")
        
        # Firm posts quotes
        prices = firm.set_price(timestep=timestep)
        quotes = firm.post_quotes(prices)
        print(f"Firm posted {len(quotes)} quotes")
        
        # Consumer makes orders
        orders = consumer.make_orders(timestep=timestep)
        print(f"Consumer made {len(orders)} orders")
        
        # Clear market
        filled_orders = market.clear(ledger)
        print(f"Market cleared: {len(filled_orders)} orders filled")
        
        # Print final state
        print(f"Final state - Consumer: cash={consumer.cash}, inventory={consumer.inventory}")
        print(f"Final state - Firm: cash={firm.cash}, inventory={firm.inventory}")
    
    print("\nComplete market simulation tests passed!")


if __name__ == "__main__":
    test_fixed_consumer_agent()
    test_firm_consumer_trading()
    test_multiple_agents_market()
    test_consumer_budget_allocation()
    test_complete_market_simulation()
    
    print("\n\nAll FixedConsumerAgent and market interaction tests passed!")
