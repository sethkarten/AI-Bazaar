# tests/test_market_integration.py
"""
Short integration test for Ledger, Market, and FixedFirmAgent.
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


def test_ledger_market_fixed_firm_integration():
    """Test Ledger, Market, and FixedFirmAgent working together."""
    
    # 1. TEST LEDGER
    print("\nTESTING LEDGER...\n")
    ledger = Ledger()
    
    # Test basic ledger operations
    print("\nCrediting firm1 and consumer1...")
    ledger.credit("firm1", 1000.0)
    ledger.credit("consumer1", 500.0)
    print("ledger.agent_money", ledger.agent_money)
    assert ledger.agent_money["firm1"] == 1000.0
    assert ledger.agent_money["consumer1"] == 500.0
    
    
    # Test money transfer
    print("\nTransferring money from consumer1 to firm1...")
    ledger.transfer_money("consumer1", "firm1", 100.0)
    print("ledger.agent_money", ledger.agent_money)
    assert ledger.agent_money["firm1"] == 1100.0
    assert ledger.agent_money["consumer1"] == 400.0
    
    
    
    # Test adding goods
    print("\nAdding goods to firm1...")
    ledger.add_good("firm1", "widget", 10.0)
    print("ledger.agent_inventories", ledger.agent_inventories)
    assert ledger.agent_inventories["firm1"]["widget"] == 10.0
    
    
    # 2. TEST MARKET
    print("\nTESTING MARKET...\n")
    market = Market()
    
    # Create and post a quote
    print("\nCreating and posting a quote...")
    
    quote = Quote("firm1", "widget", 15.0, 5.0)
    market.post_quote(quote)
    print("market.quotes", market.quotes)
    assert len(market.quotes) == 1
    assert market.quotes[0].price == 15.0
    
    
    # Create and submit an order
    print("\nCreating and submitting an order...")
    print("market.orders", market.orders)
    order = Order("consumer1", "firm1", "widget", 2.0, 20.0)
    market.submit_order(order)
    assert len(market.orders) == 1
    
    
    # Clear the market (should execute the trade)
    print("\nClearing the market...")
    filled_orders = market.clear(ledger)
    print("filled_orders", filled_orders)
    assert len(filled_orders) == 1  # Order was filled
    
    
    # Check that trade was executed
    print("\nChecking that trade was executed...")
    print("ledger.agent_money", ledger.agent_money)
    print("ledger.agent_inventories", ledger.agent_inventories)
    assert ledger.agent_money["consumer1"] == 370.0  # 400 - 2*15
    assert ledger.agent_money["firm1"] == 1130.0     # 1100 + 2*15
    assert ledger.agent_inventories["firm1"]["widget"] == 8.0  # 10 - 2
    assert ledger.agent_inventories["consumer1"]["widget"] == 2.0  # 0 + 2
    
    
    # 3. TEST FIXED FIRM AGENT
    # Create a new market and ledger for the firm test
    print("\nTESTING FIXED FIRM AGENT...\n")
    firm_ledger = Ledger()
    firm_market = Market()
    
    # Initialize the firm
    print("\nInitializing the firm...")
    firm = FixedFirmAgent(
        name="test_firm",
        goods=["widget", "gadget"],
        initial_cash=1000.0,
        ledger=firm_ledger,
        market=firm_market,
        unit_costs={"widget": 11.5, "gadget": 11.5},
        markup=0.50,
    )
    
    # Test firm initialization
    print("firm.goods", firm.goods)
    print("firm.cash", firm.cash)
    print("firm.inventory", firm.inventory)
    print("firm.ledger.agent_money", firm.ledger.agent_money)
    print("firm.ledger.agent_inventories", firm.ledger.agent_inventories)
    assert firm.goods == ["widget", "gadget"]
    assert firm.cash == 1000.0
    # Firm inventory now directly references ledger inventory, so it includes supply
    assert firm.inventory == {"widget": 0.0, "gadget": 0.0, "supply": 0.0}
    assert firm.ledger.agent_money["test_firm"] == 1000.0
    assert firm.ledger.agent_inventories["test_firm"] == {"widget": 0.0, "gadget": 0.0, "supply": 0.0}
    # Verify that firm.inventory is the same object as ledger inventory
    assert firm.inventory is firm.ledger.agent_inventories["test_firm"]
    
    
    # Test that cash and supplies properties work correctly
    print("\nTesting cash and supplies properties...")
    print("firm.cash", firm.ledger.agent_money["test_firm"])
    print("firm.supplies", firm.ledger.agent_inventories["test_firm"]["supply"])
    assert firm.cash == 1000.0
    assert firm.supplies == 0.0
    assert firm.ledger.agent_money["test_firm"] == firm.cash
    assert firm.ledger.agent_inventories["test_firm"]["supply"] == firm.supplies
    
    
    # Test setting prices (unit_cost + markup = 12)
    print("\nSetting prices...")
    prices = firm.set_price(timestep=0)
    print("prices", prices)
    assert prices == {"widget": 12.0, "gadget": 12.0}
    
    
    # Give the firm some inventory to sell
    print("\nGiving the firm some inventory to sell...")
    firm_ledger.add_good("test_firm", "widget", 5.0)
    firm_ledger.add_good("test_firm", "gadget", 3.0)
    print("firm.inventory", firm.inventory)
    print("firm_ledger.agent_inventories", firm_ledger.agent_inventories)
    assert firm.inventory["widget"] == 5.0
    assert firm.inventory["gadget"] == 3.0
    assert firm_ledger.agent_inventories["test_firm"]["widget"] == 5.0
    assert firm_ledger.agent_inventories["test_firm"]["gadget"] == 3.0
    
    
    # Test posting quotes
    print("\nPosting quotes...")
    quotes = firm.post_quotes(prices)
    print("quotes", quotes)
    assert len(quotes) == 2  # 1 quote for each good
    assert quotes[0].firm_id == "test_firm"
    assert quotes[0].price == 12
    assert quotes[0].quantity_available == 5.0
    assert quotes[0].good == "widget"
    assert quotes[1].firm_id == "test_firm"
    assert quotes[1].price == 12
    assert quotes[1].quantity_available == 3.0
    assert quotes[1].good == "gadget"
    
    
    # Verify quotes were posted to market
    print("firm_market.quotes", firm_market.quotes)
    print("firm_market.quotes[0]", firm_market.quotes[0])
    print("firm_market.quotes[1]", firm_market.quotes[1])
    assert len(firm_market.quotes) == 2
    assert firm_market.quotes[0].firm_id == "test_firm"
    assert firm_market.quotes[0].price == 12
    assert firm_market.quotes[0].quantity_available == 5.0
    assert firm_market.quotes[0].good == "widget"
    assert firm_market.quotes[1].firm_id == "test_firm"
    assert firm_market.quotes[1].price == 12
    assert firm_market.quotes[1].quantity_available == 3.0
    assert firm_market.quotes[1].good == "gadget"
    
    
    # Test purchase_supplies method
    print("\nTesting purchase_supplies method...")
    initial_cash = firm.cash
    initial_supplies = firm.supplies
    
    # Purchase some supplies
    quantity_purchased = firm.purchase_supplies(quantity_to_purchase=10.0, unit_price=10.0, timestep=0)
    expected_quantity = 10.0  # 10.0 supplies
    expected_cost = 10.0 * 10.0  # 100.0 cost
    
    print(f"Purchased {quantity_purchased} supplies for {expected_cost}")
    print(f"Quantity purchased: {quantity_purchased}, Expected quantity: {expected_quantity}")
    print(f"New cash: {firm.cash}, New supplies: {firm.supplies}")
    assert quantity_purchased == expected_quantity
    assert firm.cash == initial_cash - expected_cost
    assert firm.supplies == initial_supplies + expected_quantity
    
    
    # Test produce_goods method
    print("\nTesting produce_goods method...")
    initial_widgets = firm.inventory["widget"]
    initial_gadgets = firm.inventory["gadget"]
    supplies_before_production = firm.supplies
    
    firm.produce_goods(timestep=0)
    
    expected_production_per_good = supplies_before_production / len(firm.goods)  # 10.0 / 2 = 5.0
    
    print(f"Initial widgets: {initial_widgets}, Initial gadgets: {initial_gadgets}, Supplies before production: {supplies_before_production}")
    print(f"Produced: {expected_production_per_good}")
    print(f"New widget inventory: {firm.inventory['widget']}")
    print(f"New gadget inventory: {firm.inventory['gadget']}")
    print(f"Supplies after production: {firm.supplies}")
    assert firm.inventory["widget"] == initial_widgets + expected_production_per_good
    assert firm.inventory["gadget"] == initial_gadgets + expected_production_per_good
    assert firm.supplies == 0.0  # All supplies should be consumed
    
    print("\n\nAll tests passed: Ledger, Market, and FixedFirmAgent work correctly!")


if __name__ == "__main__":
    test_ledger_market_fixed_firm_integration()