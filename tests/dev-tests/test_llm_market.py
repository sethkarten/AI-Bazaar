# tests/test_fixed_market.py
"""
Integration test for FixedConsumerAgent and LLM FirmAgent interactions.
Tests buying and selling between firms and consumers in a market.
"""
from argparse import Namespace
from ai_bazaar.agents.firm import FirmAgent
from ai_bazaar.agents.consumer import FixedConsumerAgent
from ai_bazaar.market_core.market_core import Ledger, Market, Order, Quote

def test_firm_consumer_trading():
    """Test complete trading cycle between firm and consumer."""
    
    print("\nTESTING FIRM-CONSUMER TRADING...\n")
    
    # Create shared ledger and market
    ledger = Ledger()
    market = Market()
    
    # Initialize firm
    print("\nInitializing firm...")
    args = Namespace(bracket_setting='three', service='google-ai', max_supply_unit_cost=10.0)
    firm = FirmAgent(
        llm="gemini-2.5-flash",
        port=8000,
        name="test_firm",
        prompt_algo='io',
        history_len=10,
        timeout=5,
        goods=["widget", "gadget"],
        initial_cash=1000.0,
        ledger=ledger,
        market=market,
        args=args
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
    supplies_purchased, _ = firm.purchase_supplies(timestep=0)
    print(f"Supplies purchased: {supplies_purchased}")
    print(f"Firm cash after supply purchase: {firm.cash}")
    print(f"Firm supplies: {firm.supplies}")
    
    # assert supplies were purchased (ignore case where agent might decide not to participate)
    assert supplies_purchased > 0.0
    assert firm.cash < 1000.0 
    assert firm.supplies > 0.0
    
    supplies_after_purchase = firm.supplies
    cash_after_purchase = firm.cash
    
    # Firm produces goods
    print("\nFirm producing goods...")
    firm.produce_goods(timestep=0)
    print(f"Firm widget inventory: {firm.inventory['widget']}")
    print(f"Firm gadget inventory: {firm.inventory['gadget']}")
    print(f"Firm supplies after production: {firm.supplies}")
    
    assert firm.inventory["widget"] > 0.0
    assert firm.inventory["gadget"] > 0.0
    assert firm.supplies < supplies_after_purchase
    
    # Firm sets prices and posts quotes
    print("\nFirm setting prices and posting quotes...")
    prices = firm.set_price(timestep=0)
    quotes = firm.post_quotes(prices)
    print(f"Prices: {prices}")
    print(f"Quotes posted: {len(quotes)}")
    print(f"Market quotes: {len(market.quotes)}")
    for quote in quotes:
        print(f"Quote: {quote.good} at {quote.price}") 
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
        assert order.max_price > 0.0
    
    # Clear the market to execute trades
    print("\nClearing the market...")
    filled_orders = market.clear(ledger)
    print(f"Filled orders: {len(filled_orders)}")
    
    # Display final state
    print("\nChecking final state...")
    print(f"Consumer cash: {consumer.cash}")
    print(f"Consumer inventory: {consumer.inventory}")
    print(f"Firm cash: {firm.cash}")
    print(f"Firm inventory: {firm.inventory}")
    
    # Verify trades were executed
    assert len(filled_orders) > 0
    assert consumer.cash < 200.0  # Consumer spent money
    assert sum(consumer.inventory.values()) > 0  # Consumer has goods
    assert firm.cash > cash_after_purchase # Firm received money
    
    print("\nFirm-consumer trading tests passed!")


def test_multiple_agents_market():
    """Test multiple firms and consumers interacting in the same market."""
    
    print("\nTESTING MULTIPLE AGENTS MARKET...\n")
    
    # Create shared ledger and market
    ledger = Ledger()
    market = Market()
    
    # Initialize multiple firms
    print("\nInitializing multiple firms...")
    args = Namespace(bracket_setting='three', service='google-ai', max_supply_unit_cost=10.0)
    
    firm1 = FirmAgent(
        llm="gemini-2.5-flash",
        port=8000,
        name="firm1",
        prompt_algo='io',
        history_len=10,
        timeout=5,
        goods=["widget"],
        initial_cash=1000.0,
        ledger=ledger,
        market=market,
        args=args
    )
    
    firm2 = FirmAgent(
        llm="gemini-2.5-flash",
        port=8001,
        name="firm2",
        prompt_algo='io',
        history_len=10,
        timeout=5,
        goods=["gadget"],
        initial_cash=1000.0,
        ledger=ledger,
        market=market,
        args=args
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
    supplies1_purchased, _ = firm1.purchase_supplies(timestep=0)
    supplies2_purchased, _ = firm2.purchase_supplies(timestep=0)
    
    print(f"Firm1 purchased {supplies1_purchased} supplies")
    print(f"Firm2 purchased {supplies2_purchased} supplies")
    
    firm1.produce_goods(timestep=0)
    firm2.produce_goods(timestep=0)
    
    print(f"Firm1 widget inventory: {firm1.inventory['widget']}")
    print(f"Firm2 gadget inventory: {firm2.inventory['gadget']}")
    
    # Firms set prices and post quotes
    print("\nFirms setting prices and posting quotes...")
    firm1_prices = firm1.set_price(timestep=0)
    firm2_prices = firm2.set_price(timestep=0)
    
    print(f"Firm1 prices: {firm1_prices}")
    print(f"Firm2 prices: {firm2_prices}")
    
    firm1_quotes = firm1.post_quotes(firm1_prices)
    firm2_quotes = firm2.post_quotes(firm2_prices)
    
    print(f"Total quotes in market: {len(market.quotes)}")
    print(f"Firm1 quotes: {len(firm1_quotes)}")
    print(f"Firm2 quotes: {len(firm2_quotes)}")
    
    assert len(market.quotes) >= 2  # Could be more if firms have multiple goods
    assert len(firm1_quotes) >= 1
    assert len(firm2_quotes) >= 1
    
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
    
    # Verify trades occurred (if any orders were filled)
    print(f"\nFilled orders: {len(filled_orders)}")
    
    if len(filled_orders) > 0:
        # If trades occurred, verify the expected outcomes
        print("Trades occurred - verifying outcomes...")
        
        # Track initial values
        initial_consumer1_cash = 150.0
        initial_consumer2_cash = 200.0
        
        # Check if consumers spent money (if they made successful orders)
        if consumer1.cash < initial_consumer1_cash:
            print(f"Consumer1 spent money: ${initial_consumer1_cash - consumer1.cash:.2f}")
            assert sum(consumer1.inventory.values()) > 0, "Consumer1 should have goods if they spent money"
        
        if consumer2.cash < initial_consumer2_cash:
            print(f"Consumer2 spent money: ${initial_consumer2_cash - consumer2.cash:.2f}")
            assert sum(consumer2.inventory.values()) > 0, "Consumer2 should have goods if they spent money"
        
        # Check that at least one consumer has goods
        assert sum(consumer1.inventory.values()) > 0 or sum(consumer2.inventory.values()) > 0, \
            "At least one consumer should have goods after trading"
        
        print("Trade verification passed!")
    else:
        print("No trades occurred - this can happen if firms didn't produce or prices were too high")
    
    print("\nMultiple agents market tests passed!")



def test_multi_timestep_simulation():
    """Test a complete market simulation with multiple LLM firms over multiple rounds."""
    
    print("\nTESTING MULTI-TIMESTEP MARKET SIMULATION...\n")
    
    # Create shared ledger and market
    ledger = Ledger()
    market = Market()
    
    # Initialize LLM-based firms
    args = Namespace(bracket_setting='three', service='google-ai', max_supply_unit_cost=10.0)
    
    print("Initializing agents...")
    firm1 = FirmAgent(
        llm="gemini-2.5-flash",
        port=8000,
        name="firm1",
        prompt_algo='io',
        history_len=10,
        timeout=5,
        goods=["widget"],
        initial_cash=2000.0,
        ledger=ledger,
        market=market,
        args=args
    )
    
    firm2 = FirmAgent(
        llm="gemini-2.5-flash",
        port=8001,
        name="firm2",
        prompt_algo='io',
        history_len=10,
        timeout=5,
        goods=["gadget"],
        initial_cash=2000.0,
        ledger=ledger,
        market=market,
        args=args
    )
    
    # Initialize consumers
    consumer1 = FixedConsumerAgent(
        name="consumer1",
        income_stream=100.0,
        ledger=ledger,
        market=market,
        goods=["widget", "gadget"]
    )
    
    consumer2 = FixedConsumerAgent(
        name="consumer2",
        income_stream=150.0,
        ledger=ledger,
        market=market,
        goods=["widget", "gadget"]
    )
    
    print("Starting multi-timestep market simulation...")
    
    # Track metrics across timesteps
    timestep_results = []
    
    # Run simulation for multiple timesteps
    num_timesteps = 3
    for timestep in range(num_timesteps):
        print(f"\n{'='*60}")
        print(f"TIMESTEP {timestep}")
        print(f"{'='*60}")
        
        # Initialize message history for new timestep (for LLM agents)
        if timestep > 0:
            firm1.add_message_history_timestep(timestep)
            firm2.add_message_history_timestep(timestep)
        
        # Consumers receive income
        print(f"\n[Income Phase]")
        consumer1.receive_income(timestep=timestep)
        consumer2.receive_income(timestep=timestep)
        print(f"Consumer1 cash: ${consumer1.cash:.2f}")
        print(f"Consumer2 cash: ${consumer2.cash:.2f}")
        
        # Firms purchase supplies
        print(f"\n[Supply Purchase Phase]")
        supplies1_purchased, _ = firm1.purchase_supplies(timestep=timestep)
        supplies2_purchased, _ = firm2.purchase_supplies(timestep=timestep)
        print(f"Firm1 purchased {supplies1_purchased:.2f} supplies (cash: ${firm1.cash:.2f})")
        print(f"Firm2 purchased {supplies2_purchased:.2f} supplies (cash: ${firm2.cash:.2f})")
        
        # Firms produce goods
        print(f"\n[Production Phase]")
        firm1.produce_goods(timestep=timestep)
        firm2.produce_goods(timestep=timestep)
        print(f"Firm1 inventory: {dict(firm1.inventory)}")
        print(f"Firm2 inventory: {dict(firm2.inventory)}")
        
        # Firms set prices and post quotes
        print(f"\n[Price Setting Phase]")
        prices1 = firm1.set_price(timestep=timestep)
        prices2 = firm2.set_price(timestep=timestep)
        print(f"Firm1 prices: {prices1}")
        print(f"Firm2 prices: {prices2}")
        
        quotes1 = firm1.post_quotes(prices1)
        quotes2 = firm2.post_quotes(prices2)
        print(f"Firm1 posted {len(quotes1)} quotes")
        print(f"Firm2 posted {len(quotes2)} quotes")
        print(f"Total market quotes: {len(market.quotes)}")
        
        # Consumers make orders
        print(f"\n[Order Submission Phase]")
        orders1 = consumer1.make_orders(timestep=timestep)
        orders2 = consumer2.make_orders(timestep=timestep)
        print(f"Consumer1 made {len(orders1)} orders")
        print(f"Consumer2 made {len(orders2)} orders")
        print(f"Total market orders: {len(market.orders)}")
        
        # Clear market
        print(f"\n[Market Clearing Phase]")
        filled_orders = market.clear(ledger)
        print(f"Market cleared: {len(filled_orders)} orders filled")
        
        # Record timestep results
        timestep_data = {
            'timestep': timestep,
            'filled_orders': len(filled_orders),
            'consumer1_cash': consumer1.cash,
            'consumer2_cash': consumer2.cash,
            'consumer1_goods': sum(consumer1.inventory.values()),
            'consumer2_goods': sum(consumer2.inventory.values()),
            'firm1_cash': firm1.cash,
            'firm2_cash': firm2.cash,
            'firm1_inventory': sum(v for k, v in firm1.inventory.items() if k != 'supply'),
            'firm2_inventory': sum(v for k, v in firm2.inventory.items() if k != 'supply'),
        }
        timestep_results.append(timestep_data)
        
        # Print final state for this timestep
        print(f"\n[End of Timestep {timestep} Summary]")
        print(f"Consumer1: cash=${consumer1.cash:.2f}, goods={sum(consumer1.inventory.values()):.2f}")
        print(f"Consumer2: cash=${consumer2.cash:.2f}, goods={sum(consumer2.inventory.values()):.2f}")
        print(f"Firm1: cash=${firm1.cash:.2f}, inventory={sum(v for k, v in firm1.inventory.items() if k != 'supply'):.2f}")
        print(f"Firm2: cash=${firm2.cash:.2f}, inventory={sum(v for k, v in firm2.inventory.items() if k != 'supply'):.2f}")
    
    # Final summary across all timesteps
    print(f"\n{'='*60}")
    print(f"SIMULATION SUMMARY")
    print(f"{'='*60}")
    
    total_trades = sum(r['filled_orders'] for r in timestep_results)
    print(f"\nTotal trades across all timesteps: {total_trades}")
    
    print(f"\nFinal state:")
    print(f"  Consumer1: cash=${consumer1.cash:.2f}, total goods={sum(consumer1.inventory.values()):.2f}")
    print(f"  Consumer2: cash=${consumer2.cash:.2f}, total goods={sum(consumer2.inventory.values()):.2f}")
    print(f"  Firm1: cash=${firm1.cash:.2f}, inventory={sum(v for k, v in firm1.inventory.items() if k != 'supply'):.2f}")
    print(f"  Firm2: cash=${firm2.cash:.2f}, inventory={sum(v for k, v in firm2.inventory.items() if k != 'supply'):.2f}")
    
    # Verify the simulation ran properly
    print(f"\nVerifying simulation results...")
    
    # Check that consumers received income (should have received income at least once)
    total_consumer1_income = consumer1.income * num_timesteps
    total_consumer2_income = consumer2.income * num_timesteps
    
    # Total money in the system should be conserved
    # Initial: firm1 (2000) + firm2 (2000) + consumer1 (0) + consumer2 (0) = 4000
    # Plus income: consumer1 income * timesteps + consumer2 income * timesteps
    expected_total_money = 4000.0 + total_consumer1_income + total_consumer2_income
    actual_total_money = consumer1.cash + consumer2.cash + firm1.cash + firm2.cash
    
    print(f"Money conservation check:")
    print(f"  Expected total: ${expected_total_money:.2f}")
    print(f"  Actual total: ${actual_total_money:.2f}")
    print(f"  Difference: ${abs(expected_total_money - actual_total_money):.2f}")
    
    # Allow for small floating point errors
    assert abs(expected_total_money - actual_total_money) < 0.01, \
        f"Money not conserved! Expected ${expected_total_money:.2f}, got ${actual_total_money:.2f}"
    
    # Check that at least some economic activity occurred
    assert total_trades > 0, "No trades occurred during the simulation"
    
    # Check that consumers acquired some goods
    total_consumer_goods = sum(consumer1.inventory.values()) + sum(consumer2.inventory.values())
    assert total_consumer_goods > 0, "Consumers didn't acquire any goods"
    
    print(f"\nAll simulation checks passed!")
    print("\nMulti-timestep market simulation tests passed!")


if __name__ == "__main__":
    # test_firm_consumer_trading()
    #test_multiple_agents_market()
    test_multi_timestep_simulation()
    
    print("\n\nAll FixedConsumerAgent and FirmAgent market interaction tests passed!")
