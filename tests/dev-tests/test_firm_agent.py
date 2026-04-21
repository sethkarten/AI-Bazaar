# tests/dev-tests/test_firm_agent.py
"""
Test for FirmAgent (LLM-based firm) functionality.
"""

import sys
import os
import logging
from argparse import Namespace
from ai_bazaar.agents.firm import FirmAgent, FixedFirmAgent
from ai_bazaar.market_core.market_core import Ledger, Market, Order, Quote

# Set up logging
logging.basicConfig(level=logging.INFO)


def test_firm_agent_initialization():
    """Test FirmAgent initialization."""
    
    print("\n" + "="*70)
    print("TESTING FIRM AGENT INITIALIZATION")
    print("="*70)
    
    # Create ledger and market
    ledger = Ledger()
    market = Market()
    
    # Create mock args
    args = Namespace(
        bracket_setting='three',
        service='google-ai' # note: service only relevant for llama models
    )
    
    # Initialize firm agent (with mock LLM)
    print("\nInitializing FirmAgent...")
    try:
        firm = FirmAgent(
            llm="gemini-2.5-flash",  # Using smaller model for testing
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
        
        # Test initialization
        print(f"✓ Firm name: {firm.name}")
        print(f"✓ Goods: {firm.goods}")
        print(f"✓ Initial cash: {firm.cash}")
        print(f"✓ Inventory: {firm.inventory}")
        print(f"✓ System prompt exists: {len(firm.system_prompt) > 0}")
        
        assert firm.name == "test_firm"
        assert firm.goods == ["widget", "gadget"]
        assert firm.cash == 1000.0
        assert firm.supplies == 0.0
        assert "widget" in firm.inventory
        assert firm.inventory.get("widget", None) == 0.0
        assert "gadget" in firm.inventory
        assert firm.inventory.get("gadget", None) == 0.0
        assert "supply" in firm.inventory
        assert firm.inventory.get("supply", None) == 0.0
        assert len(firm.system_prompt) > 10 # prompt should exist
        
        print("\n✅ FirmAgent initialization tests passed!")
        return firm, ledger, market
        
    except Exception as e:
        print(f"\n❌ Error during initialization: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_firm_agent_parse_functions():
    """Test parsing functions without LLM calls."""
    
    print("\n" + "="*70)
    print("TESTING PARSE FUNCTIONS")
    print("="*70)
    
    ledger = Ledger()
    market = Market()
    args = Namespace(bracket_setting='three', service='google-ai')
    
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
    
    # Test parse_prices
    print("\n Testing parse_prices...")
    prices = firm.parse_prices(["10.50", "$25.00", "30"])
    print(f"✓ Parsed prices: {prices}")
    assert prices == (10.50, 25.0, 30.0)
    
    # Test parse_supply_purchase
    print("\nTesting parse_supply_purchase...")
    quantities = firm.parse_supply_purchase(["100 units", "50.5", "-10"])
    print(f"✓ Parsed quantities: {quantities}")
    assert quantities[0] == 100.0
    assert quantities[1] == 50.5
    assert quantities[2] == 0.0  # Negative clipped to 0
    
    # Test parse_production
    print("\nTesting parse_production...")
    percentages = firm.parse_production(["50%", "30", "120%"])
    print(f"✓ Parsed percentages: {percentages}")
    assert percentages[0] == 50.0
    assert percentages[1] == 30.0
    assert percentages[2] == 100.0
    
    print("\n✅ Parse function tests passed!")


def test_firm_agent_message_building():
    """Test message building for prompts."""
    
    print("\n" + "="*70)
    print("TESTING MESSAGE BUILDING")
    print("="*70)
    
    ledger = Ledger()
    market = Market()
    args = Namespace(bracket_setting='three', service='google-ai')
    
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
    
    # Add a timestep to message history
    firm.add_message_history_timestep(1)
    
    # Test UPDATE_PRICE message
    print("\nTesting UPDATE_PRICE message...")
    from ai_bazaar.utils.common import Message
    firm.add_message(timestep=1, m_type=Message.UPDATE_PRICE)
    print(f"Historical: {firm.message_history[1]['historical']}")
    print(f"User prompt: {firm.message_history[1]['user_prompt']}")
    assert "Cash:" in firm.message_history[1]['historical']
    assert "price_widget" in firm.message_history[1]['user_prompt']
    assert "price_gadget" in firm.message_history[1]['user_prompt']
    print("✓ UPDATE_PRICE message built correctly")
    
    # Test ACTION_PRICE message
    print("\nTesting ACTION_PRICE message...")
    firm.add_message(timestep=1, m_type=Message.ACTION_PRICE, prices={"widget": 10.0, "gadget": 15.0})
    print(f"Historical: {firm.message_history[1]['historical']}")
    assert "widget: $10.00" in firm.message_history[1]['historical']
    assert "gadget: $15.00" in firm.message_history[1]['historical']
    print("✓ ACTION_PRICE message built correctly")
    
    # Add supplies for production test
    ledger.add_good("test_firm", "supply", 100.0)
    
    # Test UPDATE_PRODUCTION message
    print("\nTesting UPDATE_PRODUCTION message...")
    firm.add_message(timestep=1, m_type=Message.UPDATE_PRODUCTION)
    print(f"User prompt: {firm.message_history[1]['user_prompt']}")
    assert "Available supply:" in firm.message_history[1]['historical']
    assert "produce_widget" in firm.message_history[1]['user_prompt']
    assert "produce_gadget" in firm.message_history[1]['user_prompt']
    print("✓ UPDATE_PRODUCTION message built correctly")
    
    print("\n✅ Message building tests passed!")


def test_firm_agent_with_fixed_behavior():
    """Test FirmAgent methods with controlled inputs (without actual LLM calls)."""
    
    print("\n" + "="*70)
    print("TESTING FIRM AGENT METHODS (WITHOUT LLM)")
    print("="*70)
    
    ledger = Ledger()
    market = Market()
    args = Namespace(bracket_setting='three', service='google-ai')
    
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
    
    print("\nInitial state:")
    print(f"  Cash: ${firm.cash}")
    print(f"  Supplies: {firm.supplies}")
    print(f"  Inventory: {firm.inventory}")
    
    # Test manual supply addition and production (bypassing LLM)
    print("\nManually adding supplies...")
    ledger.add_good("test_firm", "supply", 100.0)
    print(f"  Supplies after addition: {firm.supplies}")
    assert firm.supplies == 100.0
    
    # Manually produce goods (simulate what produce_goods would do)
    print("\nManually producing goods (50/50 split)...")
    ledger.add_good("test_firm", "widget", 50.0)
    ledger.add_good("test_firm", "gadget", 50.0)
    ledger.add_good("test_firm", "supply", -100.0)
    
    print(f"  Widget inventory: {firm.inventory['widget']}")
    print(f"  Gadget inventory: {firm.inventory['gadget']}")
    print(f"  Supplies remaining: {firm.supplies}")
    
    assert firm.inventory['widget'] == 50.0
    assert firm.inventory['gadget'] == 50.0
    assert firm.supplies == 0.0
    
    # Test post_quotes (this doesn't need LLM)
    print("\nTesting post_quotes...")
    prices = {"widget": 10.0, "gadget": 15.0}
    quotes = firm.post_quotes(prices)
    
    print(f"  Quotes posted: {len(quotes)}")
    print(f"  Market quotes: {len(market.quotes)}")
    
    assert len(quotes) == 2
    assert len(market.quotes) == 2
    assert market.quotes[0].good in ["widget", "gadget"]
    assert market.quotes[0].firm_id == "test_firm"
    print("✓ Quotes posted successfully")
    
    print("\n✅ Firm agent methods tests passed!")


def test_comparison_fixed_vs_llm_structure():
    """Test that FirmAgent and FixedFirmAgent have compatible interfaces."""
    
    print("\n" + "="*70)
    print("TESTING INTERFACE COMPATIBILITY")
    print("="*70)
    
    ledger1 = Ledger()
    market1 = Market()
    
    ledger2 = Ledger()
    market2 = Market()
    
    args = Namespace(bracket_setting='three', service='google-ai')
    
    # Create both types of firms
    fixed_firm = FixedFirmAgent(
        name="fixed_firm",
        goods=["widget", "gadget"],
        initial_cash=1000.0,
        ledger=ledger1,
        market=market1
    )
    
    llm_firm = FirmAgent(
        llm="gemini-2.5-flash",
        port=8000,
        name="llm_firm",
        prompt_algo='io',
        history_len=10,
        timeout=5,
        goods=["widget", "gadget"],
        initial_cash=1000.0,
        ledger=ledger2,
        market=market2,
        args=args
    )
    
    # Check that both have the same essential attributes
    print("\nChecking shared attributes...")
    shared_attrs = ['name', 'goods', 'ledger', 'market', 'inventory']
    for attr in shared_attrs:
        assert hasattr(fixed_firm, attr), f"FixedFirmAgent missing {attr}"
        assert hasattr(llm_firm, attr), f"FirmAgent missing {attr}"
        print(f"✓ Both have '{attr}' attribute")
    
    # Check that both have the same essential methods
    print("\nChecking shared methods...")
    shared_methods = ['cash', 'supplies', 'post_quotes']
    for method in shared_methods:
        assert hasattr(fixed_firm, method), f"FixedFirmAgent missing {method}"
        assert hasattr(llm_firm, method), f"FirmAgent missing {method}"
        print(f"✓ Both have '{method}' method")
    
    # Check that both have decision methods (with potentially different signatures)
    print("\nChecking decision methods...")
    decision_methods = ['set_price', 'purchase_supplies', 'produce_goods']
    for method in decision_methods:
        assert hasattr(fixed_firm, method), f"FixedFirmAgent missing {method}"
        assert hasattr(llm_firm, method), f"FirmAgent missing {method}"
        print(f"✓ Both have '{method}' method")
    
    print("\n✅ Interface compatibility tests passed!")


def run_all_tests():
    """Run all tests in sequence."""
    
    print("\n" + "="*70)
    print("RUNNING ALL FIRM AGENT TESTS")
    print("="*70)
    
    tests = [
        ("Initialization", test_firm_agent_initialization),
        ("Parse Functions", test_firm_agent_parse_functions),
        ("Message Building", test_firm_agent_message_building),
        ("Methods (No LLM)", test_firm_agent_with_fixed_behavior),
        ("Interface Compatibility", test_comparison_fixed_vs_llm_structure),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
            print(f"\n✅ {test_name} - PASSED")
        except Exception as e:
            failed += 1
            print(f"\n❌ {test_name} - FAILED")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print(f"TEST SUMMARY: {passed} passed, {failed} failed")
    print("="*70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

