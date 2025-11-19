# tests/dev-tests/test_ces_consumer_agent.py
"""
Test for CESConsumerAgent functionality, specifically:
- CES parameters generation
- Risk aversion generation
"""

import sys
import os
import logging
from argparse import Namespace
from llm_economist.agents.consumer import CESConsumerAgent
from llm_economist.market_core.market_core import Ledger, Market

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def test_ces_params_generation():
    """Test CES parameters generation using persona."""
    
    print("\n" + "="*70)
    print("TESTING CES PARAMETERS GENERATION")
    print("="*70)
    
    # Create ledger and market
    ledger = Ledger()
    market = Market()
    
    # Create minimal args (required by LLMAgent)
    args = Namespace(
        bracket_setting='three',
        service='google-ai'  # Only relevant for llama models
    )
    
    # Test goods list
    goods = ['food', 'clothing', 'electronics']
    
    # Test with a persona
    persona = 'teacher'  # Using a persona from ROLE_MESSAGES
    
    print(f"\nTesting CES parameter generation with persona: {persona}")
    print(f"Goods: {goods}")
    
    try:
        # Initialize consumer agent with persona
        # Note: This will make actual LLM calls if llm is provided
        consumer = CESConsumerAgent(
            name="test_consumer",
            income_stream=1000.0,
            ledger=ledger,
            market=market,
            persona=persona,
            goods=goods,
            llm="gemini-2.5-flash",  # Use a real LLM or 'None' to skip LLM calls
            port=8000,
            prompt_algo='cot',
            timeout=3,
            args=args
        )
        
        # Check that CES params were generated
        print(f"\n✓ Consumer initialized successfully")
        print(f"✓ CES parameters: {consumer.ces_params}")
        print(f"✓ Risk aversion: {consumer.risk_aversion}")
        
        # Validate CES parameters
        assert consumer.ces_params is not None, "CES parameters should be generated"
        assert len(consumer.ces_params) == len(goods), f"Expected {len(goods)} CES parameters, got {len(consumer.ces_params)}"
        
        # Check all goods are present
        for good in goods:
            assert good in consumer.ces_params, f"Missing CES parameter for {good}"
            assert consumer.ces_params[good] > 0, f"CES parameter for {good} should be positive"
            assert consumer.ces_params[good] <= 1.0, f"CES parameter for {good} should be <= 1.0"
        
        # Check that parameters sum to approximately 1.0 (allowing for floating point errors)
        total = sum(consumer.ces_params.values())
        print(f"✓ Sum of CES parameters: {total:.6f}")
        assert abs(total - 1.0) < 0.01, f"CES parameters should sum to 1.0, got {total}"
        
        # Validate risk aversion
        assert consumer.risk_aversion is not None, "Risk aversion should be generated"
        assert isinstance(consumer.risk_aversion, float), "Risk aversion should be a float"
        assert 0.0 <= consumer.risk_aversion <= 1.0, f"Risk aversion should be between 0 and 1, got {consumer.risk_aversion}"
        
        print("\n✅ CES parameters generation tests passed!")
        return consumer
        
    except Exception as e:
        print(f"\n❌ Error during CES parameter generation: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_risk_aversion_generation():
    """Test risk aversion generation separately."""
    
    print("\n" + "="*70)
    print("TESTING RISK AVERSION GENERATION")
    print("="*70)
    
    # Create ledger and market
    ledger = Ledger()
    market = Market()
    
    # Create minimal args
    args = Namespace(
        bracket_setting='three',
        service='google-ai'
    )
    
    goods = ['food', 'clothing']
    persona = 'entrepreneur'  # Different persona to test
    
    print(f"\nTesting risk aversion generation with persona: {persona}")
    
    try:
        consumer = CESConsumerAgent(
            name="test_consumer_2",
            income_stream=2000.0,
            ledger=ledger,
            market=market,
            persona=persona,
            goods=goods,
            llm="gemini-2.5-flash",
            port=8000,
            prompt_algo='cot',
            timeout=3,
            args=args
        )
        
        # Test the risk aversion value
        risk_aversion = consumer.risk_aversion
        print(f"\n✓ Risk aversion generated: {risk_aversion}")
        
        # Validate it's a float
        assert isinstance(risk_aversion, float), f"Risk aversion should be float, got {type(risk_aversion)}"
        
        # Validate range
        assert 0.0 <= risk_aversion <= 1.0, f"Risk aversion should be in [0, 1], got {risk_aversion}"
        
        print("\n✅ Risk aversion generation tests passed!")
        return consumer
        
    except Exception as e:
        print(f"\n❌ Error during risk aversion generation: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_fallback_behavior():
    """Test fallback behavior when LLM fails or is None."""
    
    print("\n" + "="*70)
    print("TESTING FALLBACK BEHAVIOR")
    print("="*70)
    
    ledger = Ledger()
    market = Market()
    
    args = Namespace(
        bracket_setting='three',
        service='google-ai'
    )
    
    goods = ['food', 'clothing', 'electronics']
    
    print("\nTesting with provided CES params (no LLM generation needed)")
    
    try:
        # When ces_params are provided, no LLM generation is needed
        # Use 'None' as string to indicate no LLM (as per LLMAgent convention)
        consumer = CESConsumerAgent(
            name="test_consumer_3",
            income_stream=1000.0,
            ledger=ledger,
            market=market,
            persona=None,
            ces_params={'food': 0.4, 'clothing': 0.3, 'electronics': 0.3},
            goods=goods,
            llm='None',  # Use string 'None' to skip LLM initialization
            port=8000,
            args=args
        )
        
        print(f"✓ Consumer initialized with provided CES params: {consumer.ces_params}")
        assert consumer.ces_params == {'food': 0.4, 'clothing': 0.3, 'electronics': 0.3}
        
        print("\n✅ Fallback behavior tests passed!")
        
    except Exception as e:
        print(f"\n❌ Error during fallback test: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_parse_functions():
    """Test the parse functions directly without LLM calls."""
    
    print("\n" + "="*70)
    print("TESTING PARSE FUNCTIONS")
    print("="*70)
    
    ledger = Ledger()
    market = Market()
    
    args = Namespace(
        bracket_setting='three',
        service='google-ai'
    )
    
    # Create consumer with provided params to avoid LLM calls
    consumer = CESConsumerAgent(
        name="test_consumer_4",
        income_stream=1000.0,
        ledger=ledger,
        market=market,
        persona=None,
        ces_params={'food': 0.5, 'clothing': 0.5},
        goods=['food', 'clothing'],
        llm='None',  # Use string 'None' to skip LLM initialization
        port=8000,
        args=args
    )
    
    # Test _parse_ces_params
    print("\nTesting _parse_ces_params...")
    test_items = ['0.4', '0.3', '0.3']
    result = consumer._parse_ces_params(test_items)
    print(f"Input: {test_items}")
    print(f"Output: {result}")
    assert isinstance(result, tuple), "Should return tuple"
    assert len(result) == 3, "Should have 3 values"
    assert all(isinstance(x, float) for x in result), "All values should be floats"
    print("✓ _parse_ces_params works correctly")
    
    # Test _parse_risk_aversion
    print("\nTesting _parse_risk_aversion...")
    test_items = ['0.75']
    result = consumer._parse_risk_aversion(test_items)
    print(f"Input: {test_items}")
    print(f"Output: {result}")
    assert isinstance(result, float), "Should return float"
    assert 0.0 <= result <= 1.0, "Should be in [0, 1]"
    print("✓ _parse_risk_aversion works correctly")
    
    # Test with different input types
    print("\nTesting _parse_risk_aversion with different input types...")
    test_cases = [
        ['0.5'],      # string
        [0.5],        # float
        [0.75],       # float
        ['0.8'],      # string
    ]
    
    for test_case in test_cases:
        result = consumer._parse_risk_aversion(test_case)
        assert isinstance(result, float), f"Should return float for input {test_case}"
        assert 0.0 <= result <= 1.0, f"Should be in [0, 1] for input {test_case}"
        print(f"  ✓ Input {test_case} -> {result}")
    
    print("\n✅ Parse functions tests passed!")


def main():
    """Run all tests."""
    
    print("\n" + "="*70)
    print("CES CONSUMER AGENT TEST SUITE")
    print("="*70)
    
    try:
        # Test parse functions first (no LLM required)
        test_parse_functions()
        
        # Test fallback behavior (no LLM required)
        test_fallback_behavior()
        
        # Test with LLM (requires API key and LLM access)
        print("\n" + "="*70)
        print("NOTE: The following tests require LLM API access.")
        print("If you don't have API keys set up, these will fail.")
        print("="*70)
        
        # Uncomment these if you have LLM access:
        test_ces_params_generation()
        test_risk_aversion_generation()
        
        print("\n" + "="*70)
        print("ALL TESTS COMPLETED")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

