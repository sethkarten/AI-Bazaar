# tests/dev-tests/test_firm_agent_parsing.py
"""
Test for FirmAgent parsing of production, prices, and supply purchase decisions.
This test is designed to help tune prompts by capturing what the LLM outputs
vs. what the parser expects.

The test captures:
1. Raw LLM JSON output
2. Expected keys vs actual keys found
3. Parsing errors with details
4. Successful parsing results
5. The prompts being sent to the LLM

USAGE:
    Run individual tests:
        python -m pytest tests/dev-tests/test_firm_agent_parsing.py::test_set_price_parsing -v
        python -m pytest tests/dev-tests/test_firm_agent_parsing.py::test_purchase_supplies_parsing -v
        python -m pytest tests/dev-tests/test_firm_agent_parsing.py::test_produce_goods_parsing -v
    
    Run all tests:
        python tests/dev-tests/test_firm_agent_parsing.py
        # or
        python -m pytest tests/dev-tests/test_firm_agent_parsing.py -v

The test creates a ParsingTestFirmAgent that captures all LLM outputs and parsing
attempts, making it easy to see what's going wrong and tune the prompts accordingly.
"""

import sys
import os
import logging
import json
from argparse import Namespace
from ai_bazaar.agents.firm import FirmAgent
from ai_bazaar.market_core.market_core import Ledger, Market
from ai_bazaar.utils.common import Message

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_parsing')


class ParsingTestFirmAgent(FirmAgent):
    """Extended FirmAgent that captures LLM outputs for testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.captured_outputs = []
    
    def call_llm(self, msg: str, timestep: int, keys: list[str], parse_func, depth: int=0, retry: bool=False, cot: bool=False, temperature: float=0.7):
        """Override to capture LLM output before parsing."""
        response_found = False
        raw_llm_output = None
        
        if cot:
            raw_llm_output, response_found = self.llm.send_msg(self.system_prompt, msg, temperature=temperature, json_format=True)
            msg = msg + raw_llm_output
        if not response_found:
            raw_llm_output, _ = self.llm.send_msg(self.system_prompt, msg + '\n{"', temperature=temperature, json_format=True)
        
        # Capture the output for testing
        capture_info = {
            'timestep': timestep,
            'depth': depth,
            'retry': retry,
            'expected_keys': keys.copy(),
            'raw_output': raw_llm_output,
            'full_message': msg,
            'system_prompt': self.system_prompt,
            'action_type': self._infer_action_type(keys),
            'parsed_successfully': False,
            'parsed_values': None,
            'error': None
        }
        
        try:
            self.logger.info(f"LLM OUTPUT RECURSE {depth}\t{raw_llm_output.strip()}")
            # parse for json braces {}
            data = json.loads(raw_llm_output)
            
            data = self.extract_keys_from_dict(data, keys)
            
            # Check which keys were found
            found_keys = list(data.keys())
            missing_keys = [k for k in keys if k not in found_keys]
            extra_keys = [k for k in found_keys if k not in keys]
            
            capture_info['found_keys'] = found_keys
            capture_info['missing_keys'] = missing_keys
            capture_info['extra_keys'] = extra_keys
            capture_info['parsed_json'] = data
            
            parsed_keys = []
            for key in keys:
                parsed_keys.append(data[key])
            output = parse_func(parsed_keys)
            
            capture_info['parsed_successfully'] = True
            capture_info['parsed_values'] = output
            
            self.captured_outputs.append(capture_info)
            return output
            
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            capture_info['parsed_successfully'] = False
            capture_info['error'] = str(e)
            capture_info['error_type'] = type(e).__name__
            
            self.logger.warning(f"JSON parsing failed (attempt {depth}): {str(e)}")
            self.logger.warning(f"LLM output was: {repr(raw_llm_output)}")
            
            if depth <= self.timeout:
                # Try to clean up the output before retrying
                cleaned_output = self._clean_json_output(raw_llm_output, keys)
                if cleaned_output != raw_llm_output:
                    self.logger.info(f"Attempting to use cleaned output: {repr(cleaned_output)}")
                    try:
                        data = json.loads(cleaned_output)
                        
                        data = self.extract_keys_from_dict(data, keys)
                        
                        parsed_keys = []
                        for key in keys:
                            parsed_keys.append(data[key])
                        output = parse_func(parsed_keys)
                        
                        capture_info['parsed_successfully'] = True
                        capture_info['parsed_values'] = output
                        capture_info['cleaned_output'] = cleaned_output
                        capture_info['used_cleaning'] = True
                        
                        self.captured_outputs.append(capture_info)
                        return output
                    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e2:
                        self.logger.warning(f"Falling through to retry: {str(e2)}")
                        capture_info['cleanup_error'] = str(e2)
                        pass  # Fall through to retry
                
                self.captured_outputs.append(capture_info)
                return super().call_llm(msg, timestep, keys, parse_func, depth=depth+1, retry=True)
            else:
                self.captured_outputs.append(capture_info)
                raise ValueError(f"Max recursion depth={depth} reached. Error parsing JSON: " + str(e))
    
    def _infer_action_type(self, keys):
        """Infer what action is being tested from the keys."""
        if any('price_' in k for k in keys):
            return 'set_price'
        elif 'supply_quantity' in keys:
            return 'purchase_supplies'
        elif any('produce_' in k for k in keys):
            return 'produce_goods'
        else:
            return 'unknown'


def print_test_header(title):
    """Print a formatted test header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def print_capture_summary(capture, show_prompt=False):
    """Print a summary of captured LLM output."""
    print(f"  Action: {capture['action_type']}")
    print(f"  Timestep: {capture['timestep']}, Depth: {capture['depth']}, Retry: {capture['retry']}")
    print(f"  Expected Keys: {capture['expected_keys']}")
    
    if 'found_keys' in capture:
        print(f"  Found Keys: {capture['found_keys']}")
        if capture.get('missing_keys'):
            print(f"  ⚠️  Missing Keys: {capture['missing_keys']}")
        if capture.get('extra_keys'):
            print(f"  ℹ️  Extra Keys: {capture['extra_keys']}")
    
    if show_prompt:
        print(f"\n  System Prompt:")
        print(f"    {capture.get('system_prompt', 'N/A')[:300]}...")
        print(f"\n  User Prompt (from message):")
        # Extract user prompt from full message if possible
        msg = capture.get('full_message', '')
        if 'Historical data:' in msg:
            parts = msg.split('Historical data:')
            if len(parts) > 1:
                print(f"    {parts[1][:300]}...")
    
    print(f"\n  Raw LLM Output:")
    print(f"    {repr(capture['raw_output'][:200])}{'...' if len(capture['raw_output']) > 200 else ''}")
    
    if 'parsed_json' in capture:
        print(f"\n  Parsed JSON:")
        print(f"    {json.dumps(capture['parsed_json'], indent=4)}")
    
    if capture['parsed_successfully']:
        print(f"\n  ✅ Parsed Successfully!")
        print(f"  Parsed Values: {capture['parsed_values']}")
    else:
        print(f"\n  ❌ Parsing Failed!")
        print(f"  Error Type: {capture.get('error_type', 'Unknown')}")
        print(f"  Error Message: {capture['error']}")
        if 'cleanup_error' in capture:
            print(f"  Cleanup Error: {capture['cleanup_error']}")
    
    print("\n" + "-"*80 + "\n")


def test_set_price_parsing():
    """Test price setting action parsing."""
    print_test_header("TESTING: set_price() Parsing")
    
    ledger = Ledger()
    market = Market()
    args = Namespace(bracket_setting='three', service='google-ai', max_supply_unit_cost=10.0)
    
    firm = ParsingTestFirmAgent(
        llm="gemini-2.5-flash",
        port=8000,
        name="test_firm_price",
        prompt_algo='io',
        history_len=10,
        timeout=3,
        goods=["widget", "gadget"],
        initial_cash=1000.0,
        ledger=ledger,
        market=market,
        args=args
    )
    
    print("Firm state:")
    print(f"  Cash: ${firm.cash:.2f}")
    print(f"  Inventory: {dict(firm.inventory)}")
    print(f"  Goods: {firm.goods}")
    print(f"  Expected keys: price_widget, price_gadget\n")
    
    try:
        print("Calling set_price()...")
        prices = firm.set_price(timestep=0)
        print(f"✅ Successfully parsed prices: {prices}")
        
        # Print all captures for this action
        price_captures = [c for c in firm.captured_outputs if c['action_type'] == 'set_price']
        print(f"\nTotal attempts: {len(price_captures)}")
        for i, capture in enumerate(price_captures):
            print(f"\n  --- Attempt {i+1} ---")
            print_capture_summary(capture, show_prompt=(i == 0))  # Show prompt for first attempt
        
    except Exception as e:
        print(f"❌ Error during set_price(): {e}")
        import traceback
        traceback.print_exc()
        
        # Print captures even if it failed
        price_captures = [c for c in firm.captured_outputs if c['action_type'] == 'set_price']
        for i, capture in enumerate(price_captures):
            print(f"\n  --- Attempt {i+1} ---")
            print_capture_summary(capture)
    
    return firm


def test_purchase_supplies_parsing():
    """Test supply purchase action parsing."""
    print_test_header("TESTING: purchase_supplies() Parsing")
    
    ledger = Ledger()
    market = Market()
    args = Namespace(bracket_setting='three', service='google-ai', max_supply_unit_cost=10.0)
    
    firm = ParsingTestFirmAgent(
        llm="gemini-2.5-flash",
        port=8000,
        name="test_firm_supply",
        prompt_algo='io',
        history_len=10,
        timeout=3,
        goods=["widget", "gadget"],
        initial_cash=1000.0,
        ledger=ledger,
        market=market,
        args=args
    )
    
    print("Firm state:")
    print(f"  Cash: ${firm.cash:.2f}")
    print(f"  Supply unit price: $10.00")
    print(f"  Expected key: supply_quantity\n")
    
    try:
        print("Calling purchase_supplies()...")
        quantity, _ = firm.purchase_supplies(timestep=0)
        print(f"✅ Successfully parsed supply quantity: {quantity}")
        print(f"  Cash after purchase: ${firm.cash:.2f}")
        print(f"  Supplies after purchase: {firm.supplies:.2f}")
        
        # Print all captures for this action
        supply_captures = [c for c in firm.captured_outputs if c['action_type'] == 'purchase_supplies']
        print(f"\nTotal attempts: {len(supply_captures)}")
        for i, capture in enumerate(supply_captures):
            print(f"\n  --- Attempt {i+1} ---")
            print_capture_summary(capture, show_prompt=(i == 0))  # Show prompt for first attempt
        
    except Exception as e:
        print(f"❌ Error during purchase_supplies(): {e}")
        import traceback
        traceback.print_exc()
        
        # Print captures even if it failed
        supply_captures = [c for c in firm.captured_outputs if c['action_type'] == 'purchase_supplies']
        for i, capture in enumerate(supply_captures):
            print(f"\n  --- Attempt {i+1} ---")
            print_capture_summary(capture)
    
    return firm


def test_produce_goods_parsing():
    """Test production action parsing."""
    print_test_header("TESTING: produce_goods() Parsing")
    
    ledger = Ledger()
    market = Market()
    args = Namespace(bracket_setting='three', service='google-ai', max_supply_unit_cost=10.0)
    
    firm = ParsingTestFirmAgent(
        llm="gemini-2.5-flash",
        port=8000,
        name="test_firm_production",
        prompt_algo='io',
        history_len=10,
        timeout=3,
        goods=["widget", "gadget"],
        initial_cash=1000.0,
        ledger=ledger,
        market=market,
        args=args
    )
    
    # Add supplies so production can happen
    ledger.add_good("test_firm_production", "supply", 100.0)
    
    print("Firm state:")
    print(f"  Available supply: {firm.supplies:.2f}")
    print(f"  Goods: {firm.goods}")
    print(f"  Expected keys: produce_widget, produce_gadget (as percentages)\n")
    
    try:
        print("Calling produce_goods()...")
        firm.produce_goods(timestep=0)
        print(f"✅ Successfully parsed production allocation")
        print(f"  Widget inventory: {firm.inventory['widget']:.2f}")
        print(f"  Gadget inventory: {firm.inventory['gadget']:.2f}")
        print(f"  Supplies remaining: {firm.supplies:.2f}")
        
        # Print all captures for this action
        prod_captures = [c for c in firm.captured_outputs if c['action_type'] == 'produce_goods']
        print(f"\nTotal attempts: {len(prod_captures)}")
        for i, capture in enumerate(prod_captures):
            print(f"\n  --- Attempt {i+1} ---")
            print_capture_summary(capture, show_prompt=(i == 0))  # Show prompt for first attempt
        
    except Exception as e:
        print(f"❌ Error during produce_goods(): {e}")
        import traceback
        traceback.print_exc()
        
        # Print captures even if it failed
        prod_captures = [c for c in firm.captured_outputs if c['action_type'] == 'produce_goods']
        for i, capture in enumerate(prod_captures):
            print(f"\n  --- Attempt {i+1} ---")
            print_capture_summary(capture)
    
    return firm


def test_all_actions_sequence():
    """Test all three actions in sequence to see how they interact."""
    print_test_header("TESTING: All Actions in Sequence")
    
    ledger = Ledger()
    market = Market()
    args = Namespace(bracket_setting='three', service='google-ai', max_supply_unit_cost=10.0)
    
    firm = ParsingTestFirmAgent(
        llm="gemini-2.5-flash",
        port=8000,
        name="test_firm_full",
        prompt_algo='io',
        history_len=10,
        timeout=3,
        goods=["widget", "gadget"],
        initial_cash=1000.0,
        ledger=ledger,
        market=market,
        args=args
    )
    
    print("Running full sequence: purchase_supplies -> produce_goods -> set_price\n")
    
    timestep = 0
    
    # 1. Purchase supplies
    print("="*80)
    print("STEP 1: Purchase Supplies")
    print("="*80)
    try:
        quantity, _ = firm.purchase_supplies(timestep=timestep)
        print(f"✅ Purchased {quantity:.2f} supplies\n")
    except Exception as e:
        print(f"❌ Failed: {e}\n")
    
    # 2. Produce goods
    print("="*80)
    print("STEP 2: Produce Goods")
    print("="*80)
    try:
        firm.produce_goods(timestep=timestep)
        print(f"✅ Produced goods\n")
    except Exception as e:
        print(f"❌ Failed: {e}\n")
    
    # 3. Set prices
    print("="*80)
    print("STEP 3: Set Prices")
    print("="*80)
    try:
        prices = firm.set_price(timestep=timestep)
        print(f"✅ Set prices: {prices}\n")
    except Exception as e:
        print(f"❌ Failed: {e}\n")
    
    # Summary of all captures
    print("="*80)
    print("PARSING SUMMARY")
    print("="*80)
    
    for action_type in ['purchase_supplies', 'produce_goods', 'set_price']:
        action_captures = [c for c in firm.captured_outputs if c['action_type'] == action_type]
        if action_captures:
            successful = sum(1 for c in action_captures if c['parsed_successfully'])
            failed = len(action_captures) - successful
            print(f"\n{action_type}:")
            print(f"  Total attempts: {len(action_captures)}")
            print(f"  ✅ Successful: {successful}")
            print(f"  ❌ Failed: {failed}")
            
            if failed > 0:
                print(f"\n  Failed attempts:")
                for i, capture in enumerate(action_captures):
                    if not capture['parsed_successfully']:
                        print(f"    Attempt {i+1}: {capture.get('error_type', 'Unknown')} - {capture.get('error', 'No error message')}")
                        print(f"      Expected: {capture['expected_keys']}")
                        if 'found_keys' in capture:
                            print(f"      Found: {capture['found_keys']}")
                        print(f"      Raw output: {repr(capture['raw_output'][:100])}...")
    
    return firm


def print_prompt_analysis(firm):
    """Analyze the prompts being sent to the LLM."""
    print_test_header("PROMPT ANALYSIS")
    
    # Get the most recent message history entries
    if len(firm.message_history) > 1:
        latest_msg = firm.message_history[-1]
        print("Latest message prompt structure:")
        print(f"\nSystem Prompt (first 500 chars):")
        print(f"  {latest_msg['system_prompt'][:500]}...")
        print(f"\nUser Prompt:")
        print(f"  {latest_msg['user_prompt']}")
        print(f"\nHistorical Context:")
        print(f"  {latest_msg['historical']}")


def run_all_tests():
    """Run all parsing tests."""
    print("\n" + "="*80)
    print("  FIRM AGENT PARSING TESTS")
    print("  Testing LLM output parsing for production, prices, and supply purchases")
    print("="*80)
    
    results = {}
    
    # Test each action individually
    try:
        results['set_price'] = test_set_price_parsing()
    except Exception as e:
        print(f"❌ set_price test crashed: {e}")
        results['set_price'] = None
    
    try:
        results['purchase_supplies'] = test_purchase_supplies_parsing()
    except Exception as e:
        print(f"❌ purchase_supplies test crashed: {e}")
        results['purchase_supplies'] = None
    
    try:
        results['produce_goods'] = test_produce_goods_parsing()
    except Exception as e:
        print(f"❌ produce_goods test crashed: {e}")
        results['produce_goods'] = None
    
    # Test all in sequence
    try:
        results['full_sequence'] = test_all_actions_sequence()
    except Exception as e:
        print(f"❌ full sequence test crashed: {e}")
        results['full_sequence'] = None
    
    # Final summary
    print("\n" + "="*80)
    print("  FINAL SUMMARY")
    print("="*80)
    
    for test_name, firm_result in results.items():
        if firm_result:
            total_captures = len(firm_result.captured_outputs)
            successful = sum(1 for c in firm_result.captured_outputs if c['parsed_successfully'])
            failed = total_captures - successful
            print(f"\n{test_name}:")
            print(f"  Total parsing attempts: {total_captures}")
            print(f"  ✅ Successful: {successful}")
            print(f"  ❌ Failed: {failed}")
            
            # Show error types
            if failed > 0:
                error_types = {}
                for capture in firm_result.captured_outputs:
                    if not capture['parsed_successfully']:
                        err_type = capture.get('error_type', 'Unknown')
                        error_types[err_type] = error_types.get(err_type, 0) + 1
                print(f"  Error breakdown: {error_types}")
    
    print("\n" + "="*80)
    print("  TESTS COMPLETE")
    print("="*80 + "\n")
    
    return results


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0)

