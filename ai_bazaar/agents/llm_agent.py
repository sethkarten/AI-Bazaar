import logging
from time import sleep
from typing import Optional
from ..utils.common import Message
import json
from ..models.openai_model import OpenAIModel
from ..models.vllm_model import VLLMModel, OllamaModel
from ..models.openrouter_model import OpenRouterModel
from ..models.gemini_model import GeminiModel, GeminiModelViaOpenRouter
from ..utils.bracket import get_num_brackets, get_default_rates
from collections import Counter
import numpy as np


class LLMAgent:
    def __init__(
        self,
        llm_type: str,
        port: int,
        name: str,
        prompt_algo: str = "io",
        history_len: int = 5,  # Reduced from 10
        timeout: int = 5,  # Reduced from 10
        K: int = 3,
        args=None,
        llm_instance=None,
    ) -> None:
        assert args is not None

        self.bracket_setting = args.bracket_setting
        self.num_brackets = get_num_brackets(self.bracket_setting)
        self.tax_rates = get_default_rates(self.bracket_setting)

        self.logger = logging.getLogger("main")
        self.name = name
        self.args = args  # Store args for access to logging flags

        # Initialize the appropriate model based on llm_type
        if llm_instance:
            self.llm = llm_instance
        else:
            self.llm = self._create_llm_model(llm_type, port, args)

        self.history_len = history_len
        self.timeout = timeout  # number of times to retry message before failing
        self.system_prompt = ""  # Initialized to empty string instead of None
        self.init_message_history()

        self.prompt_algo = prompt_algo
        self.K = K  # depth of prompt trees
        self.trajectory = []
        self.diary = []  # List of (timestep, entry) tuples

    def _create_llm_model(self, llm_type: str, port: int, args):
        """Create the appropriate LLM model based on the type."""
        if llm_type == "None":
            return None
        elif "gpt" in llm_type.lower():
            return OpenAIModel(model_name=llm_type, max_tokens=args.max_tokens)
        elif "claude" in llm_type.lower() or "anthropic" in llm_type.lower():
            return OpenRouterModel(model_name=llm_type, max_tokens=args.max_tokens)
        elif "gemini" in llm_type.lower():
            return GeminiModel(model_name=llm_type, max_tokens=args.max_tokens)
        elif (
            "llama" in llm_type.lower()
            or "gemma" in llm_type.lower()
            or "qwen" in llm_type.lower()
            or "olmo" in llm_type.lower()
            or "ministral" in llm_type.lower()
            or llm_type.startswith(".")
            or llm_type.startswith("/")
        ):
            if port == 0:
                return None  # Should be provided via llm_instance
            if args.service == "ollama":
                return OllamaModel(
                    model_name=llm_type,
                    base_url=f"http://localhost:{port}",
                    max_tokens=args.max_tokens,
                )
            else:
                return VLLMModel(
                    model_name=llm_type,
                    base_url=f"http://localhost:{port}",
                    max_tokens=args.max_tokens,
                )
        elif "/" in llm_type:  # Assume it's a model path for OpenRouter
            return OpenRouterModel(model_name=llm_type, max_tokens=args.max_tokens)
        else:
            raise ValueError(f"Invalid LLM type: {llm_type}")

    def act(self) -> str:
        raise NotImplementedError

    def interview(self, question: str) -> str:
        """Allow a user to query the agent's internal state/reasoning."""
        system_prompt = (
            f"You are {self.name}. A researcher is interviewing you about your role in the Agent Bazaar. "
            "Respond in character based on your history and current state."
        )
        history = self.get_historical_message(
            len(self.message_history) - 1, include_user_prompt=False
        )
        diary_context = "\n".join([f"Day {t}: {e}" for t, e in self.diary[-5:]])

        user_prompt = f"Agent History:\n{history}\n\nRecent Diary Entries:\n{diary_context}\n\nResearcher Question: {question}"

        response, _ = self.llm.send_msg(system_prompt, user_prompt)
        return response

    def write_diary_entry(self, timestep: int, entry: str):
        """Store a retrospective reflection for the current timestep."""
        self.diary.append((timestep, entry))

    def init_message_history(self) -> None:
        # [{timestep: i, 'system_prompt': '', 'user_prompt': 'Historical timesteps: ', 'action': '' }, ...]
        # init first timestep
        self.message_history = [
            {
                "timestep": 0,
                "system_prompt": "",
                "user_prompt": "",
                "historical": "",
                "action": "",
                "leader": "planner",
                "metric": 0,
            }
        ]
        return

    def add_message_history_timestep(self, timestep: int) -> None:
        assert self.system_prompt is not None
        new_msg_dict = {
            "timestep": timestep,
            "system_prompt": self.system_prompt,
            "user_prompt": "",
            "historical": "",
            "action": "",
            "leader": "",
            "metric": 0,
        }
        self.message_history.append(new_msg_dict)
        return

    def get_historical_message(
        self, timestep: int, retry: bool = False, include_user_prompt: bool = True
    ) -> str:
        unique_metrics = set()  # Set to store unique 'metric' values
        sorted_message_history = []  # List to store sorted unique entries

        # Sort the dictionary by 'metric' key in descending order
        for item in sorted(
            self.message_history, key=lambda x: x["metric"], reverse=True
        ):
            if str(item["metric"]) + str(item["action"]) not in unique_metrics:
                unique_metrics.add(str(item["metric"]) + str(item["action"]))
                sorted_message_history.append(item)
        output = "Historical data:\n"

        # Include diary entries as in-context memory
        if self.diary:
            output += "Your Strategic Reflections (Recent):\n"
            for t_diary, entry in self.diary[-3:]:  # Last 3 reflections
                output += f"Timestep {t_diary}: {entry}\n"
            output += "\n"

        for t in range(
            max(0, timestep - min(self.history_len, len(self.message_history))),
            timestep + 1,
        ):
            output += f"Timestep {t}:\n"
            output += self.message_history[t]["historical"]
        N = min(5, len(sorted_message_history))
        output += f"Best {N} timesteps:\n"
        for i in range(N):
            output += f"Timestep {sorted_message_history[i]['timestep']} (leader {self.message_history[t]['leader']}):\n"
            output += sorted_message_history[i]["historical"]
        if include_user_prompt:
            output += self.message_history[timestep]["user_prompt"]
        if retry:
            output += "Please enter a valid response. "
        return output

    def act_llm(
        self,
        timestep: int,
        keys: list[str],
        parse_func,
        depth: int = 0,
        retry: bool = False,
    ) -> list[float]:
        # concat user prompts from prev timesteps to get historical information for current timestep
        msg = self.get_historical_message(timestep, retry)

        # Extract expected format from message_history if available
        expected_format = None
        if timestep in self.message_history:
            expected_format = self.message_history[timestep].get(
                "expected_format", None
            )

        if self.prompt_algo == "io":
            return self.prompt_io(
                msg, timestep, keys, parse_func, expected_format=expected_format
            )
        elif self.prompt_algo == "cot":
            return self.prompt_cot(
                msg, timestep, keys, parse_func, expected_format=expected_format
            )
        elif self.prompt_algo == "sc":
            return self.prompt_sc(
                msg, timestep, keys, parse_func, expected_format=expected_format
            )
        elif self.prompt_algo == "tot":
            return self.prompt_sc(
                msg, timestep, keys, parse_func, expected_format=expected_format
            )
        elif self.prompt_algo == "mcts":
            return self.prompt_mcts(
                msg, timestep, keys, parse_func, expected_format=expected_format
            )
        else:
            raise ValueError()

    def extract_keys_from_dict(self, d, keys):
        result = {}

        # Key mapping for alternative key names that LLMs might use
        key_mapping = {
            "purchase_supply": "supply_quantity",
            "purchase_supplies": "supply_quantity",
            "buy_supply": "supply_quantity",
            "buy_food_supply": "supply_quantity",
        }

        # Handle nested structures for supply purchases
        supply_purchase_keys = ["supply_purchases", "purchase_supplies", "buy_supplies"]
        for supply_key in supply_purchase_keys:
            if supply_key in d and isinstance(d[supply_key], dict):
                # Convert supply_purchases: {"supply": 30.0} to supply_quantity: 30.0
                # Extract the value from the nested dict (usually keyed by "supply")
                if "supply_quantity" in keys and d[supply_key]:
                    # Prefer "supply" key if it exists, otherwise take the first value
                    if "supply" in d[supply_key]:
                        result["supply_quantity"] = d[supply_key]["supply"]
                    else:
                        # Take the first value from the nested dict
                        result["supply_quantity"] = next(iter(d[supply_key].values()))
                break  # Only process the first matching key

        # Handle nested structures for production_allocations/production and set_prices/pricing
        # Support multiple key names the LLM might use
        production_keys = [
            "production_allocations",
            "production",
            "produce",
            "production_proportions",
            "production_percentages",
            "production_distribution",
            "production_allocation",
            "production_percentage",
            "production_distribution",
        ]
        pricing_keys = ["set_prices", "pricing", "prices"]

        for prod_key in production_keys:
            if prod_key in d and isinstance(d[prod_key], dict):
                # Convert production_allocations/production: {"food": "100%"} to produce_food: "100%"
                for good, pct in d[prod_key].items():
                    key = f"produce_{good}"
                    if key in keys:
                        result[key] = pct
                break  # Only process the first matching key

        for price_key in pricing_keys:
            if price_key in d and isinstance(d[price_key], dict):
                # Convert set_prices/pricing: {"food": 2.00} to price_food: 2.00
                for good, price in d[price_key].items():
                    key = f"price_{good}"
                    if key in keys:
                        result[key] = price
                break  # Only process the first matching key

        # Standard extraction
        for key, value in d.items():
            # Skip nested structures we've already handled
            if key in [
                "production_allocations",
                "production",
                "produce",
                "set_prices",
                "pricing",
                "prices",
                "supply_purchases",
                "purchase_supplies",
                "buy_supplies",
            ]:
                continue

            # Check if key matches directly
            if key in keys:
                result[key] = value
            # Check if mapped key matches
            elif key in key_mapping and key_mapping[key] in keys:
                result[key_mapping[key]] = value
            # Recursively search in nested dicts
            elif isinstance(value, dict):
                nested_result = self.extract_keys_from_dict(value, keys)
                result.update(nested_result)

        return result

    def _call_parsing_agent(
        self, malformed_json: str, expected_format: str, keys: list[str]
    ) -> Optional[str]:
        """Use a parsing agent LLM to clean and reformat malformed JSON.

        Args:
            malformed_json: The malformed JSON string from the original LLM
            expected_format: The expected JSON format from the user prompt
            keys: List of expected keys

        Returns:
            Cleaned JSON string in the expected format
        """
        if (
            not hasattr(self.args, "use_parsing_agent")
            or not self.args.use_parsing_agent
        ):
            return None

        if self.llm is None:
            return None

        try:
            # Create a prompt for the parsing agent
            system_prompt = """You are a JSON parsing and reformatting assistant. Your task is to take malformed or incorrectly formatted JSON and convert it to the exact format specified.

You must:
1. Parse the malformed JSON (even if it's incomplete or has errors)
2. Extract the relevant values
3. Reformat it to match the EXACT expected format provided
4. Return ONLY valid JSON in the expected format, with no additional text or markdown

If the malformed JSON contains nested structures (like arrays of objects), convert them to the flat format expected.
If keys have different names, map them to the expected key names.
If values are in different formats (e.g., percentages vs numbers), convert them appropriately."""

            user_prompt = f"""Malformed JSON response:
{malformed_json}

Expected JSON format:
{expected_format}

Expected keys: {", ".join(keys)}

Please reformat the malformed JSON to match the expected format exactly. Return ONLY the cleaned JSON object, with no markdown, no code blocks, and no explanatory text."""

            self.logger.info(
                f"[PARSING AGENT] Sending malformed JSON to parsing agent for {self.name}"
            )
            cleaned_output, _ = self.llm.send_msg(
                system_prompt, user_prompt, temperature=0.1, json_format=True
            )
            self.logger.info(
                f"[PARSING AGENT] Replied with cleaned JSON: {cleaned_output}"
            )

            # Clean up the response (remove markdown if present)
            cleaned_output = cleaned_output.strip()
            if cleaned_output.startswith("```"):
                # Remove markdown code blocks
                import re

                cleaned_output = re.sub(r"```(?:json)?\s*", "", cleaned_output)
                cleaned_output = re.sub(r"```\s*", "", cleaned_output)
                cleaned_output = cleaned_output.strip()

            self.logger.info(
                f"[PARSING AGENT] Received cleaned JSON: {cleaned_output[:200]}..."
            )
            return cleaned_output

        except Exception as e:
            self.logger.warning(
                f"[PARSING AGENT] Failed to parse JSON with parsing agent: {e}"
            )
            return None

    def call_llm(
        self,
        msg: str,
        timestep: int,
        keys: list[str],
        parse_func,
        depth: int = 0,
        retry: bool = False,
        cot: bool = False,
        temperature: float = 0.7,
        expected_format: Optional[str] = None,
    ) -> list[float]:
        # Log when prompting an agent
        if depth == 0:
            self.logger.info(f"[PROMPT] Prompting agent: {self.name}")

        # Optionally log full prompts for firm agents (log before sending to LLM)
        if (
            hasattr(self.args, "log_firm_prompts")
            and self.args.log_firm_prompts
            and self.name.startswith("firm_")
            and depth == 0
        ):
            # Log the initial message that will be sent
            user_msg_to_log = msg if cot else (msg + '\n{"')
            self.logger.info(
                f"[FIRM PROMPT] {self.name} at timestep {timestep}:\n"
                f"System Prompt:\n{self.system_prompt}\n\n"
                f"User Message:\n{user_msg_to_log}"
            )

        response_found = False
        if cot:
            llm_output, response_found = self.llm.send_msg(
                self.system_prompt, msg, temperature=temperature, json_format=True
            )
            msg = msg + llm_output
        if not response_found:
            llm_output, _ = self.llm.send_msg(
                self.system_prompt,
                msg,  # No injection - let thinking models do CoT then output JSON
                temperature=temperature,
                json_format=True,
            )
            # Extract JSON from output (may contain thinking/reasoning before JSON)
            # The JSON extraction in unsloth_model.py will handle this

        # Record trajectory for RL training
        self.trajectory.append(
            {
                "timestep": timestep,
                "system_prompt": self.system_prompt,
                "user_prompt": msg,
                "response": llm_output,
                "keys": keys,
                "reward": None,  # To be filled by environment
            }
        )

        try:
            self.logger.info(f"LLM OUTPUT RECURSE {depth}\t{llm_output.strip()}")
            # parse for json braces {}
            data = json.loads(llm_output)

            # Extract keys with mapping support
            data = self.extract_keys_from_dict(data, keys)

            # Check if all required keys are present
            missing_keys = [key for key in keys if key not in data]
            if missing_keys:
                raise KeyError(
                    f"Missing keys: {missing_keys}. Available keys: {list(data.keys())}"
                )

            parsed_keys = []
            for key in keys:
                parsed_keys.append(data[key])
            output = parse_func(parsed_keys)

        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            self.logger.warning(f"JSON parsing failed (attempt {depth}): {str(e)}")
            self.logger.warning(f"LLM output was: {repr(llm_output)}")

            if depth <= self.timeout:
                # Try parsing agent first (if enabled)
                if self.args.use_parsing_agent and expected_format is not None:
                    self.logger.info(
                        f"[PARSING AGENT] Calling parsing agent with expected format: {expected_format}"
                    )
                    parsing_agent_output = self._call_parsing_agent(
                        llm_output, expected_format, keys
                    )
                    if parsing_agent_output:
                        try:
                            data = json.loads(parsing_agent_output)

                            # Extract keys with mapping support
                            data = self.extract_keys_from_dict(data, keys)

                            # Check if all required keys are present
                            missing_keys = [key for key in keys if key not in data]
                            if missing_keys:
                                raise KeyError(
                                    f"Missing keys: {missing_keys}. Available keys: {list(data.keys())}"
                                )

                            parsed_keys = []
                            for key in keys:
                                parsed_keys.append(data[key])
                            output = parse_func(parsed_keys)
                            self.logger.info(
                                f"[PARSING AGENT] Successfully parsed JSON using parsing agent"
                            )
                            return output
                        except (
                            json.JSONDecodeError,
                            KeyError,
                            ValueError,
                            TypeError,
                        ) as parse_error:
                            self.logger.warning(
                                f"[PARSING AGENT] Parsing agent output also failed: {parse_error}"
                            )
                else:
                    # Try to clean up the output before retrying
                    cleaned_output = self._clean_json_output(llm_output, keys)
                    if cleaned_output != llm_output:
                        self.logger.info(
                            f"Attempting to use cleaned output: {repr(cleaned_output)}"
                        )
                        try:
                            data = json.loads(cleaned_output)

                            # Extract keys with mapping support
                            data = self.extract_keys_from_dict(data, keys)

                            # Check if all required keys are present
                            missing_keys = [key for key in keys if key not in data]
                            if missing_keys:
                                raise KeyError(
                                    f"Missing keys: {missing_keys}. Available keys: {list(data.keys())}"
                                )

                            parsed_keys = []
                            for key in keys:
                                parsed_keys.append(data[key])
                            output = parse_func(parsed_keys)
                            return output
                        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                            self.logger.warning(f"Falling through to retry: {str(e)}")
                            pass  # Fall through to retry

                return self.call_llm(
                    msg,
                    timestep,
                    keys,
                    parse_func,
                    depth=depth + 1,
                    retry=True,
                    expected_format=expected_format,
                )
            else:
                raise ValueError(
                    f"Max recursion depth={depth} reached. Error parsing JSON: "
                    + str(e)
                )
        return output

    def _clean_json_output(self, output: str, keys: list[str]) -> str:
        """Try to clean up malformed JSON output from LLM."""
        import re
        import json

        # Remove markdown code blocks if present
        output = re.sub(r"```(?:json)?\s*", "", output)
        output = re.sub(r"```\s*", "", output)

        # Remove any text before the first {
        output = re.sub(r"^[^{]*", "", output)

        # If output doesn't start with { but starts with a quote, add opening brace
        # This handles cases like: "purchase_supply": 200.00,...}
        if output and not output.startswith("{") and output.strip().startswith('"'):
            output = "{" + output

        output = output.replace("\\n", " ")

        # Handle unterminated strings more intelligently
        # If we have an odd number of quotes, we have an incomplete string
        quote_count = output.count('"')
        if quote_count % 2 == 1:
            # Find incomplete key-value pairs and remove them
            # Pattern 1: "key": "incomplete_value (no closing quote)
            # Pattern 2: "incomplete_key (incomplete key name)

            # First, try to find and remove incomplete key-value pairs
            # Look for: "key": "incomplete (with colon but no closing quote on value)
            incomplete_value_pattern = r',\s*"[^"]+"\s*:\s*"[^"]*$'
            output = re.sub(incomplete_value_pattern, "", output)

            # Remove incomplete key at the end (e.g., "price_)
            output = re.sub(r',\s*"[^":]*$', "", output)

            # If still odd quotes, find the last complete key-value pair
            quote_count = output.count('"')
            if quote_count % 2 == 1:
                # Find all complete key-value pairs: "key": "value" followed by , or }
                complete_pair_pattern = r'"\w+"\s*:\s*(?:"[^"]*"|[^,}]+)(?=\s*[,}])'
                matches = list(re.finditer(complete_pair_pattern, output))

                if matches:
                    # Keep only up to the last complete pair's end position
                    # Find where the value ends (after quote or number)
                    last_match_end = matches[-1].end()
                    # Look ahead to see if there's a comma or closing brace
                    next_char_pos = last_match_end
                    while (
                        next_char_pos < len(output) and output[next_char_pos] in " \t\n"
                    ):
                        next_char_pos += 1
                    if next_char_pos < len(output) and output[next_char_pos] in ",}":
                        next_char = output[next_char_pos]
                        output = output[:next_char_pos]
                        if next_char == ",":
                            output += "}"
                        elif not output.endswith("}"):
                            output += "}"
                    else:
                        # The last match might be incomplete, try previous one
                        if len(matches) > 1:
                            prev_match = matches[-2]
                            output = output[: prev_match.end()]
                            # Add comma if next would be another field, or closing brace
                            output = re.sub(r",\s*$", "", output)
                            output += "}"
                        else:
                            # Only one match, but might be incomplete - just close the string
                            output += '"'
                            if not output.endswith("}"):
                                output = re.sub(r",\s*$", "", output)
                                output += "}"
                else:
                    # No complete pairs found, try to salvage by closing the string
                    output += '"'

        # Fix missing closing braces
        open_braces = output.count("{")
        close_braces = output.count("}")
        if open_braces > close_braces:
            # Remove trailing comma before adding closing brace
            output = re.sub(r",\s*$", "", output)
            output += "}" * (open_braces - close_braces)

        # Remove trailing commas before closing braces
        output = re.sub(r",(\s*[}])", r"\1", output)

        # Try to extract just the JSON part if there's extra text
        json_match = re.search(r"\{.*\}", output, re.DOTALL)
        if json_match:
            output = json_match.group(0)

        # Final cleanup: remove any text after the last }
        output = re.sub(r"}[^}]*$", "}", output)

        return output

    # prompting
    def prompt_io(
        self,
        msg: str,
        timestep: int,
        keys: list[str],
        parse_func,
        expected_format: Optional[str] = None,
    ) -> list[float]:
        return self.call_llm(
            msg, timestep, keys, parse_func, expected_format=expected_format
        )

    # Self-Consistency prompting
    def prompt_sc(
        self,
        msg: str,
        timestep: int,
        keys: list[str],
        parse_func,
        expected_format: Optional[str] = None,
    ) -> list[float]:
        llm_outputs = []
        for i in range(self.K):
            llm_output = self.prompt_cot(
                msg, timestep, keys, parse_func, expected_format=expected_format
            )
            llm_outputs.append(llm_output)

        def most_common(lst):
            lst_str = [str(x) for x in lst]
            data = Counter(lst_str)
            str_common = data.most_common(1)[0][0]
            str_index = lst_str.index(str_common)
            return lst[str_index]

        output = most_common(llm_outputs)
        return output

    # Chain of thought prompting
    def prompt_cot(
        self,
        msg: str,
        timestep: int,
        keys: list[str],
        parse_func,
        expected_format: Optional[str] = None,
    ) -> list[float]:
        cot_prompt = (
            " Let's think step by step. Your thought should no more than 4 sentences."
        )
        # always add json thought "thought":"<step-by-step-thinking>" response in user_prompt in agent
        return self.call_llm(
            msg + cot_prompt,
            timestep,
            keys,
            parse_func,
            cot=True,
            expected_format=expected_format,
        )

    def add_message(self, timestep: int, m_type: Message, **args) -> None:
        raise NotImplementedError

    def parse_tax(self, items: list[str]) -> tuple:
        # self.logger.info("[parse_tax]", tax_rates)
        tax_rates = items[0]
        output_tax_rates = []
        if len(tax_rates) != self.num_brackets:  # fixed to 2 tax divisions
            raise ValueError("too many tax values", tax_rates)
        for i, rate in enumerate(tax_rates):
            if isinstance(rate, str):
                rate = rate.replace("$", "").replace(",", "").replace("%", "")
            rate = float(rate)
            rate = np.clip(rate, -self.delta, self.delta)
            rate = np.round(rate / 10) * 10
            # rate = np.round(rate / 10) * 10
            if rate + self.tax_rates[i] > 100:
                rate = 100 - self.tax_rates[i]
            elif rate + self.tax_rates[i] < 0:
                rate = -self.tax_rates[i]
            # if rate > 100: rate = 100
            # if rate > 100 or rate < 0:
            #     raise ValueError(f'Rates outside bounds: 0 <= {rate} <= 100')
            output_tax_rates.append(rate)
        # return (output_tax_rates, float(items[1]))
        return (output_tax_rates,)


class TestAgent(LLMAgent):
    def __init__(self, llm: str, port: int, args):
        super().__init__(llm, port, name="TestAgent", args=args)
        max_retries = 5  # Maximum attempts (including initial call)
        initial_delay = 1  # Starting delay in seconds
        max_delay = 60  # Maximum delay between retries
        current_delay = initial_delay

        for attempt in range(max_retries):
            try:
                self.llm.send_msg("", 'This is a test. Output "test" in response.')
                print(f"Successfully connected to f{args.service} LLM service")
                return  # Exit on success
            except Exception as e:
                if attempt == max_retries - 1:  # Final attempt failed
                    raise RuntimeError(
                        f"Failed to connect after {max_retries} attempts. Last error: {str(e)}"
                    ) from e

                print(f"Attempt {attempt + 1} failed. Retrying in {current_delay}s...")
                sleep(current_delay)
                current_delay = min(
                    current_delay * 2, max_delay
                )  # Exponential backoff with cap
