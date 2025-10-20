import hashlib
import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

import tiktoken
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from tqdm import tqdm

load_dotenv()

# Disable httpx logging
logging.getLogger("httpx").setLevel(logging.WARNING)

from generation_rates import (
    cached_input_rates,
    cheaper_model_map,
    input_rates,
    output_rates,
)
from generation_utils import fill_template_file

# setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Generator:
    def __init__(
        self,
        model_name,
        show_progress: bool = True,
        verbose: bool = False,
        safe_mode: bool = True,
        seed=None,
    ):
        """
        Initializes the Generator class with the specified model name and configuration options.
        Args:
            model_name: The name of the model to use for generation.
            show_progress: Whether to display progress during generation.
            verbose: If True, enables verbose logging.
            safe_mode: If True, enables safe mode for generation.
            seed: Random seed for reproducibility.
        """
        self.model_name = model_name
        self.show_progress = show_progress
        self.tokenizer = None
        self.model = None
        self.verbose = verbose
        self.safe_mode = safe_mode
        self.seed = seed
        if seed:
            random.seed(seed)
        self.setup_client()

        try:
            self.estimater_encoder = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.estimater_encoder = tiktoken.encoding_for_model("gpt-4o")
        self.curr_batches = []

        # Load existing cache from file
        cache_path = self._get_prompt_cache_path()
        self.prompt_cache = {}
        if cache_path.exists():
            with cache_path.open("r") as f:
                self.prompt_cache = json.load(f)

    def setup_client(self):
        """
        Sets up the OpenAI client with the .env API key and organization.
        """
        logger.info("Setting up OpenAI client")
        assert os.environ.get("OPENAI_API_KEY") is not None, "OPENAI_API_KEY not set"
        has_org = os.environ.get("OPENAI_API_ORG") is not None
        if has_org:
            self.client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
                organization=os.environ.get("OPENAI_API_ORG"),
            )
        else:
            logger.info("OPENAI_API_ORG not set")
            self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

        if self.safe_mode:
            partial_api_key = os.environ["OPENAI_API_KEY"][:6] + "..."
            if input("Use API key {}? (y/n): ".format(partial_api_key)) != "y":
                assert False, "Aborting"
            if has_org:
                partial_org = os.environ["OPENAI_API_ORG"][:6] + "..."
                if input("Use org {}? (y/n): ".format(partial_org)) != "y":
                    assert False, "Aborting"

    def verify_prompt(self, questions: List[Dict[str, str]], template_file: str):
        """
        Verifies that the provided prompts are valid.
        Args:
            questions: List of prompts to verify.
            template_file: Path to the template file.
        """
        assert len(questions) > 0, "Nothing to run"
        assert os.path.exists(template_file), "Template file does not exist: {}".format(
            template_file
        )

    def create_llm_input(self, question, examples, template_file, num_shots):
        """
        Creates the input for the LLM model.
        Args:
            question: The question to generate input for.
            examples: List of examples to use.
            template_file: Path to the template file.
            num_shots: Number of shots to use.
        Returns:
            The created LLM input.
        """
        # There are two possibilities for the examples: instant-specific examples, and general examples
        # if "examples" is in the question, then use those instead of the general examples
        if "examples" not in question and examples:
            question["examples"] = examples

        llm_input = fill_template_file(
            template_file,
            question=question,
            num_shots=num_shots,
        )

        # Need to adjust for models which do not take system messages
        if (
            len(llm_input[0]) > 0
            and llm_input[0]["role"] == "system"
            and self.system_message == False
        ):
            llm_input[1]["content"] = (
                llm_input[0]["content"] + "\n" + llm_input[1]["content"]
            )
            llm_input.pop(0)

        return llm_input

    def build_prompts(
        self,
        questions: List[Dict[str, str]],
        template_file: str,
        example_file: Optional[str] = None,
        num_shots=-1,
        save_as=None,
    ):
        """
        Builds prompts from a template file and a list of questions.
        Args:
            questions: List of questions to build prompts for.
            template_file: Path to the template file.
            example_file: Optional path to a file containing example prompts.
            num_shots: Number of shots to use.
            save_as: Optional path to save the generated prompts.
        Returns:
            List of built prompts and the JSON schema.
        """
        prompts = []
        if example_file:
            with open(example_file) as f:
                examples = json.load(f)
                if num_shots > 0:
                    examples = examples[:num_shots]
        else:
            examples = []

        if len(questions) == 0:
            return [], None

        for question in questions:
            messages, json_schema = fill_template_file(template_file, question)
            prompts.append(messages)

        output = {"prompts": prompts, "json_schema": json_schema}

        if not save_as:
            # Save prompts to generation_logs directory
            log_dir = Path("generation_logs")
            log_dir.mkdir(exist_ok=True)
            prompts_hash = self._get_prompts_hash(prompts)

            save_as = log_dir / f"prompts_{prompts_hash}.json"

        with open(save_as, "w") as f:
            json.dump(output, f, ensure_ascii=False, indent=4)

        if self.verbose:
            dump = json.dumps(prompts[0], ensure_ascii=False, indent=2)
            dump = dump.replace("\\n", "\n")
            logger.info(f"First prompt:\n{dump}")
            logger.info(f"Generated {len(prompts)} prompts")
            logger.info(f"Saved prompts to: {save_as}")

            if self.safe_mode:
                input("Press enter to continue...")

        return prompts, json_schema

    def get_rates(self):
        """
        Retrieves the input, cached input, and output rates for the current model.
        Returns:
            Tuple containing input rate, cached input rate, and output rate.
        """
        input_rate = input_rates.get(self.model_name, 0)
        cached_input_rate = cached_input_rates.get(self.model_name, 0)
        output_rate = output_rates.get(self.model_name, 0)

        return input_rate, cached_input_rate, output_rate

    def calculate_cost(self, stats, verbose=False):
        """
        Calculates the cost of generating a list of prompts.
        Args:
            stats: Dictionary containing statistics for the prompts.
            verbose: If True, enables verbose logging.
        Returns:
            The calculated cost.
        """
        cached_tokens = stats.get("total_cached_tokens", 0)

        input_rate, cached_input_rate, output_rate = self.get_rates()

        input_cost = (
            (stats["total_prompt_tokens"] - cached_tokens) / 1000000 * input_rate
        )
        input_cost += cached_tokens / 1000000 * cached_input_rate
        if verbose:
            logger.info(
                "{} actual input tokens, {} cached (${:0.2f})".format(
                    stats["total_prompt_tokens"], cached_tokens, input_cost
                )
            )
        output_cost = stats["total_completion_tokens"] / 1000000 * output_rate
        if verbose:
            logger.info(
                "{} actual output tokens (${:0.2f})".format(
                    stats["total_completion_tokens"], output_cost
                )
            )
        cost = input_cost + output_cost

        if verbose:
            logger.info("Actual Total cost: ${:0.2f}".format(cost))

        return cost

    def estimate_cost_with_sample(
        self,
        prompts,
        sample_size=3,
        json_schema=None,
        verbose=True,
        max_tokens=None,
    ):
        """
        Estimates the cost of generating prompts using a sample.
        Args:
            prompts: List of prompts to estimate cost for.
            sample_size: Number of prompts to use for estimation.
            json_schema: Optional JSON schema to use for validation.
            verbose: If True, enables verbose logging.
            max_tokens: Maximum number of tokens to use.
        Returns:
            Estimated cost and cached statistics for the prompts.
        """
        sample_size = min(sample_size, len(prompts))
        if sample_size == 0:
            return None

        # Check cache first
        cached_stats = self._get_cached_stats(prompts)
        if cached_stats is not None:
            if verbose:
                logger.info("Using cached stats")
            estimated_stats = {
                "total_prompt_tokens": cached_stats["prompt_tokens"] * len(prompts),
                "total_cached_tokens": cached_stats["cached_tokens"]
                * (len(prompts) - 1),
                "total_completion_tokens": cached_stats["completion_tokens"]
                * len(prompts),
            }
            estimated_cost = self.calculate_cost(estimated_stats)
            if verbose:
                logger.info(
                    f"Average tokens per prompt: {cached_stats['prompt_tokens']:.1f} input "
                    f"({cached_stats['cached_tokens']:.1f} cached), "
                    f"{cached_stats['completion_tokens']:.1f} output"
                )
                logger.info(
                    f"Estimated total number of input tokens: {round(estimated_stats['total_prompt_tokens'])}"
                )
                logger.info(
                    f"Estimated total number of output tokens: {round(estimated_stats['total_completion_tokens'])}"
                )
                input_rate, cached_input_rate, output_rate = self.get_rates()
                logger.info(
                    f"Based on ${input_rate}/M input rate, ${cached_input_rate}/M cached input rate, ${output_rate}/M output rate"
                )
                logger.info(
                    f"Estimated cost for {len(prompts)} prompts on {self.model_name}: ${estimated_cost:.2f}"
                )
                logger.info(
                    "Note: The rates are subject to change, please double check."
                )

            return estimated_cost, cached_stats

        # Get a sample of prompts
        sample_indices = random.sample(range(len(prompts)), sample_size)
        sample_prompts = [prompts[i] for i in sample_indices]

        # Save original model name and temporarily switch to cheaper one
        original_model = self.model_name
        cheaper_model = cheaper_model_map.get(self.model_name, self.model_name)

        if verbose:
            logger.info(
                f"Estimating cost using {sample_size} samples with {cheaper_model}"
            )
        self.model_name = cheaper_model
        self.safe_mode = False

        try:
            # Run the sample
            _, _, stats = self.prompt(
                sample_prompts, json_schema=json_schema, max_tokens=max_tokens
            )

            self.safe_mode = True

            # Calculate per-prompt stats
            stats_per_prompt = {
                "prompt_tokens": stats["total_prompt_tokens"] / sample_size,
                "cached_tokens": (
                    stats["total_cached_tokens"] / (sample_size - 1)
                    if sample_size > 1
                    else 0
                ),
                "completion_tokens": stats["total_completion_tokens"] / sample_size,
            }
            # Save stats to cache
            self._save_stats_cache(prompts, stats_per_prompt)

            # Estimate total cost for actual model
            total_prompts = len(prompts)
            estimated_tokens = {
                "total_prompt_tokens": stats_per_prompt["prompt_tokens"]
                * total_prompts,
                "total_cached_tokens": stats_per_prompt["cached_tokens"]
                * (total_prompts - 1),
                "total_completion_tokens": stats_per_prompt["completion_tokens"]
                * total_prompts,
            }

            # Get costs for actual model
            self.model_name = original_model
            estimated_cost = self.calculate_cost(estimated_tokens)

            if verbose:
                logger.info(
                    f"Average tokens per prompt: {stats_per_prompt['prompt_tokens']:.1f} input "
                    f"({stats_per_prompt['cached_tokens']:.1f} cached), "
                    f"{stats_per_prompt['completion_tokens']:.1f} output"
                )
                logger.info(
                    f"Estimated total number of input tokens: {round(estimated_tokens['total_prompt_tokens'])}"
                )
                logger.info(
                    f"Estimated total number of output tokens: {round(estimated_tokens['total_completion_tokens'])}"
                )
                input_rate, cached_input_rate, output_rate = self.get_rates()
                logger.info(
                    f"Based on ${input_rate}/M input rate, ${cached_input_rate}/M cached input rate, ${output_rate}/M output rate"
                )
                logger.info(
                    f"Estimated cost for {len(prompts)} prompts on {self.model_name}: ${estimated_cost:.2f}"
                )
                logger.info(
                    "Note: These rates are subject to change, please double check."
                )

            return estimated_cost, stats_per_prompt

        finally:
            # Restore original model
            self.model_name = original_model

    def _get_prompts_hash(self, prompts):
        """
        Generates a hash for the list of prompts.
        Args:
            prompts: List of prompts to generate a hash for.
        Returns:
            The generated hash.
        """
        # Convert prompts to a string and encode
        prompts_str = json.dumps(prompts, sort_keys=True)
        return hashlib.md5(prompts_str.encode()).hexdigest()

    def _generate_cache_key(
        self, prompt, model_name, temperature, frequency_penalty, stop
    ):
        """
        Generates a cache key for the given prompt and model.
        Args:
            prompt: The prompt to generate a cache key for.
            model_name: The name of the model.
            temperature: The temperature to use.
            frequency_penalty: The frequency penalty to use.
            stop: The stop sequence to use.
        Returns:
            The generated cache key.
        """
        key = f"{prompt}-{model_name}-{temperature}-{frequency_penalty}-{stop}"
        return hashlib.sha256(key.encode()).hexdigest()

    def prompt(
        self,
        prompts: List[str],
        temperature=0.0,
        frequency_penalty=0.0,
        stop=[],
        json_schema=None,
        max_tokens=1000,
        log_probs=False,
        save_as=None,
        dont_use_cached=False,
    ):
        """
        Generates text using the provided prompts.
        Args:
            prompts: List of prompts to generate text for.
            temperature: The temperature to use.
            frequency_penalty: The frequency penalty to use.
            stop: The stop sequence to use.
            json_schema: Optional JSON schema to use for validation.
            max_tokens: Maximum number of tokens to use.
            log_probs: If True, enables logging of probabilities.
            save_as: Optional path to save the generated text.
            dont_use_cached: If True, disables the use of cached responses.
        Returns:
            List of generated text, details, and statistics.
        """
        estimated_stats = None

        if frequency_penalty == 0.0:
            print("Frequency penalty is 0.0.")
        else:
            print("Frequency penalty is {}.".format(frequency_penalty))

        if self.safe_mode and len(prompts) > 0:
            estimated_cost, estimated_stats = self.estimate_cost_with_sample(
                prompts, json_schema=json_schema, max_tokens=max_tokens
            )
            if estimated_cost is not None:
                if (
                    input(
                        f"Estimated cost on {self.model_name}: ${estimated_cost:.2f}. Continue? (y/n): "
                    )
                    != "y"
                ):
                    logger.info("Aborting due to cost")
                    exit()
            if json_schema is not None:
                logger.info("Using JSON schema: {}".format(json_schema))
            else:
                logger.info("Not using JSON schema")
            logger.info("Temperature: {}".format(temperature))
            if input(f"Continue? (y/n): ") != "y":
                logger.info("Aborting due to configuration")
                exit()
        answers = []
        completions = []

        check = 0
        stats = {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_cached_tokens": 0,
        }
        for llm_input in tqdm(prompts, disable=not self.show_progress):
            try:
                cache_key = self._generate_cache_key(
                    llm_input, self.model_name, temperature, frequency_penalty, stop
                )

                if cache_key in self.prompt_cache and not dont_use_cached:
                    logger.info("Using cached response")
                    answer = self.prompt_cache[cache_key]
                else:
                    if json_schema is not None:
                        response_format = {
                            "type": "json_schema",
                            "json_schema": json_schema,
                        }
                    else:
                        response_format = None

                    if (
                        self.safe_mode
                        and estimated_stats.get("completion_tokens", None) is not None
                    ):
                        max_tokens = min(
                            max_tokens,
                            (
                                max(100, int(estimated_stats["completion_tokens"] * 4))
                                if estimated_stats
                                else max_tokens
                            ),
                        )

                    if "5" in self.model_name:
                        completion = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=llm_input,
                            seed=self.seed,
                            response_format=response_format,
                            frequency_penalty=frequency_penalty,
                            logprobs=log_probs,
                            top_logprobs=20 if log_probs else None,
                        )
                    else:
                        completion = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=llm_input,
                            temperature=temperature,
                            stop=stop,
                            seed=self.seed,
                            response_format=response_format,
                            max_completion_tokens=max_tokens,
                            frequency_penalty=frequency_penalty,
                            logprobs=log_probs,
                            top_logprobs=20 if log_probs else None,
                        )
                    answer = completion.choices[0].message.content
                    completions.append(completion)

                    self.prompt_cache[cache_key] = answer

                    self._save_prompt_cache()

                    with open("temp.json", "w") as f:
                        f.write(json.dumps(completion.model_dump()))
                    stats["total_prompt_tokens"] += completion.usage.prompt_tokens
                    stats[
                        "total_cached_tokens"
                    ] += completion.usage.prompt_tokens_details.cached_tokens
                    stats[
                        "total_completion_tokens"
                    ] += completion.usage.completion_tokens

                    if (
                        self.safe_mode
                        and estimated_stats
                        and (
                            completion.usage.completion_tokens
                            > (estimated_stats["completion_tokens"] * 2.5)
                        )
                        and check < 3
                    ):
                        logger.warning(
                            "Actual output tokens higher than expected.({} instead of {}).".format(
                                completion.usage.completion_tokens,
                                estimated_stats["completion_tokens"],
                            )
                        )
                        logger.info(answer)
                        if input("Continue? (y/n): ") != "y":
                            logger.info("Aborting due to cost")
                            exit()
                        check += 1

            except OpenAIError as e:
                logging.error(
                    "Error with chatgpt after {} questions".format(len(answers))
                )
                logging.error(e)
                return answers, None, stats

            answers.append(answer)

        stats["per_prompt_tokens"] = stats["total_prompt_tokens"] / len(prompts)
        stats["per_cached_tokens"] = (
            stats["total_cached_tokens"] / (len(prompts) - 1) if len(prompts) > 1 else 0
        )
        stats["per_completion_tokens"] = stats["total_completion_tokens"] / len(prompts)

        stats["prompts_hash"] = self._get_prompts_hash(prompts)
        log_dir = Path("generation_logs")
        log_dir.mkdir(exist_ok=True)
        with open(log_dir / f"stats_{stats['prompts_hash']}.json", "w") as f:
            json.dump(stats, f)
        with open(log_dir / f"answers_{stats['prompts_hash']}.json", "w") as f:
            json.dump(answers, f)

        if save_as:
            with open(save_as, "w") as f:
                json.dump([str(completion) for completion in completions], f)

        if log_probs:
            details = []
            for completion in completions:
                choice_details = []
                for choice in completion.choices:
                    choice_details.append(
                        {
                            "finish_reason": choice.finish_reason,
                            "index": choice.index,
                            "logprobs": [
                                {
                                    "token": chat_completion_token_logprob.token,
                                    "bytes": chat_completion_token_logprob.bytes,
                                    "logprob": chat_completion_token_logprob.logprob,
                                }
                                for chat_completion_token_logprob in choice.logprobs.content[
                                    0
                                ].top_logprobs
                            ],
                        }
                    )
                details.append(choice_details)
        else:
            details = None

        return answers, details, stats

    def _get_stats_cache_path(self):
        """
        Gets the path to the stats cache directory.
        Returns:
            Path to the stats cache file.
        """
        cache_dir = Path("generation_logs")
        cache_dir.mkdir(exist_ok=True)
        return cache_dir / "stats_cache.json"

    def _get_prompt_cache_path(self):
        """
        Gets the path to the prompt cache directory.
        Returns:
            Path to the prompt cache file.
        """
        cache_dir = Path("generation_logs")
        cache_dir.mkdir(exist_ok=True)
        return cache_dir / "prompt_cache.json"

    def _save_prompt_cache(self):
        """
        Saves the prompt cache to file.
        """
        cache_path = self._get_prompt_cache_path()
        with cache_path.open("w") as f:
            json.dump(self.prompt_cache, f)

    def _get_cached_stats(self, prompts):
        """
        Gets cached statistics for the given prompts if they exist.
        Args:
            prompts: List of prompts to get cached statistics for.
        Returns:
            Cached statistics or None if not found.
        """
        cache_path = self._get_stats_cache_path()
        if not cache_path.exists():
            return None

        prompts_hash = self._get_prompts_hash(prompts)
        try:
            with cache_path.open("r") as f:
                cache = json.load(f)
            return cache.get(prompts_hash)
        except (json.JSONDecodeError, IOError):
            return None

    def _save_stats_cache(self, prompts, stats):
        """
        Saves statistics to the cache.
        Args:
            prompts: List of prompts to save statistics for.
            stats: Statistics to save.
        """
        cache_path = self._get_stats_cache_path()
        prompts_hash = self._get_prompts_hash(prompts)

        # Load existing cache
        cache = {}
        if cache_path.exists():
            try:
                with cache_path.open("r") as f:
                    cache = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

        # Update cache
        cache[prompts_hash] = stats

        # Save updated cache
        with cache_path.open("w") as f:
            json.dump(cache, f)
