import argparse
import json
import logging
import os

from generator import Generator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--prompt-template",
        "-pt",
        type=str,
        help="Path to prompt template",
        default="prompts/das_encode.txt",
    )
    parser.add_argument(
        "--functions-file",
        "-f",
        type=str,
        help="Path to functions file template",
        default="prompts/das_functions.json",
    )
    parser.add_argument(
        "--context-template",
        "-ct",
        type=str,
        help="Path to context template",
        default="prompts/das_context.txt",
    )

    parser.add_argument(
        "--input", "-i", type=str, help="Path to input data file", required=True
    )
    parser.add_argument("--save-full-prompts-as", "-sa", type=str, default=None)
    parser.add_argument("--model", "-m", type=str, default="gpt-4o-mini")
    parser.add_argument("--max_instances", "-n", type=int, default=100)
    parser.add_argument("--output_dir", "-o", type=str, default="results/")
    parser.add_argument("--run_name", "-r", type=str, default="dialogue_acts")
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--temperature", "-t", type=float, default=0)
    parser.add_argument("--safe-mode", "-sm", action="store_true", default=False)

    args = parser.parse_args()

    # look for "# start functions" in prompt_template file
    if args.prompt_template:
        with open(args.prompt_template) as f:
            prompt_template = f.read()
        if "# start functions" in prompt_template and not args.functions_file:
            parser.error("--functions-file is required with this prompt template")

    if args.output_dir[-1] != "/":
        args.output_dir += "/"

    # if the output_dir doesn't exist, make it
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    return args


def preprocess_conversation(conversation):
    return "\n".join(conversation["dialogue"])


def preprocess_dailydialogue(data, functions):
    conversations = [
        {"functions": functions, "conversation": preprocess_conversation(conversation)}
        for conversation in data
    ]
    return conversations


def create_encoding_prompts(data, generator, prompt_template, functions_file=None):
    assert prompt_template, "Prompt template is required"
    assert os.path.exists(
        prompt_template
    ), "Prompt template file does not exist: {}".format(prompt_template)

    if functions_file:
        assert os.path.exists(
            functions_file
        ), "Functions file does not exist: {}".format(functions_file)

        with open(functions_file) as f:
            functions = json.load(f)

        conversations = preprocess_dailydialogue(data, functions)

        prompts, json_schema = generator.build_prompts(conversations, prompt_template)

    return prompts, json_schema


def create_context_prompts(data, generator, prompt_template):
    assert prompt_template, "Prompt template is required"
    assert os.path.exists(
        prompt_template
    ), "Prompt template file does not exist: {}".format(prompt_template)

    prompts, json_schema = generator.build_prompts(data, prompt_template)

    return prompts, json_schema


def read_responses(data, responses):
    responses_decoded = []
    for j, response in enumerate(responses):
        try:
            responses_decoded.append(json.loads(response))
        except:
            logger.error("Failed to load response {}".format(j))
            assert False

    for j, line in enumerate(data):
        if "das_encoding" in responses_decoded[j]:
            line["das_encoding"] = responses_decoded[j]["das_encoding"]
        elif "context" in responses_decoded[j]:
            line["context"] = responses_decoded[j]["context"]
        else:
            logger.error("Failed to load response {}".format(j))
            assert False

    return data


def save_data(data, output_path):
    output_path = output_path + "_encoded.json"
    logger.info("Saving encoded data to: {}".format(output_path))
    with open(output_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def main():
    args = parse_args()
    with open(args.input) as f:
        data = json.load(f)

    output_path = args.output_dir + f"{args.run_name}_{args.model}"

    if args.max_instances > 0:
        data = data[: args.max_instances]

    logger.info("Number of instances: {}".format(len(data)))

    generator = Generator(
        args.model, verbose=args.safe_mode, safe_mode=args.safe_mode, seed=args.seed
    )
    assert generator, "Unable to make generator"
    logger.info("Using model: {}".format(generator.model))

    prompts, json_schema = create_encoding_prompts(
        data, generator, args.prompt_template, args.functions_file
    )

    responses, _, _ = generator.prompt(
        prompts, json_schema=json_schema, temperature=args.temperature
    )

    data = read_responses(data, responses)

    context_prompts, context_json_schema = create_context_prompts(
        data, generator, args.context_template
    )

    responses, _, _ = generator.prompt(
        context_prompts, json_schema=context_json_schema, temperature=args.temperature
    )

    data = read_responses(data, responses)

    save_data(data, output_path)


if __name__ == "__main__":
    main()
