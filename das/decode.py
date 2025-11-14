import json
import logging
import os

from generator import Generator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt-template", "-pt", type=str, help="Path to prompt template", default="prompts/das_decode.txt"
    )
    parser.add_argument(
        "--input", "-i", type=str, help="Path to input data file", required=True
    )
    parser.add_argument("--model", "-m", type=str, default="gpt-4o-mini")
    parser.add_argument("--max_instances", "-n", type=int, default=100)
    parser.add_argument("--output_dir", "-o", type=str, default="results/")
    parser.add_argument("--run_name", "-r", type=str, default="dialogue_acts")
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--temperature", "-t", type=float, default=0.2)
    parser.add_argument("--language", "-l", type=str, default="English")
    parser.add_argument("--key", "-k", type=str, default="localized_das")
    parser.add_argument("--context_key", "-c", type=str, default="localized_context")
    parser.add_argument("--safe_mode", type=bool, default=False)

    args = parser.parse_args()

    if args.output_dir[-1] != "/":
        args.output_dir += "/"

    return args


def decode(
    model,
    conversations,
    max_instances,
    prompt_template="prompts/decode.txt",
    temperature=0.2,
    seed=42,
    safe_mode=False,
):
    generator = Generator(
        model, verbose=safe_mode, safe_mode=safe_mode, seed=seed
    )
    assert generator, "Unable to make generator"

    conversations = conversations[:max_instances]

    prompts, json_schema = generator.build_prompts(
        conversations, prompt_template
    )

    # Generate responses
    responses, _, stats = generator.prompt(
        prompts, json_schema=json_schema, temperature=temperature
    )

    responses = [json.loads(r) for r in responses]

    return responses


def preprocess_conversation(conversation, key):
    turns = []
    for i, turn in enumerate(conversation[key]):
        sub_turns = []
        speaker_id = turn["speaker_id"]

        if speaker_id == "1":
            speaker_id = "A"
        elif speaker_id == "2":
            speaker_id = "B"

        functions_list = turn["functions"]
        if type(functions_list) == str:
            functions_list = functions_list.split(";")

        for function in functions_list:
            sub_turns.append(f"{speaker_id}.{function}")
        turns.append(f"{i+1}:" + "; ".join(sub_turns))
    return "\n".join(turns)


def preprocess_dailydialogue(data, key, context_key):
    conversations = [
        {
            "turns": preprocess_conversation(conversation, key),
            "context": conversation[context_key],
        }
        for conversation in data
    ]
    return conversations


def main():
    args = parse_args()
    with open(args.input) as f:
        data = json.load(f)

    assert os.path.exists(
        args.prompt_template
    ), "Prompt template file does not exist: {}".format(args.prompt_template)

    conversations = preprocess_dailydialogue(data, args.key, args.context_key)
    for conversation in conversations:
        conversation["language"] = args.language

    responses = decode(
        args.model,
        conversations,
        args.max_instances,
        args.prompt_template,
        args.temperature,
        args.seed,
    )

    # Add the responses to the input data
    for i, line in enumerate(data):
        line[f"decoded_{args.language}"] = responses[i]["generated_conversation"]

    # Write responses to file
    output_path = (
        args.output_dir + f"{args.model}_{args.language}_decoded_full.json"
    )
    logger.info("Saving responses to: {}".format(output_path))
    with open(output_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
