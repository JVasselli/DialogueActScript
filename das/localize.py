import argparse
import json
import logging
import os

from generator import Generator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt_template", "-pt", type=str, help="Path to prompt template"
    )
    parser.add_argument(
        "--input", "-i", type=str, help="Path to input data file", required=True
    )
    parser.add_argument("--model", "-m", type=str, default="gpt-4o-mini")
    parser.add_argument("--max_instances", "-n", type=int, default=100)
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--language", "-l", type=str, default="English")
    parser.add_argument("--output_dir", "-o", type=str, default="results/")
    parser.add_argument("--safe_mode", "-sm", action="store_true", default=False)
    parser.add_argument("--temperature", "-t", type=float, default=0)

    args = parser.parse_args()

    if args.output_dir[-1] != "/":
        args.output_dir += "/"

    # if the output_dir doesn't exist, make it
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    return args


def localize(
    generator,
    conversations,
    prompt_template,
    temperature=0.0,
):
    prompts, json_schema = generator.build_prompts(conversations, prompt_template)

    responses, _, _ = generator.prompt(
        prompts, json_schema=json_schema, temperature=temperature, frequency_penalty=0.0
    )

    return responses


def preprocess_conversation(conversation):
    turns = []
    for i, turn in enumerate(conversation["das_encoding"]):
        sub_turns = []
        speaker_id = turn["speaker_id"]

        if speaker_id == "1":
            speaker_id = "A"
        elif speaker_id == "2":
            speaker_id = "B"

        for function in turn["functions"]:
            sub_turns.append(f"{speaker_id}.{function}")
        turns.append(f"{i+1}:" + "; ".join(sub_turns))
    return "\n".join(turns)


def localize_and_process(generator, data, prompt_template, temperature, key):
    responses = localize(
        generator,
        data,
        prompt_template=prompt_template,
        temperature=temperature,
    )

    for i, line in enumerate(responses):
        try:
            json_line = json.loads(line)
            data[i][key] = json_line[key]
        except:
            print("Failed on index {}".format(i))

    return data


def main():
    args = parse_args()
    with open(args.input) as f:
        data = json.load(f)

    data = data[: args.max_instances]

    generator = Generator(
        args.model, verbose=args.safe_mode, safe_mode=args.safe_mode, seed=args.seed
    )
    assert generator, "Unable to make generator"

    for entry in data:
        entry["turns"] = preprocess_conversation(entry)
        entry["language"] = args.language

    data = localize_and_process(
        generator,
        data,
        prompt_template="prompts/das_localize_context.txt",
        temperature=args.temperature,
        key="localized_context",
    )

    data = localize_and_process(
        generator,
        data,
        key="localized_das",
        prompt_template="prompts/das_localize.txt",
        temperature=args.temperature,
    )

    output_path = args.output_dir + f"{args.model}_{args.language}_localized.json"
    with open(output_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
