import json
import random
from typing import Any, Dict, List


def replace_placeholders(text: str, instance: Dict[str, Any], prefix: str = "") -> str:
    """
    Replaces placeholders in a string with values from a dictionary.
    Args:
        text: string with placeholders
        instance: dictionary with values to fill in
        prefix: prefix for the keys in the dictionary (used for examples)
    """
    for key, value in instance.items():
        if type(value) == list and len(value) > 0 and type(value[0]) == str:
            value = ", ".join(value)
        if type(value) != str:
            value = str(value)
        text = text.replace(f"[{prefix}{key}]", value)
    return text


def process_examples(
    lines: List[str], examples: List[Dict[str, Any]], num_shots: int = -1
) -> str:
    """
    Processes the examples section of a template file.
    Args:
        lines: list of lines in the template file
        examples: list of dictionaries with information to fill in
        num_shots: number of examples to include in the output, -1 for all
    """
    example_template = ""
    for line in lines:
        if not line.startswith("#"):
            example_template += line.strip() + "\n"

    if num_shots > -1:
        num_examples = min(num_shots, len(examples))
        # randomly select num_examplse
        examples = random.sample(examples, num_examples)
    example_text = ""
    for i, example in enumerate(examples):
        example_text += replace_placeholders(
            example_template, example, prefix="example_"
        ).replace("[$index]", str(i + 1))
    return example_text


def process_loop(lines: List[str], items: List[Dict[str, Any]]) -> str:
    """
    Processes any loop through included data.
    Args:
        lines: list of lines in the template file
        items: list of dictionaries with information to fill in
    Returns:
        str: The processed text with all placeholders replaced
    """
    template = ""
    for line in lines:
        if not line.startswith("#"):
            template += line.strip() + "\n"
    item_text = ""
    for i, item in enumerate(items):
        item_text += replace_placeholders(template, item).replace(
            "[$index]", str(i + 1)
        )
    return item_text.rstrip()  # Remove trailing whitespace/newlines


def process_nested_section(lines: List[str], items: List[Dict[str, Any]]) -> str:
    """
    Processes a nested section with inner loops.
    Args:
        lines: list of lines in the template file
        items: list of dictionaries with outer section information
    Returns:
        str: The processed text with all placeholders replaced
    """
    result = []
    inner_section = []
    inner_section_key = ""

    # Process each item
    for i, item in enumerate(items):
        item_lines = []
        for line in lines:
            line = remove_indentation(line)

            if line.startswith("# start "):
                inner_section_key = line.replace("# start ", "")
                inner_section = []
            elif line.startswith("# end "):
                if line.replace("# end ", "") == inner_section_key:
                    if inner_section_key in item:
                        inner_text = process_loop(
                            inner_section, item[inner_section_key]
                        )
                        item_lines.append(inner_text)
                    inner_section_key = ""
            elif inner_section_key:
                inner_section.append(line)
            else:
                if type(item) == str:
                    processed_line = line.replace("[$index]", str(i + 1)).replace(
                        "[item]", str(item)
                    )
                else:
                    processed_line = replace_placeholders(line, item).replace(
                        "[$index]", str(i + 1)
                    )
                item_lines.append(processed_line)

        result.append("\n".join(line.rstrip() for line in item_lines))

    # Join all items with a single newline between them and add trailing newline
    return "\n".join(result)


def fill_template_file(
    template_file: str,
    question: Dict[str, Any],
    num_shots: int = -1,
):
    """
    Reads a template file and fills in placeholders with values from a dictionary.
    Args:
        template_file: path to the template file
        question: dictionary with values to fill in to the general prompt
        num_shots: number of examples to include, -1 for all
    Returns:
        List[Dict[str, str]]: List of message dictionaries with role and content
    """
    messages = []
    lines = [line for line in open(template_file, "r").readlines()]
    section_lines = []
    section_name = ""
    json_schema = None

    # First pass: Process all sections and store their content
    for i, line in enumerate(lines):
        line = line.rstrip()
        # Handle role markers first
        if line == "# system":
            messages.append({"role": "system", "content": ""})
            continue
        elif line == "# user":
            messages.append({"role": "user", "content": ""})
            continue
        elif line == "# assistant":
            messages.append({"role": "assistant", "content": ""})
            continue
        elif line == "# JSON schema":
            json_schema = ""
            continue

        line = remove_indentation(line)

        # Handle section markers
        if line.startswith("# start ") and section_name == "":
            section_name = line.replace("# start ", "")
        elif line.startswith("# end ") and section_name == line.replace("# end ", ""):
            if section_name == "examples":
                section_text = process_examples(
                    section_lines, question["examples"], num_shots
                )
            elif section_name in question:
                # Process the nested section if it exists in question
                section_text = process_nested_section(
                    section_lines,
                    question[section_name],
                )
            else:
                continue

            if len(messages) > 0:
                messages[-1]["content"] += section_text + "\n"
            else:
                messages.append({"role": "user", "content": section_text + "\n"})
            section_lines = []
            section_name = ""
        else:
            if section_name != "":
                section_lines.append(line)
            elif json_schema is not None:
                json_schema += line
            else:
                messages[-1]["content"] += replace_placeholders(line, question) + "\n"

    if json_schema is not None:
        json_schema = json.loads(json_schema)

    return messages, json_schema


def remove_indentation(line: str) -> str:
    if line.startswith("\t"):
        return line[1:]
    elif line.startswith("    "):
        return line[4:]
    return line
