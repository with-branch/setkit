from typing import Any, List, Mapping, Sequence, Tuple

import textwrap

TAB_SIZE = 4
TAB = "".join([" " for i in range(TAB_SIZE)])


def condense_with_elipse(string: str, max_width: int) -> str:
    if len(string) > max_width:
        string = string[: max_width - 3]
        string += "..."
    return string


def format_data_element(element: Any) -> str:
    if isinstance(element, str):
        return element.strip()
    elif isinstance(element, float):
        return f"{element:.3f}"
    elif isinstance(element, type):
        return element.__name__
    else:
        return str(element)


def format_docstring(docstring: dict, display_width: int) -> str:
    formatted_docstring = ""
    docstring_lines = docstring.split("\n")
    for line in docstring_lines:
        wrapped_lines = textwrap.wrap(line, width=display_width)
        for line in wrapped_lines:
            formatted_docstring += line + "\n"
    return formatted_docstring[:-1]


def format_statistics(
    statistics: dict, display_width: int, indent: bool = False
) -> str:
    def _format_statistics(statistics: dict, display_width: int):
        lines = []
        for key, value in statistics.items():
            if isinstance(value, Mapping):
                lines.append(f"{key}:")
                nested_lines = _format_statistics(value, display_width - TAB_SIZE)
                lines += [TAB + line for line in nested_lines]
            else:
                if isinstance(value, Sequence) and not isinstance(value, str):
                    value = [format_data_element(element) for element in value]
                else:
                    value = format_data_element(value)
                wraped_stat_lines = textwrap.wrap(
                    f"{key}: {value}", width=display_width
                )
                lines += wraped_stat_lines
        return lines

    formatted_stats_string = ""
    if indent:
        lines = _format_statistics(statistics, display_width - TAB_SIZE)
    else:
        lines = _format_statistics(statistics, display_width)
    for line in lines:
        if indent:
            formatted_stats_string += TAB
        formatted_stats_string += line + "\n"
    return formatted_stats_string[:-1]


def flatten_example(example: dict) -> list:
    id, data, target = example["id"], example["data"], example["target"]
    flat_example = [id]

    if isinstance(data, Sequence) and not isinstance(data, str):
        flat_example += data
    elif isinstance(data, Mapping):
        flat_example += data.values()
    else:
        flat_example.append(data)

    if isinstance(target, Mapping):
        flat_example += target.values()
    else:
        flat_example.append(target)

    return flat_example


def get_flat_column_names(example: dict) -> List[str]:
    _, data, target = example["id"], example["data"], example["target"]

    column_names = ["id"]
    if isinstance(data, Sequence) and not isinstance(data, str):
        column_names += [f"feature_{i}" for i in range(len(data))]
    elif isinstance(data, Mapping):
        column_names += data.keys()
    elif isinstance(data, str):
        column_names.append("text")
    else:
        column_names.append("data")

    if isinstance(target, Mapping):
        column_names += target.keys()
    else:
        column_names.append("target")

    return column_names


def format_examples_tabular(
    examples: List[dict], table_width: int, indent: bool = False
) -> str:
    if indent:
        table_width = table_width - TAB_SIZE
    column_names = get_flat_column_names(examples[0])
    examples = [flatten_example(example) for example in examples]

    num_columns = len(column_names)
    column_seperator = " "
    column_width = (
        table_width - (len(column_seperator) * (num_columns))
    ) // num_columns

    formatted_examples_string = ""

    divider_string = ""
    if indent:
        formatted_examples_string += TAB
    for column_header in column_names:
        column_header = condense_with_elipse(column_header, column_width)
        formatted_examples_string += (
            f"{column_header:<{column_width}}" + column_seperator
        )
        divider_string += "".join(["-" for i in range(column_width)]) + column_seperator
    formatted_examples_string += "\n"
    if indent:
        formatted_examples_string += TAB
    formatted_examples_string += divider_string + "\n"

    for example in examples:
        example_string = ""
        for example_element in example:
            column_formatted_element = condense_with_elipse(
                format_data_element(example_element), column_width
            )
            example_string += (
                f"{column_formatted_element:<{column_width}}" + column_seperator
            )
        if indent:
            formatted_examples_string += TAB
        formatted_examples_string += example_string + "\n"

    return formatted_examples_string[:-1]
