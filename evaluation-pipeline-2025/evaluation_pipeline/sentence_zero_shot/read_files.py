# File: read_files.py
# -------------------

import json
import pathlib
from typing import Any


def read_files(data_path: pathlib.Path, task: str, full_sentence_scores: bool) -> list[dict[str, str]]:
    """Takes the path to a data directory and a task, reads the JSONL datafiles
    in the directory and returns a list of dictionaries containing all the
    information used by the evaluation.

    Args:
        data_path(pathlib.Path): The path to a data directory containing JSONL
            files.
        task(str): The task of the data (for example blimp).

    Returns:
        list[dict[str, str]]: A list of dictionaries containing the information
            to evaluate the given task.
    """
    data = []
    for filename in data_path.iterdir():
        if filename.suffix != ".jsonl":
            continue

        with filename.open("r") as f:
            for line in f:
                data.append(decode(line, filename, task, full_sentence_scores))

    return data


def decode(line: str, file_name: pathlib.Path, task: str, full_sentence_scores: bool) -> dict[str, str]:
    """This function takes a line of a JSONL file and returns a dictionary of terms to be used by the evaluation.

    Args:
        line(str): A JSONL line from a datafile.
        file_name(pathlib.Path): The file name the line comes from.
        task(str): The task we are evaluating, this tells us what needs to be imported.

    Returns:
        dict[str, str]: A dictionary with values used for evaluation
    """

    raw_dict = json.loads(line.strip())

    if task == "blimp":
        data_dict = decode_blimp(raw_dict, file_name)
    elif task == "ewok":
        data_dict = decode_ewok(raw_dict, full_sentence_scores)
    elif task == "wug":
        data_dict = decode_wug_adj_nominalization(raw_dict)
    elif task == "entity_tracking":
        data_dict = decode_entity_tracking(raw_dict, file_name)
    else:
        raise NotImplementedError(f"The task {task} is not implemented! Please implement it or choose one of the implemented tasks.")

    return data_dict


def decode_blimp(raw_dict: dict[str, Any], file_name: pathlib.Path) -> dict[str, str]:
    """This function takes a dictionary of a single datapoint of a BLiMP datafile and returns a dictionary of terms to be used by the evaluation.

    Args:
        raw_dict(dict[str, Any]): A dictionary from a single datapoint of a BLiMP datafile.
        file_name(pathlib.Path): When no UID is mentioned, we take the file name.

    Returns:
        dict[str, str]: A dictionary with values used for evaluation
    """
    if "field" in raw_dict:
        pair = {
            "sentences": [raw_dict["sentence_good"], raw_dict["sentence_bad"]],
            "prefixes": [None, None],
            "completions": [raw_dict["sentence_good"], raw_dict["sentence_bad"]],
            "label": 0,
            "field": raw_dict["field"],
            "UID": raw_dict["UID"],
            "linguistics_term": raw_dict["linguistics_term"],
        }
        if pair["field"] == "syntax_semantics":  # Standardizing the style of this field
            pair["field"] = "syntax/semantics"
    else:  # For the supplemetal tasks, there is no field or UID
        pair = {
            "sentences": [raw_dict["sentence_good"], raw_dict["sentence_bad"]],
            "prefixes": [None, None],
            "completions": [raw_dict["sentence_good"], raw_dict["sentence_bad"]],
            "label": 0,
            "field": "supplement",
            "UID": file_name.stem,
            "linguistics_term": "supplement",
        }

    return pair


def decode_ewok(raw_dict: dict[str, Any], full_sentence_scores: bool) -> dict[str, str]:
    """This function takes a dictionary of a single datapoint of a EWoK datafile
    and returns a dictionary of terms to be used by the evaluation.

    Args:
        raw_dict(dict[str, Any]): A dictionary from a single datapoint of a
            EWoK datafile.

    Returns:
        dict[str, str]: A dictionary with values used for evaluation
    """
    if full_sentence_scores:
        completions = [" ".join([raw_dict["Context1"], raw_dict["Target1"]]), " ".join([raw_dict["Context1"], raw_dict["Target2"]])]
    else:
        completions = [raw_dict["Target1"], raw_dict["Target2"]]
    pair = {
        "sentences": [" ".join([raw_dict["Context1"], raw_dict["Target1"]]), " ".join([raw_dict["Context1"], raw_dict["Target2"]])],
        "prefixes": [raw_dict["Context1"], raw_dict["Context2"]],
        "completions": completions,
        "label": 0,
        "UID": raw_dict["Domain"],
        "context_type": raw_dict["ContextType"],
        "context_contrast": raw_dict["ContextDiff"],
        "target_contrast": raw_dict["TargetDiff"],
    }

    return pair


def decode_wug_adj_nominalization(raw_dict: dict[str, Any]) -> dict[str, str]:
    """This function takes a dictionary of a single datapoint of the wug test
    datafile and returns a dictionary of terms to be used by the evaluation.

    Args:
        raw_dict(dict[str, Any]): A dictionary from a single datapoint of a BLiMP datafile.
        file_name(pathlib.Path): When no UID is mentioned, we take the file name.

    Returns:
        dict[str, str]: A dictionary with values used for evaluation
    """
    pair = {
        "sentences": raw_dict["sentences"].split('\t'),
        "prefixes": [None],
        "completions": raw_dict["sentences"].split('\t'),
        "ratio": float(raw_dict["ratio"]),
        "label": 0,
        "UID": "wug_adj_nominalization",
    }

    return pair


def decode_entity_tracking(raw_dict: dict[str, Any], file_name: pathlib.Path) -> dict[str, str]:
    """This function takes a dictionary of a single datapoint of a Entity Tracking datafile
    and returns a dictionary of terms to be used by the evaluation.

    Args:
        raw_dict(dict[str, Any]): A dictionary from a single datapoint of a
            Entity Tracking datafile.

    Returns:
        dict[str, str]: A dictionary with values used for evaluation
    """
    subset = f'{file_name.stem}_{raw_dict["numops"]}_ops'
    pair = {
        "sentences" : [raw_dict["input_prefix"] + option for option in raw_dict["options"]],
        "prefixes": [raw_dict["input_prefix"] for _ in raw_dict["options"]],
        "completions" : [option for option in raw_dict["options"]],
        "label" : 0,
        "UID" : subset
    }

    return pair
