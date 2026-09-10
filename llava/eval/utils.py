from argparse import Namespace
import glob
import json
from logging import getLogger, Formatter, StreamHandler, DEBUG, INFO
import os
import math
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import torch
from transformers.generation.utils import GreedySearchOutput


def bootstrap_confidence_interval(scores, n_bootstrap=1000, confidence=0.95, seed=42):
    """Compute bootstrap confidence interval for the mean of *scores*.

    Parameters
    ----------
    scores : array-like
        Per-sample metric values (e.g. 0/1 for accuracy).
    n_bootstrap : int
        Number of bootstrap resamples.
    confidence : float
        Confidence level (default 95 %).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    (lower, upper) : tuple of float
        Lower and upper bounds of the confidence interval.
    """
    scores = np.asarray(scores, dtype=float)
    rng = np.random.RandomState(seed)
    n = len(scores)
    boot_means = np.array(
        [np.mean(rng.choice(scores, size=n, replace=True)) for _ in range(n_bootstrap)]
    )
    alpha = (1 - confidence) / 2
    lower = float(np.percentile(boot_means, alpha * 100))
    upper = float(np.percentile(boot_means, (1 - alpha) * 100))
    return lower, upper

def read_json(
    file: str,
    iid: str,
    learning_type: str,
    filename="meta.json",
    image_col: str = "image_id",
    loss_col: str = "loss",
    experiment_col: str = "learning_type",
    embedding_size: int = 5120,
):
    with open(file) as f:
        meta = json.load(f)
    meta[image_col] = iid
    meta[experiment_col] = learning_type
    if loss_col not in meta.keys():
        meta[loss_col] = [0.0 for _ in range(embedding_size)]
    else:
        pass
    return meta


def read_jsons(
    root_dir: str,
    filename: str = "meta.json",
    image_col: str = "image_id",
    loss_col: str = "loss",
    experiment_col: str = "learning_type",
    export_cols: List[str] = ["learning_type", "image_id", "y_matched"],
    embedding_size: int = 5120,
):
    json_files = Path(root_dir).glob(f"**/{filename}")
    jsons = []
    for json_file in json_files:
        learning_type = str(json_file).split("/")[-3]
        image_id = str(json_file).split("/")[-2]
        meta = read_json(
            json_file,
            image_id,
            learning_type,
            image_col=image_col,
            loss_col=loss_col,
            experiment_col=experiment_col,
            embedding_size=embedding_size,
        )
        jsons.append(meta)
    df_out = pd.DataFrame(jsons)[export_cols + [loss_col]]
    return df_out
    # loss_cols = [f"loss_{i}" for i in range(embedding_size)]
    # df_out[loss_cols] = pd.DataFrame(df_out[loss_col].tolist())
    # return df_out[export_cols + loss_cols]


def read_jsonl(filepath: str):
    df = pd.read_json(filepath, orient="records", lines=True, dtype={"id": str})
    return df


def read_jsonls(args, train_set: str = "train"):
    """
    Read jsonl as dataframe
    """
    df_train = read_jsonl(f"{args.memes_path}/{train_set}.jsonl")
    df_test = read_jsonl(f"{args.memes_path}/{args.test_set}.jsonl")
    return df_train.drop_duplicates(), df_test.drop_duplicates()


def _set_handler(logger, handler, verbose: bool):
    """
    Prep handler
    """
    if verbose:
        handler.setLevel(DEBUG)
    else:
        handler.setLevel(INFO)
    formatter = Formatter(
        "%(asctime)s %(name)s:%(lineno)s [%(levelname)s]: %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def get_module_logger(verbose: bool = False, level=DEBUG):
    """
    Create logger
    """
    logger = getLogger(__name__)
    logger = _set_handler(logger, StreamHandler(), False)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def makedirs_recursive(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def remove_objects_w_names(dirname, file_strs, exception_str="."):
    for file_str in file_strs:
        for f in glob.glob(f"{dirname}/*{file_str}*"):
            if exception_str not in f:
                os.remove(f)


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def add_image_token(
    args: Namespace,
    line: Dict[str, str],
    default_im_start_token: str,
    default_image_token: str,
    default_im_end_token: str,
    is_multiple_questions: bool,
    image_placeholder: str,
    model_config: Dict[str, Any],
    image_col: str = "image",
    text_col: str = "text",
    answer_col: str = "answer",
):
    """
    Add image token to the llava format question.

    For multi-input (ICL) entries the ``__sep__``-delimited fields may
    contain K+1 parts (K ICL references + 1 query).  The function
    returns:
      - ``qs``:  list of K+1 processed question strings
      - ``ans``: list of K reference answer strings (or a single string
        for legacy 1-shot)
      - ``image_file``: list of K+1 image paths
    """
    image_token_se = default_im_start_token + default_image_token + default_im_end_token
    if is_multiple_questions:
        img_parts = line[image_col].split(args.sep)
        # parts[:-1] = ICL reference images, parts[-1] = query image
        image_file = []
        for f in img_parts[:-1]:
            if f.endswith(".jpg") or f.endswith(".png"):
                image_file.append(f"{args.train_path}/{f}")
            else:
                image_file.append(f)
        image_file.append(img_parts[-1])  # query image (relative to image_folder)

        texts = line[text_col].split(args.sep)
        assert len(texts) >= 2, (
            f"When multiple questions option is set, the entry must have "
            f"at least 2 parts (got {len(texts)})"
        )
        assert answer_col in line, "When multiple questions option is set, first question must have answer"

        # Parse K reference answers (may be __sep__-delimited or a single string)
        raw_ans = line[answer_col]
        if args.sep in str(raw_ans):
            ans = raw_ans.split(args.sep)  # list of K answers
        else:
            ans = [raw_ans]  # legacy 1-shot: single answer string in a list

        # Replace image placeholder in every text part
        if image_placeholder in texts[0]:
            if model_config.mm_use_im_start_end:
                qs = [text.replace(image_placeholder, image_token_se) for text in texts]
            else:
                qs = [text.replace(image_placeholder, default_image_token) for text in texts]
        else:
            if model_config.mm_use_im_start_end:
                qs = [image_token_se + '\n' + text for text in texts]
            else:
                qs = [default_image_token + '\n' + text for text in texts]
    else:
        ans = None
        image_file = line[image_col]
        qs = line[text_col]
        if image_placeholder in qs:
            if model_config.mm_use_im_start_end:
                qs = qs.replace(image_placeholder, image_token_se)
            else:
                qs = qs.replace(image_placeholder, default_image_token)
        else:
            if model_config.mm_use_im_start_end:
                qs = image_token_se + '\n' + qs
            else:
                qs = default_image_token + '\n' + qs
    return qs, ans, image_file

def append_message(conv, qs: Union[str, List[str]], is_multiple_questions: bool, ans: Union[str, List[str], None] = None):
    """Format a conversation with optional ICL demonstrations.

    For multi-input (ICL) entries, ``qs`` is a list of K+1 strings
    (K reference questions followed by the query) and ``ans`` is a list
    of K reference answers (or a single answer string for legacy 1-shot).
    """
    if is_multiple_questions:
        # Normalise ans to a list so both legacy (str) and K-shot (list) work
        if isinstance(ans, str):
            ans_list = [ans]
        elif ans is None:
            ans_list = []
        else:
            ans_list = list(ans)

        # qs = [ref1_q, ref2_q, ..., refK_q, query_q]
        # ans_list = [ref1_a, ref2_a, ..., refK_a]
        parts = []
        for i, ref_q in enumerate(qs[:-1]):
            parts.append(ref_q)
            parts.append(conv.roles[1])
            parts.append(ans_list[i] if i < len(ans_list) else "")
            parts.append(conv.roles[0])
        parts.append(qs[-1])  # query
        conditioned_qs = "\n".join(parts)
        conv.append_message(conv.roles[0], conditioned_qs)
        conv.append_message(conv.roles[1], None)
    else:
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
    return conv

def explore_shape(element):
    # print(f"element: {element}")
    if isinstance(element, tuple) or isinstance(element, GreedySearchOutput):
        print("Tuple of length:", len(element))
        for sub_element in element:
            # print(f"sub_element: {sub_element}")
            explore_shape(sub_element)
    elif torch.is_tensor(element):
        print("Tensor with size:", element.size())
        print("---------------------------------")
    else:
        raise ValueError("Unknown type:", type(element))

def parse_filenames(file):
    file1, file2 = file.split(',')
    return file1, file2
