"""
Create LLaVA eval datasets with multiple inputs 
python test.py \
    --task MMBench \
    --config_path ../../config/llava_eval.yaml \
    --train_path ../../../LLaVA/playground/data
python multiple_input_dataset.py \
    --task MMBench \
    --config_path ../../config/llava_eval.yaml \
    --train_path ../../../LLaVA/playground/data
python multiple_input_dataset.py \
    --task GQA \
    --config_path ../../config/llava_eval.yaml \
    --train_path ../../../LLaVA/playground/data
"""

import argparse
from argparse import Namespace
import base64
import re
from io import BytesIO
import json
from logging import Logger
import math
import os
from typing import Any, Dict, List, Tuple, Union
import yaml

import numpy as np
from numpy.core.defchararray import add
import pandas as pd
from PIL import Image
import torch
from torch import Tensor
import torch.nn.functional as F
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from transformers.models.clip.modeling_clip import CLIPModel as CLIPModelClass
from transformers.models.clip.processing_clip import CLIPProcessor as CLIPProcessorClass

from utils import get_module_logger
try:
    from llava.constants import DEFAULT_IMAGE_TOKEN
except Exception as e:
    from constants import DEFAULT_IMAGE_TOKEN

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        choices=[
            "VQAv2",
            "GQA",
            "VizWiz",
            "TextVQA",
            "MMBench",
            "MM-Vet",
        ],
        help="task type",
    )
    parser.add_argument(
        "--config_path",
        "-cp",
        type=str,
        default="<path-to-llava_eval.yaml>",
    )
    parser.add_argument(
        "--train_path",
        "-tp",
        type=str,
        default="<path-to-llava_v1_5_mix665k.json>",
    )
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--num-batch-ref", type=int, default=10)
    parser.add_argument(
        "--clip-model", type=str, default="openai/clip-vit-large-patch14"
    )
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--prefix", type=str, help="Prefix for output file", default="multiple_inputs_")
    args = parser.parse_args()
    return args


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def preprocess_dataset(
    args: Namespace,
    df: pd.DataFrame,
    col_dict: Dict[str, str] = {"id": "id", "text": "first_question", "image": "image", "answer": "first_answer"},
) -> Dict[str, List[str]]:
    out = {}
    for ky in col_dict.keys():
        use_dict = df[col_dict[ky]].tolist()
        num_split = round(len(df) / args.num_batch_ref)
        out[ky] = split_list(use_dict, num_split)
    return out


def load_dataset(args: Namespace, logger: Logger) -> pd.DataFrame:
    config = yaml.load(open(args.config_path, "r"), Loader=yaml.FullLoader)
    data_path = f"{config['root_path']}/{config[args.task]['prefix']}/{config[args.task]['questions']}"
    if data_path.endswith(".tsv"):
        org_data = pd.read_table(data_path)
    elif data_path.endswith(".jsonl"):
        org_data = pd.DataFrame([json.loads(q) for q in open(data_path, "r")])
    else:
        raise NotImplementedError
    if args.debug:
        org_data = org_data.head(69)
    org_data["answer"] = ""# initialize answer column
    logger.info(f"Loaded {len(org_data)} questions from {data_path}")
    col_dict = {
        "id": config[args.task]["id_col"],
        "text": config[args.task]["text_col"],
        "answer": "answer",# initialized column
        "image": config[args.task]["image_col"],
    }
    questions = preprocess_dataset(args, org_data, col_dict)
    return org_data, questions, config, col_dict


def load_objects(
    args: Namespace
) -> Tuple[CLIPModelClass, CLIPProcessorClass]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = CLIPProcessor.from_pretrained(args.clip_model)
    model = CLIPModel.from_pretrained(args.clip_model).to(device)
    return model, processor, device

def model_pair(
    texts: List[str],
    images: List[str],
    model: CLIPModelClass,
    processor: CLIPProcessorClass,
    device: torch.device,
) -> Tensor:
    """
    Model (text, image) by CLIP
    """
    model_inputs = processor(
        text=texts, images=images, return_tensors="pt", padding=True, truncation = True
    ).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(**model_inputs)
    reprs_t, reprs_v = (
        outputs.text_embeds.to("cpu"),
        outputs.image_embeds.to("cpu"),
    )
    return torch.cat((reprs_t, reprs_v), dim=1)


def load_image(image_file, args, ds_type, config):
    if ds_type == "reference":
        image_file = f"{args.train_path}/{image_file}"
        image = Image.open(image_file).convert("RGB")
    elif ds_type =="model_vqa_loader":
        image_file = f"{args.train_path}/{config[args.task]['image_folder']}/{image_file}"
        image = Image.open(image_file).convert("RGB")
    elif ds_type == "model_vqa_mmbench":
        image = Image.open(BytesIO(base64.b64decode(image_file)))
    elif ds_type == "model_vqa":
        image_file = f"{args.train_path}/{config[args.task]['image_folder']}/{image_file}"
        image = Image.open(image_file)
    else:
        raise NotImplementedError
    return image


def load_images(image_files, args, ds_type, config):
    out = []
    for image_file in image_files:
        image = load_image(image_file, args, ds_type, config)
        out.append(image)
    return out


def model_dataset(
    objects: List[
        Union[CLIPProcessorClass, CLIPModelClass]
    ],
    dataset: Dict[str, List[Any]],
    args: Namespace,
    ds_type: str,
    config: Dict[str, any] = None,
) -> List[Tensor]:
    """logger, objects, query, query_dict, "query"
    Model a dataset by CLIP
    """
    model, processor, device = objects
    pair_reprs = []
    for t_batch, i_batch in tqdm(zip(dataset["text"], dataset["image"])):
        i_batch = load_images(i_batch, args, ds_type, config)
        pair_repr = model_pair(t_batch, i_batch, model, processor, device)
        pair_reprs.append(pair_repr)
    return pair_reprs


def calculate_similarity(repr1: Tensor, repr2: Tensor, method="cosine") -> Tensor:
    """
    Calculate similarity of two representations
    """
    if method == "cosine":
        sims = F.cosine_similarity(repr1, repr2, dim=0)
    elif method == "l2":
        sims = torch.norm(repr1 - repr2, dim=0)
    else:
        raise NotImplementedError
    return sims


def load_reference_dataset(
    args: Namespace, filename: str = "llava_v1_5_mix665k.json"
) -> pd.DataFrame:
    train_file = f"{args.train_path}/{filename}"
    train = pd.DataFrame(json.load(open(train_file, "r"))).dropna(subset="image")
    train["source"] = train["image"].apply(lambda x: x.split("/")[0])
    if args.task.lower() in train["source"].unique():
        train = train[train["source"] == args.task.lower()].reset_index(drop=True)
    else:
        train = train[train["source"] == "coco"].reset_index(drop=True)
    train["first_question"] = train["conversations"].apply(lambda x: x[0]["value"])
    train["first_question"] = train["first_question"].apply(lambda x: re.sub(DEFAULT_IMAGE_TOKEN, "", x))
    train["first_answer"] = train["conversations"].apply(lambda x: x[1]["value"])
    if args.debug:
        train = train.head(79)
    return train


def get_reference(args: Namespace, logger: Logger) -> Dict[str, List[str]]:
    org_reference = load_reference_dataset(args)
    logger.info(f"Loaded {len(org_reference)} reference questions")
    reference = preprocess_dataset(args, org_reference)
    assert len(reference["image"]) == len(reference["text"])
    assert len(reference["text"]) == len(reference["answer"])
    return org_reference, reference


def model_datasets(
    args: Namespace,
    logger: Logger,
    reference: Dict[str, List[str]],
    query: Dict[str, List[any]],
    config: Dict[str, any],
) -> Dict[str, List[Tensor]]:
    """
    Model a reference dataset by CLIP
    """
    objects = load_objects(args)
    logger.info(f"Loaded CLIP model {args.clip_model}")
    q_reprs = model_dataset(objects, query, args, config[args.task]["eval_type"], config)
    r_reprs = model_dataset(objects, reference, args, "reference")
    return q_reprs, r_reprs


def calculate_similarity(A, B, method="cosine", params=None):
    A_norm = F.normalize(A, p=2, dim=1)
    B_norm = F.normalize(B, p=2, dim=1)
    if method == "cosine":
        out = torch.mm(A, B.t())
    elif method == "rbf":
        params = params or {"gamma": 1}
        euclidean_distance = torch.cdist(A_norm, B_norm)
        out = torch.exp(-params["gamma"] * euclidean_distance**2)
    else:
        raise NotImplementedError
    return out

def extract_most_similar_col(similarity, val_arr):
    _, sim_max_idx = torch.max(similarity, dim=1)
    return val_arr[sim_max_idx.numpy()].astype(str)

def extract_most_similar_cols(similarity, data, col_dict):
    out = {}
    for ky in col_dict.keys():
        out[ky] = extract_most_similar_col(similarity, data[col_dict[ky]].to_numpy())
    return out

def extract_most_similar_reference(similarity, reference, col_dict):
    most_similar_reference = {}
    for ky in col_dict.keys():
        most_similar_reference[ky] = extract_most_similar_col(similarity, reference[col_dict[ky]].to_numpy())
    return most_similar_reference

def insert_reference(org_data, reference, org_col_dict, sep="__sep__"):
    out = org_data.copy()
    for ky in org_col_dict.keys():
        org_col = org_col_dict[ky]
        export_series = pd.Series(reference[ky].astype(str))
        org_series = out[org_col].astype(str)
        if max(org_series.str.len()) > 0:
            export_series = export_series+sep
        else:
            pass
        export_series = export_series+org_series
        out[org_col] = export_series.copy()
    return out

def create_dataset_with_reference(
    args, 
    org_col_dict, 
    org_data, 
    org_reference, 
    q_reprs, 
    r_reprs,
    ref_col_dict = {
        "id": "id", 
        "text": "first_question", 
        "answer": "first_answer", 
        "image": "image",
    },
):
    '''
    Create a dataset with reference input
    '''
    q_reprs, r_reprs = torch.cat(q_reprs, dim=0), torch.cat(r_reprs, dim=0)
    similarity = calculate_similarity(q_reprs, r_reprs, method="cosine")
    most_similar_reference = extract_most_similar_reference(similarity, org_reference, ref_col_dict)
    data_w_reference = insert_reference(org_data, most_similar_reference, org_col_dict)
    return data_w_reference

def save_dataset(args, config, data_w_reference):
    '''
    Save a dataset with reference input
    '''
    data_path = f"{config['root_path']}/{config[args.task]['prefix']}/{args.prefix}{config[args.task]['questions']}"
    assert not os.path.isfile(data_path), f"File {data_path} already exists"
    if data_path.endswith(".tsv"):
        data_w_reference.to_csv(data_path, sep="\t", index=False)
    elif data_path.endswith(".jsonl"):
        data_w_reference.to_json(data_path, orient="records", lines=True)
    else:
        raise NotImplementedError
    return data_path

if __name__ == "__main__":
    args = arg_parser()
    logger = get_module_logger()
    assert os.path.isfile(f"{args.train_path}/llava_v1_5_mix665k.json")
    org_data, dataset, config, col_dict = load_dataset(args, logger)
    logger.info(f"Dataset: loaded {len(dataset['image'])} batches w/ roughly {len(dataset['image'][0])} elements each")
    org_reference, reference = get_reference(args, logger)
    logger.info(
        f"Reference: Converted to {len(reference['image'])} batches w/ roughly {len(reference['image'][0])} elements each"
    )
    q_reprs, r_reprs = model_datasets(args, logger, reference, dataset, config)
    logger.info(f"Modelled {len(q_reprs), len(r_reprs)} questions")
    data_w_reference = create_dataset_with_reference(args, col_dict, org_data, org_reference, q_reprs, r_reprs)
    logger.info(f"Created #{len(data_w_reference)} dataset with reference")
    saved_path = save_dataset(args, config, data_w_reference)
    logger.info(f"Saved dataset with reference to {saved_path}")