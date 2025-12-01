# bin/python
import argparse
import gzip
from itertools import product
import json
import os
from pathlib import Path
import pickle
from pprint import pprint
from typing import Dict, List, Optional, Tuple
import gc
from contextlib import nullcontext
import random
import re

import numpy as np
from PIL import Image
import shortuuid
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor, to_pil_image
import wandb
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud, STOPWORDS
from scipy import stats

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle

# from llava.mm_utils import tokenizer_image_token, process_images
from llava.eval.utils import add_image_token, append_message
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from llava.mm_utils import (
    tokenizer_image_token,
    process_images,
    get_model_name_from_path,
)
from llava.model.builder import load_pretrained_model
from representation_learning import (
    CustomModelConfig,
    MixedEffectProbe,
    RandomEffectProbe,
)
from utils import get_module_logger

import torch
from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset, DataLoader


def free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_visualization_settings(settings_path: str = None) -> dict:
    """Load visualization settings from JSON file.
    
    Parameters
    ----------
    settings_path : str, optional
        Path to settings JSON file. If None, uses default settings.
        
    Returns
    -------
    dict
        Dictionary containing visualization settings.
    """
    if settings_path is None:
        # Use default settings
        settings_path = os.path.join(os.path.dirname(__file__), "settings.json")
    
    try:
        with open(settings_path, 'r') as f:
            settings = json.load(f)
        return settings
    except FileNotFoundError:
        # Return default settings if file not found
        return {
            "visualization": {
                "colors": {
                    "original": "gray",
                    "ablated": "darkblue", 
                    "random_ablated": "darkred"
                },
                "transparency": {
                    "opacity": 0.6
                },
                "histogram": {
                    "nbins": 50,
                    "shared_binning": True
                }
            },
            "wordcloud": {
                "width": 800,
                "height": 400,
                "background_color": "white"
            }
        }


def load_dataset_entries(
    dataset: str,
    data_dir: str,
    llava_dir: str,
    prefix: str,
    target_qids: Optional[List[str]] = None,
    sample_n: Optional[int] = None,
) -> Tuple[
    List[str],
    Dict[str, str],
    Dict[str, str],
    Dict[str, str],
    Dict[str, str],
    Dict[str, str],
]:
    """Load question and image data for a dataset.

    Parameters
    ----------
    dataset : str
        Dataset name such as ``dataset1`` or ``dataset2``.
    data_dir : str
        Directory containing dataset annotation files.
    llava_dir : str
        Root directory of the LLaVA repository used for path resolution.
    prefix : str
        Prefix describing the model type, e.g. ``prefix1`` or ``prefix2``.
    target_qids : list[str], optional
        Subset of question ids to keep. If ``None``, all questions are used.
    sample_n : int, optional
        If given, limit the number of returned questions to this number.

    Returns
    -------
    qids : list[str]
        Question ids found in the dataset.
    qid_texts : dict[str, str]
        Mapping from question id to its text.
    iid_texts : dict[str, str]
        Mapping from image id to its text (only available for ``prefix1`` files).
    qid_images : dict[str, str]
        Mapping from question id to the path of the associated image.
    iid_images : dict[str, str]
        Mapping from image id to the path of the auxiliary image (only for ``prefix1`` files).
    """

    if dataset == "textvqa":
        suffix = "val_v051_ocr.jsonl"
    elif dataset == "gqa":
        suffix = "testdev_balanced.jsonl"
    else:
        raise ValueError(
            f"Unsupported dataset: {dataset}. Supported datasets are 'gqa' and 'textvqa'."
        )
    file_path = Path(data_dir) / dataset / f"{prefix}llava_{dataset}_{suffix}"
    lines = [json.loads(l) for l in open(file_path, "r")]

    if target_qids is not None:
        target_qids_set = set(target_qids)
        lines = [
            l
            for l in lines
            if l.get("question_id", "").split("__sep__")[-1] in target_qids_set
        ]
    if sample_n is not None:
        lines = lines[:sample_n]

    qids: List[str] = []
    qid_texts: Dict[str, str] = {}
    iid_texts: Dict[str, str] = {}
    qid_images: Dict[str, str] = {}
    iid_images: Dict[str, str] = {}
    answers: Dict[str, str] = {}

    for entry in lines:
        qid_field = entry["question_id"]
        image_field = entry["image"]
        text_field = entry["text"]
        answer_field = entry["answer"]

        if "__sep__" in qid_field:
            iid, qid = qid_field.split("__sep__")
            iid_img, qid_img = image_field.split("__sep__")
            iid_text, qid_text = text_field.split("__sep__")
        else:
            qid = qid_field
            qid_img = image_field
            qid_text = text_field
            iid = iid_img = iid_text = None

        qids.append(qid)
        qid_texts[qid] = qid_text
        if iid is not None:
            iid_texts[qid] = iid_text

        def resolve_path(p: str, ds: str) -> str:
            if "/" in p:
                return str(Path(llava_dir) / "playground" / "data" / p)
            else:
                if ds == "textvqa":
                    return str(Path(data_dir) / dataset / "train_images" / p)
                else:
                    return str(Path(data_dir) / dataset / "data" / "images" / p)

        qid_images[qid] = resolve_path(qid_img, dataset)
        if iid is not None:
            iid_images[qid] = resolve_path(iid_img, dataset)

        if answer_field:
            answers[qid] = answer_field

    return qids, qid_texts, iid_texts, qid_images, iid_images, answers


class QIDDataset(Dataset):
    """Dataset that caches tokenized text and processed images for each qid."""

    def __init__(
        self,
        qids: List[str],
        qid_texts: Dict[str, str],
        iid_texts: Dict[str, str],
        qid_images: Dict[str, str],
        iid_images: Dict[str, str],
        answers: Dict[str, str],
        args: argparse.Namespace,
        tokenizer: PreTrainedTokenizer,
        image_processor,
        model_config,
    ) -> None:
        self.qids = qids
        self.qid_texts = qid_texts
        self.iid_texts = iid_texts
        self.qid_images = qid_images
        self.iid_images = iid_images
        self.answers = answers
        self.args = args
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.cache: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = {}

    def __len__(self) -> int:
        return len(self.qids)

    def _build_prompt(self, texts: List[str]):
        conv = conv_templates[self.args.conv_mode].copy()
        if len(texts) == 1:
            conv = append_message(conv, texts[0], False, None)
        else:
            ex_question, ex_answer, input_text = texts
            conv = append_message(conv, [ex_question, input_text], True, ex_answer)
        prompt = conv.get_prompt()
        ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        attn = torch.ones_like(ids)
        return ids, attn

    def _process_image(self, images):
        if images is None:
            return None
        if not isinstance(images, list):
            images = [images]
        return process_images(images, self.image_processor, self.model_config)

    def __getitem__(self, idx: int):
        qid = self.qids[idx]
        if qid in self.cache:
            return self.cache[qid]

        qid_text = self.qid_texts[qid]
        iid_text = self.iid_texts.get(qid)
        answer = self.answers.get(qid)

        zsl_image = load_image(self.qid_images[qid]).convert("RGB")
        zsl_text_rand = " ".join(
            np.random.permutation(qid_text.split()) if qid_text else []
        )

        icl_texts_sem = [iid_text, answer, qid_text] if iid_text else [qid_text]
        iid_text_rand = (
            " ".join(np.random.permutation(iid_text.split())) if iid_text else ""
        )
        answer_rand = (
            " ".join(np.random.permutation(answer.split())) if answer else ""
        )
        icl_texts_rand = [iid_text_rand, answer_rand, qid_text] if iid_text else [qid_text]

        if iid_text:
            org_image = load_image(self.iid_images[qid]).convert("RGB")
            org_tensor = to_tensor(org_image)
            if self.args.image_noise_pattern == "black":
                noise = to_pil_image(torch.zeros_like(org_tensor))
            elif self.args.image_noise_pattern == "white":
                noise = to_pil_image(torch.ones_like(org_tensor) * 255)
            else:
                noise = to_pil_image(torch.rand_like(org_tensor) * 255)
            icl_images_sem = [org_image, zsl_image]
            icl_images_rand = [noise, zsl_image]
        else:
            icl_images_sem = [zsl_image]
            icl_images_rand = [zsl_image]

        zsl_rand_ids, zsl_rand_attn = self._build_prompt([zsl_text_rand])
        zsl_sem_ids, zsl_sem_attn = self._build_prompt([qid_text])
        icl_rand_ids, icl_rand_attn = self._build_prompt(icl_texts_rand)
        icl_sem_ids, icl_sem_attn = self._build_prompt(icl_texts_sem)

        zsl_image_tensor = self._process_image([zsl_image])
        icl_images_rand_tensor = self._process_image(icl_images_rand)
        icl_images_sem_tensor = self._process_image(icl_images_sem)

        item = {
            "zsl_rand": {
                "input_ids": zsl_rand_ids,
                "attention_mask": zsl_rand_attn,
                "image": zsl_image_tensor,
            },
            "zsl_sem": {
                "input_ids": zsl_sem_ids,
                "attention_mask": zsl_sem_attn,
                "image": zsl_image_tensor,
            },
            "icl_rand": {
                "input_ids": icl_rand_ids,
                "attention_mask": icl_rand_attn,
                "image": icl_images_rand_tensor,
            },
            "icl_sem": {
                "input_ids": icl_sem_ids,
                "attention_mask": icl_sem_attn,
                "image": icl_images_sem_tensor,
            },
        }
        self.cache[qid] = item
        return item


def collate_qid_batch(batch):
    keys = batch[0].keys()
    out = {k: [] for k in keys}
    for b in batch:
        for k in keys:
            out[k].append(b[k])
    return out


def forward_hidden_states_batch(
    batch_inputs: List[Dict[str, torch.Tensor]],
    model: LlavaLlamaForCausalLM,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    layer_num: int = -1,
):
    if len(batch_inputs) == 0:
        return torch.empty(0)
    input_ids_list = [b["input_ids"] for b in batch_inputs]
    attn_list = [b["attention_mask"] for b in batch_inputs]
    images = [b["image"] for b in batch_inputs]
    max_len = max(t.shape[0] for t in input_ids_list)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    padded_ids = []
    padded_attn = []
    for ids, attn in zip(input_ids_list, attn_list):
        pad_len = max_len - ids.shape[0]
        if pad_len > 0:
            ids = torch.cat(
                [ids, torch.full((pad_len,), pad_id, dtype=torch.long)], dim=0
            )
            attn = torch.cat(
                [attn, torch.zeros(pad_len, dtype=torch.long)], dim=0
            )
        padded_ids.append(ids)
        padded_attn.append(attn)
    input_ids = torch.stack(padded_ids, dim=0).to(model.device)
    attention_mask = torch.stack(padded_attn, dim=0).to(model.device)
    images = [img.to(device) for img in images]
    model_dtype = torch.float32 if not torch.cuda.is_available() else torch.float16
    if torch.cuda.is_available():
        autocast = torch.cuda.amp.autocast
        with torch.inference_mode(), autocast(dtype=torch.float16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=[img.to(dtype=model_dtype, non_blocking=True) for img in images],
                output_hidden_states=True,
            )
    else:
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=[img.to(dtype=model_dtype, non_blocking=True) for img in images],
                output_hidden_states=True,
            )
    hidden = outputs.hidden_states[layer_num]
    del outputs, input_ids, attention_mask, images
    free_memory()
    return hidden

def extract_outputs(
    input_texts: str,
    input_image,
    model: LlavaLlamaForCausalLM,
    tokenizer: PreTrainedTokenizer,
    image_processor,
    max_new_tokens: int = 128,
    logger=None,
    conv_mode: str = "llava_v1",
    device: Optional[torch.device] = None,
    layer_num: int = -1,
):
    """Generate hidden representation for ``input_text`` and ``input_image``.

    Parameters
    ----------
    input_text : str
        User prompt.
    input_image : PIL.Image.Image or torch.Tensor
        Image associated with the prompt.
    model : LlavaLlamaForCausalLM
        Model used for generation.
    tokenizer : PreTrainedTokenizer
        Tokenizer corresponding to the model.
    image_processor : transformers.ImageProcessor
        Preprocessor for the image input.
    max_new_tokens : int, optional
        Maximum number of tokens to generate. Default is 128.
    layer_num : int, optional
        The index of the layer to use for decoding. Default is -1, which uses the last layer.

    Returns
    -------
    torch.FloatTensor
        Hidden states of the last decoder layer for the generated sequence.
    """
    conv = conv_templates[conv_mode].copy()

    if len(input_texts) == 1:
        input_text = input_texts[0]
        conv = append_message(conv, input_text, False, None)
    else:
        ex_question, ex_answer, input_text = input_texts
        # Example question and answer
        conv = append_message(conv, [ex_question, input_text], True, ex_answer)

    prompt = conv.get_prompt()

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(model.device)
    )
    # input_ids = tokenizer_image_token(
    #     prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    # ).to(model.device)

    stop_str = (
        conv_templates[conv_mode].sep
        if conv_templates[conv_mode].sep_style != SeparatorStyle.TWO
        else conv_templates[conv_mode].sep2
    )

    # image_tensor = process_images(input_image, image_processor, model.config)
    image_tensor = process_images(input_image, image_processor, model.config).unsqueeze(
        0
    )  # Add batch dimension

    model_dtype = torch.float32 if not torch.cuda.is_available() else torch.float16
    with torch.inference_mode():
        outputs = model.generate(
            input_ids,
            images=image_tensor.to(dtype=model_dtype, device=device, non_blocking=True),
            do_sample=False,
            temperature=0.0,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            output_hidden_states=True,
            output_scores=True,
            return_dict_in_generate=True,
            use_cache=True,
        )
    
    hidden_states = [step[layer_num] for step in outputs.hidden_states]

    output_ids = outputs.sequences

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        if logger is not None:
            logger.warning(
                f"{n_diff_input_output} output_ids are not the same as the input_ids"
            )
        else:
            print(
                f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
            )
    output_sentences = tokenizer.batch_decode(
        output_ids[:, input_token_len:], skip_special_tokens=True
    )[0]
    output_sentences = output_sentences.strip()
    if output_sentences.endswith(stop_str):
        output_sentences = output_sentences[: -len(stop_str)]
    output_sentences = output_sentences.strip()

    ans_id = shortuuid.uuid()
    ans_dict = {
        "prompt": prompt,
        "text": output_sentences,
        "answer_id": ans_id,
        "input_ids": input_ids,
        "output_ids": output_ids,
        "hidden_states": hidden_states,
    }

    return ans_dict


def extract_outputs_batch(
    batch_input_texts: List[List[str]],
    batch_input_images,
    model: LlavaLlamaForCausalLM,
    tokenizer: PreTrainedTokenizer,
    image_processor,
    max_new_tokens: int = 128,
    logger=None,
    conv_mode: str = "llava_v1",
    device: Optional[torch.device] = None,
    layer_num: int = -1,
):
    """Batch version of :func:`extract_outputs` to avoid multiple forward passes.

    Parameters
    ----------
    batch_input_texts : list[list[str]]
        A list where each element contains the text inputs for a single example
        (see ``extract_outputs`` for formatting).
    batch_input_images : list
        List of images corresponding to ``batch_input_texts``. Each element can be
        a single image or a list of images. All elements within the batch must have
        the same number of images.
    Other parameters are identical to :func:`extract_outputs`.

    Returns
    -------
    list[dict]
        A list of answer dictionaries, one per example.
    """

    if len(batch_input_texts) == 0:
        return []

    convs = []
    prompts = []
    for texts in batch_input_texts:
        conv = conv_templates[conv_mode].copy()
        if len(texts) == 1:
            conv = append_message(conv, texts[0], False, None)
        else:
            ex_question, ex_answer, input_text = texts
            conv = append_message(conv, [ex_question, input_text], True, ex_answer)
        convs.append(conv)
        prompts.append(conv.get_prompt())

    # Tokenize and pad
    input_ids_list = [
        tokenizer_image_token(p, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        for p in prompts
    ]
    max_len = max(t.shape[0] for t in input_ids_list)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    padded = []
    attention = []
    for ids in input_ids_list:
        pad_len = max_len - ids.shape[0]
        if pad_len > 0:
            ids = torch.cat([ids, torch.full((pad_len,), pad_id, dtype=torch.long)], dim=0)
        padded.append(ids)
        attention.append(ids.ne(pad_id))
    input_ids = torch.stack(padded, dim=0).to(model.device)
    attention_mask = torch.stack(attention, dim=0).to(model.device)

    # Images
    image_tensors = []
    for img in batch_input_images:
        image_tensor = process_images(img, image_processor, model.config).unsqueeze(0)
        image_tensors.append(image_tensor)
    images = torch.cat(image_tensors, dim=0)

    stop_str = (
        conv_templates[conv_mode].sep
        if conv_templates[conv_mode].sep_style != SeparatorStyle.TWO
        else conv_templates[conv_mode].sep2
    )

    model_dtype = torch.float32 if not torch.cuda.is_available() else torch.float16
    with torch.inference_mode():
        outputs = model.generate(
            input_ids,
            images=images.to(dtype=model_dtype, device=device, non_blocking=True),
            attention_mask=attention_mask,
            do_sample=False,
            temperature=0.0,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            output_hidden_states=True,
            output_scores=True,
            return_dict_in_generate=True,
            use_cache=True,
        )

    # Collect hidden states per example
    hidden_states_all = [step[layer_num].to("cpu") for step in outputs.hidden_states]

    ans_dicts = []
    for i in range(len(batch_input_texts)):
        hidden_states = [hs[i : i + 1] for hs in hidden_states_all]
        output_ids = outputs.sequences[i : i + 1].to("cpu")
        input_len = input_ids_list[i].shape[0]
        n_diff_input_output = (
            input_ids_list[i] != output_ids[:, :input_len]
        ).sum().item()
        if n_diff_input_output > 0:
            if logger is not None:
                logger.warning(
                    f"{n_diff_input_output} output_ids are not the same as the input_ids"
                )

        output_sentences = tokenizer.batch_decode(
            output_ids[:, input_len:], skip_special_tokens=True
        )[0]
        output_sentences = output_sentences.strip()
        if output_sentences.endswith(stop_str):
            output_sentences = output_sentences[: -len(stop_str)]
        output_sentences = output_sentences.strip()

        ans_dicts.append(
            {
                "prompt": prompts[i],
                "text": output_sentences,
                "answer_id": shortuuid.uuid(),
                "input_ids": input_ids[i : i + 1],
                "output_ids": output_ids,
                "hidden_states": hidden_states,
            }
        )

    return ans_dicts


def unembed_hidden_states(
    hidden_states: torch.FloatTensor,
    model: LlavaLlamaForCausalLM,
    tokenizer: PreTrainedTokenizer,
    unembed_last: bool = True,
):
    """Convert hidden states to the next token using greedy decoding.

    Parameters
    ----------
    hidden_states : torch.FloatTensor
        Hidden states returned by ``LlavaLlamaForCausalLM.forward``.
    model : LlavaLlamaForCausalLM
        The model instance that produced the hidden states.
    tokenizer : PreTrainedTokenizer
        Tokenizer used to convert token ids to strings.
    unembed_last : bool, optional
        IF ``True``, only the last hidden state is used for decoding.

    Returns
    -------
    str or list[str]
        The decoded token(s). If ``hidden_states`` is batched, a list of
        tokens is returned.
    """
    with torch.inference_mode():
        lm_device = model.lm_head.weight.device
        lm_dtype = model.lm_head.weight.dtype
        hidden_states = hidden_states.to(device=lm_device, dtype=lm_dtype)

        if unembed_last:
            # Use only the last hidden state
            logits = torch.stack(
                [model.lm_head(s[:, -1, :]) for s in hidden_states], dim=1
            )
        else:
            # Use all hidden states
            logits = torch.stack(
                [
                    model.lm_head(hidden_states[:, s_idx, :])
                    for s_idx in range(hidden_states.shape[1])
                ],
                dim=1,
            )
        ids = logits.argmax(dim=-1)
        tokens = tokenizer.batch_decode(ids, skip_special_tokens=True)

    return tokens[0] if len(tokens) == 1 else tokens


def get_embedding_file_lists(dir_path: str) -> Tuple[List[str], List[str]]:
    """
    Scan `dir_path` for .pt.gz files.
    Returns two sorted lists:
    - query_files: files whose name contains 'reprs1'
    - target_files: files whose name contains 'reprs2'
    """
    all_files = sorted(os.listdir(dir_path))
    query_files = [
        os.path.join(dir_path, fn)
        for fn in all_files
        if fn.endswith(".pt.gz") and "reprs1" in fn
    ]
    target_files = [
        os.path.join(dir_path, fn)
        for fn in all_files
        if fn.endswith(".pt.gz") and "reprs2" in fn
    ]
    return query_files, target_files


def load_embedding_pair(
    q_path: str, t_path: str, device: torch.device = torch.device("cpu")
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given the path to a reprs1 and reprs2 .pt.gz files, load them.
    Returns (query_embed, target_embed).
    """

    def _load_tensor(path: str) -> torch.Tensor:
        if path.endswith(".pt.gz"):
            with gzip.open(path, "rb") as f:
                return torch.load(f, map_location=device)
        else:
            raise ValueError(f"Unsupported extension for {path!r}, expected .pt.gz")

    query_embed = _load_tensor(q_path)
    target_embed = _load_tensor(t_path)
    return query_embed, target_embed


def index_lists(pairs: List[Tuple[str, str]]) -> Tuple[torch.Tensor, dict]:
    """
    Given a list of (task, model) tuples, map each unique tuple to an index
    and return a tensor of indices and the mapping dict.
    """
    unique = list(set(pairs))
    tuple_to_index = {t: i for i, t in enumerate(unique)}
    indices = torch.tensor([tuple_to_index[p] for p in pairs], dtype=torch.long)
    return indices, tuple_to_index


def build_model_paths(
    exp_dir: str, tags: List[str], prefix: str, model2: str, model_dir: str
) -> Tuple[str, str, str]:
    """
    Construct paths for trained, mixed, and random models.
    tags: [ts1, ts2, ts3]
    """
    ts1, ts2, ts3 = tags
    trained = os.path.join(exp_dir, f"{ts1}_{prefix}_{model2}_trained.pt")
    mixed = os.path.join(model_dir, f"{ts2}_mixed.pt")
    rand = os.path.join(model_dir, f"{ts3}_linear.pt")
    return trained, mixed, rand


def build_llava_model(
    args: argparse.Namespace, device: torch.device
) -> Tuple[nn.Module, nn.Module, nn.Module, int]:
    """
    Load a pre-trained Llava model.
    model_name: e.g., "llava-v1.5-13b"
    model_dir: directory where the model is stored
    """

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name, device=device
    )

    return tokenizer, model, image_processor, context_len


def load_image(image_file):
    image = Image.open(image_file).convert("RGB")
    return image


class FourSpaceProjector(nn.Module):
    """
    Learns shared projections over the hidden dimension:

        zsl_rand ↦ H1(x)
        zsl_sem  ↦ H1(x) + H2(x)
        icl_rand ↦ H1(x) + P1 * Δ(x)
        icl_sem  ↦ H1(x) + H2(x) + P2 * Δ(x)

    Inputs:  zsl_rand, zsl_sem, icl_rand, icl_sem with shapes [batch, tokens, hidden]
             (any of them can be None)
    Outputs: dict with keys for provided inputs; each value matches the input's shape
    """
    def __init__(
        self,
        hidden_size: int,
        *,
        bias: bool = False,
        gate: str = "scalar",        # "scalar" or "vector"
        init: str = "xavier_uniform" # "xavier_uniform" or "orthogonal"
    ):
        super().__init__()
        # Shared projections
        self.proj_h1 = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.proj_h2 = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.delta   = nn.Linear(hidden_size, hidden_size, bias=bias)

        # Gates P1, P2
        if gate == "scalar":
            self.p1 = nn.Parameter(torch.tensor(0.0))
            self.p2 = nn.Parameter(torch.tensor(0.0))
        elif gate == "vector":
            self.p1 = nn.Parameter(torch.zeros(hidden_size))
            self.p2 = nn.Parameter(torch.zeros(hidden_size))
        else:
            raise ValueError("gate must be 'scalar' or 'vector'")
        self.gate_type = gate

        self._reset_parameters(init)

    def _reset_parameters(self, init: str):
        layers = [self.proj_h1, self.proj_h2, self.delta]
        if init == "xavier_uniform":
            for m in layers:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        elif init == "orthogonal":
            for m in layers:
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # else: keep PyTorch defaults

    def _apply_gate(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        if self.gate_type == "scalar":
            return p * x
        else:
            view_shape = (1,) * (x.ndim - 1) + (x.size(-1),)
            return x * p.view(view_shape)

    def forward(
        self,
        zsl_rand: torch.Tensor | None = None,
        zsl_sem:  torch.Tensor | None = None,
        icl_rand: torch.Tensor | None = None,
        icl_sem:  torch.Tensor | None = None,
    ):
        out = {}
        dtype = self.proj_h1.weight.dtype

        if zsl_rand is not None:
            zsl_rand = zsl_rand.to(dtype=dtype)
            out["zsl_rand"] = self.proj_h1(zsl_rand)

        if zsl_sem is not None:
            zsl_sem = zsl_sem.to(dtype=dtype)
            out["zsl_sem"] = self.proj_h1(zsl_sem) + self.proj_h2(zsl_sem)

        if icl_rand is not None:
            icl_rand = icl_rand.to(dtype=dtype)
            out["icl_rand"] = self.proj_h1(icl_rand) + self._apply_gate(self.delta(icl_rand), self.p1)

        if icl_sem is not None:
            icl_sem = icl_sem.to(dtype=dtype)
            out["icl_sem"] = (
                self.proj_h1(icl_sem)
                + self.proj_h2(icl_sem)
                + self._apply_gate(self.delta(icl_sem), self.p2)
            )

        return out


# def probe_embedding(
#     args: argparse.Namespace,
#     llava_model: nn.Module,
#     tokenizer: PreTrainedTokenizer,
#     probe: nn.Module,
#     fixed_indices: torch.Tensor,
#     mapping: dict,
#     zsl_outputs: Dict[str, torch.Tensor],
#     task: str = "textvqa",
#     ood: str = "0_id",
#     logger = None,
# ):
#     fixed_index = fixed_indices[mapping[(ood, task)]]
#     zsl_in_repr = zsl_outputs["hidden_states"][0].to(torch.float32)  # [n_token, embed_dim=5120]
#     fixed_index_repr = torch.Tensor([fixed_index] * zsl_in_repr.shape[0]).to(torch.int64)  # [n_token, ]
#     zsl_out_repr = torch.concat([s[:, -1, :] for s in zsl_outputs["hidden_states"]]).unsqueeze(0).to(torch.float32)  # [n_batch, n_token, embed_dim=5120]
#     in_length = zsl_in_repr.shape[1]  # Number of input tokens
#     out_length = zsl_out_repr.shape[1]  # Number of output tokens
#     if logger is not None:
#         logger.info(f"ZSL input representation shape: {zsl_in_repr.shape}")
#         logger.info(f"ZSL output representation shape: {zsl_out_repr.shape}")
#         logger.info(f"in_length: {in_length}, out_length: {out_length}")
#     with torch.no_grad():
#         if args.probe_type == "random":
#             zsl_out = probe(zsl_in_repr, torch.Tensor([in_length]), torch.Tensor([out_length]))
#         elif args.probe_type == "mixed":
#             zsl_out = probe(zsl_in_repr, fixed_index_repr, torch.Tensor([in_length]), torch.Tensor([out_length]))
#     zsl_tokens = unembed_hidden_states(
#         zsl_out, llava_model, tokenizer, unembed_last=False
#     )
#     return zsl_tokens


def project_embedding(
    llava_model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    probe: nn.Module,
    outputs: list[Dict[str, torch.Tensor]],
    logger = None,
):
    in_reprs = []
    dtype = next(probe.parameters()).dtype
    device = next(probe.parameters()).device
    for output in outputs:
        in_repr = (
            output["hidden_states"][0]
            .to(device=device, dtype=dtype)
        )  # [n_token, embed_dim=5120]
        in_reprs.append(in_repr)
    with torch.no_grad():
        out = probe(in_reprs[0], in_reprs[1], in_reprs[2], in_reprs[3])
    del in_reprs
    free_memory()
    rand_zsl_tokens = unembed_hidden_states(
        out["zsl_rand"], llava_model, tokenizer, unembed_last=False
    )
    sem_zsl_tokens = unembed_hidden_states(
        out["zsl_sem"], llava_model, tokenizer, unembed_last=False
    )
    rand_icl_tokens = unembed_hidden_states(
        out["icl_rand"], llava_model, tokenizer, unembed_last=False
    )
    sem_icl_tokens = unembed_hidden_states(
        out["icl_sem"], llava_model, tokenizer, unembed_last=False
    )
    return {
        "hidden_states": out,
        "tokens": {
            "zsl_rand": rand_zsl_tokens,
            "zsl_sem": sem_zsl_tokens,
            "icl_rand": rand_icl_tokens,
            "icl_sem": sem_icl_tokens
        }
    }


def analyze_probe(out: Dict[str, torch.Tensor], save_dir: str, logger=None) -> None:
    """Analyze probe outputs and save distribution plots.

    This function computes two distributions:

    1. The per-dimension difference between the norms of ``out['zsl_rand']`` and
       ``out['zsl_sem']``.
    2. The per-dimension difference between the norms of ``out['icl_rand'] -
       out['zsl_rand']`` and ``out['icl_sem'] - out['zsl_sem']``.

    Each distribution is saved as a histogram plot (HTML) in ``save_dir``.
    """

    os.makedirs(save_dir, exist_ok=True)

    def _flatten(t: torch.Tensor) -> torch.Tensor:
        return t.detach().reshape(-1, t.shape[-1])

    # Difference between norms of zsl_rand and zsl_sem
    zsl_rand = _flatten(out["zsl_rand"])
    zsl_sem = _flatten(out["zsl_sem"])
    zsl_rand_norm = torch.norm(zsl_rand, dim=0)
    zsl_sem_norm = torch.norm(zsl_sem, dim=0)
    zsl_diff = (zsl_rand_norm - zsl_sem_norm).cpu().numpy()

    fig1 = go.Figure(data=[go.Histogram(x=zsl_diff)])
    fig1.update_layout(
        title="Norm difference per dimension: zsl_rand vs zsl_sem",
        xaxis_title="zsl_rand_norm - zsl_sem_norm",
        yaxis_title="Count",
    )
    zsl_path = os.path.join(save_dir, "zsl_norm_diff.html")
    fig1.write_html(zsl_path)
    if logger is not None:
        logger.info(f"Saved ZSL norm difference histogram to {zsl_path}")

    # Difference between norms of (icl_rand - zsl_rand) and (icl_sem - zsl_sem)
    icl_rand = _flatten(out["icl_rand"] - out["zsl_rand"])
    icl_sem = _flatten(out["icl_sem"] - out["zsl_sem"])
    icl_rand_norm = torch.norm(icl_rand, dim=0)
    icl_sem_norm = torch.norm(icl_sem, dim=0)
    icl_diff = (icl_rand_norm - icl_sem_norm).cpu().numpy()

    fig2 = go.Figure(data=[go.Histogram(x=icl_diff)])
    fig2.update_layout(
        title="Norm difference per dimension: (icl_rand - zsl_rand) vs (icl_sem - zsl_sem)",
        xaxis_title="(icl_rand - zsl_rand)_norm - (icl_sem - zsl_sem)_norm",
        yaxis_title="Count",
    )
    icl_path = os.path.join(save_dir, "icl_zsl_norm_diff.html")
    fig2.write_html(icl_path)
    if logger is not None:
        logger.info(f"Saved ICL/ZSL norm difference histogram to {icl_path}")


def analyze_probe_overlay(
    original_out: Dict[str, torch.Tensor], 
    ablated_out: Dict[str, torch.Tensor], 
    save_dir: str, 
    logger=None,
    settings: dict = None
) -> None:
    """Analyze probe outputs with overlay plots comparing original vs ablated distributions.
    
    This function creates overlay histograms comparing the distributions of:
    1. The per-dimension difference between the norms of ``original_out['zsl_rand']`` and ``original_out['zsl_sem']``
       vs the same for ablated outputs.
    2. The per-dimension difference between the norms of ``(icl_rand - zsl_rand)`` and ``(icl_sem - zsl_sem)``
       vs the same for ablated outputs.
    
    Each overlay plot includes statistical tests comparing the means of the distributions.
    
    Parameters
    ----------
    original_out : dict
        Dictionary containing original hidden states for each condition.
    ablated_out : dict
        Dictionary containing ablated hidden states for each condition.
    save_dir : str
        Directory where overlay plots will be saved.
    logger : optional
        Logger for status messages.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Load settings
    if settings is None:
        settings = load_visualization_settings()
    
    colors = settings["visualization"]["colors"]
    opacity = settings["visualization"]["transparency"]["opacity"]
    nbins = settings["visualization"]["histogram"]["nbins"]
    shared_binning = settings["visualization"]["histogram"]["shared_binning"]

    def _flatten(t: torch.Tensor) -> torch.Tensor:
        return t.detach().reshape(-1, t.shape[-1])

    # ZSL Analysis: Difference between norms of zsl_rand and zsl_sem
    orig_zsl_rand = _flatten(original_out["zsl_rand"])
    orig_zsl_sem = _flatten(original_out["zsl_sem"])
    orig_zsl_rand_norm = torch.norm(orig_zsl_rand, dim=0)
    orig_zsl_sem_norm = torch.norm(orig_zsl_sem, dim=0)
    orig_zsl_diff = (orig_zsl_rand_norm - orig_zsl_sem_norm).cpu().numpy()

    ablated_zsl_rand = _flatten(ablated_out["zsl_rand"])
    ablated_zsl_sem = _flatten(ablated_out["zsl_sem"])
    ablated_zsl_rand_norm = torch.norm(ablated_zsl_rand, dim=0)
    ablated_zsl_sem_norm = torch.norm(ablated_zsl_sem, dim=0)
    ablated_zsl_diff = (ablated_zsl_rand_norm - ablated_zsl_sem_norm).cpu().numpy()

    # Statistical test for ZSL
    zsl_stat, zsl_p_value = stats.ttest_ind(orig_zsl_diff, ablated_zsl_diff)
    zsl_mean_orig = np.mean(orig_zsl_diff)
    zsl_mean_ablated = np.mean(ablated_zsl_diff)

    # Create ZSL overlay plot with shared binning
    fig_zsl = go.Figure()
    
    if shared_binning:
        # Calculate shared bin edges
        all_zsl_data = np.concatenate([orig_zsl_diff, ablated_zsl_diff])
        bin_edges = np.linspace(all_zsl_data.min(), all_zsl_data.max(), nbins + 1)
        
        fig_zsl.add_trace(go.Histogram(
            x=orig_zsl_diff, 
            name='Original', 
            opacity=opacity,
            xbins=dict(start=bin_edges[0], end=bin_edges[-1], size=(bin_edges[1] - bin_edges[0])),
            marker_color=colors["original"]
        ))
        fig_zsl.add_trace(go.Histogram(
            x=ablated_zsl_diff, 
            name='Ablated', 
            opacity=opacity,
            xbins=dict(start=bin_edges[0], end=bin_edges[-1], size=(bin_edges[1] - bin_edges[0])),
            marker_color=colors["ablated"]
        ))
    else:
        fig_zsl.add_trace(go.Histogram(
            x=orig_zsl_diff, 
            name='Original', 
            opacity=opacity,
            nbinsx=nbins,
            marker_color=colors["original"]
        ))
        fig_zsl.add_trace(go.Histogram(
            x=ablated_zsl_diff, 
            name='Ablated', 
            opacity=opacity,
            nbinsx=nbins,
            marker_color=colors["ablated"]
        ))
    
    # Add statistical test results to title
    title_zsl = (f"ZSL Norm Difference: Original vs Ablated<br>"
                f"Original Mean: {zsl_mean_orig:.4f}, Ablated Mean: {zsl_mean_ablated:.4f}<br>"
                f"t-statistic: {zsl_stat:.4f}, p-value: {zsl_p_value:.4f}")
    
    fig_zsl.update_layout(
        title=title_zsl,
        xaxis_title="zsl_rand_norm - zsl_sem_norm",
        yaxis_title="Count",
        barmode='overlay',
        legend=dict(x=0.7, y=0.9)
    )
    
    zsl_path = os.path.join(save_dir, "zsl_norm_diff_overlay.html")
    fig_zsl.write_html(zsl_path)
    if logger is not None:
        logger.info(f"Saved ZSL overlay histogram to {zsl_path}")

    # ICL Analysis: Difference between norms of (icl_rand - zsl_rand) and (icl_sem - zsl_sem)
    orig_icl_rand = _flatten(original_out["icl_rand"] - original_out["zsl_rand"])
    orig_icl_sem = _flatten(original_out["icl_sem"] - original_out["zsl_sem"])
    orig_icl_rand_norm = torch.norm(orig_icl_rand, dim=0)
    orig_icl_sem_norm = torch.norm(orig_icl_sem, dim=0)
    orig_icl_diff = (orig_icl_rand_norm - orig_icl_sem_norm).cpu().numpy()

    ablated_icl_rand = _flatten(ablated_out["icl_rand"] - ablated_out["zsl_rand"])
    ablated_icl_sem = _flatten(ablated_out["icl_sem"] - ablated_out["zsl_sem"])
    ablated_icl_rand_norm = torch.norm(ablated_icl_rand, dim=0)
    ablated_icl_sem_norm = torch.norm(ablated_icl_sem, dim=0)
    ablated_icl_diff = (ablated_icl_rand_norm - ablated_icl_sem_norm).cpu().numpy()

    # Statistical test for ICL
    icl_stat, icl_p_value = stats.ttest_ind(orig_icl_diff, ablated_icl_diff)
    icl_mean_orig = np.mean(orig_icl_diff)
    icl_mean_ablated = np.mean(ablated_icl_diff)

    # Create ICL overlay plot with shared binning
    fig_icl = go.Figure()
    
    if shared_binning:
        # Calculate shared bin edges for ICL
        all_icl_data = np.concatenate([orig_icl_diff, ablated_icl_diff])
        icl_bin_edges = np.linspace(all_icl_data.min(), all_icl_data.max(), nbins + 1)
        
        fig_icl.add_trace(go.Histogram(
            x=orig_icl_diff, 
            name='Original', 
            opacity=opacity,
            xbins=dict(start=icl_bin_edges[0], end=icl_bin_edges[-1], size=(icl_bin_edges[1] - icl_bin_edges[0])),
            marker_color=colors["original"]
        ))
        fig_icl.add_trace(go.Histogram(
            x=ablated_icl_diff, 
            name='Ablated', 
            opacity=opacity,
            xbins=dict(start=icl_bin_edges[0], end=icl_bin_edges[-1], size=(icl_bin_edges[1] - icl_bin_edges[0])),
            marker_color=colors["ablated"]
        ))
    else:
        fig_icl.add_trace(go.Histogram(
            x=orig_icl_diff, 
            name='Original', 
            opacity=opacity,
            nbinsx=nbins,
            marker_color=colors["original"]
        ))
        fig_icl.add_trace(go.Histogram(
            x=ablated_icl_diff, 
            name='Ablated', 
            opacity=opacity,
            nbinsx=nbins,
            marker_color=colors["ablated"]
        ))
    
    # Add statistical test results to title
    title_icl = (f"ICL/ZSL Norm Difference: Original vs Ablated<br>"
                f"Original Mean: {icl_mean_orig:.4f}, Ablated Mean: {icl_mean_ablated:.4f}<br>"
                f"t-statistic: {icl_stat:.4f}, p-value: {icl_p_value:.4f}")
    
    fig_icl.update_layout(
        title=title_icl,
        xaxis_title="(icl_rand - zsl_rand)_norm - (icl_sem - zsl_sem)_norm",
        yaxis_title="Count",
        barmode='overlay',
        legend=dict(x=0.7, y=0.9)
    )
    
    icl_path = os.path.join(save_dir, "icl_zsl_norm_diff_overlay.html")
    fig_icl.write_html(icl_path)
    if logger is not None:
        logger.info(f"Saved ICL/ZSL overlay histogram to {icl_path}")

    # Save statistical test results to a JSON file
    stats_results = {
        "zsl_analysis": {
            "original_mean": float(zsl_mean_orig),
            "ablated_mean": float(zsl_mean_ablated),
            "t_statistic": float(zsl_stat),
            "p_value": float(zsl_p_value),
            "significant": int(zsl_p_value < 0.05)
        },
        "icl_analysis": {
            "original_mean": float(icl_mean_orig),
            "ablated_mean": float(icl_mean_ablated),
            "t_statistic": float(icl_stat),
            "p_value": float(icl_p_value),
            "significant": int(icl_p_value < 0.05)
        }
    }
    
    stats_path = os.path.join(save_dir, "statistical_tests.json")
    with open(stats_path, "w") as f:
        json.dump(stats_results, f, indent=2)
    
    if logger is not None:
        logger.info(f"Saved statistical test results to {stats_path}")
        logger.info(f"ZSL: Original mean={zsl_mean_orig:.4f}, Ablated mean={zsl_mean_ablated:.4f}, p-value={zsl_p_value:.4f}")
        logger.info(f"ICL: Original mean={icl_mean_orig:.4f}, Ablated mean={icl_mean_ablated:.4f}, p-value={icl_p_value:.4f}")


def analyze_probe_overlay_direct_comparison(
    original_out: Dict[str, torch.Tensor],
    ablated_out: Dict[str, torch.Tensor], 
    random_ablated_out: Dict[str, torch.Tensor],
    save_dir: str, 
    logger=None,
    settings: dict = None
) -> None:
    """Analyze probe outputs with direct comparisons between original, ablated, and random ablated distributions.
    
    This function creates overlay histograms comparing the distributions of:
    1. The per-dimension difference between the norms of zsl_rand and zsl_sem
       for ablated vs random ablated outputs.
    2. The per-dimension difference between the norms of (icl_rand - zsl_rand) and (icl_sem - zsl_sem)
       for ablated vs random ablated outputs.
    3. The per-dimension difference between the norms of zsl_rand and zsl_sem
       for original vs random ablated outputs.
    4. The per-dimension difference between the norms of (icl_rand - zsl_rand) and (icl_sem - zsl_sem)
       for original vs random ablated outputs.
    
    Each overlay plot includes statistical tests comparing the means of the distributions.
    
    Parameters
    ----------
    original_out : dict
        Dictionary containing original hidden states for each condition.
    ablated_out : dict
        Dictionary containing ablated hidden states for each condition.
    random_ablated_out : dict
        Dictionary containing random ablated hidden states for each condition.
    save_dir : str
        Directory where overlay plots will be saved.
    logger : optional
        Logger for status messages.
    settings : dict, optional
        Visualization settings dictionary.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Load settings
    if settings is None:
        settings = load_visualization_settings()
    
    colors = settings["visualization"]["colors"]
    opacity = settings["visualization"]["transparency"]["opacity"]
    nbins = settings["visualization"]["histogram"]["nbins"]
    shared_binning = settings["visualization"]["histogram"]["shared_binning"]

    def _flatten(t: torch.Tensor) -> torch.Tensor:
        return t.detach().reshape(-1, t.shape[-1])

    # ZSL Analysis: Difference between norms of zsl_rand and zsl_sem
    ablated_zsl_rand = _flatten(ablated_out["zsl_rand"])
    ablated_zsl_sem = _flatten(ablated_out["zsl_sem"])
    ablated_zsl_rand_norm = torch.norm(ablated_zsl_rand, dim=0)
    ablated_zsl_sem_norm = torch.norm(ablated_zsl_sem, dim=0)
    ablated_zsl_diff = (ablated_zsl_rand_norm - ablated_zsl_sem_norm).cpu().numpy()

    random_ablated_zsl_rand = _flatten(random_ablated_out["zsl_rand"])
    random_ablated_zsl_sem = _flatten(random_ablated_out["zsl_sem"])
    random_ablated_zsl_rand_norm = torch.norm(random_ablated_zsl_rand, dim=0)
    random_ablated_zsl_sem_norm = torch.norm(random_ablated_zsl_sem, dim=0)
    random_ablated_zsl_diff = (random_ablated_zsl_rand_norm - random_ablated_zsl_sem_norm).cpu().numpy()

    # Statistical test for ZSL
    zsl_stat, zsl_p_value = stats.ttest_ind(ablated_zsl_diff, random_ablated_zsl_diff)
    zsl_mean_ablated = np.mean(ablated_zsl_diff)
    zsl_mean_random = np.mean(random_ablated_zsl_diff)

    # Create ZSL direct comparison plot with shared binning
    fig_zsl = go.Figure()
    
    if shared_binning:
        # Calculate shared bin edges
        all_zsl_data = np.concatenate([ablated_zsl_diff, random_ablated_zsl_diff])
        bin_edges = np.linspace(all_zsl_data.min(), all_zsl_data.max(), nbins + 1)
        
        fig_zsl.add_trace(go.Histogram(
            x=ablated_zsl_diff, 
            name='Ablated', 
            opacity=opacity,
            xbins=dict(start=bin_edges[0], end=bin_edges[-1], size=(bin_edges[1] - bin_edges[0])),
            marker_color=colors["ablated"]
        ))
        fig_zsl.add_trace(go.Histogram(
            x=random_ablated_zsl_diff, 
            name='Random Ablated', 
            opacity=opacity,
            xbins=dict(start=bin_edges[0], end=bin_edges[-1], size=(bin_edges[1] - bin_edges[0])),
            marker_color=colors["random_ablated"]
        ))
    else:
        fig_zsl.add_trace(go.Histogram(
            x=ablated_zsl_diff, 
            name='Ablated', 
            opacity=opacity,
            nbinsx=nbins,
            marker_color=colors["ablated"]
        ))
        fig_zsl.add_trace(go.Histogram(
            x=random_ablated_zsl_diff, 
            name='Random Ablated', 
            opacity=opacity,
            nbinsx=nbins,
            marker_color=colors["random_ablated"]
        ))
    
    # Add statistical test results to title
    title_zsl = (f"ZSL Norm Difference: Ablated vs Random Ablated<br>"
                f"Ablated Mean: {zsl_mean_ablated:.4f}, Random Mean: {zsl_mean_random:.4f}<br>"
                f"t-statistic: {zsl_stat:.4f}, p-value: {zsl_p_value:.4f}")
    
    fig_zsl.update_layout(
        title=title_zsl,
        xaxis_title="zsl_rand_norm - zsl_sem_norm",
        yaxis_title="Count",
        barmode='overlay',
        legend=dict(x=0.7, y=0.9)
    )
    
    zsl_path = os.path.join(save_dir, "zsl_norm_diff_direct_comparison.html")
    fig_zsl.write_html(zsl_path)
    if logger is not None:
        logger.info(f"Saved ZSL direct comparison histogram to {zsl_path}")

    # ICL Analysis: Difference between norms of (icl_rand - zsl_rand) and (icl_sem - zsl_sem)
    ablated_icl_rand = _flatten(ablated_out["icl_rand"] - ablated_out["zsl_rand"])
    ablated_icl_sem = _flatten(ablated_out["icl_sem"] - ablated_out["zsl_sem"])
    ablated_icl_rand_norm = torch.norm(ablated_icl_rand, dim=0)
    ablated_icl_sem_norm = torch.norm(ablated_icl_sem, dim=0)
    ablated_icl_diff = (ablated_icl_rand_norm - ablated_icl_sem_norm).cpu().numpy()

    random_ablated_icl_rand = _flatten(random_ablated_out["icl_rand"] - random_ablated_out["zsl_rand"])
    random_ablated_icl_sem = _flatten(random_ablated_out["icl_sem"] - random_ablated_out["zsl_sem"])
    random_ablated_icl_rand_norm = torch.norm(random_ablated_icl_rand, dim=0)
    random_ablated_icl_sem_norm = torch.norm(random_ablated_icl_sem, dim=0)
    random_ablated_icl_diff = (random_ablated_icl_rand_norm - random_ablated_icl_sem_norm).cpu().numpy()

    # Statistical test for ICL
    icl_stat, icl_p_value = stats.ttest_ind(ablated_icl_diff, random_ablated_icl_diff)
    icl_mean_ablated = np.mean(ablated_icl_diff)
    icl_mean_random = np.mean(random_ablated_icl_diff)

    # Create ICL direct comparison plot with shared binning
    fig_icl = go.Figure()
    
    if shared_binning:
        # Calculate shared bin edges for ICL
        all_icl_data = np.concatenate([ablated_icl_diff, random_ablated_icl_diff])
        icl_bin_edges = np.linspace(all_icl_data.min(), all_icl_data.max(), nbins + 1)
        
        fig_icl.add_trace(go.Histogram(
            x=ablated_icl_diff, 
            name='Ablated', 
            opacity=opacity,
            xbins=dict(start=icl_bin_edges[0], end=icl_bin_edges[-1], size=(icl_bin_edges[1] - icl_bin_edges[0])),
            marker_color=colors["ablated"]
        ))
        fig_icl.add_trace(go.Histogram(
            x=random_ablated_icl_diff, 
            name='Random Ablated', 
            opacity=opacity,
            xbins=dict(start=icl_bin_edges[0], end=icl_bin_edges[-1], size=(icl_bin_edges[1] - icl_bin_edges[0])),
            marker_color=colors["random_ablated"]
        ))
    else:
        fig_icl.add_trace(go.Histogram(
            x=ablated_icl_diff, 
            name='Ablated', 
            opacity=opacity,
            nbinsx=nbins,
            marker_color=colors["ablated"]
        ))
        fig_icl.add_trace(go.Histogram(
            x=random_ablated_icl_diff, 
            name='Random Ablated', 
            opacity=opacity,
            nbinsx=nbins,
            marker_color=colors["random_ablated"]
        ))
    
    # Add statistical test results to title
    title_icl = (f"ICL/ZSL Norm Difference: Ablated vs Random Ablated<br>"
                f"Ablated Mean: {icl_mean_ablated:.4f}, Random Mean: {icl_mean_random:.4f}<br>"
                f"t-statistic: {icl_stat:.4f}, p-value: {icl_p_value:.4f}")
    
    fig_icl.update_layout(
        title=title_icl,
        xaxis_title="(icl_rand - zsl_rand)_norm - (icl_sem - zsl_sem)_norm",
        yaxis_title="Count",
        barmode='overlay',
        legend=dict(x=0.7, y=0.9)
    )
    
    icl_path = os.path.join(save_dir, "icl_zsl_norm_diff_direct_comparison.html")
    fig_icl.write_html(icl_path)
    if logger is not None:
        logger.info(f"Saved ICL/ZSL direct comparison histogram to {icl_path}")

    # Original vs Random Ablated Analysis
    orig_zsl_rand = _flatten(original_out["zsl_rand"])
    orig_zsl_sem = _flatten(original_out["zsl_sem"])
    orig_zsl_rand_norm = torch.norm(orig_zsl_rand, dim=0)
    orig_zsl_sem_norm = torch.norm(orig_zsl_sem, dim=0)
    orig_zsl_diff = (orig_zsl_rand_norm - orig_zsl_sem_norm).cpu().numpy()

    # Statistical test for Original vs Random ZSL
    orig_zsl_stat, orig_zsl_p_value = stats.ttest_ind(orig_zsl_diff, random_ablated_zsl_diff)
    orig_zsl_mean_orig = np.mean(orig_zsl_diff)
    orig_zsl_mean_random = np.mean(random_ablated_zsl_diff)

    # Create Original vs Random ZSL overlay plot with shared binning
    fig_orig_zsl = go.Figure()
    
    if shared_binning:
        # Calculate shared bin edges for Original vs Random ZSL
        all_orig_zsl_data = np.concatenate([orig_zsl_diff, random_ablated_zsl_diff])
        orig_zsl_bin_edges = np.linspace(all_orig_zsl_data.min(), all_orig_zsl_data.max(), nbins + 1)
        
        fig_orig_zsl.add_trace(go.Histogram(
            x=orig_zsl_diff, 
            name='Original', 
            opacity=opacity,
            xbins=dict(start=orig_zsl_bin_edges[0], end=orig_zsl_bin_edges[-1], size=(orig_zsl_bin_edges[1] - orig_zsl_bin_edges[0])),
            marker_color=colors["original"]
        ))
        fig_orig_zsl.add_trace(go.Histogram(
            x=random_ablated_zsl_diff, 
            name='Random Ablated', 
            opacity=opacity,
            xbins=dict(start=orig_zsl_bin_edges[0], end=orig_zsl_bin_edges[-1], size=(orig_zsl_bin_edges[1] - orig_zsl_bin_edges[0])),
            marker_color=colors["random_ablated"]
        ))
    else:
        fig_orig_zsl.add_trace(go.Histogram(
            x=orig_zsl_diff, 
            name='Original', 
            opacity=opacity,
            nbinsx=nbins,
            marker_color=colors["original"]
        ))
        fig_orig_zsl.add_trace(go.Histogram(
            x=random_ablated_zsl_diff, 
            name='Random Ablated', 
            opacity=opacity,
            nbinsx=nbins,
            marker_color=colors["random_ablated"]
        ))
    
    # Add statistical test results to title
    title_orig_zsl = (f"ZSL Norm Difference: Original vs Random Ablated<br>"
                     f"Original Mean: {orig_zsl_mean_orig:.4f}, Random Mean: {orig_zsl_mean_random:.4f}<br>"
                     f"t-statistic: {orig_zsl_stat:.4f}, p-value: {orig_zsl_p_value:.4f}")
    
    fig_orig_zsl.update_layout(
        title=title_orig_zsl,
        xaxis_title="zsl_rand_norm - zsl_sem_norm",
        yaxis_title="Count",
        barmode='overlay',
        legend=dict(x=0.7, y=0.9)
    )
    
    orig_zsl_path = os.path.join(save_dir, "zsl_norm_diff_original_vs_random.html")
    fig_orig_zsl.write_html(orig_zsl_path)
    if logger is not None:
        logger.info(f"Saved Original vs Random ZSL histogram to {orig_zsl_path}")

    # Original vs Random ICL Analysis
    orig_icl_rand = _flatten(original_out["icl_rand"] - original_out["zsl_rand"])
    orig_icl_sem = _flatten(original_out["icl_sem"] - original_out["zsl_sem"])
    orig_icl_rand_norm = torch.norm(orig_icl_rand, dim=0)
    orig_icl_sem_norm = torch.norm(orig_icl_sem, dim=0)
    orig_icl_diff = (orig_icl_rand_norm - orig_icl_sem_norm).cpu().numpy()

    # Statistical test for Original vs Random ICL
    orig_icl_stat, orig_icl_p_value = stats.ttest_ind(orig_icl_diff, random_ablated_icl_diff)
    orig_icl_mean_orig = np.mean(orig_icl_diff)
    orig_icl_mean_random = np.mean(random_ablated_icl_diff)

    # Create Original vs Random ICL overlay plot with shared binning
    fig_orig_icl = go.Figure()
    
    if shared_binning:
        # Calculate shared bin edges for Original vs Random ICL
        all_orig_icl_data = np.concatenate([orig_icl_diff, random_ablated_icl_diff])
        orig_icl_bin_edges = np.linspace(all_orig_icl_data.min(), all_orig_icl_data.max(), nbins + 1)
        
        fig_orig_icl.add_trace(go.Histogram(
            x=orig_icl_diff, 
            name='Original', 
            opacity=opacity,
            xbins=dict(start=orig_icl_bin_edges[0], end=orig_icl_bin_edges[-1], size=(orig_icl_bin_edges[1] - orig_icl_bin_edges[0])),
            marker_color=colors["original"]
        ))
        fig_orig_icl.add_trace(go.Histogram(
            x=random_ablated_icl_diff, 
            name='Random Ablated', 
            opacity=opacity,
            xbins=dict(start=orig_icl_bin_edges[0], end=orig_icl_bin_edges[-1], size=(orig_icl_bin_edges[1] - orig_icl_bin_edges[0])),
            marker_color=colors["random_ablated"]
        ))
    else:
        fig_orig_icl.add_trace(go.Histogram(
            x=orig_icl_diff, 
            name='Original', 
            opacity=opacity,
            nbinsx=nbins,
            marker_color=colors["original"]
        ))
        fig_orig_icl.add_trace(go.Histogram(
            x=random_ablated_icl_diff, 
            name='Random Ablated', 
            opacity=opacity,
            nbinsx=nbins,
            marker_color=colors["random_ablated"]
        ))
    
    # Add statistical test results to title
    title_orig_icl = (f"ICL/ZSL Norm Difference: Original vs Random Ablated<br>"
                     f"Original Mean: {orig_icl_mean_orig:.4f}, Random Mean: {orig_icl_mean_random:.4f}<br>"
                     f"t-statistic: {orig_icl_stat:.4f}, p-value: {orig_icl_p_value:.4f}")
    
    fig_orig_icl.update_layout(
        title=title_orig_icl,
        xaxis_title="(icl_rand - zsl_rand)_norm - (icl_sem - zsl_sem)_norm",
        yaxis_title="Count",
        barmode='overlay',
        legend=dict(x=0.7, y=0.9)
    )
    
    orig_icl_path = os.path.join(save_dir, "icl_zsl_norm_diff_original_vs_random.html")
    fig_orig_icl.write_html(orig_icl_path)
    if logger is not None:
        logger.info(f"Saved Original vs Random ICL/ZSL histogram to {orig_icl_path}")

    # Update statistical test results to include original vs random comparisons
    stats_results = {
        "ablated_vs_random_analysis": {
            "zsl_analysis": {
                "ablated_mean": float(zsl_mean_ablated),
                "random_ablated_mean": float(zsl_mean_random),
                "t_statistic": float(zsl_stat),
                "p_value": float(zsl_p_value),
                "significant": int(zsl_p_value < 0.05)
            },
            "icl_analysis": {
                "ablated_mean": float(icl_mean_ablated),
                "random_ablated_mean": float(icl_mean_random),
                "t_statistic": float(icl_stat),
                "p_value": float(icl_p_value),
                "significant": int(icl_p_value < 0.05)
            }
        },
        "original_vs_random_analysis": {
            "zsl_analysis": {
                "original_mean": float(orig_zsl_mean_orig),
                "random_ablated_mean": float(orig_zsl_mean_random),
                "t_statistic": float(orig_zsl_stat),
                "p_value": float(orig_zsl_p_value),
                "significant": int(orig_zsl_p_value < 0.05)
            },
            "icl_analysis": {
                "original_mean": float(orig_icl_mean_orig),
                "random_ablated_mean": float(orig_icl_mean_random),
                "t_statistic": float(orig_icl_stat),
                "p_value": float(orig_icl_p_value),
                "significant": int(orig_icl_p_value < 0.05)
            }
        }
    }
    
    stats_path = os.path.join(save_dir, "statistical_tests_direct_comparison.json")
    with open(stats_path, "w") as f:
        json.dump(stats_results, f, indent=2)
    
    if logger is not None:
        logger.info(f"Updated direct comparison statistical test results to {stats_path}")
        logger.info(f"Original vs Random ZSL: Original mean={orig_zsl_mean_orig:.4f}, Random mean={orig_zsl_mean_random:.4f}, p-value={orig_zsl_p_value:.4f}")
        logger.info(f"Original vs Random ICL: Original mean={orig_icl_mean_orig:.4f}, Random mean={orig_icl_mean_random:.4f}, p-value={orig_icl_p_value:.4f}")


def analyze_probe_overlay_three_way(
    original_out: Dict[str, torch.Tensor], 
    ablated_out: Dict[str, torch.Tensor], 
    random_ablated_out: Dict[str, torch.Tensor],
    save_dir: str, 
    logger=None,
    settings: dict = None
) -> None:
    """Analyze probe outputs with three-way overlay plots comparing original vs ablated vs random ablated distributions.
    
    This function creates overlay histograms comparing the distributions of:
    1. The per-dimension difference between the norms of zsl_rand and zsl_sem
       for original, ablated, and random ablated outputs.
    2. The per-dimension difference between the norms of (icl_rand - zsl_rand) and (icl_sem - zsl_sem)
       for original, ablated, and random ablated outputs.
    
    Each overlay plot includes statistical tests comparing the means of all three distributions.
    
    Parameters
    ----------
    original_out : dict
        Dictionary containing original hidden states for each condition.
    ablated_out : dict
        Dictionary containing ablated hidden states for each condition.
    random_ablated_out : dict
        Dictionary containing random ablated hidden states for each condition.
    save_dir : str
        Directory where overlay plots will be saved.
    logger : optional
        Logger for status messages.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Load settings
    if settings is None:
        settings = load_visualization_settings()
    
    colors = settings["visualization"]["colors"]
    opacity = settings["visualization"]["transparency"]["opacity"]
    nbins = settings["visualization"]["histogram"]["nbins"]
    shared_binning = settings["visualization"]["histogram"]["shared_binning"]

    def _flatten(t: torch.Tensor) -> torch.Tensor:
        return t.detach().reshape(-1, t.shape[-1])

    # ZSL Analysis: Difference between norms of zsl_rand and zsl_sem
    orig_zsl_rand = _flatten(original_out["zsl_rand"])
    orig_zsl_sem = _flatten(original_out["zsl_sem"])
    orig_zsl_rand_norm = torch.norm(orig_zsl_rand, dim=0)
    orig_zsl_sem_norm = torch.norm(orig_zsl_sem, dim=0)
    orig_zsl_diff = (orig_zsl_rand_norm - orig_zsl_sem_norm).cpu().numpy()

    ablated_zsl_rand = _flatten(ablated_out["zsl_rand"])
    ablated_zsl_sem = _flatten(ablated_out["zsl_sem"])
    ablated_zsl_rand_norm = torch.norm(ablated_zsl_rand, dim=0)
    ablated_zsl_sem_norm = torch.norm(ablated_zsl_sem, dim=0)
    ablated_zsl_diff = (ablated_zsl_rand_norm - ablated_zsl_sem_norm).cpu().numpy()

    random_ablated_zsl_rand = _flatten(random_ablated_out["zsl_rand"])
    random_ablated_zsl_sem = _flatten(random_ablated_out["zsl_sem"])
    random_ablated_zsl_rand_norm = torch.norm(random_ablated_zsl_rand, dim=0)
    random_ablated_zsl_sem_norm = torch.norm(random_ablated_zsl_sem, dim=0)
    random_ablated_zsl_diff = (random_ablated_zsl_rand_norm - random_ablated_zsl_sem_norm).cpu().numpy()

    # Statistical tests for ZSL
    zsl_stat_orig_abl, zsl_p_orig_abl = stats.ttest_ind(orig_zsl_diff, ablated_zsl_diff)
    zsl_stat_orig_rand, zsl_p_orig_rand = stats.ttest_ind(orig_zsl_diff, random_ablated_zsl_diff)
    zsl_stat_abl_rand, zsl_p_abl_rand = stats.ttest_ind(ablated_zsl_diff, random_ablated_zsl_diff)
    
    zsl_mean_orig = np.mean(orig_zsl_diff)
    zsl_mean_ablated = np.mean(ablated_zsl_diff)
    zsl_mean_random = np.mean(random_ablated_zsl_diff)

    # Create ZSL three-way overlay plot with shared binning
    fig_zsl = go.Figure()
    
    if shared_binning:
        # Calculate shared bin edges
        all_zsl_data = np.concatenate([orig_zsl_diff, ablated_zsl_diff, random_ablated_zsl_diff])
        bin_edges = np.linspace(all_zsl_data.min(), all_zsl_data.max(), nbins + 1)
        
        fig_zsl.add_trace(go.Histogram(
            x=orig_zsl_diff, 
            name='Original', 
            opacity=opacity,
            xbins=dict(start=bin_edges[0], end=bin_edges[-1], size=(bin_edges[1] - bin_edges[0])),
            marker_color=colors["original"]
        ))
        fig_zsl.add_trace(go.Histogram(
            x=ablated_zsl_diff, 
            name='Ablated', 
            opacity=opacity,
            xbins=dict(start=bin_edges[0], end=bin_edges[-1], size=(bin_edges[1] - bin_edges[0])),
            marker_color=colors["ablated"]
        ))
        fig_zsl.add_trace(go.Histogram(
            x=random_ablated_zsl_diff, 
            name='Random Ablated', 
            opacity=opacity,
            xbins=dict(start=bin_edges[0], end=bin_edges[-1], size=(bin_edges[1] - bin_edges[0])),
            marker_color=colors["random_ablated"]
        ))
    else:
        fig_zsl.add_trace(go.Histogram(
            x=orig_zsl_diff, 
            name='Original', 
            opacity=opacity,
            nbinsx=nbins,
            marker_color=colors["original"]
        ))
        fig_zsl.add_trace(go.Histogram(
            x=ablated_zsl_diff, 
            name='Ablated', 
            opacity=opacity,
            nbinsx=nbins,
            marker_color=colors["ablated"]
        ))
        fig_zsl.add_trace(go.Histogram(
            x=random_ablated_zsl_diff, 
            name='Random Ablated', 
            opacity=opacity,
            nbinsx=nbins,
            marker_color=colors["random_ablated"]
        ))
    
    # Add statistical test results to title
    title_zsl = (f"ZSL Norm Difference: Original vs Ablated vs Random Ablated<br>"
                f"Original Mean: {zsl_mean_orig:.4f}, Ablated Mean: {zsl_mean_ablated:.4f}, Random Mean: {zsl_mean_random:.4f}<br>"
                f"Orig vs Abl: p={zsl_p_orig_abl:.4f}, Orig vs Rand: p={zsl_p_orig_rand:.4f}, Abl vs Rand: p={zsl_p_abl_rand:.4f}")
    
    fig_zsl.update_layout(
        title=title_zsl,
        xaxis_title="zsl_rand_norm - zsl_sem_norm",
        yaxis_title="Count",
        barmode='overlay',
        legend=dict(x=0.7, y=0.9)
    )
    
    zsl_path = os.path.join(save_dir, "zsl_norm_diff_three_way_overlay.html")
    fig_zsl.write_html(zsl_path)
    if logger is not None:
        logger.info(f"Saved ZSL three-way overlay histogram to {zsl_path}")

    # ICL Analysis: Difference between norms of (icl_rand - zsl_rand) and (icl_sem - zsl_sem)
    orig_icl_rand = _flatten(original_out["icl_rand"] - original_out["zsl_rand"])
    orig_icl_sem = _flatten(original_out["icl_sem"] - original_out["zsl_sem"])
    orig_icl_rand_norm = torch.norm(orig_icl_rand, dim=0)
    orig_icl_sem_norm = torch.norm(orig_icl_sem, dim=0)
    orig_icl_diff = (orig_icl_rand_norm - orig_icl_sem_norm).cpu().numpy()

    ablated_icl_rand = _flatten(ablated_out["icl_rand"] - ablated_out["zsl_rand"])
    ablated_icl_sem = _flatten(ablated_out["icl_sem"] - ablated_out["zsl_sem"])
    ablated_icl_rand_norm = torch.norm(ablated_icl_rand, dim=0)
    ablated_icl_sem_norm = torch.norm(ablated_icl_sem, dim=0)
    ablated_icl_diff = (ablated_icl_rand_norm - ablated_icl_sem_norm).cpu().numpy()

    random_ablated_icl_rand = _flatten(random_ablated_out["icl_rand"] - random_ablated_out["zsl_rand"])
    random_ablated_icl_sem = _flatten(random_ablated_out["icl_sem"] - random_ablated_out["zsl_sem"])
    random_ablated_icl_rand_norm = torch.norm(random_ablated_icl_rand, dim=0)
    random_ablated_icl_sem_norm = torch.norm(random_ablated_icl_sem, dim=0)
    random_ablated_icl_diff = (random_ablated_icl_rand_norm - random_ablated_icl_sem_norm).cpu().numpy()

    # Statistical tests for ICL
    icl_stat_orig_abl, icl_p_orig_abl = stats.ttest_ind(orig_icl_diff, ablated_icl_diff)
    icl_stat_orig_rand, icl_p_orig_rand = stats.ttest_ind(orig_icl_diff, random_ablated_icl_diff)
    icl_stat_abl_rand, icl_p_abl_rand = stats.ttest_ind(ablated_icl_diff, random_ablated_icl_diff)
    
    icl_mean_orig = np.mean(orig_icl_diff)
    icl_mean_ablated = np.mean(ablated_icl_diff)
    icl_mean_random = np.mean(random_ablated_icl_diff)

    # Create ICL three-way overlay plot with shared binning
    fig_icl = go.Figure()
    
    if shared_binning:
        # Calculate shared bin edges for ICL
        all_icl_data = np.concatenate([orig_icl_diff, ablated_icl_diff, random_ablated_icl_diff])
        icl_bin_edges = np.linspace(all_icl_data.min(), all_icl_data.max(), nbins + 1)
        
        fig_icl.add_trace(go.Histogram(
            x=orig_icl_diff, 
            name='Original', 
            opacity=opacity,
            xbins=dict(start=icl_bin_edges[0], end=icl_bin_edges[-1], size=(icl_bin_edges[1] - icl_bin_edges[0])),
            marker_color=colors["original"]
        ))
        fig_icl.add_trace(go.Histogram(
            x=ablated_icl_diff, 
            name='Ablated', 
            opacity=opacity,
            xbins=dict(start=icl_bin_edges[0], end=icl_bin_edges[-1], size=(icl_bin_edges[1] - icl_bin_edges[0])),
            marker_color=colors["ablated"]
        ))
        fig_icl.add_trace(go.Histogram(
            x=random_ablated_icl_diff, 
            name='Random Ablated', 
            opacity=opacity,
            xbins=dict(start=icl_bin_edges[0], end=icl_bin_edges[-1], size=(icl_bin_edges[1] - icl_bin_edges[0])),
            marker_color=colors["random_ablated"]
        ))
    else:
        fig_icl.add_trace(go.Histogram(
            x=orig_icl_diff, 
            name='Original', 
            opacity=opacity,
            nbinsx=nbins,
            marker_color=colors["original"]
        ))
        fig_icl.add_trace(go.Histogram(
            x=ablated_icl_diff, 
            name='Ablated', 
            opacity=opacity,
            nbinsx=nbins,
            marker_color=colors["ablated"]
        ))
        fig_icl.add_trace(go.Histogram(
            x=random_ablated_icl_diff, 
            name='Random Ablated', 
            opacity=opacity,
            nbinsx=nbins,
            marker_color=colors["random_ablated"]
        ))
    
    # Add statistical test results to title
    title_icl = (f"ICL/ZSL Norm Difference: Original vs Ablated vs Random Ablated<br>"
                f"Original Mean: {icl_mean_orig:.4f}, Ablated Mean: {icl_mean_ablated:.4f}, Random Mean: {icl_mean_random:.4f}<br>"
                f"Orig vs Abl: p={icl_p_orig_abl:.4f}, Orig vs Rand: p={icl_p_orig_rand:.4f}, Abl vs Rand: p={icl_p_abl_rand:.4f}")
    
    fig_icl.update_layout(
        title=title_icl,
        xaxis_title="(icl_rand - zsl_rand)_norm - (icl_sem - zsl_sem)_norm",
        yaxis_title="Count",
        barmode='overlay',
        legend=dict(x=0.7, y=0.9)
    )
    
    icl_path = os.path.join(save_dir, "icl_zsl_norm_diff_three_way_overlay.html")
    fig_icl.write_html(icl_path)
    if logger is not None:
        logger.info(f"Saved ICL/ZSL three-way overlay histogram to {icl_path}")

    # Save statistical test results to a JSON file
    stats_results = {
        "zsl_analysis": {
            "original_mean": float(zsl_mean_orig),
            "ablated_mean": float(zsl_mean_ablated),
            "random_ablated_mean": float(zsl_mean_random),
            "orig_vs_abl": {
                "t_statistic": float(zsl_stat_orig_abl),
                "p_value": float(zsl_p_orig_abl),
                "significant": int(zsl_p_orig_abl < 0.05)
            },
            "orig_vs_random": {
                "t_statistic": float(zsl_stat_orig_rand),
                "p_value": float(zsl_p_orig_rand),
                "significant": int(zsl_p_orig_rand < 0.05)
            },
            "abl_vs_random": {
                "t_statistic": float(zsl_stat_abl_rand),
                "p_value": float(zsl_p_abl_rand),
                "significant": int(zsl_p_abl_rand < 0.05)
            }
        },
        "icl_analysis": {
            "original_mean": float(icl_mean_orig),
            "ablated_mean": float(icl_mean_ablated),
            "random_ablated_mean": float(icl_mean_random),
            "orig_vs_abl": {
                "t_statistic": float(icl_stat_orig_abl),
                "p_value": float(icl_p_orig_abl),
                "significant": int(icl_p_orig_abl < 0.05)
            },
            "orig_vs_random": {
                "t_statistic": float(icl_stat_orig_rand),
                "p_value": float(icl_p_orig_rand),
                "significant": int(icl_p_orig_rand < 0.05)
            },
            "abl_vs_random": {
                "t_statistic": float(icl_stat_abl_rand),
                "p_value": float(icl_p_abl_rand),
                "significant": int(icl_p_abl_rand < 0.05)
            }
        }
    }
    
    stats_path = os.path.join(save_dir, "statistical_tests_three_way.json")
    with open(stats_path, "w") as f:
        json.dump(stats_results, f, indent=2)
    
    if logger is not None:
        logger.info(f"Saved three-way statistical test results to {stats_path}")
        logger.info(f"ZSL: Orig={zsl_mean_orig:.4f}, Abl={zsl_mean_ablated:.4f}, Rand={zsl_mean_random:.4f}")
        logger.info(f"ICL: Orig={icl_mean_orig:.4f}, Abl={icl_mean_ablated:.4f}, Rand={icl_mean_random:.4f}")


def export_local_output(tokens: Dict[str, str], save_dir: str, logger=None) -> None:
    """Save generated tokens for the four evaluation conditions.

    Parameters
    ----------
    tokens : dict
        Dictionary with keys ``zsl_rand``, ``zsl_sem``, ``icl_rand`` and
        ``icl_sem`` mapping to the decoded token strings produced by the
        projector.
    save_dir : str
        Directory where the JSON file will be written.
    logger : optional
        Logger for status messages.
    """
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "local_output.json")
    with open(out_path, "w") as f:
        json.dump(tokens, f, indent=2)
    if logger is not None:
        logger.info(f"Saved local output tokens to {out_path}")


def create_llava_wordclouds(
    llava_outputs: Dict[str, List[Dict[str, str]]],
    save_dir: str,
    settings: dict = None,
    logger=None,
) -> None:
    """Create wordclouds for LLaVA model outputs (before projection).
    
    Parameters
    ----------
    llava_outputs : dict
        Mapping from condition name to a list of dictionaries containing
        ``id`` and ``text`` for each example from LLaVA model outputs.
    save_dir : str
        Directory where wordcloud images will be saved.
    settings : dict, optional
        Visualization settings dictionary.
    logger : optional
        Logger for status messages.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Load settings
    if settings is None:
        settings = load_visualization_settings()
    
    wc_settings = settings["wordcloud"]
    width = wc_settings["width"]
    height = wc_settings["height"]
    background_color = wc_settings["background_color"]

    # Prepare wordclouds
    base_stopwords = set(STOPWORDS)

    def tokenize(text: str) -> List[str]:
        return [w.lower() for w in re.findall(r"\b\w+\b", text)]

    word_sets: Dict[str, set] = {}
    for cond, records in llava_outputs.items():
        docs = [r["text"] for r in records]
        words = []
        for doc in docs:
            words.extend(tokenize(doc))
        word_sets[cond] = set(w for w in words if w not in base_stopwords)

    common_words = set.intersection(*word_sets.values()) if word_sets else set()
    stopwords = base_stopwords | common_words

    for cond, records in llava_outputs.items():
        docs = [r["text"] for r in records]
        text = " ".join(docs)
        if not text:
            continue
        wc = WordCloud(
            width=width,
            height=height,
            background_color=background_color,
            stopwords=stopwords,
        ).generate(text)
        wc_path = os.path.join(save_dir, f"{cond}_llava_wordcloud.png")
        wc.to_file(wc_path)
        if logger is not None:
            logger.info(f"Saved LLaVA wordcloud for {cond} to {wc_path}")


def characterize_by_word(
    token_records: Dict[str, List[Dict[str, str]]],
    save_dir: str,
    top_n: int = 20,
    logger=None,
) -> None:
    """Run TF-IDF analysis and create wordclouds for generated tokens.

    Parameters
    ----------
    token_records : dict
        Mapping from condition name to a list of dictionaries containing
        ``id`` and ``text`` for each example.
    save_dir : str
        Directory where analysis results will be saved.
    top_n : int, optional
        Number of top-scoring entries to retain per condition.
    logger : optional
        Logger for status messages.
    """
    os.makedirs(save_dir, exist_ok=True)

    tfidf_summary: Dict[str, List[Dict[str, object]]] = {}
    docs_per_cond: Dict[str, List[str]] = {}

    for cond, records in token_records.items():
        ids = [r["id"] for r in records]
        docs = [r["text"] for r in records]
        docs_per_cond[cond] = docs
        if len(docs) == 0:
            tfidf_summary[cond] = []
            continue

        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform(docs)
        feature_names = np.array(vectorizer.get_feature_names_out())
        scores = tfidf.toarray()
        max_indices = scores.argmax(axis=1)
        max_scores = scores.max(axis=1)

        cond_res: List[Dict[str, object]] = []
        for doc_id, doc_text, w_idx, s in zip(ids, docs, max_indices, max_scores):
            cond_res.append(
                {
                    "id": doc_id,
                    "text": doc_text,
                    "word": feature_names[w_idx],
                    "score": float(s),
                }
            )

        cond_res.sort(key=lambda x: x["score"], reverse=True)
        tfidf_summary[cond] = cond_res[:top_n]

    json_path = os.path.join(save_dir, f"tfidf_top_{top_n}.json")
    with open(json_path, "w") as f:
        json.dump(tfidf_summary, f, indent=2)
    if logger is not None:
        logger.info(f"Saved TF-IDF analysis to {json_path}")

    # Prepare wordclouds
    base_stopwords = set(STOPWORDS)

    def tokenize(text: str) -> List[str]:
        return [w.lower() for w in re.findall(r"\b\w+\b", text)]

    word_sets: Dict[str, set] = {}
    for cond, docs in docs_per_cond.items():
        words = []
        for doc in docs:
            words.extend(tokenize(doc))
        word_sets[cond] = set(w for w in words if w not in base_stopwords)

    common_words = set.intersection(*word_sets.values()) if word_sets else set()
    stopwords = base_stopwords | common_words

    for cond, docs in docs_per_cond.items():
        text = " ".join(docs)
        if not text:
            continue
        wc = WordCloud(
            width=800,
            height=400,
            background_color="white",
            stopwords=stopwords,
        ).generate(text)
        wc_path = os.path.join(save_dir, f"{cond}_wordcloud.png")
        wc.to_file(wc_path)
        if logger is not None:
            logger.info(f"Saved wordcloud for {cond} to {wc_path}")


def load_images(images_or_image_files):
    out = []
    if type(images_or_image_files) is not list:
        image_list = [images_or_image_files]
    else:
        image_list = images_or_image_files
    for image_file in image_list:
        if not isinstance(image_file, str):
            image = image_file
        else:
            image = load_image(image_file)
        out.append(image)
    return out


def _compute_outputs_for_qid(
    qid: str,
    qid_texts: Dict[str, str],
    iid_texts: Dict[str, str],
    qid_images: Dict[str, str],
    iid_images: Dict[str, str],
    answers: Dict[str, str],
    args: argparse.Namespace,
    tokenizer: PreTrainedTokenizer,
    llava_model: LlavaLlamaForCausalLM,
    image_processor,
    device: torch.device,
    logger=None,
):
    """Prepare model outputs for a single dataset entry."""
    qid_text = qid_texts[qid]
    iid_text = iid_texts[qid] if qid in iid_texts else None
    qid_image = qid_images[qid]
    iid_image = iid_images[qid] if qid in iid_images else None
    answer = answers[qid] if qid in answers else None

    # ZSL inputs
    zsl_image = load_image(qid_image).convert("RGB") if iid_image else None
    zsl_tensor_image = to_tensor(zsl_image) if zsl_image else None
    if args.image_noise_pattern == "black":
        _ = to_pil_image(torch.zeros_like(zsl_tensor_image))
    elif args.image_noise_pattern == "white":
        _ = to_pil_image(torch.ones_like(zsl_tensor_image) * 255)
    elif args.image_noise_pattern == "random":
        _ = to_pil_image(torch.rand_like(zsl_tensor_image) * 255)
    else:
        raise ValueError(
            f"Unsupported image noise pattern: {args.image_noise_pattern}. "
            "Supported patterns are 'black', 'white', and 'random'."
        )
    zsl_text_rand = " ".join(
        np.random.permutation(qid_text.split()) if qid_text else []
    )

    # ICL inputs
    icl_images_sem = load_images([iid_image, qid_image]) if iid_image else None
    org_image = load_image(iid_image).convert("RGB") if iid_image else None
    org_tensor_image = to_tensor(org_image) if org_image is not None else None
    if org_tensor_image is not None:
        if args.image_noise_pattern == "black":
            icl_image_rand = to_pil_image(torch.zeros_like(org_tensor_image))
        elif args.image_noise_pattern == "white":
            icl_image_rand = to_pil_image(torch.ones_like(org_tensor_image) * 255)
        elif args.image_noise_pattern == "random":
            icl_image_rand = to_pil_image(torch.rand_like(org_tensor_image) * 255)
        else:
            raise ValueError(
                f"Unsupported image noise pattern: {args.image_noise_pattern}. "
                "Supported patterns are 'black', 'white', and 'random'."
            )
        icl_images_rand = load_images([icl_image_rand, qid_image])
    else:
        icl_images_rand = None
    iid_text_rand = " ".join(
        np.random.permutation(iid_text.split()) if iid_text else []
    )
    answer_rand = " ".join(np.random.permutation(answer.split()) if answer else [])

    # Extract hidden representations using batched forward passes
    zsl_batch = extract_outputs_batch(
        batch_input_texts=[[zsl_text_rand], [qid_text]],
        batch_input_images=[zsl_image, zsl_image],
        model=llava_model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_new_tokens=128,
        logger=logger,
        conv_mode=args.conv_mode,
        device=device,
    )

    icl_texts_sem = [iid_text, answer, qid_text] if iid_text else [qid_text]
    icl_texts_rand = [iid_text_rand, answer_rand, qid_text] if iid_text else [qid_text]
    icl_batch = extract_outputs_batch(
        batch_input_texts=[icl_texts_rand, icl_texts_sem],
        batch_input_images=[icl_images_rand, icl_images_sem],
        model=llava_model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_new_tokens=128,
        logger=logger,
        conv_mode=args.conv_mode,
        device=device,
    )

    zsl_outputs_rand, zsl_outputs = zsl_batch
    icl_outputs_rand, icl_outputs_sem = icl_batch

    return zsl_outputs_rand, zsl_outputs, icl_outputs_rand, icl_outputs_sem


def train(
    args: argparse.Namespace,
    dataset: QIDDataset,
    tokenizer: PreTrainedTokenizer,
    llava_model: LlavaLlamaForCausalLM,
    device: torch.device,
    logger=None,
):
    """Train the FourSpaceProjector on all dataset entries."""
    config = CustomModelConfig()
    probe = FourSpaceProjector(
        hidden_size=config.embed_dim,
        bias=True,
        gate="scalar",
        init="xavier_uniform",
    ).to(device)
    if args.debug:
        logger.info("Debug mode enabled")
        dataset.qids = dataset.qids[: args.sample_index+10]
    logger.info(f"Training on {len(dataset)} samples")

    loader = DataLoader(
        dataset,
        batch_size=args.forward_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_qid_batch,
    )

    embeddings = []
    for batch in loader:
        zsl_inputs = batch["zsl_rand"] + batch["zsl_sem"]
        icl_inputs = batch["icl_rand"] + batch["icl_sem"]

        zsl_hidden = forward_hidden_states_batch(
            zsl_inputs, llava_model, tokenizer, device
        )
        del zsl_inputs
        if torch.cuda.is_available():
            zsl_hidden = zsl_hidden.cpu()
        free_memory()
        icl_hidden = forward_hidden_states_batch(
            icl_inputs, llava_model, tokenizer, device
        )
        del icl_inputs
        if torch.cuda.is_available():
            icl_hidden = icl_hidden.cpu()
        free_memory()

        bsz = len(batch["zsl_rand"])
        for i in range(bsz):
            zsl_rand_h = zsl_hidden[2 * i : 2 * i + 1]
            zsl_sem_h = zsl_hidden[2 * i + 1 : 2 * i + 2]
            icl_rand_h = icl_hidden[2 * i : 2 * i + 1]
            icl_sem_h = icl_hidden[2 * i + 1 : 2 * i + 2]

            def _agg(h):
                return h.mean(dim=1).squeeze(0).to(torch.float32).cpu()

            embeddings.append(
                (
                    _agg(zsl_rand_h),
                    _agg(zsl_sem_h),
                    _agg(icl_rand_h),
                    _agg(icl_sem_h),
                )
            )
        free_memory()
    logger.info(f"Finished processing {len(dataset)} samples")

    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
    best_loss = float("inf")
    os.makedirs(args.exp_dir, exist_ok=True)
    best_path = os.path.join(args.exp_dir, "four_space_projector.pt")

    wandb_run = None
    if getattr(args, "wandb_run_name", None):
        wandb_run = wandb.init(project="probe_analysis", name=args.wandb_run_name)

    global_step = 0
    for epoch in range(args.num_epoch):
        log_losses = []
        probe.train()
        for start in range(0, len(embeddings), args.batch_size):
            batch = embeddings[start : start + args.batch_size]
            dtype = next(probe.parameters()).dtype
            zsl_rand = torch.stack([b[0] for b in batch]).to(device=device, dtype=dtype)
            zsl_sem = torch.stack([b[1] for b in batch]).to(device=device, dtype=dtype)
            icl_rand = torch.stack([b[2] for b in batch]).to(device=device, dtype=dtype)
            icl_sem = torch.stack([b[3] for b in batch]).to(device=device, dtype=dtype)

            optimizer.zero_grad()
            out = probe(zsl_rand, zsl_sem, icl_rand, icl_sem)
            loss = 0.0
            loss += (1 - F.cosine_similarity(out["zsl_rand"], zsl_rand, dim=-1)).mean()
            loss += (1 - F.cosine_similarity(out["zsl_sem"], zsl_sem, dim=-1)).mean()
            loss += (1 - F.cosine_similarity(out["icl_rand"], icl_rand, dim=-1)).mean()
            loss += (1 - F.cosine_similarity(out["icl_sem"], icl_sem, dim=-1)).mean()
            loss = loss / 4
            loss.backward()
            optimizer.step()
            log_losses.append(loss.item())
            logger.info(
                f"Epoch {epoch + 1}/{args.num_epoch}, batch {start // args.batch_size}: {loss.item()}"
            )
            if wandb_run is not None:
                wandb.log(
                    {
                        "train/batch_loss": loss.item(),
                        "epoch": epoch + 1,
                        "batch": start // args.batch_size + 1,
                    },
                    step=global_step,
                )
            global_step += 1
            del zsl_rand, zsl_sem, icl_rand, icl_sem, out, loss
            free_memory()

        epoch_loss = float(np.mean(log_losses)) if log_losses else 0.0
        if wandb_run is not None:
            wandb.log(
                {
                    "train/epoch_loss": epoch_loss,
                    "epoch": epoch + 1,
                },
                step=global_step,
            )
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(probe.state_dict(), best_path)
        free_memory()

    if wandb_run is not None:
        wandb.log({"train/best_loss": best_loss}, step=global_step)
        artifact = wandb.Artifact("four_space_projector", type="model")
        artifact.add_file(best_path)
        wandb.log_artifact(artifact)
    else:
        with open(os.path.join(args.exp_dir, "train_log.json"), "w") as f:
            json.dump({"train_loss": best_loss}, f)

    probe.load_state_dict(torch.load(best_path, map_location=device))
    del embeddings
    free_memory()
    return probe


def eval_local(
    args: argparse.Namespace,
    dataset: QIDDataset,
    tokenizer: PreTrainedTokenizer,
    llava_model: LlavaLlamaForCausalLM,
    probe: FourSpaceProjector,
    device: torch.device,
    logger=None,
):
    """Evaluate the trained projector on a single dataset entry."""
    idx = args.sample_index
    item = dataset[idx]
    inputs = [
        item["zsl_rand"],
        item["zsl_sem"],
        item["icl_rand"],
        item["icl_sem"],
    ]
    hidden = forward_hidden_states_batch(inputs, llava_model, tokenizer, device)
    outputs = [
        {"hidden_states": [hidden[i : i + 1]]} for i in range(len(inputs))
    ]
    proj_output = project_embedding(
        llava_model,
        tokenizer,
        probe,
        outputs,
        logger=logger,
    )
    logger.info(f"ZSL (Rand) output tokens: {proj_output['tokens']['zsl_rand']}")
    logger.info(f"ZSL (Sem) output tokens: {proj_output['tokens']['zsl_sem']}")
    logger.info(f"ICL (Rand) output tokens: {proj_output['tokens']['icl_rand']}")
    logger.info(f"ICL (Sem) output tokens: {proj_output['tokens']['icl_sem']}")
    if getattr(args, "wandb_run_name", None):
        if args.eval_only:
            wandb.init(
                project="probe_analysis",
                name=args.wandb_run_name,
                resume="allow",
            )
        wandb.log({
            "zsl_rand": proj_output["tokens"]["zsl_rand"],
            "zsl_sem": proj_output["tokens"]["zsl_sem"],
            "icl_rand": proj_output["tokens"]["icl_rand"],
            "icl_sem": proj_output["tokens"]["icl_sem"],
        })
    export_local_output(proj_output["tokens"], args.exp_dir, logger)
    analyze_probe(proj_output["hidden_states"], args.exp_dir, logger)


def extract_vocabulary_from_dataset(
    dataset: QIDDataset,
    tokenizer: PreTrainedTokenizer,
    logger = None,
) -> List[str]:
    """Extract all unique words from the dataset for random ablation.
    
    Parameters
    ----------
    dataset : QIDDataset
        Dataset containing text samples.
    tokenizer : PreTrainedTokenizer
        Tokenizer used to decode tokens.
    logger : optional
        Logger for status messages.
        
    Returns
    -------
    list[str]
        List of unique words found in the dataset.
    """
    vocabulary = set()
    
    if logger:
        logger.info("Extracting vocabulary from dataset for random ablation...")
    
    # Sample a subset of the dataset to extract vocabulary (to avoid memory issues)
    sample_size = min(1000, len(dataset))
    sample_indices = random.sample(range(len(dataset)), sample_size)
    
    for idx in sample_indices:
        item = dataset[idx]
        
        # Extract text from all conditions
        for condition in ["zsl_rand", "zsl_sem", "icl_rand", "icl_sem"]:
            if condition in item:
                input_ids = item[condition]["input_ids"]
                # Decode tokens to text
                text = tokenizer.decode(input_ids, skip_special_tokens=True)
                # Extract words (simple word splitting)
                words = re.findall(r'\b\w+\b', text.lower())
                vocabulary.update(words)
    
    vocabulary_list = list(vocabulary)
    
    if logger:
        logger.info(f"Extracted {len(vocabulary_list)} unique words from dataset sample")
    
    return vocabulary_list


def select_random_words(
    vocabulary: List[str],
    target_count: int,
    exclude_words: Optional[List[str]] = None,
    logger = None,
) -> List[str]:
    """Select random words from vocabulary for ablation.
    
    Parameters
    ----------
    vocabulary : list[str]
        List of available words to choose from.
    target_count : int
        Number of words to select.
    exclude_words : list[str], optional
        Words to exclude from selection.
    logger : optional
        Logger for status messages.
        
    Returns
    -------
    list[str]
        List of randomly selected words.
    """
    if exclude_words is None:
        exclude_words = []
    
    # Filter out excluded words
    available_words = [word for word in vocabulary if word not in exclude_words]
    
    # Ensure we have enough words
    if len(available_words) < target_count:
        if logger:
            logger.warning(f"Not enough words available ({len(available_words)} < {target_count}). Using all available words.")
        return available_words
    
    # Randomly select words
    selected_words = random.sample(available_words, target_count)
    
    if logger:
        logger.info(f"Selected {len(selected_words)} random words for ablation: {selected_words}")
    
    return selected_words


def ablate_input_tokens(
    inputs: List[Dict[str, torch.Tensor]],
    tokenizer: PreTrainedTokenizer,
    words_to_ablate: Optional[List[str]] = None,
    logger = None,
    debug: bool = False,
) -> List[Dict[str, torch.Tensor]]:
    """Replace specified tokens with [PAD] token in input token sequences.
    
    Parameters
    ----------
    inputs : list[dict]
        List of input dictionaries, each containing 'input_ids', 'attention_mask', and 'image'.
    tokenizer : PreTrainedTokenizer
        Tokenizer used to encode the words to ablate.
    words_to_ablate : list[str], optional
        List of words to replace with [PAD] token. If None, no ablation is performed.
    logger : optional
        Logger for status messages.
    debug : bool, optional
        If True, log detailed debug information about token IDs and their decoded text.
        
    Returns
    -------
    list[dict]
        Modified inputs with specified tokens replaced by [PAD] token.
    """
    if words_to_ablate is None or len(words_to_ablate) == 0:
        return inputs
    
    # Get the PAD token ID
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    
    # Tokenize the words to ablate
    words_to_ablate_ids = set()
    for word in words_to_ablate:
        # Tokenize the word and add all token IDs to the set
        word_tokens = tokenizer.encode(word, add_special_tokens=False)
        words_to_ablate_ids.update(word_tokens)
        
        # Debug logging for words to ablate
        if debug and logger:
            logger.info(f"DEBUG: Word to ablate: '{word}' -> Token IDs: {word_tokens}")
            for token_id in word_tokens:
                decoded_text = tokenizer.decode([token_id], skip_special_tokens=True)
                logger.info(f"DEBUG: Token ID {token_id} -> Decoded text: '{decoded_text}'")
    
    # Create a copy of inputs to avoid modifying the original
    ablated_inputs = []
    
    for i, input_dict in enumerate(inputs):
        ablated_input = input_dict.copy()
        input_ids = input_dict["input_ids"].clone()
        
        # Debug logging for original token IDs
        if debug and logger:
            original_text = tokenizer.decode(input_ids, skip_special_tokens=True)
            logger.info(f"DEBUG: Input {i} - Original text: '{original_text}'")
            logger.info(f"DEBUG: Input {i} - Original token IDs: {input_ids.tolist()}")
            for j, token_id in enumerate(input_ids):
                decoded_text = tokenizer.decode([token_id], skip_special_tokens=True)
                logger.info(f"DEBUG: Input {i} - Position {j}: Token ID {token_id} -> Decoded text: '{decoded_text}'")
        
        # Create ablated token IDs by replacing matching tokens with PAD
        ablated_token_ids = []
        for token_id in input_ids:
            if token_id.item() in words_to_ablate_ids:
                if logger:
                    logger.info(f"Ablating token ID {token_id.item()} in input {i}")
                ablated_token_ids.append(pad_token_id)
            else:
                ablated_token_ids.append(token_id.item())
        
        # Convert back to tensor
        ablated_input["input_ids"] = torch.tensor(ablated_token_ids, dtype=input_ids.dtype, device=input_ids.device)
        
        # Debug logging for ablated token IDs
        if debug and logger:
            ablated_text = tokenizer.decode(ablated_input["input_ids"], skip_special_tokens=True)
            logger.info(f"DEBUG: Input {i} - Ablated text: '{ablated_text}'")
            logger.info(f"DEBUG: Input {i} - Ablated token IDs: {ablated_input['input_ids'].tolist()}")
        
        ablated_inputs.append(ablated_input)
    
    return ablated_inputs


def ablate_tokens(
    proj_output: Dict[str, any],
    tokenizer: PreTrainedTokenizer,
    words_to_ablate: Optional[List[str]] = None,
    logger = None,
    debug: bool = False,
) -> Dict[str, any]:
    """Replace specified tokens with [PAD] token in model outputs and perform Rand vs. Sem analysis.
    
    Parameters
    ----------
    proj_output : dict
        Dictionary containing 'tokens' and 'hidden_states' for each condition.
        Structure: {
            "hidden_states": {
                "zsl_rand": tensor, "zsl_sem": tensor, 
                "icl_rand": tensor, "icl_sem": tensor
            },
            "tokens": {
                "zsl_rand": str, "zsl_sem": str,
                "icl_rand": str, "icl_sem": str
            }
        }
    tokenizer : PreTrainedTokenizer
        Tokenizer used to encode the words to ablate.
    words_to_ablate : list[str], optional
        List of words to replace with [PAD] token. If None, no ablation is performed.
    logger : optional
        Logger for status messages.
    debug : bool, optional
        If True, log detailed debug information about token IDs and their decoded text.
        
    Returns
    -------
    dict
        Modified proj_output with specified tokens replaced by [PAD] token.
    """
    if words_to_ablate is None or len(words_to_ablate) == 0:
        return proj_output
    
    # Get the PAD token ID
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    
    # Tokenize the words to ablate
    words_to_ablate_ids = set()
    for word in words_to_ablate:
        # Tokenize the word and add all token IDs to the set
        word_tokens = tokenizer.encode(word, add_special_tokens=False)
        words_to_ablate_ids.update(word_tokens)
        
        # Debug logging for words to ablate
        if debug and logger:
            logger.info(f"DEBUG: Word to ablate: '{word}' -> Token IDs: {word_tokens}")
            for token_id in word_tokens:
                decoded_text = tokenizer.decode([token_id], skip_special_tokens=True)
                logger.info(f"DEBUG: Token ID {token_id} -> Decoded text: '{decoded_text}'")
    
    # Create a copy of proj_output to avoid modifying the original
    ablated_output = {
        "hidden_states": {},
        "tokens": {}
    }
    
    # Process each condition
    for condition in ["zsl_rand", "zsl_sem", "icl_rand", "icl_sem"]:
        # Assert the presence of the condition
        assert condition in proj_output["tokens"], f"Missing condition {condition} in tokens"
        assert condition in proj_output["hidden_states"], f"Missing condition {condition} in hidden_states"
        
        # Get the original tokens and hidden states
        original_tokens = proj_output["tokens"][condition]
        original_hidden_states = proj_output["hidden_states"][condition]
        
        # Tokenize the original text to get token IDs
        original_token_ids = tokenizer.encode(original_tokens, add_special_tokens=False)
        
        # Debug logging for original token IDs
        if debug and logger:
            logger.info(f"DEBUG: Condition '{condition}' - Original text: '{original_tokens}'")
            logger.info(f"DEBUG: Condition '{condition}' - Original token IDs: {original_token_ids}")
            for i, token_id in enumerate(original_token_ids):
                decoded_text = tokenizer.decode([token_id], skip_special_tokens=True)
                logger.info(f"DEBUG: Condition '{condition}' - Position {i}: Token ID {token_id} -> Decoded text: '{decoded_text}'")
        
        # Create ablated token IDs by replacing matching tokens with PAD
        ablated_token_ids = []
        for token_id in original_token_ids:
            if token_id in words_to_ablate_ids:
                if logger:
                    logger.info(f"Ablating token ID {token_id} in condition {condition}")
                ablated_token_ids.append(pad_token_id)
            else:
                ablated_token_ids.append(token_id)
        
        # Decode the ablated token IDs back to text
        ablated_tokens = tokenizer.decode(ablated_token_ids, skip_special_tokens=True)
        
        # For hidden states, we need to identify which positions correspond to the ablated tokens
        # and replace those positions with PAD token embeddings or zero vectors
        ablated_hidden_states = original_hidden_states.clone()
        
        # Get the PAD token embedding from the model's embedding layer
        # We'll use zero vectors as a proxy for PAD embeddings
        pad_embedding = torch.zeros_like(ablated_hidden_states[0, 0, :])  # [hidden_dim]
        
        # Replace hidden states at positions where tokens were ablated
        for i, (original_id, ablated_id) in enumerate(zip(original_token_ids, ablated_token_ids)):
            if original_id != ablated_id:  # Token was ablated
                if i < ablated_hidden_states.shape[1]:  # Ensure we don't go out of bounds
                    ablated_hidden_states[:, i, :] = pad_embedding
        
        # Store the ablated results
        ablated_output["tokens"][condition] = ablated_tokens
        ablated_output["hidden_states"][condition] = ablated_hidden_states
    
    return ablated_output


def eval_global(
    args: argparse.Namespace,
    dataset: QIDDataset,
    tokenizer: PreTrainedTokenizer,
    llava_model: LlavaLlamaForCausalLM,
    probe: FourSpaceProjector,
    device: torch.device,
    logger=None,
    n_samples: int=100,
):
    """Evaluate the trained projector on all dataset entries with and without ablation."""
    # Initialize data structures for original, ablated, and random ablated results
    original_aggregated: Dict[str, List[torch.Tensor]] = {
        "zsl_rand": [],
        "zsl_sem": [],
        "icl_rand": [],
        "icl_sem": [],
    }
    ablated_aggregated: Dict[str, List[torch.Tensor]] = {
        "zsl_rand": [],
        "zsl_sem": [],
        "icl_rand": [],
        "icl_sem": [],
    }
    random_ablated_aggregated: Dict[str, List[torch.Tensor]] = {
        "zsl_rand": [],
        "zsl_sem": [],
        "icl_rand": [],
        "icl_sem": [],
    }
    
    # sample dataset
    if n_samples is not None:
        indices = random.sample(range(len(dataset)), n_samples)
        dataset = [dataset[i] for i in indices]
    
    original_token_records: Dict[str, List[Dict[str, str]]] = {
        "zsl_rand": [],
        "zsl_sem": [],
        "icl_rand": [],
        "icl_sem": [],
    }
    ablated_token_records: Dict[str, List[Dict[str, str]]] = {
        "zsl_rand": [],
        "zsl_sem": [],
        "icl_rand": [],
        "icl_sem": [],
    }
    random_ablated_token_records: Dict[str, List[Dict[str, str]]] = {
        "zsl_rand": [],
        "zsl_sem": [],
        "icl_rand": [],
        "icl_sem": [],
    }
    
    # Store LLaVA outputs (before projection) for wordcloud generation
    llava_outputs: Dict[str, List[Dict[str, str]]] = {
        "zsl_rand": [],
        "zsl_sem": [],
        "icl_rand": [],
        "icl_sem": [],
    }
    
    # Check if ablation should be performed
    should_ablate = hasattr(args, 'words_to_ablate') and args.words_to_ablate
    should_random_ablate = getattr(args, 'random_ablation', False)
    words_to_ablate_list = []
    random_words_to_ablate_list = []
    
    if should_ablate:
        words_to_ablate_list = [word.strip() for word in args.words_to_ablate.split(',')]
        if logger:
            logger.info(f"Performing ablation with words: {words_to_ablate_list}")
    
    if should_random_ablate and should_ablate:
        # Extract vocabulary and select random words
        vocabulary = extract_vocabulary_from_dataset(dataset, tokenizer, logger)
        random_words_to_ablate_list = select_random_words(
            vocabulary, 
            len(words_to_ablate_list), 
            exclude_words=words_to_ablate_list,
            logger=logger
        )
        if logger:
            logger.info(f"Performing random ablation with words: {random_words_to_ablate_list}")
    
    for idx in range(len(dataset)):
        if logger and (idx + 1) % 10 == 0:
            logger.info(f"Processing sample {idx + 1}/{len(dataset)}")
        item = dataset[idx]
        qid = dataset.qids[idx] if hasattr(dataset, "qids") else str(idx)
        inputs = [
            item["zsl_rand"],
            item["zsl_sem"],
            item["icl_rand"],
            item["icl_sem"],
        ]
        
        # Process original inputs (no ablation)
        hidden = forward_hidden_states_batch(inputs, llava_model, tokenizer, device)
        outputs = [
            {"hidden_states": [hidden[i : i + 1]]} for i in range(len(inputs))
        ]
        proj_output = project_embedding(
            llava_model,
            tokenizer,
            probe,
            outputs,
            logger=logger,
        )
        
        # Store LLaVA outputs (before projection) for wordcloud generation
        # Get the tokens from the LLaVA model output before projection
        llava_tokens = unembed_hidden_states(hidden, llava_model, tokenizer, unembed_last=False)
        
        # Convert to string if it's a list
        if isinstance(llava_tokens, list):
            llava_text = " ".join(llava_tokens)
        else:
            llava_text = llava_tokens
        
        # Store original results
        for key in original_aggregated:
            original_aggregated[key].append(
                proj_output["hidden_states"][key].detach().cpu()
            )
            original_token_records[key].append({"id": qid, "text": proj_output["tokens"][key]})
            
            # Store LLaVA outputs (same for all conditions since it's before projection)
            llava_outputs[key].append({"id": qid, "text": llava_text})
        
        # Process ablated inputs if ablation is specified
        if should_ablate:
            # Ablate the input tokens with specified words
            ablated_inputs = ablate_input_tokens(inputs, tokenizer, words_to_ablate_list, logger=logger, debug=args.debug)
            
            # Get hidden states from ablated inputs
            ablated_hidden = forward_hidden_states_batch(ablated_inputs, llava_model, tokenizer, device)
            ablated_outputs = [
                {"hidden_states": [ablated_hidden[i : i + 1]]} for i in range(len(ablated_inputs))
            ]
            ablated_proj_output = project_embedding(
                llava_model,
                tokenizer,
                probe,
                ablated_outputs,
                logger=logger,
            )
            
            # Store ablated results
            for key in ablated_aggregated:
                ablated_aggregated[key].append(
                    ablated_proj_output["hidden_states"][key].detach().cpu()
                )
                ablated_token_records[key].append({"id": qid, "text": ablated_proj_output["tokens"][key]})
            
            del ablated_hidden, ablated_outputs, ablated_proj_output
            
            # Process random ablated inputs if random ablation is specified
            if should_random_ablate:
                # Ablate the input tokens with random words
                random_ablated_inputs = ablate_input_tokens(inputs, tokenizer, random_words_to_ablate_list, logger=logger, debug=args.debug)
                
                # Get hidden states from random ablated inputs
                random_ablated_hidden = forward_hidden_states_batch(random_ablated_inputs, llava_model, tokenizer, device)
                random_ablated_outputs = [
                    {"hidden_states": [random_ablated_hidden[i : i + 1]]} for i in range(len(random_ablated_inputs))
                ]
                random_ablated_proj_output = project_embedding(
                    llava_model,
                    tokenizer,
                    probe,
                    random_ablated_outputs,
                    logger=logger,
                )
                
                # Store random ablated results
                for key in random_ablated_aggregated:
                    random_ablated_aggregated[key].append(
                        random_ablated_proj_output["hidden_states"][key].detach().cpu()
                    )
                    random_ablated_token_records[key].append({"id": qid, "text": random_ablated_proj_output["tokens"][key]})
                
                del random_ablated_hidden, random_ablated_outputs, random_ablated_proj_output
        else:
            # If no ablation, copy original results to ablated
            for key in ablated_aggregated:
                ablated_aggregated[key].append(
                    proj_output["hidden_states"][key].detach().cpu()
                )
                ablated_token_records[key].append({"id": qid, "text": proj_output["tokens"][key]})
            
            # Also copy to random ablated if no ablation
            for key in random_ablated_aggregated:
                random_ablated_aggregated[key].append(
                    proj_output["hidden_states"][key].detach().cpu()
                )
                random_ablated_token_records[key].append({"id": qid, "text": proj_output["tokens"][key]})
        
        del hidden, outputs, proj_output
        free_memory()

    # Process original results
    original_combined = {}
    for k, tensors in original_aggregated.items():
        flattened = [t.reshape(-1, t.shape[-1]) for t in tensors]
        original_combined[k] = torch.cat(flattened, dim=0)
    
    # Process ablated results
    ablated_combined = {}
    for k, tensors in ablated_aggregated.items():
        flattened = [t.reshape(-1, t.shape[-1]) for t in tensors]
        ablated_combined[k] = torch.cat(flattened, dim=0)
    
    # Process random ablated results
    random_ablated_combined = {}
    for k, tensors in random_ablated_aggregated.items():
        flattened = [t.reshape(-1, t.shape[-1]) for t in tensors]
        random_ablated_combined[k] = torch.cat(flattened, dim=0)
    
    # Save results to separate directories
    base_save_dir = os.path.join(args.exp_dir, "global")
    
    # Save original results
    original_save_dir = os.path.join(base_save_dir, "original")
    analyze_probe(original_combined, original_save_dir, logger)
    characterize_by_word(original_token_records, original_save_dir, logger=logger)
    
    # Save ablated results
    ablated_save_dir = os.path.join(base_save_dir, "ablated")
    analyze_probe(ablated_combined, ablated_save_dir, logger)
    characterize_by_word(ablated_token_records, ablated_save_dir, logger=logger)
    
    # Save random ablated results if performed
    if should_random_ablate:
        random_ablated_save_dir = os.path.join(base_save_dir, "random_ablated")
        analyze_probe(random_ablated_combined, random_ablated_save_dir, logger)
        characterize_by_word(random_ablated_token_records, random_ablated_save_dir, logger=logger)
        if logger:
            logger.info(f"Random ablated analysis saved to: {random_ablated_save_dir}")
    
    # Load visualization settings
    settings = load_visualization_settings()
    
    # Create overlay plots if requested
    if getattr(args, 'overlay_ablation', False) and should_ablate:
        overlay_save_dir = os.path.join(base_save_dir, "overlay")
        if should_random_ablate:
            # Create three-way comparison overlay
            analyze_probe_overlay_three_way(original_combined, ablated_combined, random_ablated_combined, overlay_save_dir, logger, settings)
            
            # Create direct comparison between ablated and random ablated
            direct_comparison_save_dir = os.path.join(base_save_dir, "direct_comparison")
            analyze_probe_overlay_direct_comparison(original_combined, ablated_combined, random_ablated_combined, direct_comparison_save_dir, logger, settings)
            if logger:
                logger.info(f"Direct comparison analysis saved to: {direct_comparison_save_dir}")
        else:
            # Create two-way comparison overlay
            analyze_probe_overlay(original_combined, ablated_combined, overlay_save_dir, logger, settings)
        if logger:
            logger.info(f"Overlay analysis saved to: {overlay_save_dir}")
    
    # Create LLaVA wordclouds (before projection)
    llava_wordcloud_save_dir = os.path.join(base_save_dir, "llava_wordclouds")
    create_llava_wordclouds(llava_outputs, llava_wordcloud_save_dir, settings, logger)
    if logger:
        logger.info(f"LLaVA wordclouds saved to: {llava_wordcloud_save_dir}")
    
    if logger:
        logger.info(f"Original analysis saved to: {original_save_dir}")
        logger.info(f"Ablated analysis saved to: {ablated_save_dir}")


def eval(
    args: argparse.Namespace,
    dataset: QIDDataset,
    tokenizer: PreTrainedTokenizer,
    llava_model: LlavaLlamaForCausalLM,
    probe: FourSpaceProjector,
    device: torch.device,
    logger=None,
):
    """Run both local and global evaluations."""
    eval_local(
        args,
        dataset,
        tokenizer,
        llava_model,
        probe,
        device,
        logger,
    )
    logger.info("Local evaluation completed.")
    eval_global(
        args,
        dataset,
        tokenizer,
        llava_model,
        probe,
        device,
        logger,
    )
    logger.info("Global evaluation completed.")


def fix_seeds(seed: int):
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # If using distributed training, fix the seed for all processes
    if torch.distributed.is_initialized():
        torch.distributed.broadcast(torch.tensor(seed), src=0)


def main():
    parser = argparse.ArgumentParser(
        description="Run representation learning experiment",
    )
    parser.add_argument(
        "--exp-dir",
        required=True,
        help="Path to experiment directory for trained model",
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Path to directory containing mixed and random models",
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Path to directory containing embeddings (.pt.gz)",
    )
    parser.add_argument(
        "--tags", required=True, help="Comma-separated tags for models (ts1,ts2,ts3)"
    )
    parser.add_argument(
        "--model2",
        required=True,
        help="Second model identifier used in trained model name",
    )
    parser.add_argument(
        "--prefix", default="multiple_inputs_", help="Model prefix used in file names"
    )
    parser.add_argument(
        "--ood-types",
        default="0_id,1_ood",
        help="Comma-separated list of OOD types",
    )
    parser.add_argument(
        "--tasks",
        default="gqa,mm-vet,mmbench,textvqa,vizwiz,vqav2",
        help="Comma-separated list of tasks for fixed effects",
    )
    parser.add_argument(
        "--sample-index", type=int, default=10, help="Index of embedding pair to sample"
    )
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument(
        "--image-noise-pattern",
        type=str,
        default="black",
        choices=["black", "white", "random"],
    )
    parser.add_argument(
        "--probe-type",
        type=str,
        default="mixed",
        choices=["mixed", "random", "proj"],
        help="Type of probe to use for analysis",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for training",
    )
    parser.add_argument(
        "--num-epoch", type=int, default=1, help="Number of training epochs",
    )
    parser.add_argument(
        "--forward-batch-size",
        type=int,
        default=8,
        help="Batch size for hidden state extraction",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="If set, log training to a Weights & Biases run with this name",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="If set, enable debug mode",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run evaluation without training by loading a pre-trained probe",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for initialization",
    )
    parser.add_argument(
        "--words-to-ablate",
        type=str,
        default="ASSISTANT:,USER:,<image>",
        help="Comma-separated list of words to ablate (replace with [PAD] token)",
    )
    parser.add_argument(
        "--overlay-ablation",
        action="store_true",
        help="Create overlay plots comparing original vs ablated distributions with statistical tests",
    )
    parser.add_argument(
        "--random-ablation",
        action="store_true",
        help="Also perform ablation with randomly selected words (same count as specified words) for comparison",
    )
    args = parser.parse_args()

    logger = get_module_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fix_seeds(args.seed)

    task = "textvqa"
    qids, qid_texts, iid_texts, qid_images, iid_images, answers = load_dataset_entries(
        dataset=task,
        data_dir=args.data_dir.split(f"/{task}")[0],
        llava_dir=args.data_dir.split("/playground")[0],
        prefix=args.prefix,
        target_qids=None,
        sample_n=None,
    )
    logger.info(f"Loaded {len(qids)} questions")

    tokenizer, llava_model, image_processor, _ = build_llava_model(args, device)
    if device.type == "cpu":
        llava_model = llava_model.to(torch.float32)
    llava_model.eval()
    logger.info("Model loaded and ready for evaluation")

    dataset = QIDDataset(
        qids,
        qid_texts,
        iid_texts,
        qid_images,
        iid_images,
        answers,
        args,
        tokenizer,
        image_processor,
        llava_model.config,
    )

    if args.eval_only:
        logger.info("Evaluation-only mode: loading trained probe from disk")
        config = CustomModelConfig()
        probe = FourSpaceProjector(
            hidden_size=config.embed_dim,
            bias=True,
            gate="scalar",
            init="xavier_uniform",
        ).to(device)
        probe_path = os.path.join(args.exp_dir, "four_space_projector.pt")
        probe.load_state_dict(torch.load(probe_path, map_location=device))
    else:
        probe = train(
            args,
            dataset,
            tokenizer,
            llava_model,
            device,
            logger,
        )
        logger.info("Training complete")

    eval(
        args,
        dataset,
        tokenizer,
        llava_model,
        probe,
        device,
        logger,
    )
    logger.info("Evaluation complete")


if __name__ == "__main__":
    main()

