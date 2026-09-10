import argparse
import gzip
import json
import logging
import os
import pickle
import random
import sys

import numpy as np
from PIL import Image
import shortuuid
import psutil
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor, to_pil_image
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_PLACEHOLDER
from llava.conversation import conv_templates, SeparatorStyle
from llava.eval.representation_learning import learn_repr
from llava.eval.utils import add_image_token, append_message, explore_shape, get_chunk, makedirs_recursive, parse_filenames, split_list
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

def get_logger():
    logger = logging.getLogger("mcicl.model_vqa_loader")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

LOGGER = get_logger()

def log_cuda_memory(tag):
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
    max_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
    LOGGER.info(
        f"[GPU] {tag}: allocated={allocated:.1f}MB "
        f"reserved={reserved:.1f}MB max_allocated={max_allocated:.1f}MB "
        f"max_reserved={max_reserved:.1f}MB"
    )

def log_system_memory(tag):
    process = psutil.Process()
    rss = process.memory_info().rss / (1024 ** 2)
    vms = process.memory_info().vms / (1024 ** 2)
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
        max_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
        LOGGER.info(
            f"[MEM] {tag}: rss={rss:.1f}MB vms={vms:.1f}MB "
            f"gpu_allocated={allocated:.1f}MB gpu_reserved={reserved:.1f}MB "
            f"gpu_max_allocated={max_allocated:.1f}MB gpu_max_reserved={max_reserved:.1f}MB"
        )
    else:
        LOGGER.info(f"[MEM] {tag}: rss={rss:.1f}MB vms={vms:.1f}MB gpu=unavailable")

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions1, questions2, image_folder, tokenizer, image_processor, model_config):
        self.questions1 = questions1
        self.questions2 = questions2
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def load_image(self, image_file):
        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        return image
    
    def load_images(self, image_files):
        out = []
        if type(image_files) is not list:
            image_list = [image_files]
        else:
            image_list = image_files
        for image_file in image_list:
            image = self.load_image(image_file)
            out.append(image)
        return out

    def get_objects(self, questions, index, is_multiple_questions=False):
        line = questions[index]
        qs, first_answer, image_file = add_image_token(args, line, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, is_multiple_questions, IMAGE_PLACEHOLDER, self.model_config)

        conv = conv_templates[args.conv_mode].copy()
        conv = append_message(conv, qs, is_multiple_questions, first_answer)
        prompt = conv.get_prompt()

        image = self.load_images(image_file)
        image_tensor = process_images(image, self.image_processor, self.model_config)#[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor

    def __getitem__(self, index):
        input_ids1, image_tensor1 = self.get_objects(self.questions1, index, is_multiple_questions=False)
        input_ids2, image_tensor2 = self.get_objects(self.questions2, index, is_multiple_questions=True)

        return input_ids1, image_tensor1, input_ids2, image_tensor2

    def __len__(self):
        return len(self.questions1)


# DataLoader
def create_data_loader(questions1, questions2, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4, pin_memory=False, prefetch_factor=None, persistent_workers=False):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions1, questions2, image_folder, tokenizer, image_processor, model_config)
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "shuffle": False,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor
    data_loader = DataLoader(dataset, **loader_kwargs)
    return data_loader


def get_chunk_from_file(question_file, args):
    questions = [json.loads(q) for q in open(os.path.expanduser(question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    return questions


def base_outputs_exist(extract_path, index, question_id1, question_id2):
    """Check if base (generation + repr) outputs exist."""
    expected_files = [
        f"reprs1_{index}_{question_id1}.pt.gz",
        f"reprs2_{index}_{question_id2}.pt.gz",
        f"output_texts1_{index}_{question_id1}.pkl",
        f"output_texts2_{index}_{question_id2}.pkl",
    ]
    return all(os.path.exists(os.path.join(extract_path, filename)) for filename in expected_files)


def control_outputs_exist(extract_path, index, question_id1, question_id2):
    """Check if FourSpaceProjector control condition outputs exist."""
    expected_files = [
        f"four_proj_zsl_sem_{index}_{question_id1}.pt.gz",
        f"four_proj_zsl_rand_{index}_{question_id1}.pt.gz",
        f"four_proj_icl_sem_{index}_{question_id2}.pt.gz",
        f"four_proj_icl_rand_{index}_{question_id2}.pt.gz",
    ]
    return all(os.path.exists(os.path.join(extract_path, filename)) for filename in expected_files)


def mean_pool_padded_repr(reprs_pad):
    """Mean-pool a zero-padded representation tensor, ignoring padding.

    Parameters
    ----------
    reprs_pad : torch.Tensor
        Padded hidden states of shape ``[1, padding_num, hidden_dim]``.

    Returns
    -------
    torch.Tensor
        Mean-pooled vector of shape ``[hidden_dim]``.
    """
    mask = (reprs_pad.abs().sum(dim=-1) > 0)  # [1, padding_num]
    seq_len = mask.sum(dim=1, keepdim=True).clamp(min=1)  # [1, 1]
    masked = reprs_pad * mask.unsqueeze(-1).float()
    return (masked.sum(dim=1) / seq_len.float()).squeeze(0).to(torch.float32)


def extract_control_hidden_state(
    line, args, model, tokenizer, image_processor, model_config,
    device, model_dtype, image_folder, is_multiple,
    shuffle_text=False, noise_pattern=None,
):
    """Run a non-generation forward pass and return mean-pooled last-layer hidden state.

    Parameters
    ----------
    line : dict
        Question data dict (``question_id``, ``text``, ``image``, ``answer``).
    shuffle_text : bool
        If True, shuffle word order in the text.  For multi-input (ICL) lines
        only the ICL example part (before ``__sep__``) is shuffled.
    noise_pattern : str or None
        If set, replace the ICL example image with noise
        (``'black'``, ``'white'``, or ``'random'``).
    """
    sep = args.sep

    # --- Optionally shuffle text ---
    if shuffle_text:
        text = line["text"]
        answer = line.get("answer", "")
        if is_multiple and sep in text:
            parts = text.split(sep, 1)
            parts[0] = " ".join(np.random.permutation(parts[0].split()))
            text = sep.join(parts)
            if sep in answer:
                ans_parts = answer.split(sep, 1)
                if ans_parts[0]:
                    ans_parts[0] = " ".join(np.random.permutation(ans_parts[0].split()))
                answer = sep.join(ans_parts)
        else:
            text = " ".join(np.random.permutation(text.split()))
        line = {**line, "text": text, "answer": answer}

    # --- Build prompt (mirrors CustomDataset.get_objects) ---
    qs, first_answer, image_file = add_image_token(
        args, line,
        DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN,
        is_multiple, IMAGE_PLACEHOLDER, model_config,
    )
    conv = conv_templates[args.conv_mode].copy()
    conv = append_message(conv, qs, is_multiple, first_answer)
    prompt = conv.get_prompt()

    # --- Load images ---
    if isinstance(image_file, list):
        images = [Image.open(os.path.join(image_folder, f)).convert("RGB") for f in image_file]
    else:
        images = [Image.open(os.path.join(image_folder, image_file)).convert("RGB")]

    # --- Replace ICL example image with noise if requested ---
    if noise_pattern is not None and is_multiple and len(images) > 1:
        org_tensor = to_tensor(images[0])  # [C, H, W], values in [0, 1]
        if noise_pattern == "black":
            noise_tensor = torch.zeros_like(org_tensor)
        elif noise_pattern == "white":
            noise_tensor = torch.ones_like(org_tensor)
        else:  # random
            noise_tensor = torch.rand_like(org_tensor)
        images[0] = to_pil_image(noise_tensor)

    # --- Process images and tokenize ---
    image_tensor = process_images(images, image_processor, model_config)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")

    # --- Forward pass (no generation) ---
    input_ids_dev = input_ids.unsqueeze(0).to(device) if input_ids.dim() == 1 else input_ids.to(device)
    # No squeeze: process_images already returns [num_images, C, H, W] without
    # a DataLoader batch dim, matching what generate_ids_reprs passes after
    # removing the DataLoader dim with squeeze(0).
    image_tensor_dev = image_tensor.to(dtype=model_dtype, device=device, non_blocking=True)

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids_dev,
            images=image_tensor_dev,
            output_hidden_states=True,
        )

    hidden = outputs.hidden_states[-1]  # [1, seq_len, hidden_dim]
    mean_pooled = hidden.mean(dim=1).squeeze(0).to(torch.float32).cpu()

    del outputs, hidden, input_ids_dev, image_tensor_dev, image_tensor, input_ids, images
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return mean_pooled


def generate_ids_reprs(image_tensor, input_ids, line, args, model, tokenizer, device, model_dtype, model_name):
    image_tensor = image_tensor.squeeze(0)
    idx = line["question_id"]
    cur_prompt = line["text"]

    stop_str = conv_templates[args.conv_mode].sep if conv_templates[args.conv_mode].sep_style != SeparatorStyle.TWO else conv_templates[args.conv_mode].sep2
    input_ids = input_ids.to(device=device, non_blocking=True)

    with torch.inference_mode():
        outputs = model.generate(
            input_ids,
            images=image_tensor.to(dtype=model_dtype, device=device, non_blocking=True),
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            output_attentions=args.output_attentions,
            output_hidden_states=args.output_hidden_states,
            return_dict_in_generate=True,
            use_cache=True)

    output_ids = outputs.sequences
    
    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        LOGGER.warning(f"{n_diff_input_output} output_ids are not the same as the input_ids")
    output_sentences = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    output_sentences = output_sentences.strip()
    if output_sentences.endswith(stop_str):
        output_sentences = output_sentences[:-len(stop_str)]
    output_sentences = output_sentences.strip()

    ans_id = shortuuid.uuid()

    ans_dict = {
        "question_id": idx,
        "prompt": cur_prompt,
        "text": output_sentences,
        "answer_id": ans_id,
        "model_id": model_name,
        "metadata": {}
    }

    reprs = outputs.hidden_states[0][-1].to('cpu').detach()
    reprs_pad = F.pad(reprs, (0, 0, 0, args.padding_num-reprs.shape[1]), "constant", 0)

    del outputs
    del output_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return ans_dict, reprs_pad


def eval_model(args):
    # Model
    makedirs_recursive(args.extract_path)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        model_dtype = torch.float32
    else:
        if args.model_dtype == "fp32":
            model_dtype = torch.float32
        elif args.model_dtype == "bf16":
            model_dtype = torch.bfloat16
        else:
            model_dtype = torch.float16
    disable_torch_init()
    if args.tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:
            pass
    question_file1, question_file2 = parse_filenames(args.question_file)
    answers_file1, answers_file2 = parse_filenames(args.answers_file)
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, device=device)
    model = model.to(device=device, dtype=model_dtype)
    log_system_memory("after_model_load")
    if args.torch_compile:
        try:
            model = torch.compile(model)
        except Exception as exc:
            LOGGER.warning(f"torch.compile failed, continuing without compile: {exc}")

    questions1 = get_chunk_from_file(question_file1, args)
    questions2 = get_chunk_from_file(question_file2, args)
    
    answers_file1 = os.path.expanduser(answers_file1)
    answers_file2 = os.path.expanduser(answers_file2)
    os.makedirs(os.path.dirname(answers_file1), exist_ok=True)
    os.makedirs(os.path.dirname(answers_file2), exist_ok=True)
    answers_mode = "a" if args.skip_existing else "w"
    ans_file1 = open(answers_file1, answers_mode)
    ans_file2 = open(answers_file2, answers_mode)

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        LOGGER.info(f"It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.")

    data_loader = create_data_loader(
        questions1,
        questions2,
        args.image_folder,
        tokenizer,
        image_processor,
        model.config,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
    )

    assert len(questions1) == len(questions2), "questions1 and questions2 must have the same length"
    # cnt = 0
    keep_repr_in_memory = False
    reprs1 = [] if keep_repr_in_memory else None
    reprs2 = [] if keep_repr_in_memory else None
    output_texts1 = [] if keep_repr_in_memory else None
    output_texts2 = [] if keep_repr_in_memory else None
    if args.do_repr_sample:
        selected_iterations = random.sample(range(len(questions1)), args.repr_sample_num)
    else:
        # all iteration
        selected_iterations = range(len(questions1))
    skipped_existing = False
    log_every = args.log_progress_every if args.log_progress_every and args.log_progress_every > 0 else None
    do_ctrl = getattr(args, "extract_control_conditions", False)
    ctrl_noise = getattr(args, "image_noise_pattern", "black")
    skip_reprs = getattr(args, "skip_reprs", False)
    if skip_reprs:
        LOGGER.info("--skip-reprs enabled: hidden state saving disabled (generation-only mode)")
    for i, ((input_ids1, image_tensor1, input_ids2, image_tensor2), line1, line2) in enumerate(tqdm(zip(data_loader, questions1, questions2), total=len(questions1))):
        question_id1 = line1["question_id"]
        question_id2 = line2["question_id"]
        should_extract = not args.do_repr_sample or i in selected_iterations

        # --- Two-tier skip logic ---
        base_exist = False
        ctrl_exist = not do_ctrl  # True (= nothing to do) when control extraction disabled
        if should_extract and args.skip_existing and not skip_reprs:
            base_exist = base_outputs_exist(args.extract_path, i, question_id1, question_id2)
            if do_ctrl:
                ctrl_exist = control_outputs_exist(args.extract_path, i, question_id1, question_id2)
            if base_exist and ctrl_exist:
                skipped_existing = True
                continue

        # --- Base generation (skip if base already exists) ---
        did_base_gen = not (should_extract and args.skip_existing and base_exist)
        repr1_for_ctrl = None
        repr2_for_ctrl = None

        if did_base_gen:
            ans_dict1, repr1 = generate_ids_reprs(image_tensor1, input_ids1, line1, args, model, tokenizer, device, model_dtype, model_name)
            ans_dict2, repr2 = generate_ids_reprs(image_tensor2, input_ids2, line2, args, model, tokenizer, device, model_dtype, model_name)
            if should_extract and not skip_reprs:
                with gzip.open(os.path.join(args.extract_path, f"reprs1_{i}_{ans_dict1['question_id']}.pt.gz"), "wb") as f:
                    torch.save(repr1, f)
                with gzip.open(os.path.join(args.extract_path, f"reprs2_{i}_{ans_dict2['question_id']}.pt.gz"), "wb") as f:
                    torch.save(repr2, f)
                with open(os.path.join(args.extract_path, f"output_texts1_{i}_{ans_dict1['question_id']}.pkl"), "wb") as f:
                    pickle.dump(ans_dict1["text"], f)
                with open(os.path.join(args.extract_path, f"output_texts2_{i}_{ans_dict2['question_id']}.pkl"), "wb") as f:
                    pickle.dump(ans_dict2["text"], f)
                if keep_repr_in_memory:
                    reprs1.append(repr1)
                    reprs2.append(repr2)
                    output_texts1.append(ans_dict1["text"])
                    output_texts2.append(ans_dict2["text"])
                repr1_for_ctrl = repr1
                repr2_for_ctrl = repr2

            ans_file1.write(json.dumps(ans_dict1) + "\n")
            ans_file2.write(json.dumps(ans_dict2) + "\n")
            ans_file1.flush()
            ans_file2.flush()

        # --- Control condition extraction (FourSpaceProjector) ---
        if do_ctrl and should_extract and not ctrl_exist:
            # zsl_sem / icl_sem: mean-pool from base reprs
            if repr1_for_ctrl is not None:
                zsl_sem = mean_pool_padded_repr(repr1_for_ctrl)
                icl_sem = mean_pool_padded_repr(repr2_for_ctrl)
            else:
                # Load base reprs from disk (re-run with USE_FOUR_PROJ=1)
                with gzip.open(os.path.join(args.extract_path, f"reprs1_{i}_{question_id1}.pt.gz"), "rb") as f:
                    zsl_sem = mean_pool_padded_repr(torch.load(f, map_location="cpu"))
                with gzip.open(os.path.join(args.extract_path, f"reprs2_{i}_{question_id2}.pt.gz"), "rb") as f:
                    icl_sem = mean_pool_padded_repr(torch.load(f, map_location="cpu"))

            # zsl_rand: shuffled question text, same image
            zsl_rand = extract_control_hidden_state(
                line1, args, model, tokenizer, image_processor, model.config,
                device, model_dtype, args.image_folder, is_multiple=False,
                shuffle_text=True,
            )
            # icl_rand: shuffled ICL text + noise image
            icl_rand = extract_control_hidden_state(
                line2, args, model, tokenizer, image_processor, model.config,
                device, model_dtype, args.image_folder, is_multiple=True,
                shuffle_text=True, noise_pattern=ctrl_noise,
            )

            # Save all 4 conditions
            for name, tensor, qid in [
                ("zsl_sem", zsl_sem, question_id1),
                ("zsl_rand", zsl_rand, question_id1),
                ("icl_sem", icl_sem, question_id2),
                ("icl_rand", icl_rand, question_id2),
            ]:
                with gzip.open(os.path.join(args.extract_path, f"four_proj_{name}_{i}_{qid}.pt.gz"), "wb") as f:
                    torch.save(tensor, f)

            del zsl_sem, zsl_rand, icl_sem, icl_rand
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # --- Cleanup ---
        repr1_for_ctrl = None
        repr2_for_ctrl = None
        if did_base_gen:
            del repr1, repr2, ans_dict1, ans_dict2
        del input_ids1, input_ids2, image_tensor1, image_tensor2

        if log_every and (i + 1) % log_every == 0:
            log_system_memory(f"after_step_{i + 1}")
            torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()
    if not keep_repr_in_memory or args.load_repr_sample or skipped_existing:
        reprs1 = reprs2 = output_texts1 = output_texts2 = args.extract_path
    log_system_memory("before_learn_repr")
    if args.skip_repr_learning:
        LOGGER.info("Skipping representation learning and outputs (SKIP_REPR_LEARNING=1).")
        repr_model = None
    else:
        repr_model = learn_repr(args, reprs1, reprs2, output_texts1, output_texts2)
    log_system_memory("after_learn_repr")
    ans_file1.close()
    ans_file2.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--sep", type=str, default="__sep__")
    parser.add_argument("--train-path", type=str, default="path/to/train/data")
    parser.add_argument("--output-attentions", action="store_true")
    parser.add_argument("--output-hidden-states", action="store_true")
    parser.add_argument("--export-ids", action="store_true")
    parser.add_argument("--extract-path", type=str, default="path/to/extraction/dir")
    parser.add_argument("--root-path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1987)
    parser.add_argument("--padding_num", type=int, default=4096)
    parser.add_argument(
        "--contrastive-model",
        type=str,
        default="mixed",
        choices=["attention", "linear", "mixed"],
    )
    parser.add_argument("--do-sample", dest="do_sample", action="store_true", default=False)
    parser.add_argument("--sample-size", dest="sample_size", type=int, default=1000)
    parser.add_argument("--log-progress-every", dest="log_progress_every", type=int, default=100)
    parser.add_argument("--do-repr-sample", action="store_true")
    parser.add_argument("--repr-sample-num", type=int, default=4000)
    parser.add_argument("--load-repr-sample", action="store_true")
    parser.add_argument("--repr-text-device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--text-model-dtype", choices=["fp16", "bf16", "fp32"], default=None)
    parser.add_argument("--cont-model-dtype", choices=["fp16", "bf16", "fp32"], default=None)
    parser.add_argument("--skip-repr-learning", action="store_true")
    parser.add_argument("--repr-batch-size", type=int, default=32)
    parser.add_argument("--repr-num-epochs", type=int, default=500)
    parser.add_argument("--repr-val-epoch", type=int, default=10)
    parser.add_argument("--task-name", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--skip-existing", dest="skip_existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--model-dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--torch-compile", action="store_true")
    # FourSpaceProjector control condition extraction
    parser.add_argument(
        "--extract-control-conditions",
        action="store_true",
        help="Extract 4-condition hidden states for FourSpaceProjector training",
    )
    parser.add_argument(
        "--image-noise-pattern",
        type=str,
        default="black",
        choices=["black", "white", "random"],
        help="Noise pattern for ICL image replacement in control conditions",
    )
    parser.add_argument(
        "--skip-reprs",
        action="store_true",
        help="Skip saving hidden state representations (reprs1/reprs2). "
             "Answers are still generated. Used for multi-shot/ordering "
             "experiments where only evaluation accuracy is needed.",
    )
    args = parser.parse_args()

    eval_model(args)
