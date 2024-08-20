import argparse
import gzip
import json
import math
import os
import pickle
import random

from PIL import Image
import shortuuid
import torch
import torch.nn.functional as F
from tqdm import tqdm

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.eval.representation_learning import learn_repr
from llava.eval.utils import add_image_token, append_message, explore_shape, get_chunk, makedirs_recursive, parse_filenames, split_list
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)


def load_image(args, image_file, convert_rgb=False):
    if convert_rgb:
        image = Image.open(os.path.join(args.train_path, image_file)).convert("RGB")
    else:
        image = Image.open(os.path.join(args.image_folder, image_file))
    return image


def load_images(args, image_files):
    out = []
    if type(image_files) is not list:
        image_list = [image_files]
        convert_rgbs = [False]
    else:
        image_list = image_files
        convert_rgbs = [True, False]
    for image_file, convert_rgb in zip(image_list, convert_rgbs):
        image = load_image(args, image_file, convert_rgb)
        out.append(image)
    return out


def get_questions(args, question_file):
    questions = [
        json.loads(q) for q in open(os.path.expanduser(question_file), "r")
    ]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    return questions


def get_ans_file(args, answers_file):
    answers_file = os.path.expanduser(answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    return ans_file


def generate_ids_reprs(args, model, tokenizer, image_processor, line, model_name, device, model_dtype, is_multiple_questions=False):
    idx = line["question_id"]
    cur_prompt = line["text"]
    qs, first_answer, image_file = add_image_token(
        args,
        line,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_END_TOKEN,
        is_multiple_questions,
        IMAGE_PLACEHOLDER,
        model.config,
    )

    conv = conv_templates[args.conv_mode].copy()
    conv = append_message(conv, qs, is_multiple_questions, first_answer)
    prompt = conv.get_prompt()

    input_ids = (
        tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        .unsqueeze(0)
        .to(device=device)
    )

    # image_file = line["image"]
    image = load_images(args, image_file)
    image_tensor = image_processor.preprocess(image, return_tensors="pt")[
        "pixel_values"
    ]

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        outputs = model.generate(
            input_ids,
            images=image_tensor.to(device=device, dtype=model_dtype),
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            # no_repeat_ngram_size=3,
            max_new_tokens=1024,
            use_cache=True,
            output_attentions=args.output_attentions,
            output_hidden_states=args.output_hidden_states,
            return_dict_in_generate=True,
        )
        output_ids = outputs["sequences"]

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (
        (input_ids != output_ids[:, :input_token_len]).sum().item()
    )
    if n_diff_input_output > 0:
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
        "question_id": idx,
        "prompt": cur_prompt,
        "text": output_sentences,
        "answer_id": ans_id,
        "model_id": model_name,
        "metadata": {},
    }

    reprs = outputs.hidden_states[0][-1].to('cpu').detach()
    reprs_pad = F.pad(reprs, (0, 0, 0, args.padding_num-reprs.shape[1]), "constant", 0)

    return ans_dict, reprs_pad


def eval_model(args):
    # Model
    disable_torch_init()
    makedirs_recursive(args.extract_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = torch.float32 if not torch.cuda.is_available() else torch.float16
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name, device=device
    )
    model = model.to(device=device, dtype=model_dtype)

    question_file1, question_file2 = parse_filenames(args.question_file)
    answers_file1, answers_file2 = parse_filenames(args.answers_file)
    
    questions1 = get_questions(args, question_file1)
    questions2 = get_questions(args, question_file2)
    
    ans_file1 = get_ans_file(args, answers_file1)
    ans_file2 = get_ans_file(args, answers_file2)
    assert len(questions1) == len(questions2), "The number of questions must be the same"
    # cnt = 0
    reprs1, reprs2 = [], []
    output_texts1, output_texts2 = [], []
    question_ids1, question_ids2 = [], []
    if args.do_repr_sample:
        selected_iterations = random.sample(range(len(questions1)), args.repr_sample_num)
    else:
        # all iteration
        selected_iterations = range(len(questions1))
    for i, (line1, line2) in enumerate(tqdm(zip(questions1, questions2))):
        ans_dict1, repr1 = generate_ids_reprs(args, model, tokenizer, image_processor, line1, model_name, device, model_dtype, is_multiple_questions=False)
        ans_dict2, repr2 = generate_ids_reprs(args, model, tokenizer, image_processor, line2, model_name, device, model_dtype, is_multiple_questions=True)
        if not args.do_repr_sample or i in selected_iterations:
            with gzip.open(os.path.join(args.extract_path, f"reprs1_{i}_{ans_dict1['question_id']}.pt.gz"), "wb") as f:
                torch.save(repr1, f)
            with gzip.open(os.path.join(args.extract_path, f"reprs2_{i}_{ans_dict2['question_id']}.pt.gz"), "wb") as f:
                torch.save(repr2, f)
            with open(os.path.join(args.extract_path, f"output_texts1_{i}_{ans_dict1['question_id']}.pkl"), "wb") as f:
                pickle.dump(ans_dict1["text"], f)
            with open(os.path.join(args.extract_path, f"output_texts2_{i}_{ans_dict2['question_id']}.pkl"), "wb") as f:
                pickle.dump(ans_dict2["text"], f)
            reprs1.append(repr1)
            reprs2.append(repr2)
            output_texts1.append(ans_dict1["text"])
            output_texts2.append(ans_dict2["text"])
            question_ids1.append(ans_dict1['question_id'])
            question_ids2.append(ans_dict2['question_id'])
        else:
            pass

        # explore_shape(outputs1.hidden_states)
        # explore_shape(outputs2.hidden_states)
        # exit()

        ans_file1.write(
            json.dumps(
                ans_dict1,
            )
            + "\n"
        )
        ans_file1.flush()
        ans_file2.write(
            json.dumps(
                ans_dict2,
            )
            + "\n"
        )
        ans_file2.flush()
        # cnt += 1
        # if cnt > 100:
        #     break

    del model
    torch.cuda.empty_cache()
    if args.load_repr_sample:
        reprs1 = reprs2 = output_texts1 = output_texts2 = args.extract_path
    repr_model = learn_repr(args, reprs1, reprs2, output_texts1, output_texts2)
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
    parser.add_argument("--sep", type=str, default="__sep__")
    parser.add_argument("--train-path", type=str, default="path/to/train/data")
    parser.add_argument("--output-attentions", action="store_true")
    parser.add_argument("--output-hidden-states", action="store_true")
    parser.add_argument("--export-ids", action="store_true")
    parser.add_argument("--extract-path", type=str, default="path/to/extraction/dir")
    parser.add_argument("--seed", type=int, default=1987)
    parser.add_argument("--padding_num", type=int, default=4096)
    parser.add_argument("--do-repr-sample", action="store_true")
    parser.add_argument("--repr_sample_num", type=int, default=4000)
    parser.add_argument("--load-repr-sample", action="store_true")
    args = parser.parse_args()

    eval_model(args)
