import argparse
import gzip
import json
import math
import os
import pickle
import random

import pandas as pd
from PIL import Image
import shortuuid
import torch
import torch.nn.functional as F
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_PLACEHOLDER
from llava.conversation import conv_templates, SeparatorStyle
from llava.eval.representation_learning import learn_repr
from llava.eval.utils import add_image_token, append_message, explore_shape, makedirs_recursive, split_list, get_chunk
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, load_image_from_base64, get_model_name_from_path


all_options = ['A', 'B', 'C', 'D']


def is_none(value):
    if value is None:
        return True
    if type(value) is float and math.isnan(value):
        return True
    if type(value) is str and value.lower() == 'nan':
        return True
    if type(value) is str and value.lower() == 'none':
        return True
    return False

def get_options(row, options):
    parsed_options = []
    for option in options:
        option_value = row[option]
        if is_none(option_value):
            break
        parsed_options.append(option_value)
    return parsed_options

def load_image(args, image_file):
    if image_file.endswith('.jpg') or image_file.endswith('.png'):
        image = Image.open(os.path.join(args.train_path, image_file)).convert('RGB')
    else:
        image = load_image_from_base64(image_file)
    return image

def load_images(args, image_files):
    out = []
    if type(image_files) is not list:
        image_list = [image_files]
    else:
        image_list = image_files
    for image_file in image_list:
        image = load_image(args, image_file)
        out.append(image)
    return out

def get_questions(args, question_file):
    questions = pd.read_table(os.path.expanduser(question_file))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    return questions

def get_answers(args, answers_file):
    answers_file = os.path.expanduser(answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    return ans_file

def get_options_char(row, all_options=all_options):
    options = get_options(row, all_options)
    cur_option_char = all_options[:len(options)]
    return options, cur_option_char

def get_answer_dict_reprs(args, row, options, cur_option_char, model_name, model, tokenizer, image_processor, device='cuda', model_dtype=torch.float16):
    if args.all_rounds:
        num_rounds = len(options)
    else:
        num_rounds = 1

    ans_dicts, reprs_all = [], []
    # cnt = 0
    for round_idx in range(num_rounds):
        idx = row['index']
        question = row['question']
        hint = row['hint']
        is_multiple_questions = True if len(question.split(args.sep)) > 1 else False
        if not is_none(hint):
            if is_multiple_questions:
                fst, snd = question.split(args.sep)
                question = fst + args.sep + hint + '\n' + snd
            else:
                question = hint + '\n' + question
        for option_char, option in zip(all_options[:len(options)], options):
            question = question + '\n' + option_char + '. ' + option
        row['question'] = qs = cur_prompt = question
        qs, first_answer, image_file = add_image_token(args, row, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, is_multiple_questions, IMAGE_PLACEHOLDER, model.config, text_col='question')
        images = load_images(args, image_file)

        if args.single_pred_prompt:
            if is_multiple_questions:
                if args.lang == 'cn':
                    qs[-1] = qs[-1] + '\n' + "请直接回答选项字母。"
                else:
                    qs[-1] = qs[-1] + '\n' + "Answer with the option's letter from the given choices directly."
            else:
                if args.lang == 'cn':
                    qs = qs + '\n' + "请直接回答选项字母。"
                else:
                    qs = qs + '\n' + "Answer with the option's letter from the given choices directly."

        conv = conv_templates[args.conv_mode].copy()
        conv = append_message(conv, qs, is_multiple_questions, first_answer)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device=device)

        image_tensor = process_images(images, image_processor, model.config)#[0]
        # image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        image_tensor = image_tensor.half().cuda() if device == 'cuda' else image_tensor.to(device=device, dtype=model_dtype)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        with torch.inference_mode():
            outputs = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                output_hidden_states=args.output_hidden_states,
                output_attentions=args.output_attentions,
                return_dict_in_generate=True,
                use_cache=True)
            output_ids = outputs["sequences"]

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        output_sentences = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        output_sentences = output_sentences.strip()
        if output_sentences.endswith(stop_str):
            output_sentences = output_sentences[:-len(stop_str)]
        output_sentences = output_sentences.strip()

        ans_id = shortuuid.uuid()

        ans_dict = {
            "question_id": idx,
            "round_id": round_idx,
            "prompt": cur_prompt,
            "text": output_sentences,
            "options": options,
            "option_char": cur_option_char,
            "answer_id": ans_id,
            "model_id": model_name,
            "metadata": {}
        }
        # rotate options
        options = options[1:] + options[:1]
        cur_option_char = cur_option_char[1:] + cur_option_char[:1]
        # explore_shape(outputs.hidden_states)
        # exit()

        reprs = outputs.hidden_states[0][-1].to('cpu').detach()
        reprs_pad = F.pad(reprs, (0, 0, 0, args.padding_num-reprs.shape[1]), "constant", 0)

        ans_dicts.append(ans_dict)
        reprs_all.append(reprs_pad)

    return ans_dicts, torch.stack(reprs_all)

def write_answer(args, ans_file, ans_dicts):
    for ans_dict in ans_dicts:
        ans_file.write(json.dumps(ans_dict) + "\n")
        ans_file.flush()
    return

def extract_text_qid_from_ans_dict(ans_dicts):
    text = [ans_dict["text"] for ans_dict in ans_dicts]
    qid = [ans_dict["question_id"] for ans_dict in ans_dicts]
    return text, qid

def eval_model(args):
    # Model
    disable_torch_init()
    makedirs_recursive(args.extract_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = torch.float32 if not torch.cuda.is_available() else torch.float16
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, device=device)
    model = model.to(device=device, dtype=model_dtype)

    question_file1, question_file2 = args.question_file.split(',')
    questions1 = get_questions(args, question_file1)
    questions2 = get_questions(args, question_file2)
    answers_file1, answers_file2 = args.answers_file.split(',')
    ans_file1 = get_answers(args, answers_file1)
    ans_file2 = get_answers(args, answers_file2)

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    assert len(questions1) == len(questions2)
    # cnt = 0
    reprs1, reprs2 = [], []
    output_texts1, output_texts2 = [], []
    question_ids1, question_ids2 = [], []
    if args.do_repr_sample:
        selected_iterations = random.sample(range(len(questions1)), args.repr_sample_num)
    else:
        # all iteration
        selected_iterations = range(len(questions1))
    for i, ((index1, row1), (index2, row2)) in enumerate(tqdm(zip(questions1.iterrows(), questions2.iterrows()), total=len(questions1))):
        options1, cur_option_char1 = get_options_char(row1, all_options)
        options2, cur_option_char2 = get_options_char(row2, all_options)

        ans_dicts1, repr1 = get_answer_dict_reprs(args, row1, options1, cur_option_char1, model_name, model, tokenizer, image_processor, device, model_dtype)
        ans_dicts2, repr2 = get_answer_dict_reprs(args, row2, options2, cur_option_char2, model_name, model, tokenizer, image_processor, device, model_dtype)
        output_text1, question_id1 = extract_text_qid_from_ans_dict(ans_dicts1)
        output_text2, question_id2 = extract_text_qid_from_ans_dict(ans_dicts2)
        if not args.do_repr_sample or i in selected_iterations:
            with gzip.open(os.path.join(args.extract_path, f"reprs1_{i}_{question_id1[0]}.pt.gz"), "wb") as f:
                torch.save(repr1.squeeze(0), f)
            with gzip.open(os.path.join(args.extract_path, f"reprs2_{i}_{question_id2[0]}.pt.gz"), "wb") as f:
                torch.save(repr2.squeeze(0), f)
            with open(os.path.join(args.extract_path, f"output_texts1_{i}_{question_id1[0]}.pkl"), "wb") as f:
                pickle.dump(output_text1, f)
            with open(os.path.join(args.extract_path, f"output_texts2_{i}_{question_id2[0]}.pkl"), "wb") as f:
                pickle.dump(output_text2, f)
            reprs1.append(repr1.squeeze(0))
            reprs2.append(repr2.squeeze(0))
            output_texts1.append(output_text1)
            output_texts2.append(output_text2)
            question_ids1.append(question_id1)
            question_ids2.append(question_id2)
        else:
            pass

        write_answer(args, ans_file1, ans_dicts1)
        write_answer(args, ans_file2, ans_dicts2)
        # cnt += 1
        # if cnt > 100:
        #     break

    del model
    torch.cuda.empty_cache()
    # if args.load_repr_sample:
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
    parser.add_argument("--all-rounds", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--sep", type=str, default="__sep__")
    parser.add_argument("--train-path", type=str, default="path/to/train/data")
    parser.add_argument("--output-attentions", action="store_true")
    parser.add_argument("--output-hidden-states", action="store_true")
    parser.add_argument("--export-ids", action="store_true")
    parser.add_argument("--extract-path", type=str, default="path/to/extraction/dir")
    parser.add_argument("--seed", type=int, default=1987)
    parser.add_argument("--padding_num", type=int, default=4096)
    parser.add_argument("--do-repr-sample", action="store_true")
    parser.add_argument("--repr-sample-num", type=int, default=4000)
    # parser.add_argument("--load-repr-sample", action="store_true")
    args = parser.parse_args()

    eval_model(args)
