import argparse
import gzip
import json
import os
import pickle
import random

from PIL import Image
import shortuuid
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_PLACEHOLDER
from llava.conversation import conv_templates, SeparatorStyle
from llava.eval.representation_learning import learn_repr
from llava.eval.utils import add_image_token, append_message, explore_shape, get_chunk, makedirs_recursive, parse_filenames, split_list
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

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
def create_data_loader(questions1, questions2, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions1, questions2, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader


def get_chunk_from_file(question_file, args):
    questions = [json.loads(q) for q in open(os.path.expanduser(question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    return questions


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
            max_new_tokens=128,
            output_attentions=args.output_attentions,
            output_hidden_states=args.output_hidden_states,
            return_dict_in_generate=True,
            use_cache=True)

    output_ids = outputs.sequences
    
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
        "prompt": cur_prompt,
        "text": output_sentences,
        "answer_id": ans_id,
        "model_id": model_name,
        "metadata": {}
    }

    reprs = outputs.hidden_states[0][-1].to('cpu').detach()
    reprs_pad = F.pad(reprs, (0, 0, 0, args.padding_num-reprs.shape[1]), "constant", 0)

    return ans_dict, reprs_pad


def eval_model(args):
    # Model
    makedirs_recursive(args.extract_path)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = torch.float32 if not torch.cuda.is_available() else torch.float16
    disable_torch_init()
    question_file1, question_file2 = parse_filenames(args.question_file)
    answers_file1, answers_file2 = parse_filenames(args.answers_file)
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, device=device)
    model = model.to(device=device, dtype=model_dtype)

    questions1 = get_chunk_from_file(question_file1, args)
    questions2 = get_chunk_from_file(question_file2, args)
    
    answers_file1 = os.path.expanduser(answers_file1)
    answers_file2 = os.path.expanduser(answers_file2)
    os.makedirs(os.path.dirname(answers_file1), exist_ok=True)
    os.makedirs(os.path.dirname(answers_file2), exist_ok=True)
    ans_file1 = open(answers_file1, "w")
    ans_file2 = open(answers_file2, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

<<<<<<< HEAD
    data_loader = create_data_loader(questions1, questions2, args.image_folder, tokenizer, image_processor, model.config)

    assert len(questions1) == len(questions2), "questions1 and questions2 must have the same length"
    # cnt = 0
    reprs1, reprs2 = [], []
    output_texts1, output_texts2 = [], []
    question_ids1, question_ids2 = [], []
    if args.do_repr_sample:
        selected_iterations = random.sample(range(len(questions1)), args.repr_sample_num)
    else:
        # all iteration
        selected_iterations = range(len(questions1))
    for i, ((input_ids1, image_tensor1, input_ids2, image_tensor2), line1, line2) in enumerate(tqdm(zip(data_loader, questions1, questions2), total=len(questions1))):
        ans_dict1, repr1 = generate_ids_reprs(image_tensor1, input_ids1, line1, args, model, tokenizer, device, model_dtype, model_name)
        ans_dict2, repr2 = generate_ids_reprs(image_tensor2, input_ids2, line2, args, model, tokenizer, device, model_dtype, model_name)
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

        ans_file1.write(json.dumps(ans_dict1) + "\n")
        ans_file2.write(json.dumps(ans_dict2) + "\n")

        ans_file1.flush()
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
=======
    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)

    for (input_ids, image_tensor), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]

        input_ids = input_ids.to(device='cuda', non_blocking=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        # ans_file.flush()
    ans_file.close()
>>>>>>> 7775b12d6b20cd69089be7a18ea02615a59621cd

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
<<<<<<< HEAD
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
    parser.add_argument("--load-repr-sample", action="store_true")
=======
    parser.add_argument("--max_new_tokens", type=int, default=128)
>>>>>>> 7775b12d6b20cd69089be7a18ea02615a59621cd
    args = parser.parse_args()

    eval_model(args)
