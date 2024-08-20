import argparse
import torch

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO

import re


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    if type(image_files) is list:
        out = []
        for image_file in image_files:
            image = load_image(image_file)
            out.append(image)
    else:
        out = load_image(image_files)
    return out


def image_parser(args):
    if args.sep in args.image_file:
        out = args.image_file.split(args.sep)
    else:
        out = args.image_file
    return out


def eval_model(
    args,
    sanitize_dict={"'": "__single_quote__", '"': "__double_quote__"},
):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    qs = args.query
    for ky in sanitize_dict.keys():
        qs = re.sub(sanitize_dict[ky], ky, qs)
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_files = image_parser(args)
    image = load_images(image_files)
    image_tensor = (
        image_processor.preprocess(image, return_tensors="pt")["pixel_values"]
        .half()
        .cuda()
    )

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        outputs_all = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            return_dict_in_generate=True,
            output_attentions=args.output_attentions,
            output_hidden_states=args.output_hidden_states,
        )
        output_ids = outputs_all["sequences"]

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(
            f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
        )
    outputs = tokenizer.batch_decode(
        output_ids[:, input_token_len:], skip_special_tokens=True
    )[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    # print(outputs)
    with open(
        f"{args.export_path}/{args.filename}_query.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(args.query)
    with open(
        f"{args.export_path}/{args.filename}.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(outputs)
    with open(
        f"{args.export_path}/{args.filename}_all.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(f"{args.query}{outputs}")
    if "_" not in args.filename:
        torch.save(model.cpu(), f"{args.export_path}/{args.filename}_model.pkl")
        torch.save(tokenizer, f"{args.export_path}/{args.filename}_tokenizer.pkl")
        torch.save(
            image_tensor.detach().cpu(),
            f"{args.export_path}/{args.filename}_image_tensor.pkl",
        )
    del outputs
    del model
    del tokenizer
    del image_tensor
    del input_token_len
    del n_diff_input_output
    del input_ids
    del output_ids
    del args
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--output_attentions", action="store_true")
    parser.add_argument("--output_hidden_states", action="store_true")
    parser.add_argument("--sep", type=str, default="__sepsep__")
    parser.add_argument("--filename", type=str, default="Filename to be saved")
    parser.add_argument("--export_path", type=str, default="<file-export-path>")
    args = parser.parse_args()

    eval_model(args)
