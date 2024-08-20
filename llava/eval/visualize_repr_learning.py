"""
./playground/data/eval/mmbench/intermediate/$PREFIX$CKPT
/home/s2140401/script/memes_xai/models/LLaVA/playground/data/eval/mmbench/intermediate/multiple_inputs_llava-llama-2-13b-chat-lightning-preview
/home/s2140401/script/memes_xai/models/LLaVA/playground/data/eval/mmbench/intermediate/multiple_inputs_llava-v1.5-13b

/home/s2140401/script/memes_xai/models/LLaVA/playground/data/eval/gqa/intermediate/multiple_inputs_llava-llama-2-13b-chat-lightning-preview/output_texts2_2_2348284__sep__202225914.pkl
/home/s2140401/script/memes_xai/models/LLaVA/playground/data/eval/gqa/intermediate/multiple_inputs_llava-llama-2-13b-chat-lightning-preview/output_texts1_2_202225914.pkl
"""

import argparse

# from llava.eval.representation_learning import learn_repr, load_files_or_objects_w_substr
from llava.eval.representation_learning import learn_repr_tasks
from llava.eval.utils import get_module_logger


# def main(args, logger):
#     logger.info(f"Root path: {args.root_path}")
#     logger.info(f"Tasks: {args.tasks}")
#     logger.info(f"Processing tasks...")
#     inputs, reprs_model = [], []
#     for task in args.tasks.split(","):
#         logger.info(f"Processing task: {task}")
#         for model in args.models.split(","):
#             logger.info(f"Processing model: {model} for task: {task}")
#             if task == "vqav2":
#                 # <root-path>/vqav2/intermediate/multiple_inputs_llava_vqav2_mscoco_test-dev2015/<model>
#                 intermediate_path = f"{args.root_path}/{task}/intermediate/multiple_inputs_llava_vqav2_mscoco_test-dev2015/{model}"
#             else:
#                 intermediate_path = f"{args.root_path}/{task}/intermediate/multiple_inputs_{model}"
#             args.extract_path = f"{args.root_path}/extract/{task}/{model}"
#             reprs1 = reprs2 = output_texts1 = output_texts2 = intermediate_path
#             repr_model = learn_repr(args, reprs1, reprs2, output_texts1, output_texts2, logger=logger)
#             inputs.append([reprs1, reprs2, output_texts1, output_texts2])
#             reprs_model.append(repr_model)
#             logger.info(f"Model {model} for task {task} completed")
#         logger.info(f"Task {task} completed")
#     logger.info(f"Processing completed")
#     return inputs, reprs_model
def main(args, logger):
    logger.info(f"Processing tasks {args.tasks}...")
    outputs = learn_repr_tasks(args, logger)
    logger.info(f"Processing completed")
    return outputs

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-path", type=str, default="path/to/root/path")
    parser.add_argument("--tasks", type=str, default="gqa,mm-vet,mmbench,textvqa,vizwiz,vqav2")
    parser.add_argument("--models", type=str, default="llava-v1.5-13b,llava-llama-2-13b-chat-lightning-preview")
    parser.add_argument("--seed", type=int, default=1987)
    parser.add_argument("--padding-num", type=int, default=4096)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--contrastive-model", type=str, default="attention", choices=["attention", "linear", "mixed"])
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    logger = get_module_logger(__name__)

    main(args, logger)
