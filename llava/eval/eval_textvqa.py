import os
import argparse
import json
import re

from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-file', type=str)
    parser.add_argument('--result-file', type=str)
    parser.add_argument('--result-dir', type=str)
    parser.add_argument('--separator', type=str, default='__sep__')
    return parser.parse_args()


def qid_prompt_processor(org_qid, org_prompt, separator = '__sep__'):
    if separator in org_prompt:
        assert separator in org_qid, "question_id and prompt must have the same separator"
        qid = org_qid.split(separator)[-1]
        prompt = org_prompt.split(separator)[-1]
    else:
        qid = org_qid
        prompt = org_prompt
    if prompt.startswith('OCR tokens: '):
        pattern = r"Question: (.*?) Short answer:"
        match = re.search(pattern, prompt, re.DOTALL)
        question = match.group(1)
    elif 'Reference OCR token: ' in prompt and len(prompt.split('\n')) == 3:
        if prompt.startswith('Reference OCR token:'):
            question = prompt.split('\n')[1]
        else:
            question = prompt.split('\n')[0]
    elif len(prompt.split('\n')) == 2:
        question = prompt.split('\n')[0]
    else:
        assert False

    return (qid, question.lower())


def eval_single(annotation_file, result_file, separator = '__sep__'):
    experiment_name = os.path.splitext(os.path.basename(result_file))[0]
    print(experiment_name)
    annotations = json.load(open(annotation_file))['data']
    annotations = {(annotation['image_id'], annotation['question'].lower()): annotation for annotation in annotations}
    results = [json.loads(line) for line in open(result_file)]

    pred_list = []
    for result in results:
        # (result['question_id'], prompt_processor(result['prompt'], separator=separator))
        annotation = annotations[qid_prompt_processor(result['question_id'], result['prompt'], separator=separator)]
        pred_list.append({
            "pred_answer": result['text'],
            "gt_answers": annotation['answers'],
        })

    evaluator = TextVQAAccuracyEvaluator()
    print('Samples: {}\nAccuracy: {:.2f}%\n'.format(len(pred_list), 100. * evaluator.eval_pred_list(pred_list)))


if __name__ == "__main__":
    args = get_args()

    if args.result_file is not None:
        eval_single(args.annotation_file, args.result_file, args.separator)

    if args.result_dir is not None:
        for result_file in sorted(os.listdir(args.result_dir)):
            if not result_file.endswith('.jsonl'):
                print(f'Skipping {result_file}')
                continue
            eval_single(args.annotation_file, os.path.join(args.result_dir, result_file), args.separator)
