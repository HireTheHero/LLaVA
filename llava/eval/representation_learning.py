# from collections import Counter
from dataclasses import dataclass
from datetime import datetime
import gzip
import itertools
# import math
import os
import pickle
from pprint import pprint
import psutil
import random
from tqdm import tqdm

from einops import repeat
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
# import pytorch_metric_learning
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.losses import SelfSupervisedLoss, TripletMarginLoss
from pytorch_metric_learning.reducers import ThresholdReducer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.dataloader import default_collate
from transformers import LlamaForCausalLM, LlamaTokenizer

from llava.eval.utils import makedirs_recursive  # , split_list


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class CustomDataset(Dataset):
    def __init__(
        self,
        query_embed,
        target_embed,
        query_text,
        target_text,
        fixed_effect_indices=None,
    ):
        self.query_embed = query_embed
        self.target_embed = target_embed
        self.query_text = query_text
        self.target_text = target_text
        self.fixed_effect_indices = fixed_effect_indices

    def __len__(self):
        if self.fixed_effect_indices is not None:
            assert (
                len(self.query_embed)
                == len(self.target_embed)
                == len(self.query_text)
                == len(self.target_text)
                == len(self.fixed_effect_indices)
            ), f"Lengths of inputs do not match: {len(self.query_embed), len(self.target_embed), len(self.query_text), len(self.target_text), len(self.fixed_effect_indices)}"
        else:
            assert (
                len(self.query_embed)
                == len(self.target_embed)
                == len(self.query_text)
                == len(self.target_text)
            ), f"Lengths of inputs do not match: {len(self.query_embed), len(self.target_embed), len(self.query_text), len(self.target_text)}"
        return len(self.query_embed)
    
    def ensure_text_is_str(self, element):
        if isinstance(element, str):
            out = element
        elif isinstance(element, list):
            out = element[0]
        else:
            raise ValueError("Item is neither a string nor a list of strings")
        return out

    def __getitem__(self, index):
        q_embed = self.query_embed[index].squeeze(0)
        t_embed = self.target_embed[index].squeeze(0)
        # q_answers = self.query_text[index]
        q_answers = self.ensure_text_is_str(self.query_text[index])
        # t_answers = self.target_text[index]
        t_answers = self.ensure_text_is_str(self.target_text[index])
        if self.fixed_effect_indices is not None:
            idx = self.fixed_effect_indices[index]
        else:
            idx = None
        return q_embed, t_embed, q_answers, t_answers, idx


@dataclass
class CustomDatasetConfig:
    # data_size: int = 152
    # batch_size: int = 4
    # batch_size: int = 10
    # batch_size: int = 12
    batch_size: int = 16
    test_size: float = 0.2
    # num_workers: int = 4


class ExtendedCustomDataset(Dataset):
    def __init__(self, dataset, categories=[0,1,2,3]):
        """
        Initializes the dataset with an instance of CustomDataset.
        """
        # assert isinstance(dataset, CustomDataset), "dataset must be an instance of CustomDataset"
        self.dataset = dataset
        self.categories = categories
        self.item_to_categories = self.prepare_item_category_map()
        self.num_categories = {ct: len(self.item_to_categories[ct]) for ct in self.categories}

    def prepare_item_category_map(self):
        """
        Prepare a mapping from each item index to its appearances across all categories.
        """
        item_category_map = {ct: [] for ct in self.categories}
        for idx in range(len(self.dataset)):
            item_data = self.dataset[idx]
            category = item_data[-1].item()
            # item_category_map[category].append(self.format2batched(item_data))
            item_category_map[category].append(item_data)
        
        return item_category_map

    def __len__(self):
        """
        Returns the total number of unique categories.
        """
        return max(self.num_categories)

    def __getitem__(self, index):
        """
        Retrieves all items for a given category.
        """
        output = {ct: None for ct in self.categories}
        for ct in self.categories:
            if len(self.item_to_categories[ct])-1>=index:
                output[ct] = self.item_to_categories[ct][index]
            else:
                output[ct] = None
        
        return output


def correct_list2str(element):
    if isinstance(element, str):
        out = element
    elif isinstance(element, list):
        out = element[0]
    else:
        raise ValueError("Item is neither a string nor a list of strings")
    return out


def correct_list2str_batch(batch):
    out_batch = []
    for item in batch:
        a, b, should_be_text1, should_be_text2, c = item
        text1 = correct_list2str(should_be_text1)
        text2 = correct_list2str(should_be_text2)
        out_batch.append((a, b, text1, text2, c))
    return out_batch


def custom_collate(batch, logger = None,):
    batch = correct_list2str_batch(batch)
    try:
        return default_collate(batch)
    except RuntimeError as e:
        # Handle or preprocess batch elements here
        if logger:
            logger.warn(f"Iteration failure: {batch}")
        else:
            print(f"Iteration failure: {batch}")
            raise e


def create_data_loader(
    query_embed,
    target_embed,
    query_text,
    target_text,
    fixed_effect_indices_and_mapping=None,
    is_category_wise_eval=False,
    custom_collate=None,
    logger=None,
):
    if custom_collate is None:
        collate = default_collate
    else:
        collate = lambda x: custom_collate(x, logger=logger)
    if fixed_effect_indices_and_mapping:
        fixed_effect_indices, mapping = fixed_effect_indices_and_mapping
    else:
        fixed_effect_indices = None
    config = CustomDatasetConfig()
    batch_size = config.batch_size
    # query_embed_batched = split_list(query_embed, batch_size)
    dataset = CustomDataset(
        query_embed,
        target_embed,
        query_text,
        target_text,
        fixed_effect_indices,
    )
    test_size = round(len(dataset) * config.test_size)
    train_size = len(dataset) - test_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        # collate_fn=lambda x: x,
        collate_fn=collate,
    )
    if is_category_wise_eval:
        test_data = ExtendedCustomDataset(test_ds, list(set(mapping.values())))
    else:
        test_data = DataLoader(
            test_ds, 
            batch_size=batch_size, 
            shuffle=False, 
            # collate_fn=lambda x: x,
            collate_fn=collate,
        )
    return train_dataloader, test_data


@dataclass
class CustomModelConfig:
    text_model: str = "meta-llama/Llama-2-13b-chat-hf"
    # n_epochs: int = 4
    # n_epochs: int = 1000
    n_epochs: int = 500
    embed_dim: int = 5120
    num_heads: int = 8
    pad_token: str = "[PAD]"  # tokenizer.eos_token


class ContrastiveAttention(nn.Module):
    def __init__(self, config):
        super(ContrastiveAttention, self).__init__()
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.attention = nn.MultiheadAttention(
            self.embed_dim, self.num_heads, batch_first=True
        )

    def forward(self, a, b):
        output, attention_weights = self.attention(a, b, b)
        return output, attention_weights


def index_lists(all_lists):
    # Step 1: Create a unified mapping for all unique tuples
    tuple_to_index = {t: i for i, t in enumerate(set(all_lists))}

    # Step 2: Convert tuple combinations to unified indices
    indices = torch.tensor([tuple_to_index[tuple] for tuple in all_lists], dtype=torch.long)

    return indices, tuple_to_index


class MixedEffectModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_categories, hidden_dim):
        super(MixedEffectModel, self).__init__()
        # Random effects - Feed Forward Network
        self.fnn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        # Fixed effects - Embedding Layer
        self.embedding = nn.Embedding(num_categories, output_dim)

    def format2batched(self, org_input):
        if org_input.dim()==1:
            formatted_input = org_input.reshape(1,len(org_input))
        else:
            formatted_input = org_input
        return formatted_input

    def forward(self, x, z):
        # Apply the random effects model
        # print(f"x.shape: {x.shape}")
        x_random = self.fnn(x)
        # print(f"x_random.shape: {x_random.shape}")
        # Apply the random effects model
        w_fixed = self.embedding(z)
        # print(f"w_fixed.shape: {w_fixed.shape}")
        w_fixed = self.format2batched(w_fixed)
        # print(f"w_fixed.shape: {w_fixed.shape}")
        w_fixed = repeat(w_fixed, 'n d2 -> n d1 d2', d1=x_random.shape[1])
        # print(f"w_fixed.shape: {w_fixed.shape}")
        x_fixed = x_random * w_fixed
        # print(f"x_fixed.shape: {x_fixed.shape}")
        # Interaction term
        interaction = x_random * x_fixed
        # print(f"interaction.shape: {interaction.shape}")
        # Combine the fixed and random effects
        x_combined = x_random + interaction
        # print(f"x_combined.shape: {x_combined.shape}")
        # print("=====================================")
        return x_combined


class RandomEffectModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_categories, hidden_dim):
        super(RandomEffectModel, self).__init__()
        # Random effects - Feed Forward Network
        self.fnn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x, z):
        # Apply the random effects model
        # print(f"x.shape: {x.shape}")
        x_random = self.fnn(x)
        return x_random


def load_objects(
    dtype4model=torch.float32, tasks=None, models=None, model_type="attention"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = CustomModelConfig()
    # text tokenizer and model
    tokenizer = LlamaTokenizer.from_pretrained(config.text_model)
    tokenizer.pad_token = config.pad_token
    text_model = LlamaForCausalLM.from_pretrained(config.text_model)
    # contrastive model
    if tasks is not None and models is not None:
        fixed_effect_indices, mapping = index_lists(list(zip(tasks, models)))
    else:
        fixed_effect_indices = None
    if model_type == "attention":
        contrastive_model = ContrastiveAttention(config).to(
            device=device, dtype=dtype4model
        )
    elif model_type == "linear":
        contrastive_model = RandomEffectModel(
            config.embed_dim, config.embed_dim, len(mapping), 512
        ).to(
            device=device, dtype=dtype4model
        )
    elif model_type == "mixed":
        contrastive_model = MixedEffectModel(
            config.embed_dim, config.embed_dim, len(mapping), 512
        ).to(device=device, dtype=dtype4model)
    else:
        raise ValueError(f"Contrastive model {model_type} not supported")
    distance = CosineSimilarity()
    reducer = ThresholdReducer(low=0)
    cont_criterion = SelfSupervisedLoss(
        TripletMarginLoss(distance=distance, reducer=reducer)
    )
    cont_optimizer = optim.AdamW(contrastive_model.parameters(), lr=0.001)
    # return config, contrastive_model, None, None, cont_criterion, cont_optimizer, distance, device, dtype4model
    return (
        config,
        contrastive_model,
        tokenizer,
        text_model,
        cont_criterion,
        cont_optimizer,
        distance,
        device,
        dtype4model,
        fixed_effect_indices,
        mapping,
    )


def embed_sentences(sentences, tokenizer, model, padding_num=4096):
    # if len(sentences) == CustomDatasetConfig().batch_size:
    #     sentences_reshaped = sentences
    # else:
    #     sentences_reshaped = list(itertools.chain(*sentences))
    input_ids = tokenizer(
        sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
    )["input_ids"]
    # input_ids = tokenizer(
    #     sentences_reshaped,
    #     return_tensors="pt",
    #     padding=True,
    #     truncation=True,
    #     max_length=4096,
    # )["input_ids"]
    with torch.no_grad():
        sentences_embed = model(input_ids, return_dict=True, output_hidden_states=True)[
            "hidden_states"
        ]
    sentences_embed_pad = F.pad(
        sentences_embed[-1],
        (0, 0, 0, padding_num - sentences_embed[-1].shape[1]),
        "constant",
        0,
    )
    return sentences_embed_pad.unsqueeze(0)  # [0].view(sentences_embed[0].shape[0], -1)


def output_contextual_embeddings(
    model_type,
    contrastive_model,
    query_embed,
    target_embed,
    fixed_effect_idx=None,
):
    if model_type in ("mixed", "linear"):
        output = contrastive_model(query_embed, fixed_effect_idx)
    elif model_type == "attention":
        output, att = contrastive_model(query_embed, target_embed)
    else:
        raise ValueError(f"Contrastive model {model_type} not supported")
    return output


def train_loop(args, objects, train_dataloader, logger=None):
    (
        config,
        contrastive_model,
        _,
        _,
        cont_criterion,
        cont_optimizer,
        _,
        device,
        dtype4model,
        _,
        _,
    ) = objects
    n_epochs = config.n_epochs
    contrastive_model.train()
    # text_model.eval()
    best_loss = None
    for epoch in range(n_epochs):
        total_loss = 0.0
        for i, batch in enumerate(train_dataloader):
            query_embed, target_embed, query_text, _, fixed_effect_idx = batch
            query_embed = query_embed.to(device=device, dtype=dtype4model)
            target_embed = target_embed.to(device=device, dtype=dtype4model)
            if fixed_effect_idx is not None:
                fixed_effect_idx = fixed_effect_idx.to(device=device)
            output = output_contextual_embeddings(
                args.contrastive_model,
                contrastive_model,
                query_embed,
                target_embed,
                fixed_effect_idx=fixed_effect_idx,
            )
            loss = cont_criterion(
                output.reshape(output.shape[0], -1),
                target_embed.reshape(target_embed.shape[0], -1),
            )
            total_loss += loss.item()
            cont_optimizer.zero_grad()
            loss.backward()
            cont_optimizer.step()
            del query_embed
            del target_embed
            del output
            # process = psutil.Process()
            # log_or_print(f"Memory used in train loop: {process.memory_info().rss}", logger)
        log_or_print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {total_loss}", logger)
        process = psutil.Process()
        log_or_print(f"Memory used in train loop: {process.memory_info().rss}", logger)

        if best_loss is None or total_loss < best_loss:
            best_loss = total_loss
            best_model = contrastive_model.state_dict()

            # Get the current datetime
            now = datetime.now()
            # Format the datetime as yymmddhhmmss
            timestamp_str = now.strftime('%y%m%d%H%M%S')

            save_dir_fin = f"{args.root_path}/contrastive_model"
            makedirs_recursive(save_dir_fin)
            torch.save(best_model, f"{save_dir_fin}/{timestamp_str}_{args.contrastive_model}.pt")
        elif total_loss == 0.0:
            log_or_print("Loss is 0.0. Stopping training.", logger)
            break
        else:
            pass
    return best_model


def calculate_sample_wise_similarity(query_embed, target_embed):
    """
    Calculate sample-wise similarity between query and target embeddings.
    # Input
    query_embed: torch.Tensor w/ shape (n_samples, step, n_features)
    target_embed: torch.Tensor w/ shape (n_samples, step, n_features)
    # Output
    similarity: torch.Tensor w/ shape (n_samples)
    """
    # reshape to (n_samples, n_features)
    query_embed = query_embed.reshape(query_embed.shape[0], -1)
    target_embed = target_embed.reshape(target_embed.shape[0], -1)
    # calculate cosine similarity
    similarity = F.cosine_similarity(query_embed, target_embed, dim=1)
    return similarity


def ensure_idx_format(item_or_tensor):
    assert isinstance(item_or_tensor, torch.Tensor), f"Input type should be tensor but is {type(item_or_tensor)}"
    if torch.numel(item_or_tensor)==1:
        output_tensor = torch.tensor([item_or_tensor.item()])
    else:
        output_tensor = item_or_tensor.clone()
    return output_tensor

def eval_loop(args, objects, test_data, is_category_wise_eval=False, logger=None):
    (
        _,
        model,
        tokenizer,
        text_model,
        cont_criterion,
        _,
        _,
        device,
        dtype4model,
        _,
        _,# mapping,        
    ) = objects
    model = model.to(device=device, dtype=dtype4model)
    model.eval()
    losses = []
    if is_category_wise_eval:
        log_or_print("Category-wise evaluation", logger)
        outputs = {}
    else:
        log_or_print("Sample-wise evaluation", logger)
        outputs = {"query_target": [], "score_target": [], "answers": [], "fixed_effect_idx": []}
    with torch.no_grad():
        if is_category_wise_eval:
            for entry in tqdm(test_data):
                for ct in entry.keys():
                    entry_ct = entry[ct]
                    if not entry_ct:
                        pass
                    else:
                        query_embed, target_embed, query_text, target_text, fixed_effect_idx = entry_ct
                        # todo: not necessarily on device
                        query_embed = query_embed.unsqueeze(0).to(device=device, dtype=dtype4model)
                        target_embed = target_embed.unsqueeze(0).to(device=device, dtype=dtype4model)
                        query_answer_embed = embed_sentences(
                            [query_text], tokenizer, text_model, padding_num=args.padding_num
                        )[0].to(device=device, dtype=dtype4model)
                        target_answer_embed = embed_sentences(
                            [target_text], tokenizer, text_model, padding_num=args.padding_num
                        )[0].to(device=device, dtype=dtype4model)
                        fixed_effect_idx = ensure_idx_format(fixed_effect_idx).to(device=device)
                        output = output_contextual_embeddings(
                            args.contrastive_model,
                            model,
                            query_embed,
                            target_embed,
                            fixed_effect_idx=fixed_effect_idx,
                        )
                        dist_query_tgt = calculate_sample_wise_similarity(
                            query_embed, target_embed
                        ).to("cpu")
                        dist_score_tgt = calculate_sample_wise_similarity(output, target_embed).to(
                            "cpu"
                        )
                        dist_ans = calculate_sample_wise_similarity(
                            query_answer_embed, target_answer_embed
                        ).to("cpu")
                        loss = cont_criterion(
                            output.reshape(output.shape[0], -1),
                            target_embed.reshape(target_embed.shape[0], -1),
                        )
                        del query_embed
                        del target_embed
                        del output
                        del query_answer_embed
                        del target_answer_embed
                    if not outputs.get(f"{ct}_query_target"):
                        outputs = {
                            f"{ct}_query_target": [dist_query_tgt],
                            f"{ct}_score_target": [dist_score_tgt],
                            f"{ct}_answers": [dist_ans],
                            f"{ct}_loss": [loss.item()],
                        }
                    else:
                        outputs[f"{ct}_query_target"].append(dist_query_tgt)
                        outputs[f"{ct}_score_target"].append(dist_score_tgt)
                        outputs[f"{ct}_answers"].append(dist_ans)
                        outputs[f"{ct}_loss"].append(loss.item())

        else:
            for batch in tqdm(test_data):
                query_embed, target_embed, query_text, target_text, fixed_effect_idx = batch
                query_embed = query_embed.to(device=device, dtype=dtype4model)
                target_embed = target_embed.to(device=device, dtype=dtype4model)
                query_answer_embed = embed_sentences(
                    query_text, tokenizer, text_model, padding_num=args.padding_num
                )[0].to(device=device, dtype=dtype4model)
                target_answer_embed = embed_sentences(
                    target_text, tokenizer, text_model, padding_num=args.padding_num
                )[0].to(device=device, dtype=dtype4model)
                fixed_effect_idx = ensure_idx_format(fixed_effect_idx).to(device=device)
                output = output_contextual_embeddings(
                    args.contrastive_model,
                    model,
                    query_embed,
                    target_embed,
                    fixed_effect_idx=fixed_effect_idx,
                )
                dist_query_tgt = calculate_sample_wise_similarity(
                    query_embed, target_embed
                ).to("cpu")
                outputs["query_target"].append(dist_query_tgt)
                dist_score_tgt = calculate_sample_wise_similarity(output, target_embed).to(
                    "cpu"
                )
                outputs["score_target"].append(dist_score_tgt)
                dist_ans = calculate_sample_wise_similarity(
                    query_answer_embed, target_answer_embed
                ).to("cpu")
                outputs["answers"].append(dist_ans)
                outputs["fixed_effect_idx"].append(fixed_effect_idx)
                loss = cont_criterion(
                    output.reshape(output.shape[0], -1),
                    target_embed.reshape(target_embed.shape[0], -1),
                )
                losses.append(loss.item())
    if is_category_wise_eval:
        pass
    else:
        total_loss = sum(losses) / len(losses)
    return outputs, total_loss


def evaluate_embeddings(model_outputs):
    # colors = ["darkblue" if l==1 else "lightgray" for l in labels]
    if isinstance(model_outputs["query_target"][0], torch.Tensor):
        query_target = torch.cat(model_outputs["query_target"]).reshape(-1).numpy()
        score_target = torch.cat(model_outputs["score_target"]).reshape(-1).numpy()
        answers = torch.cat(model_outputs["answers"]).reshape(-1).numpy()
    else:
        query_target = model_outputs["query_target"]
        score_target = model_outputs["score_target"]
        answers = model_outputs["answers"]

    input_dict = {
        "query_target": query_target,
        "score_target": score_target,
        "answers": answers,
    }
    input_df = pd.DataFrame(input_dict)

    # fig_before = go.Figure(data=[go.Scatter3d(x=query_target, y=query_answer, z=query_target_answer, mode='markers')])
    # fig_after = go.Figure(data=go.Scatter(x=score_target, y=score_target_answer, mode='markers'))
    # fig = go.Figure(data=go.Scatter(x=query_target, y=score_target, mode='markers', marker=dict(color=colors)))
    fig_3d = go.Figure(
        data=[go.Scatter3d(x=input_df["query_target"], y=input_df["score_target"], z=input_df["answers"], mode="markers")]
    )
    fig = sp.make_subplots(rows=1, cols=3)
    # 各subplotにデータをプロット
    fig.add_trace(
        go.Scatter(
            x=input_df["query_target"],
            y=input_df["answers"],
            mode="markers",
            name="X-Y",
            marker=dict(color="lightblue"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=input_df["query_target"],
            y=input_df["score_target"],
            mode="markers",
            name="X-Z",
            marker=dict(color="blue"),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=input_df["score_target"],
            y=input_df["answers"],
            mode="markers",
            name="Y-Z",
            marker=dict(color="darkblue"),
        ),
        row=1,
        col=3,
    )

    return fig_3d, fig, input_df
    # return fig_before, fig_after

def evaluate_embeddings_by_ctgry(model_outputs_dict):
    out_dict = {}
    for ky in model_outputs_dict.keys():
        out_dict[ky] = evaluate_embeddings(model_outputs_dict[ky])
    return out_dict


def load_file_or_object(file_or_object):
    """
    Load file or object.
    input 1: str *.pt.gz -> load gzip torch file
    input 2: str *.pkl -> load pickle
    input 3: tensor -> return tensor
    input 4: list -> return list
    """
    if isinstance(file_or_object, str):
        if file_or_object.endswith(".pt.gz"):
            # load gzip torch file
            with gzip.open(file_or_object, "rb") as f:
                obj = torch.load(f)
        elif file_or_object.endswith(".pkl"):
            with open(file_or_object, "rb") as f:
                obj = pickle.load(f)
        else:
            raise ValueError(f"File type {file_or_object} not supported")
    elif isinstance(file_or_object, torch.Tensor) or isinstance(file_or_object, list):
        obj = file_or_object
    else:
        raise ValueError(f"Input type {type(file_or_object)} not supported")
    return obj


def log_or_print(msg, logger=None):
    if logger:
        logger.info(msg)
    else:
        print(msg)
    return


def load_files_or_objects_w_substr(
    dirname_or_objects,
    extension=".pt.gz",
    file_substr="reprs1",
    do_sample=None,
    sample_size=1000,
):
    """
    Load files or objects with substring.
    input 1: dirname string -> load load files with extension (.pt.gz or .pkl) in dirname
    input 2: list of objects -> return list of objects
    """
    if isinstance(dirname_or_objects, str):
        # load files with extension (.pt.gz or .pkl) in dirname
        filenames = sorted(os.listdir(dirname_or_objects))
        files = [f for f in filenames if f.endswith(extension) and file_substr in f]
        if do_sample:
            # to maintain file order across multiple runs
            files = files[:sample_size]
        objs = [load_file_or_object(f"{dirname_or_objects}/{f}") for f in files]
    elif isinstance(dirname_or_objects, list):
        if do_sample:
            # to maintain file order across multiple runs
            objs = dirname_or_objects[:sample_size]
        else:
            objs = dirname_or_objects
    else:
        raise ValueError(f"Input type {type(dirname_or_objects)} not supported")
    return objs


def learn_repr(args, query_embed, target_embed, query_text, target_text, logger=None):
    makedirs_recursive(args.extract_path)
    query_embed = load_files_or_objects_w_substr(
        query_embed,
        extension=".pt.gz",
        file_substr="reprs1",
        do_sample=args.do_sample,
        sample_size=args.sample_size,
    )
    log_or_print(f"query_embed loaded with size: {len(query_embed)}", logger)
    target_embed = load_files_or_objects_w_substr(
        target_embed,
        extension=".pt.gz",
        file_substr="reprs2",
        do_sample=args.do_sample,
        sample_size=args.sample_size,
    )
    log_or_print(f"target_embed loaded with size: {len(target_embed)}", logger)
    query_text = load_files_or_objects_w_substr(
        query_text,
        extension=".pkl",
        file_substr="output_texts1",
        do_sample=args.do_sample,
        sample_size=args.sample_size,
    )
    log_or_print(f"query_text loaded with size: {len(query_text)}", logger)
    target_text = load_files_or_objects_w_substr(
        target_text,
        extension=".pkl",
        file_substr="output_texts2",
        do_sample=args.do_sample,
        sample_size=args.sample_size,
    )
    log_or_print(f"target_text loaded with size: {len(target_text)}", logger)
    assert len(query_embed) == len(target_embed) == len(query_text) == len(target_text), f"Lengths of inputs do not match: {len(query_embed), len(target_embed), len(query_text), len(target_text)}"
    # labels = [1 if q==t else 0 for q,t in zip(query_text, target_text)]
    # assert set(labels) == {0, 1}, f"Labels not binary: {set(labels)}"
    # if logger:
    #     log_or_print(f"Labels: {Counter(labels)}")
    # else:
    #     print(f"Labels: {Counter(labels)}")
    set_seed(args.seed)
    train_dataloader, test_dataloader = create_data_loader(
        query_embed, target_embed, query_text, target_text, None
    )
    log_or_print(
        f"Dataset created with size: {len(train_dataloader.dataset), len(test_dataloader.dataset)}",
        logger,
    )
    objects = load_objects(model_type=args.contrastive_model)
    log_or_print("Objects loaded", logger)
    model_state_dict = train_loop(args, objects, train_dataloader, logger=logger)
    objects[1].load_state_dict(model_state_dict)
    log_or_print("Model trained", logger)
    eval_outputs, total_loss = eval_loop(args, objects, test_dataloader, logger=logger)
    log_or_print(f"Test loss: {total_loss}", logger)
    # fig_before, fig_after = evaluate_embeddings(eval_outputs)
    # fig_before.write_html(f"{args.extract_path}/repr_before.html")
    # fig_after.write_html(f"{args.extract_path}/repr_after.html")
    # fig = evaluate_embeddings(eval_outputs, labels)
    fig_3d, fig, df = evaluate_embeddings(eval_outputs)

    # fig_3d.write_html(f"{args.extract_path}/repr_plot_3d.html")
    # fig.write_html(f"{args.extract_path}/repr_plot_2d.html")
    # df.to_csv(f"{args.extract_path}/repr_df.csv", index=False)

    log_or_print("Model evaluated", logger)
    return eval_outputs


def load_inputs_tasks(args, logger=None):
    tasks, models = args.tasks.split(","), args.models.split(",")
    (
        query_embeds,
        target_embeds,
        query_texts,
        target_texts,
        tasks_sample,
        models_sample,
    ) = ([], [], [], [], [], [])
    for task in tasks:
        log_or_print(f"Data loading for task: {task}", logger)
        for model in models:
            log_or_print(f"Processing model: {model} for task: {task}", logger)
            # define path
            if task == "vqav2":
                # <root-path>/vqav2/intermediate/multiple_inputs_llava_vqav2_mscoco_test-dev2015/<model>
                intermediate_path = f"{args.root_path}/{task}/intermediate/multiple_inputs_llava_vqav2_mscoco_test-dev2015/{model}"
            else:
                intermediate_path = (
                    f"{args.root_path}/{task}/intermediate/multiple_inputs_{model}"
                )
            args.extract_path = f"{args.root_path}/extract/{task}/{model}"
            makedirs_recursive(args.extract_path)

            # load data
            query_embed = load_files_or_objects_w_substr(
                intermediate_path,
                extension=".pt.gz",
                file_substr="reprs1",
                do_sample=args.do_sample,
                sample_size=args.sample_size,
            )
            log_or_print(
                f"query_embed for {(task, model)} loaded with size: {len(query_embed), query_embed[0].shape}",
                logger,
            )
            target_embed = load_files_or_objects_w_substr(
                intermediate_path,
                extension=".pt.gz",
                file_substr="reprs2",
                do_sample=args.do_sample,
                sample_size=args.sample_size,
            )
            log_or_print(
                f"target_embed for {(task, model)} loaded with size: {len(target_embed), target_embed[0].shape}",
                logger,
            )
            query_text = load_files_or_objects_w_substr(
                intermediate_path,
                extension=".pkl",
                file_substr="output_texts1",
                do_sample=args.do_sample,
                sample_size=args.sample_size,
            )
            # log_or_print(
            #     f"query_text for {(task, model)} loaded with size: {len(query_text)}",
            #     logger,
            # )
            target_text = load_files_or_objects_w_substr(
                intermediate_path,
                extension=".pkl",
                file_substr="output_texts2",
                do_sample=args.do_sample,
                sample_size=args.sample_size,
            )
            # log_or_print(
            #     f"target_text for {(task, model)} loaded with size: {len(target_text)}",
            #     logger,
            # )
            # log_or_print(f"Data loading for {(task, model)} completed", logger)
            assert len(query_embed) == len(target_embed) == len(query_text) == len(
                target_text
            ), f"Lengths of inputs do not match: {len(query_embed), len(target_embed), len(query_text), len(target_text)}"
            # for qe, te, qt, tt in zip(query_embed, target_embed, query_text, target_text):
            for qe, te in zip(query_embed, target_embed):
                assert qe.shape == te.shape, f"Shapes do not match: {qe.shape, te.shape}"
                # assert len(qt) == len(tt), f"Lengths of texts do not match: \n========\n{qt}\n========\n{tt}"

            query_embeds += query_embed

            target_embeds += target_embed
            query_texts += query_text
            target_texts += target_text
            tasks_sample += [task] * len(query_embed)
            models_sample += [model] * len(query_embed)
        log_or_print(f"Data loading for {task} completed", logger)
    log_or_print(f"Data loadings completed", logger)
    return (
        query_embeds,
        target_embeds,
        query_texts,
        target_texts,
        tasks_sample,
        models_sample,
    )

def create_idx_wise_dict(input_dict, index_col="fixed_effect_idx", mapping=None):
    """
    Create a dictionary with indices as keys.
    """
    
    # Convert Tensors to lists and flatten the lists
    interm_dict = {}
    for key in input_dict:
        interm_dict[key] = [item.tolist() for sublist in input_dict[key] for item in sublist]

    # Convert it to a DataFrame
    df = pd.DataFrame(interm_dict)

    # Create a dictionary with groupby
    if mapping is None:
        output_dict = {f'{index_col}_{int(k)}': v.drop(index_col, axis=1).to_dict('list') for k, v in df.groupby(index_col)}
    else:
        mapping_flipped = {v: k for k, v in mapping.items()}
        output_dict = {"_".join(mapping_flipped[int(k)]): v.drop(index_col, axis=1).to_dict('list') for k, v in df.groupby(index_col)}

    return output_dict

def learn_repr_tasks(args, logger=None, is_category_wise_eval=False):
    log_or_print(f"Processing tasks {args.tasks}...", logger)
    query_embed, target_embed, query_text, target_text, tasks, models = (
        load_inputs_tasks(args, logger)
    )
    # log_or_print(f"tasks: {tasks}", logger)
    # log_or_print(f"models: {models}", logger)
    log_or_print(f"Data loading completed", logger)
    log_or_print(f"Sample query shape: {query_embed[0].shape}", logger)
    log_or_print(f"Sample target shape: {target_embed[0].shape}", logger)
    set_seed(args.seed)
    objects = load_objects(
        model_type=args.contrastive_model,
        tasks=tasks,
        models=models,
    )
    log_or_print("Objects loaded", logger)
    # log_or_print(f"Category indices: {objects[-2]}", logger)
    # log_or_print(f"Category map: {objects[-1]}", logger)
    train_dataloader, test_data = create_data_loader(
        query_embed, 
        target_embed, 
        query_text, 
        target_text, 
        fixed_effect_indices_and_mapping=[objects[-2], objects[-1]], 
        is_category_wise_eval=is_category_wise_eval,
        custom_collate=None,
        logger=logger,
    )
    # test_ds = ExtendedCustomDataset(tmp, list(set(objects[-1].values())))
    # log_or_print("Dataloader created", logger)
    log_or_print(
        f"Dataloader created with # of batches: {len(train_dataloader.dataset), len(test_data.dataset)}",
        logger,
    )
    model_state_dict = train_loop(args, objects, train_dataloader, logger=logger)
    # args.extract_path = f"{args.root_path}/contrastive_model"
    # log_or_print(f"Loading model from {args.extract_path}", logger)
    # model_state_dict = torch.load(f"{args.extract_path}/240705110100_mixed.pt")
    objects[1].load_state_dict(model_state_dict)#, map_location=torch.device('cpu')
    log_or_print("Model trained", logger)
    eval_outputs, total_loss = eval_loop(
        args, 
        objects, 
        test_data, 
        is_category_wise_eval=is_category_wise_eval, 
        logger=logger,
    )
    # log_or_print(f"eval_outputs: \n{eval_outputs}", logger)
    log_or_print(f"Test loss: {total_loss}", logger)
    idx_wise_outputs = create_idx_wise_dict(eval_outputs, mapping=objects[-1])
    # log_or_print(f"idx_wise_outputs: \n{idx_wise_outputs}", logger)
    log_or_print(f"Index-wise outputs created", logger)
    figs_dict = evaluate_embeddings_by_ctgry(idx_wise_outputs)
    export_path = f"{args.root_path}/export/{args.contrastive_model}"
    makedirs_recursive(export_path)
    for ky in figs_dict.keys():
        figs_dict[ky][0].write_html(f"{export_path}/repr_plot_3d_{ky}.html")
        figs_dict[ky][1].write_html(f"{export_path}/repr_plot_2d_{ky}.html")
        figs_dict[ky][2].to_csv(f"{export_path}/repr_df_{ky}.csv", index=False)
    # fig = evaluate_embeddings(eval_outputs, labels)
    # fig_3d, fig, df = evaluate_embeddings(eval_outputs)
    # # fig_3d.write_html(f"{args.extract_path}/repr_plot_3d.html")
    # # fig.write_html(f"{args.extract_path}/repr_plot_2d.html")
    # # df.to_csv(f"{args.extract_path}/repr_df.csv", index=False)

    log_or_print("Model evaluated", logger)

    return eval_outputs
