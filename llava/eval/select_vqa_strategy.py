import argparse
from copy import deepcopy
import csv
import datetime
import glob
import gzip
import os
import math
import multiprocessing
import random

import numpy as np
import optuna
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, recall_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from tqdm.contrib import tenumerate
import wandb


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a model to predict labels from tensors."
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        required=True,
        help="Directory containing result CSV files.",
    )
    parser.add_argument(
        "--intermediate-dir",
        type=str,
        required=True,
        help="Directory containing tensor files.",
    )
    parser.add_argument("--task", type=str, required=True, help="Task name.")
    parser.add_argument("--prefix", type=str, default="", help="Prefix for file names.")
    parser.add_argument("--model", type=str, required=True, help="Model name.")
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for DataLoader."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="Learning rate for optimizer."
    )
    parser.add_argument(
        "--label-type",
        type=str,
        default="single_source",
        help="Labeling strategy.",
        choices=["intersection", "single_source", "both"],
    )
    parser.add_argument(
        "--wandb-name", type=str, default="vqa_strategy", help="WandB run name."
    )
    parser.add_argument(
        "--do-sample", action="store_true", help="Whether to sample data."
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of samples to use for training.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--probe-type",
        type=str,
        default="linear",
        choices=["linear", "cnn", "transformer"],
        help="Probe type.",
    )
    parser.add_argument("--deactivate-wandb", action="store_true", help="Deactivate WandB.")
    parser.add_argument("--loss-type", type=str, default="cross_entropy", choices=["cross_entropy", "focal"], help="Loss function type.")
    parser.add_argument("--use-optuna", action="store_true", help="Use Optuna for hyperparameter search.")
    parser.add_argument("--gamma", type=float, default=2.0, help="Gamma value for Focal Loss.")
    parser.add_argument("--probe-val-epoch", type=int, default=1, help="Log validation every N epochs.")
    parser.add_argument("--test-only", action="store_true", help="Only test the model without training.")
    parser.add_argument("--train-timestamp", type=str, default=None, help="Timestamp of the training run to load the model from.")
    parser.add_argument("--weight-loss", action="store_true", help="Use weighted loss.")
    parser.add_argument("--positive-weight", type=float, default=10.0, help="Weights for positive samples.")
    parser.add_argument("--confidence-interval", action="store_true",
                        help="Compute and print 95%% bootstrap confidence interval for accuracy metrics.")
    # --- Separate train/test data (USE_TRAIN_TO_SELECT mode) ---
    parser.add_argument("--train-tensor-dir", type=str, default=None,
                        help="Directory with training hidden states (overrides internal split).")
    parser.add_argument("--train-csv-file1", type=str, default=None,
                        help="Training correctness CSV (ICL model, i.e. prefix+model).")
    parser.add_argument("--train-csv-file2", type=str, default=None,
                        help="Training correctness CSV (ZSL model, i.e. model only).")
    parser.add_argument("--test-tensor-dir", type=str, default=None,
                        help="Directory with test hidden states (overrides internal split).")
    parser.add_argument("--test-csv-file1", type=str, default=None,
                        help="Test correctness CSV (ICL model).")
    parser.add_argument("--test-csv-file2", type=str, default=None,
                        help="Test correctness CSV (ZSL model).")
    args = parser.parse_args()
    return args


class TensorDataset(Dataset):
    def __init__(self, ids, labels, labels_ref, tensor_dir, probe_type="linear"):
        self.probe_type = probe_type
        self.labels = {}
        self.labels_ref = {} if labels_ref is not None else None
        self.tensor_dir = tensor_dir
        self.ids = []
        for sample_id in ids:
            tensor_pattern = os.path.join(
                self.tensor_dir, f"reprs1_*_{sample_id}.pt.gz"
            )
            files = glob.glob(tensor_pattern)
            if len(files) == 1:
                self.ids.append(sample_id)
                self.labels[sample_id] = labels[sample_id]
                if labels_ref is not None:
                    self.labels_ref[sample_id] = labels_ref[sample_id]
            elif len(files) == 0:
                pass
            else:
                # print(f"Multiple files found for ID {sample_id}. Using the first one.")
                self.ids.append(sample_id)
                self.labels[sample_id] = labels[sample_id]
                if labels_ref is not None:
                    self.labels_ref[sample_id] = labels_ref[sample_id]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        label = self.labels[sample_id]
        if self.labels_ref is not None:
            label_ref = self.labels_ref[sample_id]
        tensor = self.load_tensor(sample_id)
        return (
            tensor,
            torch.tensor(label, dtype=torch.long),
            (
                torch.tensor(label_ref, dtype=torch.long)
                if self.labels_ref is not None
                else -1 # -1 indicates no reference label
            ),
        )

    def load_tensor(self, sample_id):
        tensor_pattern = os.path.join(self.tensor_dir, f"reprs1_*_{sample_id}.pt.gz")
        files = glob.glob(tensor_pattern)
        file_path = files[0]
        with gzip.open(file_path, "rb") as f:
            tensor = torch.load(f)
        if self.probe_type == "cnn":
            tensor = tensor.view(1, 4096, 5120).float()  # Ensure tensor is float32
        elif self.probe_type == "transformer":
            tensor = tensor.view(5120, 4096).float()
        else:
            tensor = tensor.view(-1).float()  # Ensure tensor is float32
        return tensor


def tensor_exists(tensor_dir, sample_id):
    tensor_pattern = os.path.join(tensor_dir, f"reprs1_*_{sample_id}.pt.gz")
    files = glob.glob(tensor_pattern)
    return len(files) >= 1  # Returns True if tensor file exists


class SimpleClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleClassifier, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.fc(x)
        return out


class CNNClassifier(nn.Module):
    def __init__(self, num_token, depth, num_classes=2):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Compute the size after the two pooling layers
        h_out = num_token // 4  # Divide by 2 twice
        w_out = depth // 4

        # Ensure h_out and w_out are integers
        h_out = int(h_out)
        w_out = int(w_out)

        self.fc1 = nn.Linear(32 * h_out * w_out, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)  # Output size: [batch_size, 16, H/2, W/2]

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)  # Output size: [batch_size, 32, H/4, W/4]

        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)  # Outputs logits of shape [batch_size, num_classes]
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Positional Encoding module from "Attention is All You Need"
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Compute positional encodings once in log space
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(
            1
        )  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # [d_model // 2]
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(1)  # [max_len, 1, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: [seq_length, batch_size, d_model]
        seq_length = x.size(0)
        pe = self.pe[:seq_length, :]  # [seq_length, 1, d_model]
        x = x + pe
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_classes=2,
        num_layers=1,
        num_heads=8,
        dropout=0.1,
        num_tokens=5120,
    ):
        super(TransformerClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_tokens = num_tokens
        self.pos_encoder = PositionalEncoding(hidden_size, dropout, max_len=num_tokens)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=num_layers
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: [batch_size, seq_length, hidden_size]
        x = x.permute(1, 0, 2)  # Shape: [seq_length, batch_size, hidden_size]
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)  # Output: [seq_length, batch_size, hidden_size]
        # Use mean pooling over the sequence dimension
        x = x.mean(dim=0)  # Shape: [batch_size, hidden_size]
        x = self.dropout(x)
        x = self.fc(x)  # Shape: [batch_size, num_classes]
        return x


def load_original_labels(csv_file):
    labels = {}
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = row["id"]
            label = row["label"].strip()  # Keep the label as a string
            labels[sample_id] = label
    return labels


def split_data(ids, labels=None, labels_ref=None):
    if labels is None:
        random.shuffle(ids)
        total = len(ids)
        train_size = int(0.8 * total)
        eval_size = int(0.1 * total)
        train_ids = ids[:train_size]
        eval_ids = ids[train_size : train_size + eval_size]
        test_ids = ids[train_size + eval_size :]
    elif labels_ref is not None:
        # Separate ids based on labels
        positive_ids = [id for id in ids if labels[id] == 1 or labels_ref[id] == 1]
        negative_ids = [id for id in ids if labels[id] == 0 and labels_ref[id] == 0]

        # Shuffle each list independently
        random.shuffle(positive_ids)
        random.shuffle(negative_ids)

        # Calculate sizes for each set
        total_positive = len(positive_ids)
        total_negative = len(negative_ids)

        train_size_positive = int(0.8 * total_positive)
        eval_size_positive = int(0.1 * total_positive)

        train_size_negative = int(0.8 * total_negative)
        eval_size_negative = int(0.1 * total_negative)

        # Split each list into train, eval, and test sets
        train_ids = positive_ids[:train_size_positive] + negative_ids[:train_size_negative]
        eval_ids = positive_ids[train_size_positive:train_size_positive + eval_size_positive] + negative_ids[train_size_negative:train_size_negative + eval_size_negative]
        test_ids = positive_ids[train_size_positive + eval_size_positive:] + negative_ids[train_size_negative + eval_size_negative:]

        # Shuffle the combined sets to mix positive and negative samples
        random.shuffle(train_ids)
        random.shuffle(eval_ids)
        random.shuffle(test_ids)
    else:
        # Separate ids based on labels
        positive_ids = [id for id in ids if labels[id] == 1]
        negative_ids = [id for id in ids if labels[id] == 0]

        # Shuffle each list independently
        random.shuffle(positive_ids)
        random.shuffle(negative_ids)

        # Calculate sizes for each set
        total_positive = len(positive_ids)
        total_negative = len(negative_ids)

        train_size_positive = int(0.8 * total_positive)
        eval_size_positive = int(0.1 * total_positive)

        train_size_negative = int(0.8 * total_negative)
        eval_size_negative = int(0.1 * total_negative)

        # Split each list into train, eval, and test sets
        train_ids = positive_ids[:train_size_positive] + negative_ids[:train_size_negative]
        eval_ids = positive_ids[train_size_positive:train_size_positive + eval_size_positive] + negative_ids[train_size_negative:train_size_negative + eval_size_negative]
        test_ids = positive_ids[train_size_positive + eval_size_positive:] + negative_ids[train_size_negative + eval_size_negative:]

        # Shuffle the combined sets to mix positive and negative samples
        random.shuffle(train_ids)
        random.shuffle(eval_ids)
        random.shuffle(test_ids)

    return train_ids, eval_ids, test_ids

def train_model(
    model,
    train_loader,
    eval_loader,
    criterion,
    optimizer,
    epochs,
    train_path="best_model.pt",
    deactivate_wandb=False,
    val_every=1,
    plot_path=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_eval_loss = float("inf")
    model.to(device)
    val_every = max(1, val_every)
    train_losses = []
    eval_losses = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i, (tensors, labels, _) in tenumerate(
            train_loader, desc=f"Training Epoch {epoch + 1}"
        ):
            tensors, labels = tensors.to(device), labels.to(device)
            # labels_ref = labels_ref.to(device) if labels_ref is not None else None
            optimizer.zero_grad()
            outputs = model(tensors)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if not deactivate_wandb:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "batch": i + 1,
                        "loss": loss.item(),
                    }
                )
            else:
                print(
                    f"Epoch {epoch + 1}, Batch {i + 1}, Loss: {loss.item():.4f}"
                )
            # clear memory
            del tensors, labels
        avg_train_loss = total_loss / len(train_loader)
        avg_eval_loss = evaluate_model(model, eval_loader, criterion)
        train_losses.append(avg_train_loss)
        eval_losses.append(avg_eval_loss)
        if (epoch + 1) % val_every == 0 or (epoch + 1) == epochs:
            if not deactivate_wandb:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "train_loss": avg_train_loss,
                        "eval_loss": avg_eval_loss,
                    }
                )
            else:
                print(
                    f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Eval Loss: {avg_eval_loss:.4f}"
                )
        # Save the best model
        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            torch.save(model.state_dict(), train_path)
    # Load the best model
    model.load_state_dict(torch.load(train_path))
    if plot_path:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(1, len(train_losses) + 1)), y=train_losses, mode="lines", name="train_loss"))
        fig.add_trace(go.Scatter(x=list(range(1, len(eval_losses) + 1)), y=eval_losses, mode="lines+markers", name="val_loss"))
        fig.write_html(plot_path)


def train_models(
    model1,
    model2,
    train_loader,
    eval_loader,
    criterion,
    optimizer,
    epochs,
    train_pathes=["best_model1.pt", "best_model2.pt"],
    deactivate_wandb=False,
    val_every=1,
    plot_path=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_eval_loss1, best_eval_loss2 = float("inf"), float("inf")
    model1.to(device)
    model2.to(device)
    # copy initialized optimizer
    optimizer2 = optim.Adam(
        model2.parameters(), lr=optimizer.param_groups[0]["lr"]
    )
    val_every = max(1, val_every)
    train_losses1 = []
    train_losses2 = []
    eval_losses1 = []
    eval_losses2 = []
    for epoch in range(epochs):
        model1.train()
        model2.train()
        total_loss = 0
        total_loss2 = 0
        for i, (tensors, labels, labels_ref) in tenumerate(
            train_loader, desc=f"Training Epoch {epoch + 1}"
        ):
            tensors, labels = tensors.to(device), labels.to(device)
            labels_ref = labels_ref.to(device)
            # model1
            optimizer.zero_grad()
            outputs1 = model1(tensors)
            loss1 = criterion(outputs1, labels)
            loss1.backward()
            optimizer.step()
            total_loss += loss1.item()
            # model2
            optimizer2.zero_grad()
            outputs2 = model2(tensors)
            loss2 = criterion(outputs2, labels_ref)
            loss2.backward()
            optimizer2.step()
            total_loss2 += loss2.item()
            if not deactivate_wandb:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "batch": i + 1,
                        "loss_icl": loss1.item(),
                        "loss_zsl": loss2.item(),
                    }
                )
            else:
                print(
                    f"Epoch {epoch + 1}, Batch {i + 1}, Loss ICL: {loss1.item():.4f}, Loss ZSL: {loss2.item():.4f}"
                )
            # clear memory
            del tensors, labels, labels_ref
        avg_train_loss = total_loss / len(train_loader)
        avg_train_loss2 = total_loss2 / len(train_loader)
        avg_eval_loss1, avg_eval_loss2 = evaluate_models(
            model1, model2, eval_loader, criterion
        )
        train_losses1.append(avg_train_loss)
        train_losses2.append(avg_train_loss2)
        eval_losses1.append(avg_eval_loss1)
        eval_losses2.append(avg_eval_loss2)
        if (epoch + 1) % val_every == 0 or (epoch + 1) == epochs:
            if not deactivate_wandb:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "train_loss_icl": avg_train_loss,
                        "train_loss_zsl": avg_train_loss2,
                        "eval_loss_icl": avg_eval_loss1,
                        "eval_loss_zsl": avg_eval_loss2,
                    }
                )
            else:
                print(
                    f"Epoch {epoch + 1}, Train Loss ICL: {avg_train_loss:.4f}, Train Loss ZSL: {avg_train_loss2:.4f}, Eval Loss ICL: {avg_eval_loss1:.4f}, Eval Loss ZSL: {avg_eval_loss2:.4f}"
                )
        # Save the best model
        if avg_eval_loss1 < best_eval_loss1:
            best_eval_loss1 = avg_eval_loss1
            torch.save(model1.state_dict(), train_pathes[0])
        if avg_eval_loss2 < best_eval_loss2:
            best_eval_loss2 = avg_eval_loss2
            torch.save(model2.state_dict(), train_pathes[1])
    # Load the best model
    model1.load_state_dict(torch.load(train_pathes[0]))
    model2.load_state_dict(torch.load(train_pathes[1]))
    if plot_path:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(1, len(train_losses1) + 1)), y=train_losses1, mode="lines", name="train_loss_icl"))
        fig.add_trace(go.Scatter(x=list(range(1, len(eval_losses1) + 1)), y=eval_losses1, mode="lines+markers", name="val_loss_icl"))
        fig.add_trace(go.Scatter(x=list(range(1, len(train_losses2) + 1)), y=train_losses2, mode="lines", name="train_loss_zsl"))
        fig.add_trace(go.Scatter(x=list(range(1, len(eval_losses2) + 1)), y=eval_losses2, mode="lines+markers", name="val_loss_zsl"))
        fig.write_html(plot_path)


def evaluate_model(model, data_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for tensors, labels, _ in tqdm(data_loader, desc="Evaluating"):
            tensors, labels = tensors.to(device), labels.to(device)
            # labels_ref = labels_ref.to(device) if labels_ref is not None else None
            outputs = model(tensors)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            del tensors, labels, outputs
    avg_loss = total_loss / len(data_loader)
    return avg_loss


def evaluate_models(model1, model2, data_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model1.to(device)
    model1.eval()
    model2.to(device)
    model2.eval()
    total_loss1 = 0
    total_loss2 = 0
    with torch.no_grad():
        for tensors, labels, labels_ref in tqdm(data_loader, desc="Evaluating"):
            tensors, labels = tensors.to(device), labels.to(device)
            labels_ref = labels_ref.to(device)
            outputs1 = model1(tensors)
            outputs2 = model2(tensors)
            loss1 = criterion(outputs1, labels)
            loss2 = criterion(outputs2, labels_ref)
            total_loss1 += loss1.item()
            total_loss2 += loss2.item()
            del tensors, labels, labels_ref, outputs1, outputs2
    avg_loss1 = total_loss1 / len(data_loader)
    avg_loss2 = total_loss2 / len(data_loader)
    return avg_loss1, avg_loss2


def calculate_task_accuracy(all_labels, all_preds, all_labels_ref):
    task_success = 0
    total_cases = len(all_preds)
    for i in range(total_cases):
        if all_labels[i] == 1 and all_preds[i] == 1:
            task_success += 1
        elif all_labels_ref and all_labels_ref[i] == 1 and all_preds[i] == 0:
            task_success += 1
        else:
            task_success += 0

    task_accuracy = task_success / total_cases if total_cases > 0 else 0
    return task_accuracy


def test_model(model, test_loader, deactivate_wandb=False, ts=None, args=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    all_labels_ref = []
    with torch.no_grad():
        for tensors, labels, labels_ref in test_loader:
            tensors, labels, labels_ref = (
                tensors.to(device),
                labels.to(device),
                labels_ref.to(device) if labels_ref is not None else None,
            )
            outputs = model(tensors)
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.cpu().numpy()
            labels = labels.cpu().numpy()
            all_preds.extend(predicted.tolist())
            all_labels.extend(labels.tolist())
            if labels_ref is not None:
                labels_ref = labels_ref.cpu().numpy()
                all_labels_ref.extend(labels_ref.tolist())
            # clear memory
            del tensors, labels, predicted, outputs
            if labels_ref is not None:
                del labels_ref

    accuracy = accuracy_score(all_labels, all_preds)

    # Calculate task_accuracy
    task_accuracy = calculate_task_accuracy(all_labels, all_preds, all_labels_ref)

    # Log the results
    if not deactivate_wandb:
        wandb.log(
            {
                "test_accuracy": accuracy,
                "task_accuracy": task_accuracy,
            }
        )
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Task Accuracy: {task_accuracy:.4f}")

    # Optionally compute and print confidence intervals
    if args is not None and getattr(args, "confidence_interval", False):
        try:
            from llava.eval.utils import bootstrap_confidence_interval
        except ImportError:
            from utils import bootstrap_confidence_interval
        # Per-sample accuracy scores (1 if correct, 0 otherwise)
        per_sample_acc = [1.0 if p == l else 0.0 for p, l in zip(all_preds, all_labels)]
        acc_lo, acc_hi = bootstrap_confidence_interval(per_sample_acc)
        print(f"Test Accuracy 95% CI: [{acc_lo:.4f}, {acc_hi:.4f}]")
        # Per-sample task accuracy scores
        per_sample_task = []
        for i in range(len(all_preds)):
            if all_labels[i] == 1 and all_preds[i] == 1:
                per_sample_task.append(1.0)
            elif all_labels_ref and all_labels_ref[i] == 1 and all_preds[i] == 0:
                per_sample_task.append(1.0)
            else:
                per_sample_task.append(0.0)
        task_lo, task_hi = bootstrap_confidence_interval(per_sample_task)
        print(f"Task Accuracy 95% CI: [{task_lo:.4f}, {task_hi:.4f}]")


def calculate_dual_accuracy(all_labels, all_labels_ref, all_preds1, all_preds2):
    task_success = 0
    model_success = 0
    total_cases = len(all_preds1)
    for i in range(total_cases):
        if all_labels[i] == 1 and all_preds1[i] == 1:
            task_success += 1
            model_success += 1
        elif all_labels_ref[i] == 1 and all_preds2[i] == 1:
            task_success += 1
            model_success += 1
        elif all_labels[i] == 0 and all_preds1[i] == 0:
            model_success += 1
        else:
            pass

    task_accuracy = task_success / total_cases if total_cases > 0 else 0
    model_accuracy = model_success / total_cases if total_cases > 0 else 0
    return task_accuracy, model_accuracy


def test_models(model1, model2, test_loader, deactivate_wandb=False, ts=None, args=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model1.to(device)
    model1.eval()
    model2.to(device)
    model2.eval()
    all_preds1 = []
    all_preds2 = []
    all_labels = []
    all_labels_ref = []
    with torch.no_grad():
        for tensors, labels, labels_ref in test_loader:
            tensors, labels, labels_ref = (
                tensors.to(device),
                labels.to(device),
                labels_ref.to(device),
            )
            outputs1, outputs2 = model1(tensors), model2(tensors)
            _, predicted1 = torch.max(outputs1.data, 1)
            _, predicted2 = torch.max(outputs2.data, 1)
            predicted1 = predicted1.cpu().numpy()
            predicted2 = predicted2.cpu().numpy()
            labels = labels.cpu().numpy()
            labels_ref = labels_ref.cpu().numpy()
                
            all_preds1.extend(predicted1.tolist())
            all_preds2.extend(predicted2.tolist())
            all_labels.extend(labels.tolist())
            all_labels_ref.extend(labels_ref.tolist())
            # clear memory
            del tensors, labels, predicted1, predicted2, outputs1, outputs2, labels_ref

    accuracy = accuracy_score(all_labels, all_preds1)
    accuracy_ref = accuracy_score(all_labels_ref, all_preds2)

    # if ts is not None, save the predictions and labels to one csv
    if ts is not None:
        with open(os.path.join(args.result_dir, args.task, f"predictions_{ts}.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "label", "label_ref", "pred1", "pred2"])
            for i in range(len(all_labels)):
                writer.writerow([i, all_labels[i], all_labels_ref[i], all_preds1[i], all_preds2[i]])
    # Calculate task_accuracy
    task_accuracy, model_accuracy = calculate_dual_accuracy(
        all_labels, all_labels_ref, all_preds1, all_preds2
    )

    # Log the results
    if not deactivate_wandb:
        wandb.log(
            {
                "test_accuracy": accuracy,
                "test_accuracy_ref": accuracy_ref,
                "model_accuracy": model_accuracy,
                "task_accuracy": task_accuracy,
            }
        )
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Accuracy (ref): {accuracy_ref:.4f}")
    print(f"Model Accuracy: {model_accuracy:.4f}")
    print(f"Task Accuracy: {task_accuracy:.4f}")

    # Optionally compute and print confidence intervals
    if args is not None and getattr(args, "confidence_interval", False):
        try:
            from llava.eval.utils import bootstrap_confidence_interval
        except ImportError:
            from utils import bootstrap_confidence_interval
        # Per-sample accuracy scores
        per_sample_acc = [1.0 if p == l else 0.0 for p, l in zip(all_preds1, all_labels)]
        acc_lo, acc_hi = bootstrap_confidence_interval(per_sample_acc)
        print(f"Test Accuracy 95% CI: [{acc_lo:.4f}, {acc_hi:.4f}]")
        per_sample_acc_ref = [1.0 if p == l else 0.0 for p, l in zip(all_preds2, all_labels_ref)]
        acc_ref_lo, acc_ref_hi = bootstrap_confidence_interval(per_sample_acc_ref)
        print(f"Test Accuracy (ref) 95% CI: [{acc_ref_lo:.4f}, {acc_ref_hi:.4f}]")
        # Per-sample model accuracy scores
        per_sample_model = []
        for i in range(len(all_preds1)):
            if all_labels[i] == 1 and all_preds1[i] == 1:
                per_sample_model.append(1.0)
            elif all_labels_ref[i] == 1 and all_preds2[i] == 1:
                per_sample_model.append(1.0)
            elif all_labels[i] == 0 and all_preds1[i] == 0:
                per_sample_model.append(1.0)
            else:
                per_sample_model.append(0.0)
        model_lo, model_hi = bootstrap_confidence_interval(per_sample_model)
        print(f"Model Accuracy 95% CI: [{model_lo:.4f}, {model_hi:.4f}]")
        # Per-sample task accuracy scores
        per_sample_task = []
        for i in range(len(all_preds1)):
            if all_labels[i] == 1 and all_preds1[i] == 1:
                per_sample_task.append(1.0)
            elif all_labels_ref[i] == 1 and all_preds2[i] == 1:
                per_sample_task.append(1.0)
            else:
                per_sample_task.append(0.0)
        task_lo, task_hi = bootstrap_confidence_interval(per_sample_task)
        print(f"Task Accuracy 95% CI: [{task_lo:.4f}, {task_hi:.4f}]")


def label_intersection(labels1, labels2, tensor_dir):
    # Initialize a dictionary to store the mismatched records and assigned labels
    mismatched_labels = {}
    sample_ids = labels1.keys() & labels2.keys()
    for sample_id in tqdm(sample_ids, desc="Labeling records"):
        if not tensor_exists(tensor_dir, sample_id):
            continue
        label1 = labels1[sample_id]
        label2 = labels2[sample_id]
        if label1 != label2:
            if label2 == "Incorrect" and label1 == "Correct":
                mismatched_labels[sample_id] = 1  # Positive label
            elif label2 == "Correct" and label1 == "Incorrect":
                mismatched_labels[sample_id] = 0  # Negative label
            else:
                print(
                    f"Unexpected label combination for ID {sample_id}: ({label2}, {label1})"
                )
        else:
            # Labels match; exclude this ID
            continue
    return mismatched_labels


def label_by_single_source(labels, labels_ref, tensor_dir):
    # Initialize a dictionary to store the mismatched records and assigned labels
    binary_labels, binary_labels_ref = {}, {}
    sample_ids = labels.keys()
    for sample_id in tqdm(sample_ids, desc="Labeling records"):
        if not tensor_exists(tensor_dir, sample_id):
            continue
        label = labels[sample_id]
        label_ref = labels_ref[sample_id]
        assert label in ["Correct", "Incorrect"], f"Unexpected label: {label}"
        binary_label = 1 if label == "Correct" else 0
        binary_label_ref = 1 if label_ref == "Correct" else 0
        binary_labels[sample_id] = binary_label
        binary_labels_ref[sample_id] = binary_label_ref
    return binary_labels, binary_labels_ref


def label_data(
    labels1, labels2, label_type="intersection", tensor_dir=None, sample_num=None
):
    if label_type == "intersection":
        labels = label_intersection(labels1, labels2, tensor_dir)
        labels_ref = None
    elif label_type in ["single_source", "both"]:
        labels, labels_ref = label_by_single_source(labels1, labels2, tensor_dir)
    else:
        raise ValueError("Invalid label type. Use 'intersection' or 'single_source'.")
    if label_type in ["single_source", "both"] and sample_num is not None:
        # Sample a subset of the labels
        sample_ids = random.sample(list(labels.keys()), sample_num)
        labels = {k: labels[k] for k in sample_ids}
        labels_ref = {k: labels_ref[k] for k in sample_ids}
    return labels, labels_ref


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean', weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight

    def forward(self, inputs, targets):
        logp = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        p = torch.exp(-logp)
        loss = self.alpha * (1 - p) ** self.gamma * logp
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def focal_objective(trial, model, train_loader, val_loader, num_epochs=5, args=None, alpha=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Suggest hyperparameters
    if alpha is None:
        alpha = trial.suggest_float('alpha', 0.0, 5.0)
    gamma = trial.suggest_float('gamma', 0.0, 5.0)
    # Initialize wandb in each trial if not deactivated
    if not args.deactivate_wandb:
        wandb.init(
            project=args.wandb_name,
            name=f'trial_{trial.number}',
            config={'alpha': alpha, 'gamma': gamma, 'trial_number': trial.number},
            reinit=True,
            mode='offline',
        )
    # Initialize model, optimizer, criterion
    opt_model = deepcopy(model)
    opt_model.to(device)
    optimizer = optim.Adam(opt_model.parameters(), lr=0.001)
    criterion = FocalLoss(alpha=alpha, gamma=gamma)
    # Training loop
    for epoch in range(num_epochs):
        opt_model.train()
        for tensors, labels, labels_ref in train_loader:
            labels_or = torch.maximum(labels, labels_ref)
            tensors, labels_or = tensors.to(device), labels_or.to(device)
            optimizer.zero_grad()
            outputs = opt_model(tensors)
            loss = criterion(outputs, labels_or)
            loss.backward()
            optimizer.step()
            del tensors, labels_or, labels_ref, outputs
        # Optionally log training metrics per epoch
        if not args.deactivate_wandb:
            wandb.log({'epoch': epoch, 'loss': loss.item()})
    # Evaluation
    opt_model.eval()
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        for tensors, labels, labels_ref in val_loader:
            labels_or = torch.maximum(labels, labels_ref)
            tensors, labels_or = tensors.to(device), labels_or.to(device)
            outputs = opt_model(tensors)
            probabilities = torch.softmax(outputs, dim=1)[:, 1]
            predictions = (probabilities >= 0.5).long()
            all_targets.extend(labels_or.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            del tensors, labels_or, labels_ref, outputs, probabilities
    recall = recall_score(all_targets, all_predictions)
    # Log metrics and finish wandb run
    if not args.deactivate_wandb:
        wandb.log({'recall': recall})
        wandb.finish()
    # Return negative recall for minimization
    return -recall


def optimize_hyperparameters(model, train_loader, val_loader, storage_name="sqlite:///focal.db", args=None):
    study = optuna.create_study(
        direction='minimize',
        storage=storage_name,
        study_name='focal_loss_optimization',
        load_if_exists=True
    )
    study.optimize(
        lambda trial: focal_objective(trial, model, train_loader, val_loader, args=args),
        n_trials=10,
        n_jobs=4,  # Number of parallel jobs
    )
    # Get best hyperparameters
    best_hyperparams = study.best_params
    best_recall = -study.best_value
    return best_hyperparams, best_recall


def main():
    """
    # todo
    - maximize task accuracy or recall
    - unembed the difference into vocab
    - isolating fixed negative components and random positive components
    """
    args = parse_args()
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if not args.test_only:
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")# add current time as prefix to the model name
        if not args.deactivate_wandb:
            # initialize wandb
            ## finish if the previous run is still running
            wandb.finish()
            ## start a new run
            wandb.init(
                project=args.wandb_name,
                name=f"{args.prefix}{args.model}_{args.label_type}",
                config={
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                    "label_type": args.label_type,
                    "task": args.task,
                    "prefix": args.prefix,
                    "model": args.model,
                    "probe_type": args.probe_type,
                    "loss_type": args.loss_type,
                    "weight_loss": args.weight_loss,
                    "use_optuna": args.use_optuna,
                    "model_prefix": f"{ts}_{args.prefix}{args.model}"
                },
            )
        use_separate_train = args.train_tensor_dir is not None

        if use_separate_train:
            # --- USE_TRAIN_TO_SELECT mode: separate train & test data ---
            print("Using separate training data for probe (USE_TRAIN_TO_SELECT mode)")
            # Load training labels
            train_labels1 = load_original_labels(args.train_csv_file1)
            train_labels2 = load_original_labels(args.train_csv_file2)
            train_labels, train_labels_ref = label_data(
                train_labels1,
                train_labels2,
                label_type=args.label_type,
                tensor_dir=args.train_tensor_dir,
                sample_num=args.num_samples if args.do_sample else None,
            )
            train_all_ids = list(train_labels.keys())
            if len(train_all_ids) == 0:
                print("No records found in training data.")
                return
            print(f"Found {len(train_all_ids)} training records.")
            # Create full training dataset (filters to IDs with tensors)
            train_full_dataset = TensorDataset(
                train_all_ids, train_labels, train_labels_ref, args.train_tensor_dir, probe_type=args.probe_type
            )
            train_filtered_ids = train_full_dataset.ids
            # Split training data 90/10 into train/eval
            random.shuffle(train_filtered_ids)
            split_point = int(0.9 * len(train_filtered_ids))
            train_ids = train_filtered_ids[:split_point]
            eval_ids = train_filtered_ids[split_point:]
            train_dataset = TensorDataset(
                train_ids, train_labels, train_labels_ref, args.train_tensor_dir, probe_type=args.probe_type
            )
            eval_dataset = TensorDataset(
                eval_ids, train_labels, train_labels_ref, args.train_tensor_dir, probe_type=args.probe_type
            )
            # Load test labels (full eval split)
            test_csv1 = args.test_csv_file1 or os.path.join(
                args.result_dir, args.task, f"{args.prefix}{args.model}.csv"
            )
            test_csv2 = args.test_csv_file2 or os.path.join(
                args.result_dir, args.task, f"{args.model}.csv"
            )
            test_tensor_dir = args.test_tensor_dir or os.path.join(
                args.intermediate_dir, "eval", args.task, "intermediate",
                f"{args.prefix}{args.model}",
            )
            test_labels1 = load_original_labels(test_csv1)
            test_labels2 = load_original_labels(test_csv2)
            test_labels, test_labels_ref = label_data(
                test_labels1,
                test_labels2,
                label_type=args.label_type,
                tensor_dir=test_tensor_dir,
                sample_num=None,  # Use ALL eval data for testing
            )
            test_all_ids = list(test_labels.keys())
            test_dataset = TensorDataset(
                test_all_ids, test_labels, test_labels_ref, test_tensor_dir, probe_type=args.probe_type
            )
            print(f"Test set: {len(test_dataset)} records (full eval split)")
            # For model initialization, use train dataset
            tensor_dir = args.train_tensor_dir
            filtered_ids = train_filtered_ids
            labels = train_labels
            labels_ref = train_labels_ref
        else:
            # --- Original mode: single data source with 80/10/10 split ---
            # Paths for the CSV files
            csv_file1 = os.path.join(
                args.result_dir, args.task, f"{args.prefix}{args.model}.csv"
            )
            csv_file2 = os.path.join(args.result_dir, args.task, f"{args.model}.csv")
            # Load labels from both CSV files as string labels
            labels1 = load_original_labels(csv_file1)
            labels2 = load_original_labels(csv_file2)
            # Paths for the tensor files
            tensor_dir = os.path.join(
                args.intermediate_dir,
                "eval",
                args.task,
                "intermediate",
                f"{args.prefix}{args.model}",
            )
            # Label the data based on the intersection of labels / single source
            labels, labels_ref = label_data(
                labels1,
                labels2,
                label_type=args.label_type,
                tensor_dir=tensor_dir,
                sample_num=args.num_samples if args.do_sample else None,
            )
            # Use only the mismatched IDs
            all_ids = list(labels.keys())
            if len(all_ids) == 0:
                print("No mismatched records found between the two CSV files.")
                return
            print(f"Found {len(all_ids)} records.")
            # Create dataset using mismatched labels
            dataset = TensorDataset(
                all_ids, labels, labels_ref, tensor_dir, probe_type=args.probe_type
            )
            filtered_ids = dataset.ids

            # Split IDs into train, eval, and test sets
            train_ids, eval_ids, test_ids = split_data(filtered_ids, labels=labels)
            # Create datasets and loaders
            train_dataset = TensorDataset(
                train_ids, labels, labels_ref, tensor_dir, probe_type=args.probe_type
            )
            eval_dataset = TensorDataset(
                eval_ids, labels, labels_ref, tensor_dir, probe_type=args.probe_type
            )
            test_dataset = TensorDataset(
                test_ids, labels, labels_ref, tensor_dir, probe_type=args.probe_type
            )
        num_positive_train = sum(
            1 for label in train_dataset.labels.values() if label == 1
        )
        num_positive_eval = sum(
            1 for label in eval_dataset.labels.values() if label == 1
        )
        num_positive_test = sum(
            1 for label in test_dataset.labels.values() if label == 1
        )
        # record labels_ref count if available
        if labels_ref is not None:
            num_positive_train_ref = sum(
                1 for label in train_dataset.labels_ref.values() if label == 1
            )
            num_positive_eval_ref = sum(
                1 for label in eval_dataset.labels_ref.values() if label == 1
            )
            num_positive_test_ref = sum(
                1 for label in test_dataset.labels_ref.values() if label == 1
            )
        if not args.deactivate_wandb:
            wandb.log(
                {
                    "num_train_samples": len(train_dataset),
                    "num_eval_samples": len(eval_dataset),
                    "num_test_samples": len(test_dataset),
                    "num_positive_train": num_positive_train,
                    "num_positive_eval": num_positive_eval,
                    "num_positive_test": num_positive_test,
                }
            )
            if labels_ref is not None:
                wandb.log(
                    {
                        "num_positive_train_ref": num_positive_train_ref,
                        "num_positive_eval_ref": num_positive_eval_ref,
                        "num_positive_test_ref": num_positive_test_ref,
                    }
                )
        else:
            print(
                f"Number of training samples: {len(train_dataset)}\n"
                f"Number of evaluation samples: {len(eval_dataset)}\n"
                f"Number of test samples: {len(test_dataset)}\n"
                f"Number of positive training samples: {num_positive_train}\n"
                f"Number of positive evaluation samples: {num_positive_eval}\n"
                f"Number of positive test samples: {num_positive_test}"
            )
            if labels_ref is not None:
                print(
                    f"Number of positive training samples (ref): {num_positive_train_ref}\n"
                    f"Number of positive evaluation samples (ref): {num_positive_eval_ref}\n"
                    f"Number of positive test samples (ref): {num_positive_test_ref}"
                )

        # Determine input size using an example tensor
        example_id = filtered_ids[0]
        example_tensor = train_dataset.load_tensor(example_id)
        # Initialize the model
        if args.probe_type == "cnn":
            num_token, depth = example_tensor.shape[-2], example_tensor.shape[-1]
            model = CNNClassifier(num_token, depth, num_classes=2)
        elif args.probe_type == "transformer":
            num_token, depth = example_tensor.shape[-2], example_tensor.shape[-1]
            model = TransformerClassifier(
                hidden_size=depth,
                num_classes=2,
                num_layers=1,
                num_heads=8,
                dropout=0.1,
                num_tokens=num_token,
            )
        else:
            input_size = example_tensor.numel()
            model = SimpleClassifier(input_size=input_size, num_classes=2)
        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        print(
            "DataLoaders created with batch size:",
            args.batch_size,
            "num_workers:",
            args.num_workers,
        )
        # Define the criterion and optimizer
        if args.weight_loss:
            if args.deactivate_wandb:
                print(f"Using class weights: {args.positive_weight}")
            else:
                wandb.log({"positive_weight": args.positive_weight})
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            class_weights = torch.tensor(
                [1.0, args.positive_weight], device=device
            )
        else:
            class_weights = None
        if args.loss_type == "focal":
            if args.use_optuna:
                # optimize hyperparameters
                best_hyperparams, best_recall = optimize_hyperparameters(
                    model, train_loader, eval_loader, args=args
                )
                criterion = FocalLoss(
                    alpha=best_hyperparams["alpha"],
                    gamma=best_hyperparams["gamma"],
                )
                if args.deactivate_wandb:
                    print(f"Best hyperparameters: {best_hyperparams}")
                    print(f"Best recall: {best_recall}")
                else:
                    wandb.log(
                        {
                            "best_hyperparams": best_hyperparams,
                            "best_recall": best_recall,
                        }
                    )
            else:
                # Use default hyperparameters
                criterion = FocalLoss(alpha=1.0, gamma=args.gamma, weight=class_weights)
                if args.deactivate_wandb:
                    print(f"Using default hyperparameters: alpha=1.0, gamma={args.gamma}")
                else:
                    wandb.log(
                        {
                            "alpha": 1.0,
                            "gamma": args.gamma,
                        }
                    )
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        # Train the model
        save_path = os.path.join(
            args.result_dir,
            args.task,
            f"{ts}_{args.prefix}{args.model}_trained.pt",
        )
        loss_plot_path = os.path.join(
            args.result_dir,
            args.task,
            f"{ts}_{args.prefix}{args.model}_loss_plot.html",
        )
        if args.label_type == "both":
            # copy model1 / prep path2
            model2 = deepcopy(model)
            save_path2 = os.path.join(
                args.result_dir,
                args.task,
                f"{ts}_{args.prefix}{args.model}_trained_ref.pt",
            )
            loss_plot_path = os.path.join(
                args.result_dir,
                args.task,
                f"{ts}_{args.prefix}{args.model}_loss_plot.html",
            )
            # Train both models
            train_models(
                model,
                model2,
                train_loader,
                eval_loader,
                criterion,
                optimizer,
                args.epochs,
                train_pathes=[save_path, save_path2],
                deactivate_wandb=args.deactivate_wandb,
                val_every=args.probe_val_epoch,
                plot_path=loss_plot_path,
            )
        else:
            train_model(
                model,
                train_loader,
                eval_loader,
                criterion,
                optimizer,
                args.epochs,
                train_path=save_path,
                deactivate_wandb=args.deactivate_wandb,
                val_every=args.probe_val_epoch,
                plot_path=loss_plot_path,
            )
        print("Model training completed.")
        # Save the model
        if not args.label_type == "both":
            torch.save(model.state_dict(), save_path)
            if not args.deactivate_wandb:
                wandb.save(save_path)
                wandb.log({"model_path": save_path})
        else:
            # Save both models
            model2.load_state_dict(torch.load(save_path2))
            torch.save(model2.state_dict(), save_path2)
            if not args.deactivate_wandb:
                wandb.save(save_path2)
                wandb.log({"model_path_ref": save_path2})
        # Test the model
        if args.label_type == "both":
            test_models(
                model,
                model2,
                test_loader,
                deactivate_wandb=args.deactivate_wandb,
            )
        else:
            test_model(model, test_loader, deactivate_wandb=args.deactivate_wandb)
        if not args.deactivate_wandb:
            wandb.finish()
        print("Model testing completed.")
    else:
        # Load the test data
        use_separate_test = args.test_tensor_dir is not None

        if use_separate_test:
            # --- USE_TRAIN_TO_SELECT mode: test on full eval split ---
            print("Using separate test data for probe (USE_TRAIN_TO_SELECT mode)")
            test_csv1 = args.test_csv_file1 or os.path.join(
                args.result_dir, args.task, f"{args.prefix}{args.model}.csv"
            )
            test_csv2 = args.test_csv_file2 or os.path.join(
                args.result_dir, args.task, f"{args.model}.csv"
            )
            tensor_dir = args.test_tensor_dir
            labels1 = load_original_labels(test_csv1)
            labels2 = load_original_labels(test_csv2)
            labels, labels_ref = label_data(
                labels1,
                labels2,
                label_type=args.label_type,
                tensor_dir=tensor_dir,
                sample_num=None,  # Use ALL eval data
            )
            all_ids = list(labels.keys())
            test_dataset = TensorDataset(
                all_ids, labels, labels_ref, tensor_dir, probe_type=args.probe_type
            )
            filtered_ids = test_dataset.ids
            print(f"Test set: {len(test_dataset)} records (full eval split)")
        else:
            # --- Original mode: split and use 10% test set ---
            # Paths for the CSV files
            csv_file1 = os.path.join(
                args.result_dir, args.task, f"{args.prefix}{args.model}.csv"
            )
            csv_file2 = os.path.join(args.result_dir, args.task, f"{args.model}.csv")
            # Load labels from both CSV files as string labels
            labels1 = load_original_labels(csv_file1)
            labels2 = load_original_labels(csv_file2)
            # Paths for the tensor files
            tensor_dir = os.path.join(
                args.intermediate_dir,
                "eval",
                args.task,
                "intermediate",
                f"{args.prefix}{args.model}",
            )
            # Label the data based on the intersection of labels / single source
            labels, labels_ref = label_data(
                labels1,
                labels2,
                label_type=args.label_type,
                tensor_dir=tensor_dir,
                sample_num=args.num_samples if args.do_sample else None,
            )
            print(f"Data labeled with {args.label_type} method.")
            # Use only the mismatched IDs
            all_ids = list(labels.keys())
            # Create dataset using mismatched labels
            dataset = TensorDataset(
                all_ids, labels, labels_ref, tensor_dir, probe_type=args.probe_type
            )
            filtered_ids = dataset.ids
            print(f"Dataset created with {len(filtered_ids)} records.")

            # Split IDs into train, eval, and test sets
            _, _, test_ids = split_data(filtered_ids, labels=labels)
            # Create datasets and loaders
            test_dataset = TensorDataset(
                test_ids, labels, labels_ref, tensor_dir, probe_type=args.probe_type
            )
        # Create DataLoaders
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        print(
            "DataLoader created with batch size:",
            args.batch_size,
            "num_workers:",
            args.num_workers,
        )

        # Determine input size using an example tensor
        example_id = filtered_ids[0]
        example_tensor = test_dataset.load_tensor(example_id)
        # Load the model
        ts = args.train_timestamp
        save_path = os.path.join(
            args.result_dir,
            args.task,
            f"{ts}_{args.prefix}{args.model}_trained.pt",
        )
        if args.label_type == "both":
            save_path2 = os.path.join(
                args.result_dir,
                args.task,
                f"{ts}_{args.prefix}{args.model}_trained_ref.pt",
            )
        if args.probe_type == "cnn":
            num_token, depth = example_tensor.shape[-2], example_tensor.shape[-1]
            model = CNNClassifier(num_token, depth, num_classes=2)
        elif args.probe_type == "transformer":
            num_token, depth = example_tensor.shape[-2], example_tensor.shape[-1]
            model = TransformerClassifier(
                hidden_size=depth,
                num_classes=2,
                num_layers=1,
                num_heads=8,
                dropout=0.1,
                num_tokens=num_token,
            )
        else:
            input_size = example_tensor.numel()
            model = SimpleClassifier(input_size=input_size, num_classes=2)
        model.load_state_dict(torch.load(save_path))
        if args.label_type == "both":
            model2 = deepcopy(model)
            model2.load_state_dict(torch.load(save_path2))
        else:
            pass
        print("Model loaded from:", save_path)
        # Test the model
        if args.label_type == "both":
            test_models(
                model,
                model2,
                test_loader,
                deactivate_wandb=True,
                ts=ts,
                args=args,
            )
        else:
            test_model(model, test_loader, deactivate_wandb=True, ts=ts, args=args)
        print("Model testing completed.")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
