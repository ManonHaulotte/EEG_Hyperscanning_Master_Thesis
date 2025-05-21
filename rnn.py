# -*- coding: utf-8 -*-
"""
Created on Mon May 19 12:59:32 2025

@author: Moi
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import itertools

torch.manual_seed(1)


# === 1) Data Preparation ===

def prepare_sequence(merged_df):
    """
    Convert microstate sequences and metadata into:
    - sequences: dict of (dyad, trial) -> tensor of label indices
    - targets: dict of (dyad, trial) -> int label (0/1)
    - label_to_idx: mapping of combined labels to indices
    """
    merged_df['combined_label'] = (
        merged_df['sender_label'].fillna('nan') + '_' + merged_df['receiver_label'].fillna('nan')
    )
    vocab = sorted(merged_df['combined_label'].unique())
    label_to_idx = {label: idx for idx, label in enumerate(vocab)}
    merged_df['label_idx'] = merged_df['combined_label'].map(label_to_idx)

    sequences = {}
    targets = {}
    for (dyad, trial), group in merged_df.groupby(['dyad', 'trial']):
        group_sorted = group.sort_values('timestamp')
        idxs = group_sorted['label_idx'].values
        sequences[(dyad, trial)] = torch.tensor(idxs, dtype=torch.long)
        targets[(dyad, trial)] = group_sorted['result'].iloc[0]

    return sequences, targets, label_to_idx


def one_hot_encode_sequence(seq, vocab_size):
    """
    One-hot encode a sequence of label indices.
    Input: seq (LongTensor of shape [seq_len])
    Output: tensor of shape [seq_len, vocab_size]
    """
    one_hot = torch.zeros(seq.size(0), vocab_size)
    one_hot.scatter_(1, seq.unsqueeze(1), 1)
    return one_hot


def prepare_batch(sequences):
    """
    Prepare batch tensor of shape (max_seq_len, batch_size),
    padding shorter sequences with 0.
    Returns padded tensor and ordered keys list.
    """
    idx_seqs = []
    keys = []
    for key in sequences:
        idx_seqs.append(sequences[key])
        keys.append(key)

    padded_seqs = pad_sequence(idx_seqs, batch_first=False)  # (max_seq_len, batch_size)
    return padded_seqs, keys



def prepare_targets(targets, keys):
    """
    Convert target dict to tensor in order matching keys.
    """
    target_list = [targets[k] for k in keys]
    return torch.tensor(target_list, dtype=torch.long)


# === 2) Model Definition ===

class LSTM_attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers,
                 output_dim=2, bi_dir=True, n_head=4, drop_out=0.2, seed=1, device='cpu'):
        super(LSTM_attention, self).__init__()
        torch.manual_seed(seed)
        self.device = device
        self.num_directions = 2 if bi_dir else 1
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_dim, hidden_dim)

        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=bi_dir, dropout=drop_out)

        self.q = nn.Linear(hidden_dim * self.num_directions, hidden_dim * self.num_directions)
        self.k = nn.Linear(hidden_dim * self.num_directions, hidden_dim * self.num_directions)
        self.v = nn.Linear(hidden_dim * self.num_directions, hidden_dim * self.num_directions)

        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * self.num_directions,
            num_heads=n_head,
            dropout=drop_out,
            batch_first=True
        )

        self.norm = nn.LayerNorm(hidden_dim * self.num_directions)
        self.fc = nn.Linear(hidden_dim * self.num_directions, output_dim)

    def attention(self, lstm_hidden):
        query = self.q(lstm_hidden)
        key = self.k(lstm_hidden)
        value = self.v(lstm_hidden)
        attention_out, weights = self.multihead_attention(query, key, value)
        return attention_out, weights

    def forward(self, x):
        x = self.embedding(x)  # shape: (batch_size, seq_len, hidden_dim)
        batch_size = x.size(0)

        h0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim).to(self.device)

        lstm_out, _ = self.lstm(x, (h0, c0))  # (batch, seq_len, hidden_dim*num_directions)
        attention_out, weights = self.attention(lstm_out)
        norm_out = self.norm(attention_out + lstm_out)

        pooled = torch.mean(norm_out, dim=1)  # mean over time
        out = self.fc(pooled)
        return out, weights


# === 3) Training and Evaluation ===

def train(model, optimizer, criterion, train_loader, val_loader, device, epochs=20):
    model.to(device)
    best_val_acc = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output, _ = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = output.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

        train_acc = correct / total
        val_acc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/total:.4f} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

    print("Training complete. Best val acc:", best_val_acc)


def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            output, _ = model(batch_x)
            preds = output.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

    return correct / total


# === 4) Dataset and DataLoader Creation ===

def create_data_loaders(inputs, labels, batch_size=16, val_split=0.2):
    dataset = TensorDataset(inputs, labels)
    train_size = int(len(dataset) * (1 - val_split))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader


# === Execution Pipeline ===

def run(merged_df, device='cpu'):
    sequences, targets, label_to_idx = prepare_sequence(merged_df)

    inputs, keys = prepare_batch(sequences)
    labels = prepare_targets(targets, keys)
    vocab_size = len(label_to_idx)
    inputs, labels = inputs.to(device), labels.to(device)

    inputs = inputs.transpose(0, 1)  # (batch_size, seq_len)
    train_loader, val_loader = create_data_loaders(inputs, labels, batch_size=16, val_split=0.2)

    model = LSTM_attention(input_dim=vocab_size, hidden_dim=40, num_layers=1, output_dim=2, device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train(model, optimizer, criterion, train_loader, val_loader, device, epochs=20)

    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()


# === Optional - Hyperparameter Grid Search ===

def hyperparameter_grid_search(merged_df, device='cpu'):
    param_grid = {
    'hidden_dim': [20, 40, 60],
    'num_layers': [1, 2],
    'lr': [1e-3, 5e-4],
    'batch_size': [16],
    'drop_out': [0.1, 0.2, 0.3, 0.5],
    'n_head': [2, 4]
}

    sequences, targets, label_to_idx = prepare_sequence(merged_df)
    vocab_size = len(label_to_idx)
    
    inputs, keys = prepare_batch(sequences)
    labels = prepare_targets(targets, keys)
    inputs, labels = inputs.to(device), labels.to(device)
    
    inputs = inputs.transpose(0, 1)  # (batch_size, seq_len)

    best_val_acc = 0
    best_params = {}

    for hidden_dim, num_layers, lr, batch_size, drop_out, n_head in itertools.product(
        param_grid['hidden_dim'],
        param_grid['num_layers'],
        param_grid['lr'],
        param_grid['batch_size'],
        param_grid['drop_out'],
        param_grid['n_head']):
    
        print(f"Training with hidden_dim={hidden_dim}, num_layers={num_layers}, "
              f"lr={lr}, batch_size={batch_size}, drop_out={drop_out}, n_head={n_head}")
        train_loader, val_loader = create_data_loaders(inputs, labels, batch_size=batch_size)
    
        model = LSTM_attention(
            input_dim=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=2,
            drop_out=drop_out,
            n_head=n_head,
            device=device
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        train(model, optimizer, criterion, train_loader, val_loader, device, epochs=10)

        val_acc = evaluate(model, val_loader, device)
        print(f"Validation accuracy: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = {
                'hidden_dim': hidden_dim,
                'num_layers': num_layers,
                'lr': lr,
                'batch_size': batch_size,
                'drop_out': drop_out,
                'n_head': n_head
            }
            torch.save(model.state_dict(), 'best_model_grid.pth')

    print(f"Best params: {best_params}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
