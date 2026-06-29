"""IndoBERT fine-tuning pipeline for short answer grading regression.

Implements a robust regression training loop with support for:
    - Layer-wise Learning Rate Decay (LLRD)
    - Gradual Unfreezing of bottom encoder layers
    - Multi-Sample Dropout
    - R-Drop consistency regularization
    - FP16 Mixed Precision training (for T4 GPU optimization)
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup


def seed_everything(seed: int):
    """Ensure reproducibility by seeding all random number generators."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class PairDataset(Dataset):
    """PyTorch Dataset for sentence pair scoring.

    Tokenizes reference_answer and student_answer as a sentence pair.
    """

    def __init__(self, text_a: list, text_b: list, scores: list, tokenizer, max_len: int = 128):
        self.text_a = text_a
        self.text_b = text_b
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.text_a[idx],
            self.text_b[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "score": torch.tensor(self.scores[idx], dtype=torch.float),
        }


class BertRegressorModule(nn.Module):
    """IndoBERT Regression model with Multi-Sample Dropout."""

    def __init__(self, model_name: str, dropout_rate: float = 0.1, num_dropouts: int = 5):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.num_dropouts = num_dropouts
        
        if num_dropouts > 1:
            self.dropouts = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(num_dropouts)])
        else:
            self.drop = nn.Dropout(dropout_rate)
            
        self.head = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Extract CLS token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        if self.training and self.num_dropouts > 1:
            # Average regression predictions across multiple dropout masks
            preds = torch.stack([self.head(drop_layer(cls_output)).squeeze(-1) for drop_layer in self.dropouts])
            return preds.mean(dim=0), cls_output
        else:
            dropped = self.drop(cls_output) if hasattr(self, "drop") else cls_output
            return self.head(dropped).squeeze(-1), cls_output


def get_llrd_parameter_groups(model: nn.Module, base_lr: float, weight_decay: float, decay_rate: float = 0.95) -> list:
    """Build parameter groups with Layer-wise Learning Rate Decay (LLRD).

    Top layers (regressor head) get base_lr * 2, encoder layers get geometrically decayed rates.
    """
    no_decay = ["bias", "LayerNorm.weight"]
    
    # 1. Regressor Head parameters
    head_params = []
    for name, param in model.named_parameters():
        if "head" in name or "drop" in name:
            head_params.append(param)
            
    parameter_groups = [
        {"params": head_params, "lr": base_lr * 2, "weight_decay": weight_decay}
    ]
    
    # 2. Encoder layer parameters (from top layer 11 down to bottom layer 0)
    num_layers = len(model.bert.encoder.layer)
    for idx in range(num_layers - 1, -1, -1):
        layer_lr = base_lr * (decay_rate ** (num_layers - 1 - idx))
        
        decay_params = []
        nodecay_params = []
        for name, param in model.bert.encoder.layer[idx].named_parameters():
            if any(nd in name for nd in no_decay):
                nodecay_params.append(param)
            else:
                decay_params.append(param)
                
        parameter_groups.extend([
            {"params": decay_params, "lr": layer_lr, "weight_decay": weight_decay},
            {"params": nodecay_params, "lr": layer_lr, "weight_decay": 0.0}
        ])
        
    # 3. Embedding parameters
    embed_lr = base_lr * (decay_rate ** num_layers)
    decay_params = []
    nodecay_params = []
    for name, param in model.bert.embeddings.named_parameters():
        if any(nd in name for nd in no_decay):
            nodecay_params.append(param)
        else:
            decay_params.append(param)
            
    parameter_groups.extend([
        {"params": decay_params, "lr": embed_lr, "weight_decay": weight_decay},
        {"params": nodecay_params, "lr": embed_lr, "weight_decay": 0.0}
    ])
    
    return parameter_groups


def set_encoder_freezing(model: nn.Module, num_freeze_layers: int, freeze: bool = True):
    """Freeze or unfreeze the bottom n encoder layers and embeddings."""
    for param in model.bert.embeddings.parameters():
        param.requires_grad = not freeze
        
    encoder_layers = model.bert.encoder.layer
    for idx in range(min(num_freeze_layers, len(encoder_layers))):
        for param in encoder_layers[idx].parameters():
            param.requires_grad = not freeze


def train_one_fold(
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    llrd_decay: float = 0.95,
    n_freeze_layers: int = 4,
    unfreeze_epoch: int = 2,
    rdrop_alpha: float = 1.0,
    seed: int = 42,
    fold: int = 0,
    save_path: str = None
) -> tuple:
    """Train the model on a single cross-validation fold.

    Optimized for Google Colab T4 execution using FP16 mixed precision.
    """
    seed_everything(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create datasets & loaders
    train_dataset = PairDataset(
        train_df["reference_answer"].tolist(),
        train_df["student_answer"].tolist(),
        train_df["score_norm"].tolist(),
        tokenizer
    )
    val_dataset = PairDataset(
        val_df["reference_answer"].tolist(),
        val_df["student_answer"].tolist(),
        val_df["score_norm"].tolist(),
        tokenizer
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False)
    
    # Initialize module
    model = BertRegressorModule(model_name, dropout_rate=0.1, num_dropouts=5)
    model = model.to(device)
    
    # Initial layer freezing
    if n_freeze_layers > 0:
        set_encoder_freezing(model, n_freeze_layers, freeze=True)
        
    # Optimizer & Scheduler setup
    if llrd_decay is not None:
        param_groups = get_llrd_parameter_groups(model, lr, weight_decay, llrd_decay)
        optimizer = torch.optim.AdamW(param_groups)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    loss_fn = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda")) # For FP16 Mixed Precision
    
    best_loss = float("inf")
    best_preds = None
    best_embeddings = None
    history = []
    
    for epoch in range(epochs):
        # Gradual Unfreezing trigger (Executed once at the target epoch)
        if n_freeze_layers > 0 and epoch == unfreeze_epoch:
            set_encoder_freezing(model, n_freeze_layers, freeze=False)
            # Re-initialize optimizer with unfrozen parameters (preserving state where possible)
            if llrd_decay is not None:
                param_groups = get_llrd_parameter_groups(model, lr, weight_decay, llrd_decay)
                optimizer.param_groups.clear()
                for group in param_groups:
                    optimizer.add_param_group(group)
            print(f"  [Info] Unfroze bottom {n_freeze_layers} encoder layers at epoch {epoch+1}")
                    
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            scores = batch["score"].to(device)
            
            # Autocast FP16 forward pass
            with torch.cuda.amp.autocast():
                if rdrop_alpha is not None and rdrop_alpha > 0:
                    # Run two passes with different dropout masks
                    p1, _ = model(ids, mask)
                    p2, _ = model(ids, mask)
                    
                    task_loss = (loss_fn(p1, scores) + loss_fn(p2, scores)) / 2.0
                    consistency_loss = F.mse_loss(p1, p2)
                    loss = task_loss + rdrop_alpha * consistency_loss
                else:
                    preds, _ = model(ids, mask)
                    loss = loss_fn(preds, scores)
                    
            # Scale loss and backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_preds = []
        val_embs = []
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                scores = batch["score"].to(device)
                
                with torch.cuda.amp.autocast():
                    preds, cls_output = model(ids, mask)
                    loss = loss_fn(preds, scores)
                    
                val_loss += loss.item()
                val_preds.extend(preds.cpu().numpy())
                val_embs.extend(cls_output.cpu().numpy())
                
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"  [Seed {seed} Fold {fold}] Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        history.append(f"{epoch+1},{avg_train_loss:.6f},{avg_val_loss:.6f}")
        
        # Save best model checkpoints
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_preds = np.array(val_preds)
            best_embeddings = np.array(val_embs)
            if save_path:
                torch.save(model.state_dict(), save_path)
                
    # Save training history to CSV
    if save_path:
        csv_path = save_path.replace(".pt", "_history.csv")
        try:
            with open(csv_path, "w") as f:
                f.write("epoch,train_loss,val_loss\n")
                f.write("\n".join(history) + "\n")
            print(f"  [Info] Saved training history to: {csv_path}")
        except Exception as e:
            print(f"  [Warning] Failed to save history to {csv_path}: {e}")
            
    # Free GPU VRAM
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return best_preds, best_embeddings
