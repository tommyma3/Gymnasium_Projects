import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
import yaml
import os
import time
import matplotlib.pyplot as plt

from network import ADTransformer
from ad_dataset import HistoryDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def main():
    with open("hyperparameters.yml", "r") as file:
        hp_all = yaml.safe_load(file)
        hp = hp_all["config"]

    dim = hp["dim"]
    horizon = hp["H"]
    num_epochs = hp["num_epochs"]
    lr_init = hp["lr_init"]
    lr_peak = hp["lr_peak"]
    batch_size = hp["batch_size"]
    beta_1 = hp["beta_1"]
    beta_2 = hp["beta_2"]
    n_embd = hp["embd"]
    n_layer = hp["layer"]
    n_head = hp["head"]
    dropout = hp["dropout"]
    seed = hp["seed"]
    max_seq_len = hp['max_seq_len']
    train_history_len = hp['train_history_len']
    grad_clip = hp['grad_clip']

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dataset = HistoryDataset(
        "history_set/history_state.pkl",
        "history_set/history_action.pkl",
        "history_set/history_reward.pkl",
        seq_len=train_history_len,
        action_dim=5
    )
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    state_dim = 2  # Darkroom states = (x, y)
    action_dim = 5  # 5 discrete actions

    model = ADTransformer(
        state_dim=state_dim,
        action_dim=action_dim,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        dropout=dropout,
        max_seq_len=max_seq_len
    ).to(device)

    checkpoint_path = "models/ad_final.pt"
    if os.path.exists(checkpoint_path):
        print(f"Loading model weights from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path))
    else:
        print("No checkpoint found, starting from scratch.")

    print(f"Max sequence length for transformer: {max_seq_len}")
    print(f"Sequence length for each sampled history: {train_history_len}")

    optimizer = optim.Adam(model.parameters(), lr=lr_peak, betas=(beta_1, beta_2), weight_decay=1e-5)

    # Calculate total number of training steps (batches per epoch × number of epochs)
    num_training_steps = len(train_loader) * num_epochs
    num_warmup = int(0.1 * num_training_steps)  # Convert to int for safer scheduling
    num_decay = num_training_steps - num_warmup

    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=(lr_init / lr_peak),
        end_factor=1.0,
        total_iters=num_warmup
    )

    decay_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_decay,
        eta_min=lr_init
    )

    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, decay_scheduler],
        milestones=[num_warmup]
    )
    loss_fn = nn.CrossEntropyLoss()

    os.makedirs("models", exist_ok=True)
    os.makedirs("figs", exist_ok=True)

    lr_history = []

    train_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for batch in train_loader:
            states = batch["states"].to(device)
            actions = batch["actions"].to(device)
            rewards = batch["rewards"].to(device)

            pred = model(states, actions[:, :-1, :], rewards[:, :-1])
            pred = pred.view(-1, action_dim)

            actions = actions.view(-1, action_dim)
            target_idx = torch.argmax(actions, dim=-1)
            loss = loss_fn(pred, target_idx)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            
            # Step scheduler and track learning rate after each batch
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

        avg_loss = epoch_loss / len(train_loader)
        end_time = time.time()

        # Record learning rate at end of epoch
        lr_history.append(current_lr)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | LR: {current_lr:.6f} | Time: {end_time - start_time:.2f}s")

        # Save checkpoints
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"models/ad_epoch{epoch+1}.pt")

    # --- Save final model ---
    torch.save(model.state_dict(), "models/ad_final.pt")

    # --- Plot loss ---
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("AD Transformer Training Loss")
    
    plt.subplot(1, 2, 2)
    plt.plot(lr_history)
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    
    plt.tight_layout()
    plt.savefig("figs/ad_training.png")


if __name__ == "__main__":
    main()
