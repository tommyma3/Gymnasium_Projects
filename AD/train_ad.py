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

from network import ADTransformerInterleaved
from ad_dataset import HistoryDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def main():
    # --- Load hyperparameters ---
    with open("hyperparameters.yml", "r") as file:
        hp_all = yaml.safe_load(file)
        hp = hp_all["config"]

    dim = hp["dim"]
    horizon = hp["H"]
    num_epochs = hp["num_epochs"]
    lr = hp["lr"]
    n_embd = hp["embd"]
    n_layer = hp["layer"]
    n_head = hp["head"]
    dropout = hp["dropout"]
    seed = hp["seed"]

    # --- Seeds ---
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # --- Dataset ---
    dataset = HistoryDataset(
        "history_set/history_state.pkl",
        "history_set/history_action.pkl",
        "history_set/history_reward.pkl",
        seq_len=horizon,
        action_dim=5
    )
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # --- Model ---
    state_dim = 2  # Darkroom states = (x, y)
    action_dim = 5  # 5 discrete actions

    model = ADTransformerInterleaved(
        state_dim=state_dim,
        action_dim=action_dim,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        dropout=dropout,
        max_seq_len=3 * horizon + 1
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    os.makedirs("models", exist_ok=True)
    os.makedirs("figs/loss", exist_ok=True)

    # --- Training ---
    train_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for batch in train_loader:
            states = batch["states"].to(device)
            actions = batch["actions"].to(device)
            rewards = batch["rewards"].to(device)
            target_action = batch["target_action"].to(device)

            pred = model(states, actions, rewards)

            target_idx = torch.argmax(target_action, dim=-1)
            loss = loss_fn(pred, target_idx)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        end_time = time.time()

        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Time: {end_time - start_time:.2f}s")

        # Save checkpoints
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"models/ad_interleaved_epoch{epoch+1}.pt")

    # --- Save final model ---
    torch.save(model.state_dict(), "models/ad_interleaved_final.pt")

    # --- Plot loss ---
    plt.plot(train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("AD Interleaved Transformer Training Loss")
    plt.savefig("figs/loss/ad_interleaved_loss.png")
    plt.show()

if __name__ == "__main__":
    main()
