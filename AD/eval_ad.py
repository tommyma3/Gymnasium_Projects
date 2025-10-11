import os
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from envs import darkroom_env
from dataset import Dataset
from net import Transformer
from utils import build_darkroom_model_filename, build_darkroom_data_filename

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# replace offline_evaluate with this
@torch.no_grad()
def offline_evaluate(model, dataloader, loss_fn, action_dim, horizon):
    """
    Offline evaluation: compares model predictions to teacher/expert actions.
    Handles model outputs of shape [B, A] or [B, horizon, A].
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in tqdm(dataloader, desc="Offline Evaluation"):
        # move to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # teacher actions: one-hot [B, A]
        true_actions = batch['teacher_actions']

        # model output: could be [B, A] or [B, horizon, A]
        pred_actions = model(batch)

        # if sequence output, take last timestep (common for transformers)
        if pred_actions.dim() == 3:
            pred = pred_actions[:, -1, :]   # [B, A]
        else:
            pred = pred_actions             # [B, A]

        # convert one-hot target -> index
        target_idx = torch.argmax(true_actions, dim=-1)  # [B]

        # cross-entropy expects (N, C) and (N,)
        loss = loss_fn(pred, target_idx)
        total_loss += loss.item() * pred.size(0)

        pred_idx = torch.argmax(pred, dim=-1)
        total_correct += (pred_idx == target_idx).sum().item()
        total_samples += pred.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


# replace rollout_evaluate with this
@torch.no_grad()
def rollout_evaluate(model, dim, horizon, num_goals=10, episodes_per_goal=20):
    """
    Online rollout evaluation:
    - Spawns DarkroomEnv for several random goals.
    - Lets the trained Transformer act in the environment.
    - Measures success rate (fraction of episodes reaching goal).
    Works with both Gymnasium and classic env APIs.
    """
    model.eval()
    # sample random goals
    goals = [np.random.randint(0, dim, size=2) for _ in range(num_goals)]
    success_rates = []

    for goal in tqdm(goals, desc="Online Evaluation (Rollouts)"):
        env = darkroom_env.DarkroomEnv(dim, goal, horizon)

        successes = 0
        for _ in range(episodes_per_goal):
            # handle env.reset() that can return obs or (obs, info)
            reset_out = env.reset()
            if isinstance(reset_out, tuple) or isinstance(reset_out, list):
                state = reset_out[0]
            else:
                state = reset_out
            state = np.array(state)

            # prepare zero context (batch size = 1)
            context_states = torch.zeros((1, horizon, env.state_dim), dtype=torch.float32, device=device)
            context_actions = torch.zeros((1, horizon, env.action_space.n), dtype=torch.float32, device=device)
            context_next_states = torch.zeros((1, horizon, env.state_dim), dtype=torch.float32, device=device)
            context_rewards = torch.zeros((1, horizon, 1), dtype=torch.float32, device=device)
            zeros_vec = torch.zeros((1, env.state_dim**2 + env.action_space.n + 1), dtype=torch.float32, device=device)

            done = False
            step = 0

            while (not done) and (step < horizon):
                # query state shape [1, state_dim]
                query_state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

                batch = {
                    'context_states': context_states,         # [1, H, state_dim]
                    'context_actions': context_actions,       # [1, H, action_dim]
                    'context_next_states': context_next_states,
                    'context_rewards': context_rewards,       # [1, H, 1]
                    'query_states': query_state,              # [1, state_dim]
                    'zeros': zeros_vec,                       # [1, vector_len]
                }

                pred = model(batch)  # could be [1, A] or [1, H, A]

                # use last timestep if seq output
                if pred.dim() == 3:
                    pred_step = pred[:, -1, :]   # [1, A]
                else:
                    pred_step = pred            # [1, A]

                pred_step = pred_step.squeeze(0)  # [A]
                action_idx = int(torch.argmax(pred_step, dim=-1).item())

                # convert to one-hot as env expects
                action_oh = np.zeros(env.action_space.n, dtype=np.float32)
                action_oh[action_idx] = 1.0

                # step: support both gym and gymnasium step signatures
                step_out = env.step(action_oh)
                if len(step_out) == 5:
                    # gymnasium: obs, reward, terminated, truncated, info
                    next_state, reward, terminated, truncated, info = step_out
                    done = bool(terminated or truncated)
                else:
                    # classic gym: obs, reward, done, info
                    next_state, reward, done, info = step_out

                next_state = np.array(next_state)

                # update context (so model gets in-episode context)
                if step < horizon:
                    context_states[0, step] = torch.tensor(state, dtype=torch.float32, device=device)
                    context_actions[0, step] = torch.tensor(action_oh, dtype=torch.float32, device=device)
                    context_next_states[0, step] = torch.tensor(next_state, dtype=torch.float32, device=device)
                    context_rewards[0, step] = torch.tensor([reward], dtype=torch.float32, device=device)

                state = next_state
                step += 1

                # success condition (reached goal)
                if np.array_equal(state, goal):
                    successes += 1
                    break

        success_rate = successes / episodes_per_goal
        success_rates.append(success_rate)
        print(f"Goal {goal.tolist() if isinstance(goal, np.ndarray) else goal} | Success Rate: {success_rate * 100:.1f}%")

    avg_success = float(np.mean(success_rates))
    print(f"\nAverage success rate across {num_goals} goals: {avg_success * 100:.2f}%")
    return avg_success



def main():
    # === Load hyperparameters ===
    with open("hyperparameters.yml", "r") as file:
        all_hyperparameter_sets = yaml.safe_load(file)
        hyperparameters = all_hyperparameter_sets["config"]

    env = hyperparameters["env"]
    n_envs = hyperparameters["envs"]
    n_hists = hyperparameters["hists"]
    n_samples = hyperparameters["samples"]
    horizon = hyperparameters["H"]
    dim = hyperparameters["dim"]
    n_embd = hyperparameters["embd"]
    n_layer = hyperparameters["layer"]
    n_head = hyperparameters["head"]
    lr = hyperparameters["lr"]
    dropout = hyperparameters["dropout"]
    shuffle = hyperparameters["shuffle"]
    seed = hyperparameters["seed"]

    state_dim = 2
    action_dim = 5

    # === Prepare dataset and model configs ===
    dataset_config = {
        "n_hists": n_hists,
        "n_samples": n_samples,
        "horizon": horizon,
        "dim": dim,
        "rollin_type": "uniform",
    }

    model_config = {
        "shuffle": shuffle,
        "lr": lr,
        "dropout": dropout,
        "n_embd": n_embd,
        "n_layer": n_layer,
        "n_head": n_head,
        "n_envs": n_envs,
        "n_hists": n_hists,
        "n_samples": n_samples,
        "horizon": horizon,
        "dim": dim,
        "seed": seed,
    }

    # === Paths ===
    eval_path = build_darkroom_data_filename(env, 100, dataset_config, mode=2)
    model_name = build_darkroom_model_filename(env, model_config)
    model_path = f"models/{model_name}.pt"

    print(f"Loading model from {model_path}")
    print(f"Loading evaluation data from {eval_path}")

    # === Load model ===
    model = Transformer({
        "horizon": horizon,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "n_layer": n_layer,
        "n_embd": n_embd,
        "n_head": n_head,
        "shuffle": shuffle,
        "dropout": dropout,
        "test": True,
        "store_gpu": True,
    }).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # === Load eval dataset ===
    eval_dataset = Dataset(eval_path, {
        "shuffle": shuffle,
        "horizon": horizon,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "store_gpu": True,
    })
    eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False)

    # === Loss function ===
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

    # === Offline evaluation ===
    avg_loss, accuracy = offline_evaluate(model, eval_loader, loss_fn, action_dim, horizon)
    print(f"\nOffline Evaluation Results:")
    print(f"Average Cross-Entropy Loss: {avg_loss:.6f}")
    print(f"Action Accuracy: {accuracy * 100:.2f}%")

    # === Online rollout evaluation ===
    print("\nPerforming online rollout evaluation...")
    avg_success = rollout_evaluate(model, dim, horizon, num_goals=10, episodes_per_goal=100)
    print(f"Final Average Success Rate: {avg_success * 100:.2f}%")


if __name__ == "__main__":
    main()
