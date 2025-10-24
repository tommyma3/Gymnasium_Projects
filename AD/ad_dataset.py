import torch
from torch.utils.data import Dataset
import random
import numpy as np

class HistoryDataset(Dataset):
    """
    Prepares interleaved sequences for Algorithm Distillation:
        Input:  (s0, a0, r0, s1, a1, r1, ..., s_t)
        Target: a_t
    """

    def __init__(self, state_file, action_file, reward_file, seq_len=200, action_dim=5):
        # Load data
        self.history_state = torch.load(state_file, weights_only=False)
        self.history_action = torch.load(action_file, weights_only=False)
        self.history_reward = torch.load(reward_file, weights_only=False)
        
        for tid in self.history_state.keys():
            if isinstance(self.history_state[tid], list):
                self.history_state[tid] = np.array(self.history_state[tid], dtype=np.float32)
            if isinstance(self.history_action[tid], list):
                self.history_action[tid] = np.array(self.history_action[tid], dtype=np.int64)
            if isinstance(self.history_reward[tid], list):
                self.history_reward[tid] = np.array(self.history_reward[tid], dtype=np.float32)

        self.seq_len = seq_len
        self.action_dim = action_dim

        self.task_ids = list(self.history_state.keys())
        self.task_lengths = {tid: len(self.history_state[tid]) for tid in self.task_ids}

    def __len__(self):
        return sum(max(1, self.task_lengths[tid] // self.seq_len) for tid in self.task_ids)

    def __getitem__(self, idx):
        # Randomly sample a task
        task_id = random.choice(self.task_ids)
        # Convert numpy arrays to tensors directly
        states = torch.from_numpy(self.history_state[task_id]).float()
        actions = torch.from_numpy(self.history_action[task_id]).long()
        rewards = torch.from_numpy(self.history_reward[task_id]).float()

        # Make one-hot actions for embedding
        actions = torch.nn.functional.one_hot(actions, num_classes=self.action_dim).float()

        # Sample a subsequence of up to seq_len transitions
        T = len(states)
        if T <= 2:
            raise ValueError(f"Not enough data for task {task_id}: len={T}")

        if T > self.seq_len:
            start = random.randint(0, T - self.seq_len - 1)
            end = start + self.seq_len
        else:
            start, end = 0, T  

        # Build subsequence
        seq_states = states[start:end]  # includes final s_t
        seq_actions = actions[start:end]
        seq_rewards = rewards[start:end]
        
        return {
            "states": seq_states,     # (T+1, state_dim)
            "actions": seq_actions,   # (T, action_dim)
            "rewards": seq_rewards,   # (T,)
        }
