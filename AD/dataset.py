import numpy as np
import torch
from utils import convert_to_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Dataset(torch.utils.data.Dataset):

    def __init__(self, path, config):
        self.shuffle = config['shuffle']
        self.horizon = config['horizon']
        self.store_gpu = config['store_gpu']
        self.config = config

        if not isinstance(path, list):
            path = [path]

        self.trajs = []

        for p in path:
            with open(p, 'rb') as f:
                self.trajs += torch.load(f, weights_only=False)  # Explicitly set weights_only=False for full loading

        context_states = []
        context_actions = []
        context_next_states = []
        context_rewards = []
        query_state = []
        teacher_action = []

        for traj in self.trajs:
            # Handle context data with consistent keys
            context_states.append(traj['context_states'])
            context_actions.append(traj['context_actions'])
            context_next_states.append(traj['context_next_states'])
            context_rewards.append(traj['context_rewards'])
            
            # Handle query state with either old or new format
            query_key = 'query_state'
            query_state.append(traj[query_key])
            
            # Handle teacher actions with either old or new format
            action_key = 'teacher_action'
            teacher_action.append(traj[action_key])
        
        context_states = np.array(context_states)
        context_actions = np.array(context_actions)
        context_next_states = np.array(context_next_states)
        context_rewards = np.array(context_rewards)
        if len(context_rewards.shape) < 3:
            context_rewards = context_rewards[:, :, None]
        query_state = np.array(query_state)
        teacher_action = np.array(teacher_action)

        self.dataset = {
            'query_state': convert_to_tensor(query_state, store_gpu=self.store_gpu),
            'teacher_action': convert_to_tensor(teacher_action, store_gpu=self.store_gpu),
            'context_states': convert_to_tensor(context_states, store_gpu=self.store_gpu),
            'context_actions': convert_to_tensor(context_actions, store_gpu=self.store_gpu),
            'context_next_states': convert_to_tensor(context_next_states, store_gpu=self.store_gpu),
            'context_rewards': convert_to_tensor(context_rewards, store_gpu=self.store_gpu),
        }

        self.zeros = np.zeros(
            config['state_dim'] ** 2 + config['action_dim'] + 1
        )
        self.zeros = convert_to_tensor(self.zeros, store_gpu=self.store_gpu)

    def __len__(self):
        return len(self.dataset['query_state'])
    
    def __getitem__(self, index):
        res = {
            'context_states': self.dataset['context_states'][index],
            'context_actions': self.dataset['context_actions'][index],
            'context_next_states': self.dataset['context_next_states'][index],
            'context_rewards': self.dataset['context_rewards'][index],
            'query_state': self.dataset['query_state'][index],
            'teacher_action': self.dataset['teacher_action'][index],
            'zeros': self.zeros,
        }

        if self.shuffle:
            perm = torch.randperm(self.horizon)
            res['context_states'] = res['context_states'][perm]
            res['context_actions'] = res['context_actions'][perm]
            res['context_next_states'] = res['context_next_states'][perm]
            res['context_rewards'] = res['context_rewards'][perm]

        return res

