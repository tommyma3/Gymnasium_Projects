import yaml
import os
import pickle

import matplotlib as plt
import torch
from IPython import embed

from net import Transformer
from utils import build_darkroom_model_filename, build_darkroom_data_filename

import numpy as np
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    with open("hyperparameters.yml", "r") as f:
        all_hyperparameter_sets = yaml.safe_load(f)
        hyperparameters = all_hyperparameter_sets["config"]
    
    n_envs = hyperparameters['envs']
    n_hists = hyperparameters['hists']
    n_samples = hyperparameters['samples']
    H = hyperparameters['H']
    dim = hyperparameters['dim']
    state_dim = dim
    action_dim = dim
    n_embd = hyperparameters['embd']
    n_head = hyperparameters['head']
    n_layer = hyperparameters['layer']
    lr = hyperparameters['lr']
    epoch = hyperparameters['epoch']
    shuffle = hyperparameters['shuffle']
    dropout = hyperparameters['dropout']
    var = hyperparameters['var']
    cov = hyperparameters['cov']
    test_cov = hyperparameters['test_cov']
    envname = hyperparameters['env']
    horizon = hyperparameters['horizon']
    n_eval = hyperparameters['n_eval']
    seed = hyperparameters['seed']
    lin_d = hyperparameters['lin_d']

    tmp_seed = seed
    if seed == -1:
        tmp_seed = 0

    torch.manual_seed(tmp_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(tmp_seed)
    np.random.seed(tmp_seed)

    if test_cov < 0:
        test_cov = cov
    if horizon < 0:
        horizon = H
    
    model_config = {
        'shuffle': shuffle,
        'lr': lr,
        'dropout': dropout,
        'n_embd': n_embd,
        'n_layer': n_layer,
        'n_head': n_head,
        'n_envs': n_envs,
        'n_hists': n_hists,
        'n_samples': n_samples,
        'horizon': horizon,
        'dim': dim,
        'seed': seed,
    }

    if not envname.startswith('darkroom'):
        raise NotImplementedError
    
    state_dim = 2
    action_dim = 5
    filename = build_darkroom_model_filename(envname, model_config)

    config = {
        'horizon': H,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'n_layer': n_layer,
        'n_embd': n_embd,
        'n_head': n_head,
        'dropout': dropout,
        'test': True,
    }


    model = Transformer(config).to(device)

    tmp_filename = filename
    if epoch < 0:
        model_path = f'models/{tmp_filename}.pt'
    else:
        model_path = f'models/{tmp_filename}_epoch{epoch}.pt'

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    model.eval()

    dataset_config = {
        'horizon': horizon,
        'dim': dim
    }

    if envname != 'darkroom_heloout':
        raise ValueError(f'Environment {envname} not supported')
    
    dataset_config.update({'rollin_type': "uniform"})
    eval_filepath = build_darkroom_data_filename(envname, 0, n_eval, dataset_config, mode=2)
    save_filename = f'{filename}_hor{horizon}.pkl'

    with open(eval_filepath, 'rb') as f:
        eval_trajs = pickle.load(f)

    n_eval = min(n_eval, len(eval_trajs))

    evals_filename = f'evals_epoch{epoch}'
    if not os.path.exists(f'figs/{evals_filename}'):
        os.makedirs(f'figs/{evals_filename}', exist_ok=True)
    if not os.path.exists(f'figs/{evals_filename}/bar'):
        os.makedirs(f'figs/{evals_filename}/bar', exist_ok=True)
    if not os.path.exists(f'figs/{evals_filename}/online'):
        os.makedirs(f'figs/{evals_filename}/online', exist_ok=True)
    if not os.path.exists(f'figs/{evals_filename}/graph'):
        os.makedirs(f'figs/{evals_filename}/graph', exist_ok=True)

    config = {
        'Heps': 40,
        'horizon': horizon,
        'H': H,
        'n_eval': min(20, n_eval),
        'dim': dim,
        'permuted': True if envname == 'darkroom_permuted' else False
    }

    
    