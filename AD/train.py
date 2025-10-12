import torch.multiprocessing as mp

if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method('spawn', force=True)

import argparse
import os
import time
from IPython import embed

import matplotlib.pyplot as plt
import torch
from torchvision.transforms import transforms

import numpy as np
import random
from dataset import Dataset
from net import Transformer
import yaml

from utils import build_darkroom_data_filename, build_darkroom_model_filename

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if __name__ == '__main__':
    
    if not os.path.exists('figs/loss'):
        os.makedirs('figs/loss', exist_ok=True)
    if not os.path.exists('models'):
        os.makedirs('models', exist_ok=True)

    with open("hyperparameters.yml", 'r') as file:
        all_hyperparameter_sets = yaml.safe_load(file)
        hyperparameters = all_hyperparameter_sets['config']

        env = hyperparameters['env']
        n_envs = hyperparameters['envs']
        n_hists = hyperparameters['hists']
        n_samples = hyperparameters['samples']
        horizon = hyperparameters['H']
        dim = hyperparameters['dim']
        state_dim = dim
        action_dim = dim
        n_embd = hyperparameters['embd']
        n_head = hyperparameters['head']
        n_layer = hyperparameters['layer']
        lr = hyperparameters['lr']
        shuffle = hyperparameters['shuffle']
        dropout = hyperparameters['dropout']
        var = hyperparameters['var']
        cov = hyperparameters['cov']
        num_epochs = hyperparameters['num_epochs']
        seed = hyperparameters['seed']
        lin_d = hyperparameters['lin_d']

        tmp_seed = seed
        if seed == -1:
            tmp_seed = 0

        torch.manual_seed(tmp_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(tmp_seed)
            torch.cuda.manual_seed_all(tmp_seed)
            
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(tmp_seed)
        random.seed(tmp_seed)

        dataset_config = {
            'n_hists': n_hists,
            'n_samples': n_samples,
            'horizon': horizon,
            'dim': dim,
        }
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

        if env.startswith('darkroom'):
            state_dim = 2
            action_dim = 5
            dataset_config.update({'rollin_type': 'uniform'})
            path_train = build_darkroom_data_filename(env, n_envs, dataset_config, mode=0)
            path_test = build_darkroom_data_filename(env, n_envs, dataset_config, mode=1)
            filename = build_darkroom_model_filename(env, model_config)

        config = {
            'horizon': horizon,
            'state_dim': state_dim,
            'action_dim': action_dim,
            'n_layer': n_layer,
            'n_embd': n_embd,
            'n_head': n_head,
            'shuffle': shuffle,
            'dropout': dropout,
            'test': False,
            'store_gpu': True,
        }
        
        model = Transformer(config).to(device)

        params = {
            'batch_size': 64,
            'shuffle': True,
        }

        log_filename = f'figs/loss/{filename}_logs.txt'
        with open(log_filename, 'w') as f:
            pass

        def printw(message):
            print(message)
            with open(log_filename, 'a') as f:
                print(message, file=f)
        
        train_dataset = Dataset(path_train, config)
        test_dataset = Dataset(path_test, config)

        train_loader = torch.utils.data.DataLoader(train_dataset, **params)
        test_loader = torch.utils.data.DataLoader(test_dataset, **params)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

        test_loss = []
        train_loss = []

        printw("Num train batches: " + str(len(train_loader)))
        printw("Num test batches: " + str(len(test_loader)))

        for epoch in range(num_epochs):
            printw(f"Epoch {epoch + 1}")
            start_time = time.time()
            with torch.no_grad():
                epoch_test_loss = 0.0
                for i, batch in enumerate(test_loader):
                    print(f"Batch {i} of {len(test_loader)}", end='\r')
                    batch = {k: v.to(device) for k, v in batch.items()}
                    true_actions = batch['teacher_action']
                    pred_actions = model(batch)
                    true_actions = true_actions.unsqueeze(1).repeat(1, pred_actions.shape[1], 1)
                    true_actions = true_actions.reshape(-1, action_dim)
                    pred_actions = pred_actions.reshape(-1, action_dim)

                    loss = loss_fn(pred_actions, true_actions)
                    epoch_test_loss += loss.item() / horizon

            test_loss.append(epoch_test_loss / len(test_loader))
            end_time = time.time()
            printw(f"\tTest loss: {test_loss[-1]}")
            printw(f"\tEval time: {end_time - start_time}")



            epoch_train_loss = 0.0
            start_time = time.time()

            for i, batch in enumerate(train_loader):
                print(f"Batch {i} of {len(train_loader)}", end='\r')
                batch = {k: v.to(device) for k, v in batch.items()}
                true_actions = batch['teacher_action']
                pred_actions = model(batch)
                true_actions = true_actions.unsqueeze(1).repeat(1, pred_actions.shape[1], 1)
                true_actions = true_actions.reshape(-1, action_dim)
                pred_actions = pred_actions.reshape(-1, action_dim)

                loss = loss_fn(pred_actions, true_actions)
                epoch_train_loss += loss.item() / horizon

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()            
                epoch_train_loss += loss.item() / horizon
            
            train_loss.append(epoch_train_loss / len(train_loader))
            end_time = time.time()

            printw(f"\tTrain loss: {train_loss[-1]}")
            printw(f"\tTrain time: {end_time - start_time}")

            if (epoch + 1) % 10 == 0:
                torch.save(model.state_dict(), f'models/{filename}_epoch{epoch+1}.pt')

            if (epoch + 1) % 10 == 0:
                printw(f"Epoch: {epoch + 1}")
                printw(f"Test Loss:        {test_loss[-1]}")
                printw(f"Train Loss:       {train_loss[-1]}")
                printw("\n")

                plt.yscale('log')
                plt.plot(train_loss[1:], label="Train Loss")
                plt.plot(test_loss[1:], label="Test Loss")
                plt.legend()
                plt.savefig(f"figs/loss/{filename}_train_loss.png")
                plt.clf()

        torch.save(model.state_dict(), f'models/{filename}.pt')
        print("Done.")
