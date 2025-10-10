import gymnasium as gym
import numpy as np
import yaml
import random
import os
import pickle

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from envs import darkroom_env
from utils import build_darkroom_data_filename


def rollin(env, rollin_type, ppo_agent=None):
    states = []
    actions = []
    next_states = []
    rewards = []

    state = env.reset()
    for _ in range(env.horizon):
        if rollin_type == 'uniform':
            state = env.sample_state()
            action = env.sample_action()
        elif rollin_type == 'expert':
            action = env.opt_action(state)
        elif rollin_type == 'ppo':
            obs = np.array(state).reshape(1, -1)
            act_idx, _ = ppo_agent.predict(obs, deterministic=False)
            onehot = np.zeros(env.actions_space.n)
            onehot[act_idx] = 1
            action = onehot
        else:
            raise NotImplementedError
        next_state, reward = env.transit(state, action)

        states.append(state)
        actions.append(action)
        next_states.append(next_state)
        rewards.append(reward)
        state = next_state

    states = np.array(states)
    actions = np.array(actions)
    next_states = np.array(next_states)
    rewards = np.array(rewards)

    return states, actions, next_states, rewards

def rand_pos_and_dir(env):
    pos_vec = np.random.uniform(0, env.size, size=3)
    pos_vec[1] = 0.0
    dir_vec = np.random.uniform(0, 2 * np.pi)
    return pos_vec, dir_vec

def generate_mdp_histories_from_envs(envs, n_hists, n_samples, rollin_type, ppo_agent=None, update_id=None):
    trajs = []
    for env in envs:
        for j in range(n_hists):
            (
                context_states,
                context_actions,
                context_next_states,
                context_rewards,
            ) = rollin(env, rollin_type=rollin_type, ppo_agent=ppo_agent)
            for k in range(n_samples):
                query_state = env.sample_state()
                optimal_action = env.opt_action(query_state)

                traj = {
                    'query_state': query_state,
                    'optimal_action': optimal_action,
                    'context_states': context_states,
                    'context_actions': context_actions,
                    'context_next_states': context_next_states,
                    'context_rewards': context_rewards,
                    'goal': env.goal,
                    'update': update_id
                }

                # Add perm_index for DarkroomEnvPermuted
                if hasattr(env, 'perm_index'):
                    traj['perm_index'] = env.perm_index

                trajs.append(traj)
    return trajs


def generate_darkroom_histories(goals, dim, horizon, **kwargs):
    envs = [darkroom_env.DarkroomEnv(dim, goal, horizon) for goal in goals]
    trajs = generate_mdp_histories_from_envs(envs, **kwargs)
    return trajs


def generate_darkroom_permuted_histories(indices, dim, horizon, **kwargs):
    envs = [darkroom_env.DarkroomEnvPermuted(
        dim, index, horizon) for index in indices]
    trajs = generate_mdp_histories_from_envs(envs, **kwargs)
    return trajs


if __name__ == '__main__':
    np.random.seed(0)
    random.seed(0)

    with open("hyperparameters.yml", 'r') as file:
        all_hyperparameter_sets = yaml.safe_load(file)
        hyperparameters = all_hyperparameter_sets['collect_config']

        env = hyperparameters['env']
        n_envs = hyperparameters['envs']
        n_enal_envs = hyperparameters['envs_eval']
        n_hists = hyperparameters['hists']
        n_samples = hyperparameters['samples']
        horizon = hyperparameters['H']
        dim = hyperparameters['dim']
        var = hyperparameters['var']
        cov = hyperparameters['cov']
        env_id_start = hyperparameters['env_id_start']
        env_id_end = hyperparameters['env_id_end']
        lin_d = hyperparameters['lin_d']

    n_train_envs = int(.8 * n_envs)
    n_test_envs = n_envs - n_train_envs

    config = {
        'n_hists': n_hists,
        'n_samples': n_samples,
        'horizon': horizon,
    }


    if env != 'darkroom_heldout':
        raise NotImplementedError
    
    
    config.update({'dim': dim, 'rollin_type': 'uniform'})
    goals = np.array([[(j, i) for i in range(dim)]
                        for j in range(dim)]).reshape(-1, 2)
    np.random.RandomState(seed=0).shuffle(goals)
    train_test_split = int(.8 * len(goals))
    train_goals = goals[:train_test_split]
    test_goals = goals[train_test_split:]

    eval_goals = np.array(test_goals.tolist() * int(100 // len(test_goals)))
    # train_goals = np.repeat(train_goals, n_envs // (dim * dim), axis=0)
    # test_goals = np.repeat(test_goals, n_envs // (dim * dim), axis=0)


    total_updates = 50
    all_train, all_test, all_eval = [], [], []
    for task_id, goal in enumerate(train_goals):
        print(f"Training PPO teacher on task {task_id} with goal {goal}")

        base_env = lambda: darkroom_env.DarkroomEnv(dim, goal, horizon)
        train_env = DummyVecEnv([base_env])
        ppo_agent = PPO("MlpPolicy", train_env, verbose=0, n_steps=256)

        for update in range(total_updates):
            ppo_agent.learn(total_timesteps=ppo_agent.n_steps, reset_num_timesteps=False)
            train_trajs = generate_darkroom_histories(
                [goal], dim, horizon,
                n_hists=n_hists, n_samples=n_samples,
                rollin_type='ppo', ppo_agent=ppo_agent,
                update_id=update, task_id=task_id
            )
            all_train.extend(train_trajs)
        
    for task_id, goal in enumerate(test_goals, start=len(train_goals)):
        print(f"Training PPO teacher on TEST task {task_id} with goal {goal}")
        base_env = lambda: darkroom_env.DarkroomEnv(dim, goal, horizon=horizon)
        test_env = DummyVecEnv([base_env])
        ppo_agent = PPO("MlpPolicy", test_env, verbose=0, n_steps=256)

        for update in range(total_updates):
            ppo_agent.learn(total_timesteps=ppo_agent.n_steps, reset_num_timesteps=False)
            test_trajs = generate_darkroom_histories(
                [goal], dim, horizon,
                n_hists=n_hists, n_samples=n_samples,
                rollin_type='ppo', ppo_agent=ppo_agent,
                update_id=update, task_id=task_id
            )
            all_test.extend(test_trajs)

    for task_id, goal in enumerate(eval_goals, start=len(train_goals) + len(test_goals)):
        print(f"Training PPO teacher on EVAL task {task_id} with goal {goal}")
        base_env = lambda: darkroom_env.DarkroomEnv(dim, goal, horizon=horizon)
        eval_env = DummyVecEnv([base_env])
        ppo_agent = PPO("MlpPolicy", eval_env, verbose=0, n_steps=256)

        for update in range(total_updates):
            ppo_agent.learn(total_timesteps=ppo_agent.n_steps, reset_num_timesteps=False)
            eval_trajs = generate_darkroom_histories(
                [goal], dim, horizon,
                n_hists=n_hists, n_samples=n_samples,
                rollin_type='ppo', ppo_agent=ppo_agent,
                update_id=update, task_id=task_id
            )
            all_eval.extend(eval_trajs)        

    
    #train_trajs = generate_darkroom_histories(train_goals, **config)
    #test_trajs = generate_darkroom_histories(test_goals, **config)
    #eval_trajs = generate_darkroom_histories(eval_goals, **config)

    train_filepath = build_darkroom_data_filename(env, n_envs, config, mode=0)
    test_filepath = build_darkroom_data_filename(env, n_envs, config, mode=1)
    eval_filepath = build_darkroom_data_filename(env, 100, config, mode=2)
        

    if not os.path.exists('datasets'):
        os.makedirs('datasets', exist_ok=True)
    with open(train_filepath, 'wb') as file:
        pickle.dump(all_train, file)
    with open(test_filepath, 'wb') as file:
        pickle.dump(all_test, file)
    with open(eval_filepath, 'wb') as file:
        pickle.dump(all_eval, file)

    print(f"Saved to {train_filepath}.")
    print(f"Saved to {test_filepath}.")
    print(f"Saved to {eval_filepath}.")