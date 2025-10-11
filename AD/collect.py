import gymnasium as gym
import numpy as np
import yaml
import random
import os
import pickle
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from envs import darkroom_env
from utils import build_darkroom_data_filename


def rollin(env, rollin_type, ppo_agent):
    states = []
    actions = []
    next_states = []
    rewards = []

    state, _ = env.reset()  # Unpack the state from the new gymnasium API
    state = np.array(state, dtype=np.float32)  # Ensure state is a numpy array with float32 dtype
    for _ in range(env.horizon):
        if rollin_type == 'uniform':
            # Random action
            action_idx = env.action_space.sample()
            action = np.zeros(env.action_space.n)
            action[action_idx] = 1
        elif rollin_type == 'expert':
            # Get optimal action and convert to one-hot
            action_idx = env.opt_action(state)
            action = np.zeros(env.action_space.n)
            action[action_idx] = 1
        elif rollin_type == 'ppo':
            obs = state.reshape(1, -1)  # State is already a numpy array
            action, _ = ppo_agent.predict(obs, deterministic=False)
            # Convert to one-hot for consistency with transformer training
            onehot = np.zeros(env.action_space.n)
            onehot[action] = 1
            action = onehot
        else:
            raise NotImplementedError
        # Convert one-hot action to integer for step method
        action_idx = np.argmax(action)
        next_state, reward, terminated, truncated, _ = env.step(action_idx)

        states.append(state)
        actions.append(action)
        next_states.append(next_state)
        rewards.append(reward)
        state = next_state

        if terminated or truncated:
            break

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

def generate_mdp_histories_from_envs(envs, n_hists, n_samples, rollin_type, ppo_agent=None):
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
                opt_act_idx = env.opt_action(query_state)
                # Convert to one-hot for transformer training
                optimal_action = np.zeros(env.action_space.n)
                optimal_action[opt_act_idx] = 1

                traj = {
                    'query_state': query_state,
                    'optimal_action': optimal_action,
                    'context_states': context_states,
                    'context_actions': context_actions,
                    'context_next_states': context_next_states,
                    'context_rewards': context_rewards,
                    'goal': env.goal,
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

def evaluate_ppo(env, agent, n_episodes=20):
    """
    Evaluate PPO agent in the environment.
    Returns average reward and success rate.
    """
    rewards = []
    successes = 0

    for _ in range(n_episodes):
        obs = env.reset()
        if isinstance(obs, tuple):  # Gymnasium returns (obs, info)
            obs = obs[0]
        total_reward = 0
        done = False
        steps = 0

        while not done and steps < env.horizon:
            action, _ = ppo_agent.predict(obs, deterministic=True)
            step_result = env.step(action)  # Use integer action directly for environment step
            if len(step_result) == 5:
                obs, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                obs, reward, done, _ = step_result

            total_reward += reward
            steps += 1
            if np.array_equal(obs, env.goal):
                successes += 1
                break
        rewards.append(total_reward)

    avg_reward = float(np.mean(rewards))
    success_rate = successes / n_episodes
    return avg_reward, success_rate


if __name__ == '__main__':
    np.random.seed(0)
    random.seed(0)

    with open("hyperparameters.yml", 'r') as file:
        all_hyperparameter_sets = yaml.safe_load(file)
        hyperparameters = all_hyperparameter_sets['config']

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


    total_updates = 100

    min_updates = 10
    max_updates = 500
    patience = 5


    all_train, all_test, all_eval = [], [], []
    for task_id, goal in enumerate(train_goals):
        print(f"Training PPO teacher on task {task_id} with goal {goal}")

        base_env = lambda: darkroom_env.DarkroomEnv(dim, goal, horizon)
        train_env = DummyVecEnv([base_env])
        ppo_agent = PPO("MlpPolicy", train_env, verbose=0, n_steps=256, ent_coef=0.1)

        best_reward = 0
        no_improve_counter = 0

        for update in range(max_updates):
            # one PPO update
            ppo_agent.learn(total_timesteps=ppo_agent.n_steps, reset_num_timesteps=False)

            # Evaluate PPO teacher
            avg_reward, success_rate = evaluate_ppo(base_env(), ppo_agent)
            print(f"Task {task_id} | Update {update:03d} | "
                f"Avg Reward: {avg_reward:.3f} | Success Rate: {success_rate*100:.1f}%")

            if avg_reward > best_reward:
                best_reward = avg_reward
                no_improve_counter = 0
            elif avg_reward == 1:
                no_improve_counter += 1
            else:
                no_improve_counter = 0

            # collect histories at this stage
            train_trajs = generate_darkroom_histories(
                [goal], dim, horizon,
                n_hists=n_hists, n_samples=n_samples,
                rollin_type='ppo', ppo_agent=ppo_agent,
            )
            all_train.extend(train_trajs)

            if (update >= min_updates) and (no_improve_counter >= patience):
                print(f"Task {task_id} converged at update {update}")
                break
        
    for task_id, goal in enumerate(test_goals, start=len(train_goals)):
        print(f"Training PPO teacher on TEST task {task_id} with goal {goal}")
        base_env = lambda: darkroom_env.DarkroomEnv(dim, goal, horizon=horizon)
        test_env = DummyVecEnv([base_env])
        ppo_agent = PPO("MlpPolicy", test_env, verbose=0, n_steps=256, ent_coef=0.1)

        best_reward = 0
        no_improve_counter = 0

        for update in range(total_updates):
            ppo_agent.learn(total_timesteps=ppo_agent.n_steps, reset_num_timesteps=False)

            avg_reward, success_rate = evaluate_ppo(base_env(), ppo_agent)
            print(f"Task {task_id} | Update {update:03d} | "
                f"Avg Reward: {avg_reward:.3f} | Success Rate: {success_rate*100:.1f}%")

            if avg_reward > best_reward:
                best_reward = avg_reward
                no_improve_counter = 0
            elif avg_reward == 1:
                no_improve_counter += 1
            else:
                no_improve_counter = 0


            test_trajs = generate_darkroom_histories(
                [goal], dim, horizon,
                n_hists=n_hists, n_samples=n_samples,
                rollin_type='ppo', ppo_agent=ppo_agent,
            )
            all_test.extend(test_trajs)
            if (update >= min_updates) and (no_improve_counter >= patience):
                print(f"Task {task_id} converged at update {update}")
                break

    for task_id, goal in enumerate(eval_goals, start=len(train_goals) + len(test_goals)):
        print(f"Training PPO teacher on EVAL task {task_id} with goal {goal}")
        base_env = lambda: darkroom_env.DarkroomEnv(dim, goal, horizon=horizon)
        eval_env = DummyVecEnv([base_env])
        ppo_agent = PPO("MlpPolicy", eval_env, verbose=0, n_steps=256, ent_coef=0.1)

        best_reward = 0
        no_improve_counter = 0

        for update in range(total_updates):
            ppo_agent.learn(total_timesteps=ppo_agent.n_steps, reset_num_timesteps=False)

            avg_reward, success_rate = evaluate_ppo(base_env(), ppo_agent)
            print(f"Task {task_id} | Update {update:03d} | "
                f"Avg Reward: {avg_reward:.3f} | Success Rate: {success_rate*100:.1f}%")

            if avg_reward > best_reward:
                best_reward = avg_reward
                no_improve_counter = 0
            elif avg_reward == 1:
                no_improve_counter += 1
            else:
                no_improve_counter = 0

            eval_trajs = generate_darkroom_histories(
                [goal], dim, horizon,
                n_hists=n_hists, n_samples=n_samples,
                rollin_type='ppo', ppo_agent=ppo_agent,
            )
            all_eval.extend(eval_trajs)
            if (update >= min_updates) and (no_improve_counter >= patience):
                print(f"Task {task_id} converged at update {update}")
                break        

    
    #train_trajs = generate_darkroom_histories(train_goals, **config)
    #test_trajs = generate_darkroom_histories(test_goals, **config)
    #eval_trajs = generate_darkroom_histories(eval_goals, **config)

    train_filepath = build_darkroom_data_filename(env, n_envs, config, mode=0)
    test_filepath = build_darkroom_data_filename(env, n_envs, config, mode=1)
    eval_filepath = build_darkroom_data_filename(env, 100, config, mode=2)
        

    if not os.path.exists('datasets'):
        os.makedirs('datasets', exist_ok=True)
    with open(train_filepath, 'wb') as file:
        torch.save(train_trajs, file, pickle_protocol=2)  # Use older pickle protocol for compatibility
    with open(test_filepath, 'wb') as file:
        torch.save(test_trajs, file, pickle_protocol=2)  # Use older pickle protocol for compatibility
    with open(eval_filepath, 'wb') as file:
        torch.save(eval_trajs, file, pickle_protocol=2)  # Use older pickle protocol for compatibility

    print(f"Saved to {train_filepath}.")
    print(f"Saved to {test_filepath}.")
    print(f"Saved to {eval_filepath}.")