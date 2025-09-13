import argparse
import time
from torchvision import datasets, transforms
from policy import Policy
from environment import MNISTEnv

import torch
from torch import nn
import torch.nn.functional as F

import itertools
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import yaml
import datetime
import random
from datetime import datetime, timedelta

DATE_FORMAT = "%Y%m%d-%H%M%S"


RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

class Agent:

    def __init__(self, hyperparameter_set):
        with open("hyperparameters.yml", "r") as f:
            all_hyperparameter_sets = yaml.safe_load(f)
            hp = all_hyperparameter_sets[hyperparameter_set]

            self.hyperparameter_set = hyperparameter_set

            self.split = hp.get("split", "train")  
            self.eval_split = hp.get("eval_split", "test")
            self.render_eval = hp.get("render_eval", False)
            self.env_device = hp.get("env_device", "cpu")
            self.lr = hp.get("learning_rate", 3e-4)
            self.entropy_coef = hp.get("entropy_coef", 0.01)
            self.baseline_tau = hp.get("baseline_tau", 0.05)
            self.eval_interval_episodes = hp.get("eval_interval_episodes", 200)
            self.target_accuracy = hp.get("target_accuracy", None)  
            self.patience = hp.get("patience", 3)
            self.max_wall_seconds = hp.get("max_wall_seconds", None)
            self.seed = hp.get("seed", 42)

            if self.max_wall_seconds == "None":
                self.max_wall_seconds = None


            self.loss_fn = None
            self.optimizer = None

            self.LOG_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
            self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
            self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')

            self.loss_history = []
            self.eval_reward_history = []
            self.eval_accuracy_history = []
            self.reward_history = []


    def _obs_to_tensor(self, obs):
        x = torch.tensor(obs, dtype=torch.float32, device=device)
        return x

    def evaluate(self, policy, env, steps=10_000, render=False):
        with torch.no_grad():
            policy.eval()
            correct = 0
            total = 0
            total_reward = 0.0

            for _ in range(steps):
                obs, info = env.reset()
                x = self._obs_to_tensor(obs).unsqueeze(0)  
                logits = policy(x)
                action = logits.argmax(dim=-1).item()     
                obs2, reward, terminated, truncated, info2 = env.step(action)
                total_reward += float(reward)
                correct += int(info2["correct"])
                total += 1

                if render and (_ % 500 == 0):  
                    env.render()

            avg_reward = total_reward / max(1, total)             
            accuracy = correct / max(1, total)                     
            policy.train()
            return avg_reward, accuracy 
    


    def run(self, is_training=True, render=False):
        
        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time 
            log_message = f"{start_time.strftime(DATE_FORMAT)} Starting training with hyperparameter set '{self.hyperparameter_set}'"
            print(log_message)
            with open(self.LOG_FILE, 'a') as file:
                file.write(log_message + '\n')

        env = MNISTEnv(split=self.split, seed=self.seed, device=self.env_device)
        eval_env = MNISTEnv(split=self.eval_split, seed=self.seed + 999, device=self.env_device)

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        policy = Policy().to(device)

        if is_training:
            self.optimizer = torch.optim.Adam(policy.parameters(), lr=self.lr)
            baseline = 0.0
            best_eval = -float("inf")
            good_streak = 0
            start_wall = time.time()
        else:
            policy.load_state_dict(torch.load(self.MODEL_FILE))
            policy.eval()

        for episode in itertools.count():
            obs, _ = env.reset()
            obs_t =self._obs_to_tensor(obs)

            logits = policy(obs_t.unsqueeze(0))
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample().item()

            next_obs, reward, terminated, truncated, info = env.step(action)
            reward_f = float(reward)
            self.reward_history.append(reward_f)
            
            if is_training:
                log_probs = F.log_softmax(logits, dim=-1)
                chosen_logp = log_probs[0, action]
                entropy = -(probs * log_probs).sum(dim=-1).mean()

                baseline = (1 - self.baseline_tau) * baseline + self.baseline_tau * reward_f
                adv = torch.tensor(reward_f - baseline, dtype=torch.float32, device=device)

                policy_loss = -(chosen_logp * adv.detach())
                loss = policy_loss - self.entropy_coef * entropy
                self.loss_history.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                self.optimizer.step()

                if (episode + 1) % self.eval_interval_episodes == 0:
                    eval_reward, eval_acc = self.evaluate(policy, eval_env, steps=10000)
                    self.eval_reward_history.append(eval_reward)
                    self.eval_accuracy_history.append(eval_acc)

                    msg = (
                        f"{datetime.now().strftime(DATE_FORMAT)} "
                        f"Episode {episode+1} | mean_train_reward(last {self.eval_interval_episodes})="
                        f"{np.mean(self.reward_history[-self.eval_interval_episodes:]):.4f} | "
                        f"eval_reward={eval_reward:.4f} | eval_accuracy={eval_acc:.4f} | baseline={baseline:.3f}"
                    )
                    print(msg)
                    with open(self.LOG_FILE, "a") as file:
                        file.write(msg + "\n")

                    if eval_acc > best_eval:
                        torch.save(policy.state_dict(), self.MODEL_FILE)
                        best_eval = eval_acc
                        best_msg = f"{datetime.now().strftime(DATE_FORMAT)} New best eval accuracy {best_eval:.4f} at episode {episode+1}"
                        print(best_msg)
                        with open(self.LOG_FILE, "a") as file:
                            file.write(best_msg + "\n")

                    if self.target_accuracy is not None:
                        if eval_acc >= self.target_accuracy:
                            good_streak += 1
                        else:
                            good_streak = 0
                        if good_streak >= self.patience:
                            stop_msg = (
                                f"Early stop: eval_accuracy ≥ {self.target_accuracy:.3f} "
                                f"for {self.patience} consecutive evals."
                            )
                            print(stop_msg)
                            with open(self.LOG_FILE, "a") as file:
                                file.write(stop_msg + "\n")
                            break

                    if self.max_wall_seconds is not None and (time.time() - start_wall) > self.max_wall_seconds:
                        print("Stop due to time limit.")
                        break

                    self.save_graph(self.reward_history, self.eval_accuracy_history)
            
            else:
                eval_reward, eval_acc = self.evaluate(policy, eval_env, steps=10000, render=render)
                print(f"Avg reward: {eval_reward:.4f} | Accuracy: {eval_acc:.4f}")
                break
        
        if is_training:
            torch.save(policy.state_dict(), self.MODEL_FILE)



    def save_graph(self, rewards_per_episode, eval_accuracy_history):
        fig = plt.figure(figsize=(19, 4))

        ax1 = fig.add_subplot(1, 2, 1)
        ax1.set_title("Episode Reward")
        ax1.set_ylabel("Reward")
        ax1.set_xlabel("Episode")
        if len(rewards_per_episode) > 0:
            window = 100
            xs = np.arange(len(rewards_per_episode))
            if len(rewards_per_episode) >= window:
                ma = np.convolve(rewards_per_episode, np.ones(window) / window, mode="valid")
                ax1.plot(np.arange(window - 1, window - 1 + len(ma)), ma, label="Mean reward")
            ax1.plot(xs, rewards_per_episode, color="gray", alpha=0.3, label="Reward")
            ax1.legend()

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.set_title("Evaluation Accuracy")
        ax2.set_ylabel("Accuracy")
        ax2.set_xlabel(f"Every {self.eval_interval_episodes} episodes")
        if len(eval_accuracy_history) > 0:
            ax2.plot(
                np.arange(1, len(eval_accuracy_history) + 1) * self.eval_interval_episodes,
                eval_accuracy_history,
                marker="o",
            )

        fig.tight_layout()
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    agent = Agent(hyperparameter_set=args.hyperparameters)

    if args.train:
        agent.run(is_training=True)
    else:
        agent.run(is_training=False, render=True)
