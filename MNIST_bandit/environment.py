import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
from torchvision import datasets, transforms
from gymnasium.utils.env_checker import check_env

class MNISTEnv(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 30}


    def __init__(self, split="train", seed: int | None = None, device="cpu", ):
        super().__init__()
        self.device = torch.device(device)

        tfms = [transforms.ToTensor()]
        self.dataset = datasets.MNIST(root="./data", train=(split == "train"), download=True, transform=transforms.Compose(tfms))

        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(1, 28, 28), dtype=np.float32)

        self.action_space = spaces.Discrete(10)

        self.rng = np.random.default_rng(seed)
        self.current_index = None
        self.current_obs = None
        self.current_label = None
        self.episode_step = 0

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def _get_sample(self):
        idx = self.rng.integers(0, len(self.dataset))
        img, label = self.dataset[idx]

        obs = img.numpy().astype(np.float32)
        return obs, int(label), int(idx)
    
    def reset(self, *, seed: int | None = None, options=None):

        super().reset(seed=seed)

        if seed is not None:
            self.seed(seed)
        
        self.current_obs, self.current_label, self.current_index = self._get_sample()
        self.episode_step = 0
        info = {"index": self.current_index, "label": self.current_label}

        return self.current_obs, info
    
    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"
        self.episode_step += 1

        correct = int(action == self.current_label)
        reward = 1.0 if correct else -1.0
        terminated = True
        truncated = False
        info = {"correct": bool(correct), "label": self.current_label}

        self.current_obs, self.current_label, self.current_index = self._get_sample()

        return self.current_obs, reward, terminated, truncated, info
    
    def render(self):
        if self.current_obs is None:
            print("Env not reset")
            return
        import matplotlib.pyplot as plt
        img = self.current_obs
        plt.imshow(img[0], cmap="gray")
        plt.title(f"Label: {self.current_label}")
        plt.axis("off")
        plt.show()

    def close(self):
        pass

if __name__ == "__main__":
    env = MNISTEnv(split="train", seed=42)
    obs, info = env.reset()
    print("Obs shape:", obs.shape, "Label:", info["label"])

    total = 0
    correct = 0
    for _ in range(100):
        action = env.action_space.sample()  # random policy
        obs, reward, terminated, truncated, info = env.step(action)
        total += 1
        correct += int(info["correct"])
        if terminated:
            env.reset()
    print("Random policy accuracy:", correct / total)