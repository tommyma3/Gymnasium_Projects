import flappy_bird_gymnasium
import gymnasium
# env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=True)
env = gymnasium.make("CartPole-v1", render_mode="human")

obs, _ = env.reset()
while True:
    # Next action:
    # (feed the observation to your agent here)
    action = env.action_space.sample()

    # Processing:
    obs, reward, terminated, _, info = env.step(action)
    
    # Checking if the player is still alive
    if terminated:
        break

env.close()