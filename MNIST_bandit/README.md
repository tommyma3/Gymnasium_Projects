# Hand-Written Number Classification using Bandit

## Environment
- Used *MNIST* dataset. Train on the training set, and evaluate the policy using the testing set.
- **Observation**: Hand-Written Number gray scale images, with observation space (1, 28, 28).
- **Action**: Discrete action space with 10 choices, each corresponding to a number. 
- **Reward**: +1 for correct number, -1 for incorrect number.

## Agent
- Policy-based agent
- Using the running average as the baseline.
- Use Policy Gradient to optimize the policy.
- 98.7% accuracy

## Policy Network
- First Convolutional Layer:
    - Input: 1 channel
    - Output: 32 feature maps
    - Kernel Size: 3 * 3, with padding = 1
- Second Convolutional Layer:
    - Input: 32 channels
    - Output: 64 feature maps
    - Kernel Size: 3 * 3, with padding = 1
- Pooling Layer:
    - 2 * 2 max-pooling
- Dropout:
    - Randomly zeroes 25% of features
- Fully Connected Layer:
    - Flatten the feature map
    - Hidden dimension: 128
- Fully Connect Layer:
    - Output Dimension: 10


