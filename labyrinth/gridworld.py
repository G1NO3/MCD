import gym
import torch
from torch.distributions import Categorical
from torch.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt

# # Create a sample 4x4 matrix
# matrix = np.array([
#     [1, 2, 3, 4],
#     [5, 6, 7, 8],
#     [9, 10, 11, 12],
#     [13, 14, 15, 16]
# ])

# # Step 1: Transpose the matrix (to swap axes)
# matrix_transposed = matrix.T

# # Step 2: Flip the y-axis (to move the origin to the bottom-left)
# matrix_cartesian = np.flipud(matrix_transposed)

# # Plot using imshow for comparison
# plt.figure(figsize=(10, 5))

# # Original imshow coordinate system
# plt.subplot(1, 2, 1)
# plt.imshow(matrix, cmap='viridis')
# plt.title("Imshow Coordinate System")
# plt.colorbar()

# # Cartesian coordinate system
# plt.subplot(1, 2, 2)
# plt.matshow(matrix, cmap='viridis')
# plt.title("Matshow Coordinate System")
# plt.colorbar()

# plt.show()

env = gym.make('SimpleGridWorld-v0')
env.reset()
print(env.observation_space.sample())
print(env.action_space.shape)
print(env.reset().shape)
while True:
    action = int(input('Enter action: '))
    state, reward, done, _ = env.step(action)
    print('State:{a},reward:{b},done:{c}'.format(a=state,b=reward,c=done))
    if done:
        env.random_reset()
    env_grid = np.zeros((env.n_width, env.n_height))
    env_grid[state[0], state[1]] = 0.5
    reward_loc = env.rewards[0]
    env_grid[reward_loc[0], reward_loc[1]] = 1  
    end = env.ends[0]
    env_grid[end[0], end[1]] = -0.5
    env.render()
    plt.imshow(env_grid, cmap='viridis')
    plt.show()


# p = torch.tensor(np.arange(12).reshape(3,4))
# x = torch.take_along_dim(p, torch.tensor([[0,1]]), -1)
# print(p)
# print(x)
