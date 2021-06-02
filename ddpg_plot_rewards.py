import numpy as np
import matplotlib.pyplot as plt

episode = 2063
ddpg_file = f"./results/DDPG_BipedalWalker-v3_0_{episode}.npy"
td3_data = np.load("./results/TD3_BipedalWalker-v3_0.npy")
ddpg_data = np.load( ddpg_file)
print(f"{ddpg_file} has size: {ddpg_data.size}")
plt.plot(td3_data, 'b', label='TD3')
plt.plot(ddpg_data, 'r', label='DDPG')
plt.title("DDPG vs TD3: Ave Rewards vs Timesteps")
plt.xlabel('timesteps')
plt.ylabel('ave rewards')

plt.legend()
plt.grid()
# plt.savefig(f'./plots/td3_vs_ddpg_{episode}.png')
plt.show()
