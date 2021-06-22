import numpy as np
import matplotlib.pyplot as plt


ddpg_td3_data = np.load("./results/DDPG_abc_BipedalWalker-v3_0_commented.npy")
ddpg_td3_len = ddpg_td3_data.size
print(f"TD3 ran for {ddpg_td3_len*10} episodes.")
ddpg_td3_eps = [x for x in range(ddpg_td3_len * 10) if x % 10 == 0]

ddpg_b1e5_data = np.load("./results/DDPG_abc_BipedalWalker-v3_0_b1e5.npy")
ddpg_b1e5_len = ddpg_b1e5_data.size
print(f"DDPG-b1e5 ran for {ddpg_b1e5_len * 10} episodes.")
ddpg_b1e5_eps = [x for x in range(ddpg_b1e5_len * 10) if x % 10 == 0]

ddpg_b1e4_data = np.load("./results/DDPG_abc_b1e4_BipedalWalker-v3_0.npy")
ddpg_b1e4_len = ddpg_b1e4_data.size
print(f"DDPG_b1e4 ran for {ddpg_b1e4_len * 10} episodes.")
ddpg_b1e4_eps = [x for x in range(ddpg_b1e4_len * 10) if x % 10 == 0]

#plt.plot(ddpg_td3_eps, ddpg_td3_data, 'r', label='TD3')
plt.plot(ddpg_b1e5_eps, ddpg_b1e5_data, 'y', label='TD3_b1e5')
plt.plot(ddpg_b1e4_eps, ddpg_b1e4_data, 'g', label='TD3_b1e4')
plt.title("TD3_B1e5 vs TD3_B1e4: Ave Rewards vs Training Episodes")
plt.xlabel('Training Episodes')
plt.ylabel('ave rewards')

plt.legend()
plt.grid()
plt.savefig('./plots/td3_b1e5_vs_td3_b1e4_ave_rewards_vs_eps.png')
plt.show()
