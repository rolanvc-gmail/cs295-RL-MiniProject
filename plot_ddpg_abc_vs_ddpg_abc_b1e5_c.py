import numpy as np
import matplotlib.pyplot as plt


ddpg_abc_data = np.load("./results/DDPG_abc_BipedalWalker-v3_0_commented.npy")
ddpg_b1e5_data = np.load("./results/DDPG_abc_BipedalWalker-v3_0_b1e5.npy")
ddpg_abc_len = ddpg_abc_data.size
ddpg_b1e5_len = ddpg_b1e5_data.size
print(f"DDPG ran for {ddpg_abc_len*10} episodes.")
print(f"DDPG_b5e5 ran for {ddpg_b1e5_len*10} episodes.")
ddpg_abc_eps = [x for x in range(ddpg_abc_len * 10) if x % 10 == 0]
ddpg_b5e5_eps = [x for x in range(ddpg_b1e5_len * 10) if x % 10 == 0]
plt.plot(ddpg_abc_eps, ddpg_abc_data, 'r', label='TD3')
plt.plot(ddpg_b5e5_eps, ddpg_b1e5_data, 'g', label='DDPG_b1e5')
plt.title("TD3_1e6 vs TD3_B1e5: Ave Rewards vs Training Episodes")
plt.xlabel('Training Episodes')
plt.ylabel('ave rewards')

plt.legend()
plt.grid()
plt.savefig('./plots/td3_vs_td3_b1e5_ave_rewards_vs_eps.png')
plt.show()
