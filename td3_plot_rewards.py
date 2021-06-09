import numpy as np
import matplotlib.pyplot as plt


td3_data = np.load("./results/TD3_BipedalWalker-v3_0.npy")
ddpg_data = np.load("./results/DDPG_BipedalWalker-v3_0.npy")
td3_len = td3_data.size # 118 elements
ddpg_len = ddpg_data.size #116
print(f"TD3 ran for {td3_len*10} episodes.")
print(f"DDPG ran for {ddpg_len*10} episodes.")
td3_eps = [x for x in range(td3_len * 10) if x%10==0]
ddpg_eps = [x for x in range(ddpg_len * 10) if x%10==0]
plt.plot(td3_eps, td3_data, 'b', label='TD3')
plt.plot(ddpg_eps, ddpg_data, 'r', label='DDPG')
plt.title("DDPG vs TD3: Ave Rewards vs Training Episodes")
plt.xlabel('Training Episodes')
plt.ylabel('ave rewards')

plt.legend()
plt.grid()
plt.savefig('./plots/td3_vs_ddpg_ave_rewards_vs_eps.png')
plt.show()
