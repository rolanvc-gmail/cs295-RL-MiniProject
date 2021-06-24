import numpy as np
import matplotlib.pyplot as plt


td3_100k_data = np.load("./results/DDPG_abc_b1e5_BipedalWalker-v3_0_b1e5.npy")
td3_100k_len = td3_100k_data.size
print(f"TD3 ran for {td3_100k_len*10} episodes.")
td3_100k_eps = [x for x in range(td3_100k_len * 10) if x % 10 == 0]

td3_100k_cer_data = np.load("./results/DDPG_abc_b1e5_CER_BipedalWalker-v3_0_b1e5.npy")
td3_100k_cer_len = td3_100k_cer_data.size
print(f"DDPG-b1e5 ran for {td3_100k_cer_len * 10} episodes.")
td3_100k_cer_eps = [x for x in range(td3_100k_cer_len * 10) if x % 10 == 0]

#ddpg_b1e4_data = np.load("./results/DDPG_abc_b1e4_BipedalWalker-v3_0.npy")
#ddpg_b1e4_len = ddpg_b1e4_data.size
#print(f"DDPG_b1e4 ran for {ddpg_b1e4_len * 10} episodes.")
#ddpg_b1e4_eps = [x for x in range(ddpg_b1e4_len * 10) if x % 10 == 0]

#plt.plot(ddpg_td3_eps, ddpg_td3_data, 'r', label='TD3')
plt.plot(td3_100k_eps, td3_100k_data, 'r', label='TD3_100k')
plt.plot(td3_100k_cer_eps, td3_100k_cer_data, 'b', label='TD3_100k_CER')
plt.title("TD3_100k vs TD3_100k_CER: Ave Rewards vs Training Episodes")
plt.xlabel('Training Episodes')
plt.ylabel('ave rewards')

plt.legend()
plt.grid()
plt.savefig('./plots/td3_100k_vs_td3_100k_CER_ave_rewards_vs_eps.png')
plt.show()
