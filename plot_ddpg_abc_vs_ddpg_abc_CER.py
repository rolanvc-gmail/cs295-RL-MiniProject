import numpy as np
import matplotlib.pyplot as plt


ddpg_td3_data = np.load("./results/TD3_BipedalWalker-v3_0.npy")
ddpg_td3_len = ddpg_td3_data.size
print(f"TD3 ran for {ddpg_td3_len*10} episodes.")
ddpg_td3_eps = [x for x in range(ddpg_td3_len * 10) if x % 10 == 0]

ddpg_cer_data = np.load("./results/TD3-CER_BipedalWalker-v3_0.npy")
ddpg_cer_len = ddpg_cer_data.size
print(f"DDPG-b1e5 ran for {ddpg_cer_len * 10} episodes.")
ddpg_cer_eps = [x for x in range(ddpg_cer_len * 10) if x % 10 == 0]

#ddpg_b1e4_data = np.load("./results/DDPG_abc_b1e4_BipedalWalker-v3_0.npy")
#ddpg_b1e4_len = ddpg_b1e4_data.size
#print(f"DDPG_b1e4 ran for {ddpg_b1e4_len * 10} episodes.")
#ddpg_b1e4_eps = [x for x in range(ddpg_b1e4_len * 10) if x % 10 == 0]

#plt.plot(ddpg_td3_eps, ddpg_td3_data, 'r', label='TD3')
plt.plot(ddpg_td3_eps, ddpg_td3_data, 'r', label='TD3_1M')
plt.plot(ddpg_cer_eps, ddpg_cer_data, 'b', label='TD3_1M_CER')
plt.title("TD3_1M vs TD3_1M_CER: Ave Rewards vs Training Episodes")
plt.xlabel('Training Episodes')
plt.ylabel('ave rewards')

plt.legend()
plt.grid()
plt.savefig('./plots/td3_!M_vs_td3_!M_CER_ave_rewards_vs_eps.png')
plt.show()
