import numpy as np
import matplotlib.pyplot as plt

ddpg_data = np.load("./results/DDPG_BipedalWalker-v3_0.npy")
c_ddpg_data = np.load("./results/OurDDPG_BipedalWalker-v3_0_commented.npy")
ddpg_len = ddpg_data.size #116
ourddpg_data_len = c_ddpg_data.size # 118 elements
print(f"DDPG ran for {ourddpg_data_len*10} episodes.")
print(f"C_DDPG ran for {ddpg_len*10} episodes.")
td3_eps = [x for x in range(ourddpg_data_len* 10) if x%10==0]
ddpg_eps = [x for x in range(ddpg_len * 10) if x%10==0]
plt.plot(td3_eps, c_ddpg_data, 'b', label='Commented DDPG')
plt.plot(ddpg_eps, ddpg_data, 'r', label='DDPG')
plt.title("DDPG vs Commented DDPG: Ave Rewards vs Training Episodes")
plt.xlabel('Training Episodes')
plt.ylabel('ave rewards')

plt.legend()
plt.grid()
plt.savefig('./plots/td3_vs_ddpg_ave_rewards_vs_eps.png')
plt.show()
