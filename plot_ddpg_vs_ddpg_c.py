import numpy as np
import matplotlib.pyplot as plt


ddpg_data = np.load("./results/OurDDPG_BipedalWalker-v3_0_commented.npy")
ddpg_c_data = np.load("./results/DDPG_c_BipedalWalker-v3_0_commented.npy")
ddpg_len = ddpg_data.size
ddpg_c_len = ddpg_c_data.size
print(f"DDPG ran for {ddpg_len*10} episodes.")
print(f"DDPG_c ran for {ddpg_c_len*10} episodes.")
ddpg_eps = [x for x in range(ddpg_len * 10) if x % 10 == 0]
ddpg_c_eps = [x for x in range(ddpg_c_len * 10) if x % 10 == 0]
plt.plot(ddpg_eps, ddpg_data, 'r', label='DDPG')
plt.plot(ddpg_c_eps, ddpg_c_data, 'g', label='DDPG_c')
plt.title("DDPG vs DDPG_C: Ave Rewards vs Training Episodes")
plt.xlabel('Training Episodes')
plt.ylabel('ave rewards')

plt.legend()
plt.grid()
plt.savefig('./plots/ddpg_vs_ddpg_c_ave_rewards_vs_eps.png')
plt.show()
