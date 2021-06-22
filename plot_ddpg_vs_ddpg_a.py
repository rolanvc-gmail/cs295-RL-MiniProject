import numpy as np
import matplotlib.pyplot as plt


ddpg_data = np.load("./results/OurDDPG_BipedalWalker-v3_0_commented.npy")
ddpg_a_data = np.load("./results/DDPG_a_BipedalWalker-v3_0_commented.npy")
ddpg_len = ddpg_data.size
ddpg_a_len = ddpg_a_data.size
print(f"DDPG ran for {ddpg_len*10} episodes.")
print(f"DDPG_a ran for {ddpg_a_len*10} episodes.")
ddpg_eps = [x for x in range(ddpg_len * 10) if x%10==0]
ddpg_a_eps = [x for x in range(ddpg_a_len * 10) if x%10==0]
plt.plot(ddpg_eps, ddpg_data, 'b', label='TD3')
plt.plot(ddpg_a_eps, ddpg_a_data, 'r', label='DDPG_A')
plt.title("DDPG vs DDPG_A: Ave Rewards vs Training Episodes")
plt.xlabel('Training Episodes')
plt.ylabel('ave rewards')

plt.legend()
plt.grid()
plt.savefig('./plots/ddpg_vs_ddpg_a_ave_rewards_vs_eps.png')
plt.show()
