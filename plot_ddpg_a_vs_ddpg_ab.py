import numpy as np
import matplotlib.pyplot as plt


ddpg_a_data = np.load("./results/DDPG_a_BipedalWalker-v3_0_commented.npy")
ddpg_ab_data = np.load("./results/DDPG_ab_BipedalWalker-v3_0_commented.npy")
ddpg_a_len = ddpg_a_data.size
ddpg_ab_len = ddpg_ab_data.size
print(f"DDPG_A ran for {ddpg_a_len*10} episodes.")
print(f"DDPG_AB ran for {ddpg_ab_len*10} episodes.")
ddpg_a_eps = [x for x in range(ddpg_a_len * 10) if x % 10 == 0]
ddpg_ab_eps = [x for x in range(ddpg_ab_len * 10) if x % 10 == 0]
plt.plot(ddpg_a_eps, ddpg_a_data, 'b', label='DDPG_A')
plt.plot(ddpg_ab_eps, ddpg_ab_data, 'r', label='DDPG_AB')
plt.title("DDPG_A vs DDPG_AB: Ave Rewards vs Training Episodes")
plt.xlabel('Training Episodes')
plt.ylabel('ave rewards')

plt.legend()
plt.grid()
plt.savefig('./plots/ddpg_a_vs_ddpg_ab_ave_rewards_vs_eps.png')
plt.show()
