import numpy as np
import matplotlib.pyplot as plt

ddpg_data = np.load("./results/OurDDPG_BipedalWalker-v3_0_commented.npy")
ddpg_len = ddpg_data.size
print(f"DDPG-A ran for {ddpg_len*10} episodes.")
ddpg_eps = [x for x in range(ddpg_len* 10) if x%10==0]
plt.plot(ddpg_eps, ddpg_data, 'b', label='DDPG')
plt.title("DDPG: Ave Rewards vs Training Episodes")
plt.xlabel('Training Episodes')
plt.ylabel('ave rewards')

plt.legend()
plt.grid()
plt.savefig('./plots/ddpg_only_ave_rewards_vs_eps.png')
plt.show()
