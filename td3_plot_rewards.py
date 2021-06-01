import numpy as np
import matplotlib.pyplot as plt


td3_data = np.load("./results/TD3_BipedalWalker-v3_0.npy")
len = td3_data.size # 118 elements
eps = [x for x in range(118 *10) if x%10==0]
print(f"TD3 ran for {len*10} episodes.")
#ddpg_data = np.load("./results/DDPG_BipedalWalker-v3_0.npy")
plt.plot(eps, td3_data, 'b', label='TD3')
#plt.plot(ddpg_data, 'r', label='DDPG')
plt.title("DDPG vs TD3: Ave Rewards vs Training Episodes")
plt.xlabel('Training Episodes')
plt.ylabel('ave rewards')

plt.legend()
plt.grid()
plt.savefig('./plots/td3_ave_rewards_vs_eps.png')
plt.show()
