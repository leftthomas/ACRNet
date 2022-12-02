import matplotlib.pyplot as plt
import numpy as np

x = np.array([8, 16, 24, 32, 40, 48, 56, 64])

mil_250 = np.array([31.5, 33.0, 33.2, 33.3, 33.4, 33.2, 33.3, 32.9])
mil_500 = np.array([35.9, 36.9, 37.3, 37.5, 37.4, 37.2, 37.2, 37.1])
mil_750 = np.array([36.7, 37.6, 38.0, 38.1, 38.2, 38.2, 38.1, 38.1])
cluster_250 = np.array([33.6, 33.6, 33.6, 33.6, 33.6, 33.6, 33.6, 33.6])
cluster_500 = np.array([38.2, 38.2, 38.2, 38.2, 38.2, 38.2, 38.2, 38.2])
cluster_750 = np.array([39.0, 39.0, 39.0, 39.0, 39.0, 39.0, 39.0, 39.0])

plt.figure(figsize=(10, 5))
plt.plot(x, mil_250, color='red', linewidth=1.0, label='250', marker='o', markerfacecolor='white', markersize=5)
for a, b in zip(x, mil_250):
    plt.text(a, b - 0.4, str(b), color='red')
plt.plot(x, mil_500, color='blue', linewidth=1.0, label='500', marker='D', markerfacecolor='white', markersize=4)
for a, b in zip(x, mil_500):
    plt.text(a, b - 0.4, str(b), color='blue')
plt.plot(x, mil_750, color='green', linewidth=1.0, label='750', marker='*', markerfacecolor='white')
for a, b in zip(x, mil_750):
    plt.text(a, b - 0.4, str(b), color='green')

plt.plot(x, cluster_250, color='red', linewidth=1.0, linestyle='--')
plt.text(64.1, 33.5, str(33.6), color='red')
plt.plot(x, cluster_500, color='blue', linewidth=1.0, linestyle='--')
plt.text(64.1, 38.1, str(38.2), color='blue')
plt.plot(x, cluster_750, color='green', linewidth=1.0, linestyle='--')
plt.text(64.1, 38.9, str(39.0), color='green')

plt.xticks(x)
plt.xlim((7.7, 64.3))
plt.ylim(bottom=31.0)
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.legend(loc='best')
# plt.show()

plt.savefig('result/cluster.pdf', bbox_inches='tight', pad_inches=0.1)
