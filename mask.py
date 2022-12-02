import matplotlib.pyplot as plt
import numpy as np
import torch

fig = plt.figure(figsize=(10, 5))

# BaseballPitch
cluster = torch.load('result/video_test_0000324_cluster.pth').squeeze(dim=0)[:, 0].cpu().numpy()
mask = torch.load('result/video_test_0000324_all.pth').squeeze(dim=0)[:, 0].cpu().numpy()
gt = np.zeros_like(mask)
gt[77:83] = 1
gt[183:191] = 1

data = np.stack([gt, mask, cluster], axis=0)

# fig.set_title('Adaptive Clustering')
#
# fig.set_title('Adaptive Clustering + Mask Refining')
#
# fig.set_title('Ground-Truths')

c = plt.pcolor(data)
fig.colorbar(c)
plt.xlim((0, 232))
plt.yticks([])
plt.show()
plt.savefig('result/mask.pdf', bbox_inches='tight', pad_inches=0.1)
