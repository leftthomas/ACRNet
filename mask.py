import matplotlib.pyplot as plt
import numpy as np
import torch

# BaseballPitch
cluster = torch.load('result/video_test_0000324_cluster.pth').squeeze(dim=0)[:, 0].cpu().numpy()
mask = torch.load('result/video_test_0000324_all.pth').squeeze(dim=0)[:, 0].cpu().numpy()
gt = np.zeros_like(mask)
gt[77:83] = 1
gt[183:191] = 1
frame_indexes = np.arange(0, len(gt))

# cluster = grouping(np.where(cluster == 1.0)[0])
# mask = grouping(np.where(mask == 1.0)[0])
# gt = grouping(np.where(gt == 1.0)[0])

fig, axs = plt.subplots(3, 1, figsize=(7, 3))

axs[0].fill_between(frame_indexes, gt, color='red')
axs[1].fill_between(frame_indexes, cluster, color='blue')
axs[2].fill_between(frame_indexes, mask, color='red')
plt.setp(axs, xticks=[], yticks=[], xlim=(0, len(gt)), ylim=(0, 1))
# fig.set_title('Ground-Truths')
# fig.set_title('Adaptive Clustering')
# fig.set_title('Adaptive Clustering + Mask Refining')

plt.savefig('result/mask.pdf', bbox_inches='tight')
