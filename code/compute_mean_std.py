import glob
import os
import sys
from multiprocessing.dummy import Pool as ThreadPool

import cv2
import numpy as np

progress = 0
search_path = os.path.join('../tcdata/suichang_round1_train_210120/*.tif')
files = glob.glob(search_path)
files.sort()

means = [0.356, 0.069, 0.086, 0.077]
stds = [0.075, 0.031, 0.034, 0.029]


def processing_data(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    image = np.asarray(image, np.float32) / 255.0
    for i in range(4):
        means[i] += image[:, :, i].mean()
        stds[i] += image[:, :, i].std()
    global progress
    progress += 1
    print("\rProgress: {:>3} %".format(progress * 100 / len(files)), end=' ')
    sys.stdout.flush()


print('\nProcessing {} images'.format(len(files)))
print("Progress: {:>3} %".format(progress * 100 / len(files)), end=' ')
pool = ThreadPool()
pool.map(processing_data, files)
pool.close()
pool.join()
means = np.asarray(means) / len(files)
stds = np.asarray(stds) / len(files)
print(means, stds)
