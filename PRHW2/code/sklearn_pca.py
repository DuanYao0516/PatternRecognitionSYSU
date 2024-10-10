import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os

# Ensure the directories exist
if not os.path.exists('../results/PCA'):
    os.makedirs('../results/PCA')

# Load data
mat_data = sio.loadmat('../data/faces.mat')
X = mat_data['X']

# Only take the first 1024 columns and reshape for processing
# 先取所有数据的前1024个特征，然后将数据reshape为32*32，然后将数据进行转置（交换第二三个维度），之后重新reshape回原有形状
X = X[:, :1024].reshape(-1, 32, 32).transpose(0, 2, 1).reshape(-1, 1024)

# 假设 X 已经按照你前面的方法进行处理
# 对数据集应用 PCA
pca = PCA(n_components=49)
pca.fit(X)

# 获得前 49 个主成分
components = pca.components_

# 可视化前 49 个主成分
fig, axes = plt.subplots(7, 7, figsize=(8, 8))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(components[i].reshape(32, 32), cmap='gray')
    ax.axis('off')
plt.show()