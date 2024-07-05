from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 使用Scikit-learn内置样例图像
#china = load_sample_image("china.jpg")

# 或者从本地加载图像
image = Image.open("bird_small.png")
image = np.array(image)

# 归一化像素值到0-1之间
image_array = image / 255.0

# 将图像展平为一维数组，每行为一个像素点的RGB值
flat_image = image_array.reshape((-1, 3))
# 设置聚类数量，例如将色彩减少到16种
n_colors = 16

# 实例化KMeans模型并拟合数据
kmeans = KMeans(n_clusters=n_colors)
kmeans.fit(flat_image)

# 获取每个像素点所属的聚类中心索引
labels = kmeans.predict(flat_image)
# 将聚类中心索引映射回实际颜色值
compressed_image = kmeans.cluster_centers_[labels]

# 重塑为原始图像尺寸
compressed_image = compressed_image.reshape(image_array.shape)

# 显示原图和压缩后的图像
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(image)
ax[0].set_title('Original Image')
ax[1].imshow(compressed_image)
ax[1].set_title('Compressed Image ({} Colors)'.format(n_colors))
plt.show()

