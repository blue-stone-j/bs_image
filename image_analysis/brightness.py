import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt

def main(image_path):
    # 加载图像
    if image_path is None:
        # image = cv2.imread('./asserts/1.pgm')
        image = cv2.imread('./asserts/sky.jpg')
    else:
        image = cv2.imread(image_path)

    if image is None:
        print("图像加载失败")
        return -1

    # 转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算直方图
    histSize = 256
    histRange = (0, 256)
    histogram = cv2.calcHist([gray_image], [0], None, [histSize], histRange)

    # 计算CDF
    cdf = histogram.cumsum()
    cdf_normalized = cdf / cdf.max()

    # 寻找最低亮度20%和最高亮度20%的阈值
    low_thresh = 0.2
    high_thresh = 0.8
    low_val = np.searchsorted(cdf_normalized, low_thresh)  # 获取20%亮度的索引
    high_val = np.searchsorted(cdf_normalized, high_thresh)  # 获取80%亮度的索引

    print(f"最低亮度的20%阈值是: {low_val}")
    print(f"最高亮度的20%阈值是: {high_val}")

    # 绘制直方图
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('image')
    plt.axis('off')
    
    plt.subplot(122)
    plt.plot(histogram)
    plt.plot([low_val, low_val], [0, cdf_normalized[low_val]], color='red', linestyle='--')
    plt.plot([high_val, high_val], [0, cdf_normalized[high_val]], color='blue', linestyle='--')
    plt.title('histogram')
    plt.xlabel('Intensity')  # 横坐标标题
    plt.ylabel('Pixel Count')  # 纵坐标标题
    plt.xlim([0, 256])

    plt.show()

if __name__ == '__main__':
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(image_path)

