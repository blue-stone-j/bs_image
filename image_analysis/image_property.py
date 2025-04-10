import cv2
import numpy as np
from PIL import Image

def compute_percentile(data, percentile):
    return np.percentile(data, percentile)

def analyze_image(image_path):
    # 加载图像
    image = Image.open(image_path)
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # 特性1: 分辨率
    resolution = image_cv.shape[0] * image_cv.shape[1]
    print(f"分辨率: {image_cv.shape[0]}x{image_cv.shape[1]} ({resolution} 像素)[1920x1080]")

    # 特性2: 颜色多样性（评估图像中颜色的数量）
    unique_colors = len(np.unique(image_cv.reshape(-1, image_cv.shape[2]), axis=0))
    print(f"唯一颜色数: {unique_colors}[>10000]")

    # 特性3: 噪声水平评估（计算图像的标准差）
    noise_level = np.std(image_cv)
    print(f"噪声水平（标准差）: {noise_level:.2f}[<10]")

    # 特性4: 检查动态范围
    gray_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    p5 = compute_percentile(gray_image, 5)
    p95 = compute_percentile(gray_image, 95)
    robust_dynamic_range = p95 - p5
    print(f"动态范围 (使用5th和95th百分位数): {robust_dynamic_range} (P5: {p5}, P95: {p95})[>150]")
    # min_pixel, max_pixel = image_cv.min(), image_cv.max()
    # dynamic_range = max_pixel - min_pixel
    # print(f"动态范围: {dynamic_range} (最小值: {min_pixel}, 最大值: {max_pixel})")

    # 特性5: 纹理分析（使用Laplacian算子计算图像的二阶导数）
    laplacian_var = cv2.Laplacian(image_cv, cv2.CV_64F).var()
    print(f"纹理锐度（Laplacian方差）: {laplacian_var}[>100]")

    # 特性6: 光照条件分析（计算图像的平均亮度）
    hsv_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)
    brightness = hsv_image[:,:,2].mean()
    print(f"平均亮度: {brightness}[=128,0-255]")

def main():
    image_path = "./asserts/1.pgm"  # 更改为您的图片路径
    # image_path = "./asserts/sky.jpg"  # 更改为您的图片路径
    analyze_image(image_path)

if __name__ == "__main__":
    main()
