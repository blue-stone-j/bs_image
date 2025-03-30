

#include "canny.h"

int main(int argc, char **argv)
{
  cv::Mat img = imread("../assets/woman.jpg", cv::IMREAD_GRAYSCALE); // 从文件中加载灰度图像

  // 读取图片失败，则停止
  if (img.empty())
  {
    printf("读取图像文件失败");
    system("pause");
    return 0;
  }

  // 高斯滤波
  cv::Mat gauss_img;
  gaussianFilter(img, gauss_img); // 高斯滤波器

  // 用一阶偏导有限差分计算梯度幅值和方向
  cv::Mat gradXY, theta;
  getGrandient(gauss_img, gradXY, theta);

  // 局部非极大值抑制
  cv::Mat local_img;
  nonLocalMaxValue(gradXY, theta, local_img);

  // 用双阈值算法检测和连接边缘
  cv::Mat dst;
  doubleThreshold(40, 80, local_img, dst);

  // 图像显示
  cv::Mat outImg;
  mergeImg(outImg, img, dst); // 图像拼接
  cv::namedWindow("img");
  cv::imshow("img", outImg); // 图像显示
  // cv::imwrite("canny算法.jpg", outImg);

  cv::waitKey(); // 等待键值输入
  return 0;
}