#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char **argv)
{
  // 加载图像
  cv::Mat image;
  if (argc == 1)
  {
    image = cv::imread("../assets/1.pgm");
  }
  else
  {
    image = cv::imread(argv[1]);
  }
  if (image.empty())
  {
    std::cout << "图像加载失败" << std::endl;
    return -1;
  }

  // 转换为灰度图像
  cv::Mat grayImage;
  cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

  // 计算直方图
  cv::Mat histogram;
  int histSize           = 256;      // 从0到255
  float range[]          = {0, 256}; // 灰度级的范围
  const float *histRange = {range};
  bool uniform = true, accumulate = false;
  calcHist(&grayImage, 1, 0, cv::Mat(), histogram, 1, &histSize, &histRange, uniform, accumulate);

  // 计算CDF
  cv::Mat cdf;
  histogram.copyTo(cdf);
  for (int i = 1; i < histSize; i++)
  {
    cdf.at<float>(i) += cdf.at<float>(i - 1);
  }
  cdf /= image.total(); // 归一化

  // 寻找最低亮度20%和最高亮度20%的阈值
  float low_thresh = 0.2, high_thresh = 0.8;
  float low_val = 0, high_val = 0;
  for (int i = 0; i < histSize; i++)
  {
    if (cdf.at<float>(i) < low_thresh)
    {
      low_val = i;
    }
    if (cdf.at<float>(i) < high_thresh)
    {
      high_val = i;
    }
  }
  std::cout << "最低亮度的20%阈值是: " << low_val << std::endl;
  std::cout << "最高亮度的20%阈值是: " << high_val << std::endl;

  // 绘制直方图（在C++中通常需要额外的库如OpenCV HighGUI或其他图形库）
  int hist_w = 512, hist_h = 400;
  int bin_w = cvRound((double)hist_w / histSize);
  cv::Mat histImage(hist_h, hist_w, CV_8UC1, cv::Scalar(0, 0, 0));
  normalize(histogram, histogram, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());

  for (int i = 1; i < histSize; i++)
  {
    line(histImage,
         cv::Point(bin_w * (i - 1), hist_h - cvRound(histogram.at<float>(i - 1))),
         cv::Point(bin_w * (i), hist_h - cvRound(histogram.at<float>(i))),
         cv::Scalar(255, 0, 0), 2, 8, 0);
  }

  // 显示原图和直方图
  cv::imshow("原始图像", image);
  cv::imshow("灰度直方图", histImage);
  cv::waitKey(0);

  return 0;
}
