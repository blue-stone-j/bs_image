#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

/**
 * @brief Enhance grayscale image detail around a specified gray level while compressing low/high gray levels.
 * 
 * @param img_gray Input grayscale image (CV_8UC1)
 * @param x0 Gray level to enhance (default 125)
 * @param k Steepness of the sigmoid (default 0.08); increase for steeper contrast or decrease for smoother enhancement.
 * @return cv::Mat Enhanced grayscale image
 */
cv::Mat enhanceGrayDetail(const cv::Mat &img_gray, double x0 = 125.0, double k = 0.08)
{
  CV_Assert(img_gray.type() == CV_8UC1);

  cv::Mat lut(1, 256, CV_8UC1); // LUT ensures high performance even on large images.
  for (int i = 0; i < 256; ++i)
  {
    double y         = 255.0 / (1.0 + std::exp(-k * (static_cast<double>(i) - x0)));
    lut.at<uchar>(i) = static_cast<uchar>(std::clamp(y, 0.0, 255.0));
  }

  cv::Mat enhanced;
  cv::LUT(img_gray, lut, enhanced);
  return enhanced;
}

int main(int argc, char **argv)
{
  if (argc < 2)
  {
    std::cout << "Usage: ./enhance_gray_image <gray_image_path>" << std::endl;
    return -1;
  }

  cv::Mat img_gray = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
  if (img_gray.empty())
  {
    std::cerr << "Failed to load image: " << argv[1] << std::endl;
    return -1;
  }

  cv::Mat enhanced_img = enhanceGrayDetail(img_gray, 125.0, 0.08);

  cv::imshow("Original", img_gray);
  cv::imshow("Enhanced", enhanced_img);
  //   cv::imwrite("enhanced_gray_image.png", enhanced_img);

  cv::waitKey(0);
  return 0;
}
