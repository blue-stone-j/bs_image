#include <opencv2/opencv.hpp>
#include <iostream>

int main()
{
  // Load grayscale image
  cv::Mat img = cv::imread("../assets/sky.jpg", cv::IMREAD_GRAYSCALE);
  if (img.empty())
  {
    std::cerr << "Failed to load image." << std::endl;
    return -1;
  }
  cv::imshow("Original", img);


  // Method 1: Uses Laplacian to capture edges and subtracts it from the original to enhance edges.
  cv::Mat laplacian_sharpened;
  cv::Mat laplacian;
  cv::Laplacian(img, laplacian, CV_16S, 3);
  cv::Mat laplacian_abs;
  cv::convertScaleAbs(laplacian, laplacian_abs);
  cv::addWeighted(img, 1.0, laplacian_abs, -1.0, 0, laplacian_sharpened);
  cv::imshow("laplacian", laplacian_sharpened);

  // Method 2: Uses a sharpening kernel that enhances the center pixel while subtracting neighboring pixel contributions.
  cv::Mat kernel_sharpened;
  cv::Mat kernel = (cv::Mat_<float>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
  cv::filter2D(img, kernel_sharpened, img.depth(), kernel);
  cv::imshow("kernel", kernel_sharpened);

  cv::Mat blurred;
  cv::Mat sharpened;
  cv::GaussianBlur(img, blurred, cv::Size(0, 0), 1);
  cv::addWeighted(img, 1.5, blurred, -0.5, 0, sharpened);
  cv::imshow("blurred", sharpened);

  // Show result
  cv::waitKey(0);

  return 0;
}
