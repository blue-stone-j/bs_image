#include <opencv2/opencv.hpp>

int main(int argc, char **argv)
{
  // Load input image in grayscale
  cv::Mat src = cv::imread("../../assets/sky.jpg", cv::IMREAD_GRAYSCALE);
  if (src.empty())
  {
    std::cerr << "Error: Could not load image." << std::endl;
    return -1;
  }

  // Calculate gradients using Sobel operator
  cv::Mat grad_x, grad_y;
  cv::Sobel(src, grad_x, CV_32F, 1, 0, 3);
  cv::Sobel(src, grad_y, CV_32F, 0, 1, 3);
  // use cv::Scharr instead of cv::Sobel for better edge sensitivity
  // cv::Scharr(src, grad_x, CV_32F, 1, 0);
  // cv::Scharr(src, grad_y, CV_32F, 0, 1);

  // Compute gradient magnitude
  cv::Mat grad_mag;
  cv::magnitude(grad_x, grad_y, grad_mag);

  // Normalize to 0-255 and convert to CV_8U for display
  cv::Mat grad_display;
  cv::normalize(grad_mag, grad_display, 0, 255, cv::NORM_MINMAX);
  grad_display.convertTo(grad_display, CV_8U);

  // Save and show result
  cv::imwrite("gradient_image.jpg", grad_display);
  cv::imshow("Gradient Image", grad_display);
  cv::waitKey(0);

  return 0;
}
