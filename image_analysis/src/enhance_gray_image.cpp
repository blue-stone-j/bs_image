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

// enhance dark regions of an image in-place using gamma correction
void gammaEnhanceInPlace(cv::Mat &img, double gamma = 0.6)
{
  CV_Assert(!img.empty());
  const int depth = img.depth();

  if (depth == CV_8U)
  {
    // 8-bit: LUT-based gamma (in-place overwrite, type preserved)
    cv::Mat lut(1, 256, CV_8U);
    for (int i = 0; i < 256; ++i)
    {
      double v                = std::pow(i / 255.0, gamma) * 255.0;
      lut.at<std::uint8_t>(i) = static_cast<std::uint8_t>(std::clamp(std::lround(v), 0L, 255L));
    }
    cv::LUT(img, lut, img); // overwrites img, keeps CV_8U/CV_8UCn
    return;
  }

  if (depth == CV_16U)
  {
    // 16-bit: convert to float, apply gamma, convert back (type preserved at end)
    cv::Mat tmp32f;
    img.convertTo(tmp32f, CV_32F, 1.0 / 65535.0);
    cv::pow(tmp32f, gamma, tmp32f);
    tmp32f.convertTo(img, CV_16U, 65535.0); // overwrite original; still CV_16U(Cn)
    return;
  }

  if (depth == CV_32F || depth == CV_64F)
  {
    // Float types: direct pow, then write back into same Mat type
    cv::pow(img, gamma, img); // overwrites and preserves CV_32F/CV_64F
    return;
  }

  CV_Error(cv::Error::StsUnsupportedFormat, "Unsupported image depth for gammaEnhanceInPlace");
}

// improve local contrast in dark regions
void claheEnhanceInPlace(cv::Mat &img, double clipLimit = 2.0, cv::Size tileGridSize = {8, 8})
{
  CV_Assert(!img.empty());
  CV_Assert(img.depth() == CV_8U);

  cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clipLimit, tileGridSize);

  if (img.channels() == 1)
  {
    clahe->apply(img, img); // overwrites, keeps CV_8U
    return;
  }

  if (img.channels() == 3)
  {
    // Work in Lab to enhance luminance only; type remains CV_8UC3
    cv::Mat lab;
    cv::cvtColor(img, lab, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> ch;
    cv::split(lab, ch);
    clahe->apply(ch[0], ch[0]); // enhance L
    cv::merge(ch, lab);
    cv::cvtColor(lab, img, cv::COLOR_Lab2BGR); // overwrite original
    return;
  }

  CV_Error(cv::Error::StsBadArg, "CLAHE example expects 1 or 3 channels of CV_8U");
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
