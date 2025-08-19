/*
judge whether image is blurry and how blurry it is

| Method                    | Advantages          | Limitations                          |
| ------------------------- | ------------------- | ------------------------------------ |
| Variance of Laplacian     | Simple, effective   | Sensitive to noise, threshold tuning |
| Tenengrad                 | Sensitive to edges  | May misjudge low-texture images      |
| Frequency Domain Analysis | Theoretical clarity | Computationally heavy                |
| Deep Learning-based       | Adaptable, precise  | Requires dataset & training          |
*/


#include <opencv2/opencv.hpp>
#include <iostream>

// Compute high-frequency energy ratio
double computeBlurFFT(const cv::Mat &gray, double radiusRatio = 0.1)
{
  // radiusRatio = 0.1 means we exclude the innermost 10% (i.e., the low frequencies) when summing "high frequency" energy.
  // Convert to float
  cv::Mat floatImg;
  gray.convertTo(floatImg, CV_32F);

  // Expand to complex matrix
  cv::Mat planes[] = {floatImg, cv::Mat::zeros(gray.size(), CV_32F)};
  cv::Mat complexImg;
  cv::merge(planes, 2, complexImg);

  // Perform DFT
  cv::dft(complexImg, complexImg);

  // Compute magnitude
  cv::split(complexImg, planes);
  cv::magnitude(planes[0], planes[1], planes[0]);
  cv::Mat mag = planes[0];

  // Shift zero-frequency to center
  int cx = mag.cols / 2;
  int cy = mag.rows / 2;
  cv::Mat q0(mag, cv::Rect(0, 0, cx, cy));
  cv::Mat q1(mag, cv::Rect(cx, 0, cx, cy));
  cv::Mat q2(mag, cv::Rect(0, cy, cx, cy));
  cv::Mat q3(mag, cv::Rect(cx, cy, cx, cy));
  cv::Mat tmp;
  q0.copyTo(tmp);
  q3.copyTo(q0);
  tmp.copyTo(q3);
  q1.copyTo(tmp);
  q2.copyTo(q1);
  tmp.copyTo(q2);

  // Compute high-frequency energy ratio
  double totalEnergy = 0.0, highFreqEnergy = 0.0;
  int rows = mag.rows, cols = mag.cols;
  double r_max = radiusRatio * std::sqrt(rows * rows + cols * cols) / 2.0;

  for (int y = 0; y < rows; ++y)
  {
    float *row = mag.ptr<float>(y);
    for (int x = 0; x < cols; ++x)
    {
      double dx   = x - cols / 2;
      double dy   = y - rows / 2;
      double dist = std::sqrt(dx * dx + dy * dy);
      double val  = row[x];
      totalEnergy += val;
      if (dist > r_max)
      {
        highFreqEnergy += val;
      }
    }
  }

  return highFreqEnergy / totalEnergy;
}

int main()
{
  cv::Mat img = cv::imread("../../assets/桂林.jpg", cv::IMREAD_GRAYSCALE);
  if (img.empty())
  {
    std::cerr << "Image not found\n";
    return -1;
  }

  // Variance of Laplacian
  {
    cv::Mat lap;
    cv::Laplacian(img, lap, CV_64F);
    cv::Scalar mu, sigma;
    cv::meanStdDev(lap, mu, sigma);
    double variance = sigma.val[0] * sigma.val[0];

    std::cout << "Variance of Laplacian: " << variance << std::endl;

    double threshold = 100.0; // Example value, tune for your camera
    if (variance < threshold)
    {
      std::cout << "Image is blurry\n";
    }
    else
    {
      std::cout << "Image is sharp\n";
    }
  }

  // Frequency domain analysis
  {
    double blurScore = computeBlurFFT(img);
    std::cout << "High-frequency ratio: " << blurScore << std::endl;

    double threshold = 0.05; // depends on your image size and dataset
    if (blurScore < threshold)
    {
      std::cout << "Image is blurry.\n";
    }
    else
    {
      std::cout << "Image is sharp.\n";
    }
  }


  return 0;
}
