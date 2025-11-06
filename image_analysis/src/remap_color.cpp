/*
get color remapping from a single reference patch
*/

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>

/* ---- sRGB <-> linear helpers ---- */
static inline float srgbToLinear(float c)
{
  return (c <= 0.04045f) ? (c / 12.92f) : std::pow((c + 0.055f) / 1.055f, 2.4f);
}
static inline float linearToSrgb(float c)
{
  return (c <= 0.0031308f) ? (c * 12.92f) : (1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f);
}

static cv::Mat toLinearSRGB(const cv::Mat &bgr8)
{
  cv::Mat f32;
  bgr8.convertTo(f32, CV_32F, 1.0 / 255.0);
  cv::Mat lin = f32.clone();
  lin.forEach<cv::Vec3f>([](cv::Vec3f &px, const int *) {
    for (int c = 0; c < 3; ++c) px[c] = srgbToLinear(px[c]);
  });
  return lin;
}
static cv::Mat toSRGB8(const cv::Mat &lin)
{
  cv::Mat srgb = lin.clone();
  srgb.forEach<cv::Vec3f>([](cv::Vec3f &px, const int *) {
    for (int c = 0; c < 3; ++c)
      px[c] = linearToSrgb(std::max(0.0f, std::min(1.0f, px[c])));
  });
  cv::Mat u8;
  srgb.convertTo(u8, CV_8U, 255.0);
  return u8;
}

/**
 * Map colors from the new environment to the standard environment
 * using ONE reference patch (possibly non-neutral).
 *
 * @param srcBGR8      CV_8UC3, sRGB-encoded, BGR order.
 * @param measuredRGB  Reference in the new env (RGB, 0..255).
 * @param targetRGB    Desired value in the standard env (RGB, 0..255).
 */
static cv::Mat remapFromSinglePatch(
    const cv::Mat &srcBGR8,
    const cv::Vec3b &measuredRGB,
    const cv::Vec3b &targetRGB)
{
  // Whole image to linear (BGR float)
  cv::Mat linBGR = toLinearSRGB(srcBGR8);

  auto toLin = [](int v) -> float { return srgbToLinear(v / 255.0f); };
  // Gains computed in RGB order
  const float eps = 1e-9f;
  cv::Vec3f measLinRGB(toLin(measuredRGB[0]), toLin(measuredRGB[1]), toLin(measuredRGB[2]));
  cv::Vec3f targLinRGB(toLin(targetRGB[0]), toLin(targetRGB[1]), toLin(targetRGB[2]));
  cv::Vec3f gRGB(
      targLinRGB[0] / std::max(eps, measLinRGB[0]),
      targLinRGB[1] / std::max(eps, measLinRGB[1]),
      targLinRGB[2] / std::max(eps, measLinRGB[2]));

  // Apply diagonal in linear space (remember linBGR is BGR)
  linBGR.forEach<cv::Vec3f>([&](cv::Vec3f &px, const int *) {
    px[0] *= gRGB[2]; // B channel gets g_B
    px[1] *= gRGB[1]; // G channel gets g_G
    px[2] *= gRGB[0]; // R channel gets g_R
  });

  // Clamp, then encode back to sRGB
  cv::min(linBGR, 1.0, linBGR);
  cv::max(linBGR, 0.0, linBGR);
  return toSRGB8(linBGR);
}

int main(int argc, char **argv)
{
  std::string path = "../../assets/sky.jpg";
  cv::Mat img      = cv::imread(path, cv::IMREAD_COLOR); // BGR, sRGB
  if (img.empty())
  {
    std::cerr << "Cannot read image\n";
    return 1;
  }

  //! note that these RGB values are in RGB order (not BGR)
  cv::Vec3b measuredRGB(90, 100, 110);
  cv::Vec3b targetRGB(125, 100, 90);

  cv::Mat mapped = remapFromSinglePatch(img, measuredRGB, targetRGB);
  cv::imwrite("mapped_to_standard.png", mapped);
  return 0;
}