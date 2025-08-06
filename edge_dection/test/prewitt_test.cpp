
#include <gtest/gtest.h>

#include "prewitt.h"

TEST(PrewittTest, test1)
{
  cv::Mat img = cv::imread("../assets/woman.jpg", cv::IMREAD_GRAYSCALE);
  if (img.empty())
  {
    printf("读取图像文件失败");
    return;
  }
  cv::Mat output_img;
  img.copyTo(output_img);

  float values_x[9] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
  cv::Mat kernel_x  = cv::Mat_<float>(3, 3, values_x);

  float values_y[9] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};
  cv::Mat kernel_y  = cv::Mat_<float>(3, 3, values_y);

  Prewitt(img, output_img, kernel_x, kernel_y);

  // cv::imwrite(R"(../assets/woman_.jpg)", output_img);

  cv::imshow("img", img);
  cv::imshow("output_img", output_img);
  cv::waitKey(0);
}

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
