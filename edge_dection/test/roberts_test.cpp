
#include "roberts.h"

int main(int argc, char **argv)
{
  cv::Mat src, src_binary, src_gray;
  src = cv::imread("../assets/woman.jpg");
  if (src.empty())
  {
    printf("读取图像文件失败");
    system("pause");
    return 0;
  }
  cv::imshow("原图", src);
  cv::cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);
  cv::GaussianBlur(src_gray, src_binary, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
  cv::Mat dstImage = roberts(src_binary);

  cv::imshow("dstImage", dstImage);

  cv::waitKey(0);
  return 0;
}