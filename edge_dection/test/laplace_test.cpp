#include "laplace.h"

int main(int argc, char **argv)
{
  cv::Mat src = cv::imread("../assets/woman.jpg", 0);
  if (!src.data || src.channels() != 1)
  {
    fprintf(stderr, "read image fail\n");
    return -1;
  }
  int width{400}, height{400};
  cv::resize(src, src, cv::Size(width, height));

  std::unique_ptr<unsigned char[]> data1(new unsigned char[width * height]), data2(new unsigned char[width * height]);
  float elapsed_time1{0.f}, elapsed_time2{0.f}; // milliseconds
  int ksize{1};

  // fprintf(stdout, "gray image edge detection: laplacian: cpu run time: %f ms, gpu run time: %f ms\n", elapsed_time1, elapsed_time2);

  cv::Mat dst;
  cv::Laplacian(src, dst, src.depth(), ksize);
  cv::namedWindow("img");
  cv::imshow("img", dst); // 图像显示
  // cv::imwrite("../assets/woman_l.jpg", dst);

  // save_image(src, dst, width, height / 2, "E:/GitCode/CUDA_Test/test_data/images/laplacian_result.png");
  cv::waitKey(); // 等待键值输入
  return 0;
}