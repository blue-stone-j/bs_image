

#ifndef HARRIS_HPP_
#define HARRIS_HPP_


#include <iostream>
#include <opencv2/opencv.hpp>

class Harris
{
 public:
  Harris();
  ~Harris();

  void detect(const cv::Mat &img_, std::vector<cv::Point> &corners_);

 private:
  void get_corners(const cv::Mat &res_, const float thresh_, std::vector<cv::Point> &corners_);

  cv::Mat score_img(const cv::Mat &in_);

  void get_gradient(const cv::Mat &in_);

  cv::Mat filter_uchar(const cv::Mat &in_, const cv::Mat &kernel_);

  cv::Mat filter_float(const cv::Mat &in_, const cv::Mat &kernel_);

  float gaussion(int x, int y, float theta);

  // generate kernel
  cv::Mat gen_gaussion_kernel();

 private:
  cv::Mat Ixx, Iyy, Ixy;
  int _kernel_size = 5;
  float _delta     = 1.4;
  // 增大 α 的值，将减小角点响应值 R ，减少被检测角点的数量；减小 α 的值，将增大角点响应值 R ，增加被检测角点的数量。
  float alpha = 0.05;
};

#endif