#ifndef UTILITIES_H
#define UTILITIES_H

#include <opencv2/opencv.hpp>

#include <Eigen/Dense>

/**
 将两个图像拼接，以便在同一个窗口显示
 dst 输出的拼接后的图像
 src1 拼接的第一幅图
 src2 拼接的第二幅图
 */
void mergeImg(cv::Mat &dst, cv::Mat &src1, cv::Mat &src2);

#endif