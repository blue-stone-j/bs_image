#include "common/utilities.h"


/**
 将两个图像拼接，以便在同一个窗口显示
 dst 输出的拼接后的图像
 src1 拼接的第一幅图
 src2 拼接的第二幅图
 */
 void mergeImg(cv::Mat &dst, cv::Mat &src1, cv::Mat &src2)
 {
   int rows = src1.rows;
   int cols = src1.cols + 5 + src2.cols;
   dst.create(rows, cols, src1.type());
   src1.copyTo(dst(cv::Rect(0, 0, src1.cols, src1.rows)));
   src2.copyTo(dst(cv::Rect(src1.cols + 5, 0, src2.cols, src2.rows)));
 }