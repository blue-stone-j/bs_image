
#include "./common/utilities.h"

/**
 一维高斯卷积，对每行进行高斯卷积
 img 输入原图像
 dst  一维高斯卷积后的输出图像
 */
void gaussianConvolution(cv::Mat &img, cv::Mat &dst);

/**
 高斯滤波器，利用3*3的高斯模版进行高斯卷积
 img 输入原图像
 dst  高斯滤波后的输出图像
*/
void gaussianFilter(cv::Mat &img, cv::Mat &dst);

/**
 用一阶偏导有限差分计算梯度幅值和方向
 img 输入原图像
 gradXY 输出的梯度幅值
 theta 输出的梯度方向
 */
void getGrandient(cv::Mat &img, cv::Mat &gradXY, cv::Mat &theta);

/**
 局部非极大值抑制
 gradXY 输入的梯度幅值
 theta 输入的梯度方向
 dst 输出的经局部非极大值抑制后的图像
 */
void nonLocalMaxValue(cv::Mat &gradXY, cv::Mat &theta, cv::Mat &dst);

/**
 弱边缘点补充连接强边缘点
 img 弱边缘点补充连接强边缘点的输入和输出图像
 */
void doubleThresholdLink(cv::Mat &img);

/**
 用双阈值算法检测和连接边缘
 low 输入的低阈值
 high 输入的高阈值
 img 输入的原图像
 dst 输出的用双阈值算法检测和连接边缘后的图像
 */
void doubleThreshold(double low, double high, cv::Mat &img, cv::Mat &dst);