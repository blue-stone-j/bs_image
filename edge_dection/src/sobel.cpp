#include "sobel.h"

void Sobel(cv::Mat &input_img, cv::Mat &output_img, cv::Mat &kernel_x, cv::Mat &kernel_y)
{
  int height = input_img.rows;
  int width = input_img.cols;

  int height_x = kernel_x.rows;
  int width_x = kernel_x.cols;

  int height_y = kernel_y.rows;
  int width_y = kernel_y.cols;

  for (int row = 1; row < height - 1; row++)
  {
    for (int col = 1; col < width - 1; col++)
    {
      float G_X = 0;
      for (int h = 0; h < height_x; h++)
      {
        for (int w = 0; w < width_x; w++)
        {
          G_X += input_img.at<uchar>(row + h - 1, col + w - 1) * kernel_x.at<float>(h, w);
        }
      }

      float G_Y = 0;
      for (int h = 0; h < height_y; h++)
      {
        for (int w = 0; w < width_y; w++)
        {
          G_Y += input_img.at<uchar>(row + h - 1, col + w - 1) * kernel_y.at<float>(h, w);
        }
      }

      output_img.at<uchar>(row, col) = cv::saturate_cast<uchar>(cv::abs(G_X) + cv::abs(G_Y));
      // output_img.at<uchar>(row, col) = saturate_cast<uchar>(cv::abs(G_Y));
    }
  }
}