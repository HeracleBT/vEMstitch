#ifndef _ELASTIC_h_
#define _ELASTIC_H_

#include<iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <cmath>
#include <random>
#include <algorithm>
std::tuple<cv::Mat,cv::Mat,cv::Mat,cv::Mat,cv::Mat,cv::Mat,cv::Mat> local_TPS(cv::Mat& im1,cv::Mat& im2, cv::Mat& H, cv::Mat& X1_ok,cv::Mat& X2_ok,
                cv::Mat& im1_mask,cv::Mat& im2_mask,std::string& mode);
#endif