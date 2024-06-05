#ifndef _REFINEMENT_h_
#define _REFINEMENT_H_

#include<iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
double calculateStdDev(cv::Mat& region);
void find_overlap(cv::Mat& im1_mask,cv::Mat& im2_mask,cv::Mat& H,cv::Mat& im_mask_overlap);
std::tuple<cv::Mat,cv::Mat,cv::Mat,cv::Mat> fast_brief(cv::Mat& im1, cv::Mat& im2, cv::Mat& im1_mask, cv::Mat& im2_mask, cv::Mat& X1, cv::Mat& X2,
                int height, std::vector<int>& im1_region, std::vector<int>& im2_region, std::string& mode);
std::tuple<cv::Mat,cv::Mat,cv::Mat> refinement_local(cv::Mat& im1,cv::Mat& im2, cv::Mat& H,cv::Mat& X1,cv::Mat& X2,
                    cv::Mat& ok,cv::Mat& im1_mask,cv::Mat& im2_mask,std::string& mode);
std::tuple<int,std::vector<int>,std::vector<int>>find_region(cv::Mat& im1,cv::Mat& im2,cv::Mat& X1,cv::Mat& X2,cv::Mat& ok,cv::Mat& im_mask_overlap);
#endif