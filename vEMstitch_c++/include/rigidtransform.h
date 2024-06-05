#ifndef _RIGIDTRANSFORM_h_
#define _RIGIDTRANSFORM_H_

#include<iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

std::vector<int> getRandomSubset(int point_num, int subset_size);
void getSubsetMatrices(cv::Mat& x1, std::vector<int>& subset, cv::Mat& x1_subset);
std::tuple<cv::Mat,cv::Mat> RANSAC(cv::Mat& ps1, cv::Mat& ps2, int iter_num, double min_dis);
std::tuple<cv::Mat,cv::Mat,cv::Mat,cv::Mat>rigid_transform(std::vector<cv::KeyPoint>& kp1,cv::Mat& dsp1, 
                                                        std::vector<cv::KeyPoint>& kp2, cv::Mat& dsp2,
                                                        cv::Mat& im1_mask, cv::Mat& im2_mask,
                                                        std::string& mode);
std::tuple<cv::Mat,cv::Mat,cv::Mat,cv::Mat,cv::Mat,cv::Mat,cv::Mat> stitching_global(cv::Mat& im1,cv::Mat& im2, cv::Mat& H, cv::Mat& X1_ok,cv::Mat& X2_ok,
                cv::Mat& im1_mask,cv::Mat& im2_mask,std::string& mode);

#endif