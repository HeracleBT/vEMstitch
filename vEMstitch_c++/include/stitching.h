#ifndef _STITCHING_h_
#define _STITCHING_H_

#include<iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <cmath>
#include <list>
#include "omp.h"
std::tuple<cv::Mat, cv::Mat, cv::Mat> stitching_pair(cv::Mat& im1, cv::Mat& im2, cv::Mat& im1_mask, cv::Mat& im2_mask, std::string& mode);
std::tuple<cv::Mat, cv::Mat, cv::Mat> stitching_pair(cv::Mat& im1, cv::Mat& im2, cv::Mat& im1_mask, cv::Mat& im2_mask, std::string& mode, double overlap);
std::tuple<cv::Mat,cv::Mat,cv::Mat> stitching_rows(cv::Mat& im1,cv::Mat& im2,cv::Mat& im1_mask,cv::Mat& im2_mask,std::string& mode,bool refine_flag);
std::tuple<cv::Mat,cv::Mat,cv::Mat> stitching_rows(cv::Mat& im1,cv::Mat& im2,cv::Mat& im1_mask,cv::Mat& im2_mask,std::string& mode,double overlap,bool refine_flag);
void two_stitching(const std::string& data_path, const std::string& store_path, const std::string& top_num, bool refine_flag = false);
void two_stitching_seq(const std::string& data_path, const std::string& store_path, const std::string& top_num, bool refine_flag = false);
void two_stitching(const std::string& data_path, const std::string& store_path, const std::string& top_num, double overlap_rate, bool refine_flag = false);
void three_stitching(const std::string& data_path, const std::string& store_path, const std::string& top_num, bool refine_flag);

#endif