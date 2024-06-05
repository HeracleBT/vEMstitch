#ifndef _UTILS_h_
#define _UTILS_H_

#include<iostream>
#include <opencv2/opencv.hpp>
#include <numeric> 
#include <string>
#include <cmath>
#include <iomanip>
#include <vector> 

std::vector<double> range(double start, double end, double step=1);
cv::Mat normalize_img(cv::Mat& img);
cv::Mat apply_threshold(cv::Mat& img, double threshold);
cv::Mat loG(cv::Mat& img);
std::tuple<cv::Mat, cv::Mat, bool> direct_stitch(cv::Mat& im1, cv::Mat& im2, cv::Mat& im1_mask, cv::Mat& im2_mask);
bool rigidity_cons(const cv::Mat& x, const cv::Mat& y, const cv::Mat& x_, const cv::Mat& y_);
std::tuple<std::vector<cv::KeyPoint>, cv::Mat, std::vector<cv::KeyPoint>, cv::Mat> SIFT_(cv::Mat im1, cv::Mat im2);
std::tuple<std::vector<cv::KeyPoint>, cv::Mat, std::vector<cv::KeyPoint>, cv::Mat> SIFT_(cv::Mat im1, cv::Mat im1_mask, cv::Mat im2, cv::Mat im2_mask);
std::tuple<cv::Mat,cv::Mat> removeDuplicatePoints(cv::Mat& srcdsp, cv::Mat& tgtdsp);
std::pair<cv::Mat, cv::Mat> filterIsolate(cv::Mat& src, cv::Mat& tgt);
std::tuple<cv::Mat, cv::Mat,std::vector<int>> filterGeometry(cv::Mat& src, cv::Mat& tgt, 
                                                            std::pair<std::string, double> shifting ,int windowSize = 3, bool indexFlag = false);

std::tuple<cv::Mat, cv::Mat> flann_match(std::vector<cv::KeyPoint>& kp1, cv::Mat& dsp1,
                                        std::vector<cv::KeyPoint>& kp2,  cv::Mat& dsp2,
                                        double ratio,  cv::Mat& im1_mask, cv::Mat& im2_mask, 
                                        std::pair<std::string, double> shifting);
std::tuple<cv::Mat,cv::Mat,cv::Mat,cv::Mat> stitch_add_mask_linear_per_border(cv::Mat& mask1,cv::Mat& mask2);
std::tuple<cv::Mat,cv::Mat,cv::Mat,cv::Mat> stitch_add_mask_linear_border(cv::Mat& mask1,cv::Mat& mask2,std::string& mode);
void saveMatToFile(const cv::Mat& mat, const std::string& filename);
cv::Mat loadMatFromFile(const std::string& filename);
void createSIFTMask(cv::Mat& mask, double overlap_rate, char mode);
#endif