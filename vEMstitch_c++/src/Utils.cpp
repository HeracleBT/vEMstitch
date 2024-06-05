#include"Utils.h"


std::vector<double> range(double start, double stop, double step) {
    std::vector<double> values;
    for (double value = start; value < stop; value += step)
        values.push_back(value);
    return values;
}

cv::Mat normalize_img(cv::Mat& img) {
    img.convertTo(img,CV_64FC1);
    cv::Mat normalized_img;
    double min_val, max_val;
    cv::minMaxLoc(img, &min_val, &max_val);
    normalized_img = (img - (double)min_val) / ((double)max_val -  (double)min_val);
    return normalized_img;
}
cv::Mat loG(cv::Mat& img) {
    cv::Mat blurred_img, laplacian_img, abs_laplacian_img, res_img;
    cv::GaussianBlur(img, blurred_img, cv::Size(7, 7), 0);
    cv::Laplacian(blurred_img,laplacian_img,CV_16SC1,3);
    blurred_img.release();
    cv::convertScaleAbs(laplacian_img, abs_laplacian_img);
    laplacian_img.release();
    res_img = normalize_img(abs_laplacian_img);
    abs_laplacian_img.release();
    cv::threshold(res_img, res_img, 0.1, 0, cv::THRESH_TOZERO);
    return res_img;
}

std::tuple<cv::Mat, cv::Mat, bool> direct_stitch(cv::Mat& im1, cv::Mat& im2, cv::Mat& im1_mask, cv::Mat& im2_mask) {
    cv::Size im1_shape = im1.size();
    cv::Size im2_shape = im2.size();
    int dis_h = (im1_shape.height - im2_shape.height) / 2;
    if (im1_mask.empty()) {
        im1_mask = cv::Mat::ones(im1.size(), CV_64FC1);
    }
    im1.convertTo(im1,CV_64FC1);
    im2.convertTo(im2,CV_64FC1);

    int dis_w = int(im1_shape.width * 0.1);
    int h = im1_shape.height;
    int extra_w = im2_shape.width - dis_w;
    int w = im1_shape.width + extra_w;

    cv::Mat stitching_im1_res = cv::Mat::zeros(h, w, CV_64FC1);
    cv::Mat stitch_im1_mask = cv::Mat::zeros(h, w, CV_64FC1);
    
    im1.copyTo(stitching_im1_res(cv::Range(0,stitching_im1_res.rows),cv::Range(0,im1_shape.width)));
    im1_mask.copyTo(stitch_im1_mask(cv::Range(0,stitching_im1_res.rows),cv::Range(0,im1_shape.width)));
    
    cv::Mat stitch_im2_res = cv::Mat::zeros(h, w, CV_64FC1);
    cv::Mat stitch_im2_mask = cv::Mat::zeros(h, w, CV_64FC1);
    stitch_im2_mask(cv::Range(dis_h,im2_shape.height+dis_h),cv::Range(im1_shape.width-dis_w,stitch_im2_mask.cols)) = 1.0;
    stitch_im2_mask = stitch_im2_mask.mul(-stitch_im1_mask+1.0);
    im2.copyTo(stitch_im2_res(cv::Range(dis_h,im2_shape.height+dis_h),cv::Range(im1_shape.width-dis_w,stitch_im2_mask.cols)));

    cv::Mat stitching_res = stitching_im1_res.mul(stitch_im1_mask) + stitch_im2_res.mul(stitch_im2_mask);
    cv::Mat mass = stitch_im1_mask + stitch_im2_mask;
    return std::make_tuple(stitching_res, mass, false);
}

template<typename T>
std::vector<size_t> argsort(const std::vector<T> array) {
    std::vector<size_t> indices(array.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::stable_sort(indices.begin(), indices.end(),
              [&array](int left, int right) -> bool {
                  return array[left] < array[right];
              });

    return indices;
}

template<typename _Tp>
std::vector<_Tp> convertMat2Vector(const cv::Mat &mat)
{
	return (std::vector<_Tp>)(mat);
}
 
template<typename _Tp>
cv::Mat convertVector2Mat(std::vector<_Tp> v)
{
	cv::Mat mat = cv::Mat(v);
	return mat;
}

std::pair<cv::Mat, cv::Mat> filterIsolate(cv::Mat& src, cv::Mat& tgt) {

    std::vector<double> src0_vec = convertMat2Vector<double>(src.col(0));
    std::vector<size_t> srcRowSortedIdxVec0 = argsort(src0_vec);
    std::vector<int> srcRowSortedIdxVec(srcRowSortedIdxVec0.begin(), srcRowSortedIdxVec0.end());
    cv::Mat srcRowSortedIdx = cv::Mat(srcRowSortedIdxVec);
    cv::Mat ttsrcRowSortedIdx;
    cv::sortIdx(src.col(0), ttsrcRowSortedIdx, cv::SORT_EVERY_COLUMN + cv::SORT_ASCENDING);
    cv::Mat srcRow(src.rows, 1, src.type());
    for (int i = 0; i < src.rows; ++i) {
        srcRow.ptr<double>(i)[0] = src.ptr<double>(srcRowSortedIdx.ptr<int>(i)[0])[0];
    } 
    
    cv::Mat dis(srcRow.rows - 4, 1, CV_64FC1);
    for (int i = 0; i < dis.rows; ++i) {
        dis.ptr<double>(i)[0] = srcRow.ptr<double>(i+4)[0] - srcRow.ptr<double>(i)[0];
    }
    
    double meanDis = (cv::mean(dis).val[0] / 2.0);
    

    std::vector<int> index;
    int i = 0;
    while (i < srcRow.rows) {
        if (i > srcRow.rows - 3) {
            if (std::abs(srcRow.ptr<double>(i)[0] - srcRow.ptr<double>(i-2)[0]) <= meanDis * 3) {
                index.push_back(i);
            }
            i++;
        } else {
            if (std::abs(srcRow.ptr<double>(i)[0] - srcRow.ptr<double>(i+2)[0]) <= meanDis * 3) {
                index.push_back(i);
                index.push_back(i + 1);
                index.push_back(i + 2);
                i = i + 3;
            } else {
                i++;
            }
        }
    }
    cv::Mat filteredSrc;
    cv::Mat filteredTgt;
    for (size_t i = 0; i < srcRowSortedIdx.rows; ++i) {
        filteredSrc.push_back(src.row(srcRowSortedIdx.ptr<int>(i)[0]));
        filteredTgt.push_back(tgt.row(srcRowSortedIdx.ptr<int>(i)[0]));
    }
    cv::Mat filteredSrc2;
    cv::Mat filteredTgt2;
    for (size_t i = 0; i < index.size(); ++i) {
        filteredSrc2.push_back(filteredSrc.row(index[i]));
        filteredTgt2.push_back(filteredTgt.row(index[i]));
    }
    src = filteredSrc2;
    tgt = filteredTgt2;


    // Filter by y-axis

    std::vector<double> src1_vec = convertMat2Vector<double>(src.col(1));
    std::vector<size_t> srcColSortedIdxVec0 = argsort(src1_vec);
    std::vector<int> srcColSortedIdxVec(srcColSortedIdxVec0.begin(), srcColSortedIdxVec0.end());
    cv::Mat srcColSortedIdx = cv::Mat(srcColSortedIdxVec);

    cv::Mat ttsrcColSortedIdx;
    cv::sortIdx(src.col(1), ttsrcColSortedIdx, cv::SORT_EVERY_COLUMN + cv::SORT_ASCENDING);
    cv::Mat srcCol(src.rows, 1, src.type());
    for (int i = 0; i < src.rows; ++i) {
        srcCol.ptr<double>(i)[0] = src.ptr<double>(srcColSortedIdx.ptr<int>(i)[0])[1];
    } 
    cv::Mat dis2(srcCol.rows - 4, 1, CV_64FC1);
    for (int i = 0; i < dis2.rows; ++i) {
        dis2.ptr<double>(i)[0] = srcCol.ptr<double>(i+4)[0] - srcCol.ptr<double>(i)[0];
    }
    meanDis = (cv::mean(dis2).val[0] / 2);
    index.clear();
    i = 0;
    while (i < srcCol.rows) {
        if (i > srcCol.rows - 3) {
            if (std::abs(srcCol.ptr<double>(i)[0] - srcCol.ptr<double>(i-2)[0]) <= meanDis * 3) {
                index.push_back(i);
            }
            i++;
        } else {
            if (std::abs(srcCol.ptr<double>(i)[0] - srcCol.ptr<double>(i+2)[0]) <= meanDis * 3) {
                index.push_back(i);
                index.push_back(i + 1);
                index.push_back(i + 2);
                i = i + 3;
            } else {
                i++;
            }
        }
    }
    filteredSrc.release();
    filteredTgt.release();
    for (int i = 0; i < srcColSortedIdx.rows; ++i) {
        filteredSrc.push_back(src.row(srcColSortedIdx.ptr<int>(i)[0]));
        filteredTgt.push_back(tgt.row(srcColSortedIdx.ptr<int>(i)[0]));
    }
    filteredSrc2.release();
    filteredTgt2.release();
    for (int i = 0; i < index.size(); ++i) {
        filteredSrc2.push_back(filteredSrc.row(index[i]));
        filteredTgt2.push_back(filteredTgt.row(index[i]));
    }
    src = filteredSrc2;
    tgt = filteredTgt2;
    return std::make_pair(src, tgt);
}


std::tuple<cv::Mat, cv::Mat,std::vector<int>> filterGeometry(cv::Mat& src, cv::Mat& tgt,
                                                            std::pair<std::string, double> shifting, int windowSize, bool indexFlag) {
    cv::Mat newTgt = tgt.clone();
    if (shifting.first != "") {
        std::string mode = shifting.first;
        double d = shifting.second;
        if (mode == "l") {
            newTgt.col(0) -= d;
        } else if (mode == "r") {
            newTgt.col(0) += d;
        } else if (mode == "d") {
            newTgt.col(1) += d;
        }
    } else {
        newTgt = tgt.clone();
    }


    cv::Mat dis;
    cv::sqrt((src.col(0) - newTgt.col(0)).mul(src.col(0) - newTgt.col(0)) + 
             (src.col(1) - newTgt.col(1)).mul(src.col(1) - newTgt.col(1)), dis);
    double globalMeanDis = cv::mean(dis).val[0];
    int radius = windowSize / 2;
    std::vector<int> index;
    cv::Mat subDis;
    for (int i = 0; i < src.rows; ++i) {
        double disM;
        if (i <= radius - 1) {
            subDis = dis.rowRange(0, windowSize);
            disM = cv::mean(subDis).val[0];
            if (dis.ptr<double>(i)[0] <= disM * 1.5 && dis.ptr<double>(i)[0] <= globalMeanDis * 1.5) {
                index.push_back(i);
            }

        } else {
            subDis = dis.rowRange(i - radius, std::min(i + radius + 1,dis.rows));
            disM = cv::mean(subDis).val[0];
            if (dis.ptr<double>(i)[0] <= disM * 1.5 && dis.ptr<double>(i)[0] <= globalMeanDis * 1.5) {
                index.push_back(i);
            }
        }

        
    }
    if (!indexFlag) {
        return std::make_tuple(src.rowRange(cv::Range(index.front(), index.back() + 1)), 
                                tgt.rowRange(cv::Range(index.front(), index.back() + 1)),
                                index);
    } else {
        return std::make_tuple(cv::Mat(),cv::Mat(),index);
    }
}

bool rigidity_cons(const cv::Mat& x, const cv::Mat& y, const cv::Mat& x_, const cv::Mat& y_) {
    bool flag = true;
    for (int i = 0; i < 4; ++i) {
        double V = (x.ptr<double>((i + 1) % 4)[0] - x.ptr<double>(i)[0]) * (y.ptr<double>((i + 2) % 4)[0] - y.ptr<double>((i + 1) % 4)[0]) -
                    (y.ptr<double>((i + 1) % 4)[0] - y.ptr<double>(i)[0]) * (x.ptr<double>((i + 2) % 4)[0] - x.ptr<double>((i + 1) % 4)[0]);
        double V_ = (x_.ptr<double>((i + 1) % 4)[0] - x_.ptr<double>(i)[0]) * (y_.ptr<double>((i + 2) % 4)[0] - y_.ptr<double>((i + 1) % 4)[0]) -
                     (y_.ptr<double>((i + 1) % 4)[0] - y_.ptr<double>(i)[0]) * (x_.ptr<double>((i + 2) % 4)[0] - x_.ptr<double>((i + 1) % 4)[0]);
        int V_s = (V > 0) ? 1 : ((V < 0) ? -1 : 0);
        int V_s_ = (V_ > 0) ? 1 : ((V_ < 0) ? -1 : 0);
        if (V_s != V_s_) {
            flag = false;
            break;
        }
    }
    return flag;
}

std::tuple<std::vector<cv::KeyPoint>, cv::Mat, std::vector<cv::KeyPoint>, cv::Mat> SIFT_(cv::Mat im1, cv::Mat im2) {
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat dsp1, dsp2;
    
    sift->detectAndCompute(im1, cv::noArray(), kp1, dsp1);
    sift->detectAndCompute(im2, cv::noArray(), kp2, dsp2);

    return std::make_tuple(kp1, dsp1, kp2, dsp2);
}

std::tuple<std::vector<cv::KeyPoint>, cv::Mat, std::vector<cv::KeyPoint>, cv::Mat> SIFT_(cv::Mat im1, cv::Mat im1_mask, cv::Mat im2, cv::Mat im2_mask) {
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat dsp1, dsp2;
    
    sift->detectAndCompute(im1, im1_mask, kp1, dsp1);
    sift->detectAndCompute(im2, im2_mask, kp2, dsp2);

    return std::make_tuple(kp1, dsp1, kp2, dsp2);
}

std::tuple<cv::Mat,cv::Mat> removeDuplicatePoints(cv::Mat& srcdsp, cv::Mat& tgtdsp) {
    std::vector<std::vector<double>> srcdspVec, tgtdspVec;

    for (int i = 0; i < srcdsp.rows; ++i) {
        std::vector<double> row;
        row.push_back(srcdsp.ptr<double>(i)[0]);
        row.push_back(srcdsp.ptr<double>(i)[1]);
        srcdspVec.push_back(row);
        row.clear();
        row.push_back(tgtdsp.ptr<double>(i)[0]);
        row.push_back(tgtdsp.ptr<double>(i)[1]);
        tgtdspVec.push_back(row);
    }
    std::vector<int> index;
    std::vector<double> uniqueValues;
    for (size_t i = 0; i < srcdspVec.size(); ++i) {
        if (std::find(uniqueValues.begin(), uniqueValues.end(), srcdspVec[i][0]) == uniqueValues.end()) {
            index.push_back(i);
            uniqueValues.push_back(srcdspVec[i][0]);
        }
    }
    std::stable_sort(index.begin(), index.end());

    cv::Mat sortedSrcdsp(index.size(), 2, CV_64FC1);
    cv::Mat sortedTgtdsp(index.size(), 2, CV_64FC1);
    for (size_t i = 0; i < index.size(); ++i) {
        sortedSrcdsp.ptr<double>(i)[0] = srcdspVec[index[i]][0];
        sortedSrcdsp.ptr<double>(i)[1] = srcdspVec[index[i]][1];
        sortedTgtdsp.ptr<double>(i)[0] = tgtdspVec[index[i]][0];
        sortedTgtdsp.ptr<double>(i)[1] = tgtdspVec[index[i]][1];
    }
    return std::make_tuple(sortedSrcdsp,sortedTgtdsp);
}


std::tuple<cv::Mat, cv::Mat> flann_match(std::vector<cv::KeyPoint>& kp1, cv::Mat& dsp1,
                                        std::vector<cv::KeyPoint>& kp2,  cv::Mat& dsp2,
                                        double ratio,  cv::Mat& im1_mask, cv::Mat& im2_mask, 
                                        std::pair<std::string, double> shifting) {


    const int FLANN_INDEX_KDTREE = 1;
    cv::Ptr<cv::flann::IndexParams> indexParams = cv::makePtr<cv::flann::KDTreeIndexParams>(5);
    cv::Ptr<cv::flann::SearchParams> searchParams = cv::makePtr<cv::flann::SearchParams>(50);
    cv::FlannBasedMatcher flann(indexParams, searchParams);

    std::vector<std::vector<cv::DMatch>> matches;
    flann.knnMatch(dsp1, dsp2, matches, 2);

    std::vector<std::vector<cv::Point2f>> good_matches;
    cv::Point2f im1_pt,im2_pt;
    for (const auto& match_pair : matches) {
        if(match_pair.size()==2){
            if (match_pair[0].distance < ratio * match_pair[1].distance) {
                if (!im1_mask.empty() && !im2_mask.empty()) {
                    im1_pt = kp1[match_pair[0].queryIdx].pt;
                    im2_pt = kp2[match_pair[0].trainIdx].pt;
                    int im1_x = cvRound(im1_pt.x);
                    int im1_y = cvRound(im1_pt.y);
                    int im2_x = cvRound(im2_pt.x);
                    int im2_y = cvRound(im2_pt.y);

                    if (im1_y < im1_mask.rows && im1_x < im1_mask.cols && im2_y < im2_mask.rows && im2_x < im2_mask.cols) {
                        if (im1_mask.ptr<double>(im1_y)[im1_x]!=0 && im2_mask.ptr<double>(im2_y)[im2_x]!=0) {
                            good_matches.push_back({im1_pt, im2_pt});
                        }
                    }
                } else {
                    im1_pt = kp1[match_pair[0].queryIdx].pt;
                    im2_pt = kp2[match_pair[0].trainIdx].pt;
                    good_matches.push_back({im1_pt, im2_pt});
                }
            }
        }
    }

    std::vector<cv::Point2f> srcdsp_point, tgtdsp_point;
    for (const auto& match : good_matches) {
        srcdsp_point.push_back(match[0]);
        tgtdsp_point.push_back(match[1]);
    }
    cv::Mat srcdsp(srcdsp_point.size(), 2, CV_64FC1);
    for (size_t i = 0; i < srcdsp_point.size(); ++i) {
        srcdsp.ptr<double>(i)[0] = double(srcdsp_point[i].x); 
        srcdsp.ptr<double>(i)[1] = double(srcdsp_point[i].y); 
    }
    cv::Mat tgtdsp(tgtdsp_point.size(), 2, CV_64FC1);
    for (size_t i = 0; i < tgtdsp_point.size(); ++i) {
        tgtdsp.ptr<double>(i)[0] = double(tgtdsp_point[i].x); 
        tgtdsp.ptr<double>(i)[1] = double(tgtdsp_point[i].y); 
    }
    //clear
    std::vector<std::vector<cv::DMatch>>().swap(matches);
    std::vector<std::vector<cv::Point2f>>().swap(good_matches);
    std::vector<cv::Point2f>().swap(srcdsp_point);
    std::vector<cv::Point2f>().swap(tgtdsp_point);

    int kp_length = srcdsp.rows;
    if (kp_length <= 2) {
        std::cout << "feature number = " << srcdsp.rows << std::endl;
        if (srcdsp.rows == 1) {
            return std::make_pair(cv::Mat(), cv::Mat());
        }
        return std::make_pair(srcdsp, srcdsp);
    }


    std::tie(srcdsp,tgtdsp)=removeDuplicatePoints(srcdsp, tgtdsp);

    if (srcdsp.rows >= 8) {
        auto filtered_src_tgt = filterIsolate(srcdsp, tgtdsp);
        srcdsp = filtered_src_tgt.first;
        tgtdsp = filtered_src_tgt.second;
        auto filtered_tgt_src = filterIsolate(tgtdsp, srcdsp);
        tgtdsp = filtered_tgt_src.first;
        srcdsp = filtered_tgt_src.second;
    }


    std::tie(srcdsp,tgtdsp,std::ignore) = filterGeometry(srcdsp, tgtdsp,shifting);
    return std::make_tuple(srcdsp,tgtdsp);
}


std::tuple<cv::Mat,cv::Mat,cv::Mat,cv::Mat> stitch_add_mask_linear_per_border(cv::Mat& mask1,cv::Mat& mask2){
    int height = mask1.rows;
    int width = mask1.cols;
    cv::Mat mask_added = mask1+mask2;

    cv::Mat mask_super,mask_overlap;
    cv::threshold(mask_added,mask_super,0,1.0,cv::THRESH_BINARY);
    cv::threshold(mask_added,mask_overlap,1.0,1.0,cv::THRESH_BINARY);
    double radius_ratio = 0.15;
  
    cv::Mat mass_overlap_1 = cv::Mat::zeros(mask_overlap.size(),CV_64FC1);

    int path_width = width/3;
    for(int i=0;i<3;i++){
        int left_w = i*path_width;
        int right_w;
        if(i<2){
            right_w=(i+1)*path_width;
        }else{
            right_w = width;
        }
       
        cv::Mat temp_overlap = mask_overlap(cv::Range(0,mask_overlap.rows),cv::Range(left_w,right_w));
        cv::Mat temp_x_map;
        cv::reduce(temp_overlap, temp_x_map, 1, cv::REDUCE_SUM);
        int temp_x_u=0;
        int temp_x_d=0;
        for (int i = 0; i < temp_x_map.rows; ++i) {
            if (temp_x_map.ptr<double>(i)[0] >= 1.0) {
                temp_x_u = i;
                break;
            }
        }
        for (int i = temp_x_map.rows - 1; i >= 0; --i) {
            if (temp_x_map.ptr<double>(i)[0] >= 1.0) {
                temp_x_d = i;
                break;
            }
        }
        int temp_median = int(temp_x_u + temp_x_d)/2;
        int temp_radius = int(double(temp_x_d - temp_median) * radius_ratio);
        if (temp_median==0||temp_radius==0){
            continue;
        }
        cv::Mat row_vec = cv::Mat::zeros(1, 2 * temp_radius, CV_64FC1);
        for (int i = 0; i < row_vec.cols; ++i) {
            row_vec.ptr<double>(0)[i] = 1.0 - double(i) / double(2.0 * temp_radius - 1);
        }
        cv::Mat temp_map = cv::repeat(row_vec, right_w - left_w, 1);
        temp_map = temp_map.t();
        temp_map.copyTo(mass_overlap_1(cv::Range(temp_median-temp_radius,temp_median + temp_radius),cv::Range(left_w,right_w)));
        mass_overlap_1(cv::Range(temp_x_u,temp_median - temp_radius),cv::Range(left_w,right_w))=1.0;

        temp_overlap.release(),row_vec.release(),temp_map.release();
    } 
    mass_overlap_1 = mass_overlap_1.mul(mask_overlap);
    cv::Mat mass_overlap_2 = (-mass_overlap_1+1.0).mul(mask_overlap);
    mass_overlap_1 += mask1-mask_overlap;
    mass_overlap_2 += mask2-mask_overlap;
    return std::make_tuple(mass_overlap_1,mass_overlap_2,mask_super,mask_overlap);
}

std::tuple<cv::Mat,cv::Mat,cv::Mat,cv::Mat> stitch_add_mask_linear_border(cv::Mat& mask1,cv::Mat& mask2,std::string& mode){
    
    int height=mask1.rows;
    int width=mask2.cols;
    cv::Mat x_map,y_map;
    cv::reduce(mask1, x_map, 1, cv::REDUCE_SUM);//n*1
    cv::reduce(mask1, y_map, 0, cv::REDUCE_SUM);//1*m
    int y_l,y_r,x_u,x_d;
    for (int i = 0; i < y_map.cols; ++i) {
            if (y_map.ptr<double>(0)[i] >= 1.0) {
                y_l = i;
                break;
            }
    }
    for (int i = y_map.cols - 1; i >= 0; --i) {
        if (y_map.ptr<double>(0)[i] >= 1.0) {
            y_r = i;
            break;
        }
    }
    for (int i = 0; i < x_map.rows; ++i) {
            if (x_map.ptr<double>(i)[0] >= 1.0) {
                x_u = i;
                break;
            }
    }
    for (int i = x_map.rows - 1; i >= 0; --i) {
        if (x_map.ptr<double>(i)[0] >= 1.0) {
            x_d = i;
            break;
        }
    }

    cv::Mat mask_added = mask1+mask2;
    cv::Mat mask_super,mask_overlap;
    cv::threshold(mask_added,mask_super,0,1.0,cv::THRESH_BINARY);
    cv::threshold(mask_added,mask_overlap,1.0,1.0,cv::THRESH_BINARY);
    cv::Mat o_x_map,o_y_map;
    cv::reduce(mask_overlap, o_x_map, 1, cv::REDUCE_SUM);//n*1
    cv::reduce(mask_overlap, o_y_map, 0, cv::REDUCE_SUM);//1*m

    int o_y_l=0,o_y_r=0,o_x_u=0,o_x_d=0;
    for (int i = 0; i < o_y_map.cols; ++i) {
            if (o_y_map.ptr<double>(0)[i] >= 1.0) {
                o_y_l = i;
                break;
            }
    }
    for (int i = o_y_map.cols - 1; i >= 0; --i) {
        if (o_y_map.ptr<double>(0)[i] >= 1.0) {
            o_y_r = i;
            break;
        }
    }
    for (int i = 0; i < o_x_map.rows; ++i) {
            if (o_x_map.ptr<double>(i)[0] >= 1.0) {
                o_x_u = i;
                break;
            }
    }
    for (int i = o_x_map.rows - 1; i >= 0; --i) {
        if (o_x_map.ptr<double>(i)[0] >= 1.0) {
            o_x_d = i;
            break;
        }
    }

    double radius_ratio = 0.15;

    int x_median = (o_x_u + o_x_d) / 2;
    int x_radius = int((o_x_d - x_median) * radius_ratio);
    int y_median = (o_y_l + o_y_r) / 2;
    int y_radius = int(double(o_y_r - y_median) * radius_ratio);
    
    cv::Mat mass_overlap_1 = cv::Mat::zeros(mask_overlap.size(),mask_overlap.type());
    if(mode == ""){
        if(abs(o_x_u-x_u)<=3){
            cv::Mat row_vec = cv::Mat::zeros(1,o_x_d-x_u+1,CV_64FC1);
            for(int i=0;i<row_vec.cols;++i){
                row_vec.ptr<double>(0)[i] = double(i) / double(o_x_d - x_u);
            }
            cv::Mat temp_map = cv::repeat(row_vec, width, 1);
            temp_map = temp_map.t();
            temp_map.copyTo(mass_overlap_1(cv::Range(x_u,o_x_d),cv::Range(0,mass_overlap_1.cols)));
        }else if(abs(o_x_d-x_d)<=3){
            cv::Mat row_vec = cv::Mat::zeros(1,o_x_d - o_x_u + 1,CV_64FC1);
            for(int i=0;i<row_vec.cols;++i){
                row_vec.ptr<double>(0)[i] = 1.0 - (double)i / double(o_x_d - o_x_u);
            }
            cv::Mat temp_map = cv::repeat(row_vec, width, 1);
            temp_map = temp_map.t();
            temp_map.copyTo(mass_overlap_1(cv::Range(o_x_u,o_x_d),cv::Range(0,mass_overlap_1.cols)));
        }else if(abs(o_y_l-y_l)<=3){
            cv::Mat row_vec = cv::Mat::zeros(1,o_y_r - y_l + 1,CV_64FC1);
            for(int i=0;i<row_vec.cols;++i){
                row_vec.ptr<double>(0)[i] =  double(i) / double(o_y_r - y_l);
            }
            cv::Mat temp_map = cv::repeat(row_vec, height, 1);
            temp_map.copyTo(mass_overlap_1(cv::Range(0,mass_overlap_1.rows),cv::Range(y_l,o_y_r+1)));
        }else{
            cv::Mat row_vec = cv::Mat::zeros(1,o_y_r - o_y_l + 1,CV_64FC1);
            for(int i=0;i<row_vec.cols;++i){
                row_vec.ptr<double>(0)[i] = 1.0 - (double)i / double(o_y_r - o_y_l);
            }
            cv::Mat temp_map = cv::repeat(row_vec, height, 1);
            temp_map.copyTo(mass_overlap_1(cv::Range(0,mass_overlap_1.rows),cv::Range(o_y_l,o_y_r+1)));
        }
    }else{
        if(mode=="u"){
            cv::Mat row_vec = cv::Mat::zeros(1,2*x_radius,CV_64FC1);
            for(int i=0;i<row_vec.cols;++i){
                row_vec.ptr<double>(0)[i] = double(i) / double(2*x_radius-1);
            }
            cv::Mat temp_map = cv::repeat(row_vec, width, 1);
            temp_map = temp_map.t();
            temp_map.copyTo(mass_overlap_1(cv::Range(x_median - x_radius,x_median + x_radius),cv::Range(0,mass_overlap_1.cols)));
            mass_overlap_1(cv::Range(x_median + x_radius,o_x_d),cv::Range(0,mass_overlap_1.cols))=1.0;
        }else if(mode=="d"){
            cv::Mat row_vec = cv::Mat::zeros(1,2*x_radius,CV_64FC1);
            for(int i=0;i<row_vec.cols;++i){
                row_vec.ptr<double>(0)[i] = 1.0 - (double)i / double(2*x_radius-1);
            }
            cv::Mat temp_map = cv::repeat(row_vec, width, 1);
            temp_map = temp_map.t();
            temp_map.copyTo(mass_overlap_1(cv::Range(x_median - x_radius,x_median + x_radius),cv::Range(0,mass_overlap_1.cols)));
            mass_overlap_1(cv::Range(o_x_u,x_median - x_radius),cv::Range(0,mass_overlap_1.cols))=1.0;
        }else if(mode=="l"){
            cv::Mat row_vec = cv::Mat::zeros(1,2*y_radius,CV_64FC1);
            for(int i=0;i<row_vec.cols;++i){
                row_vec.ptr<double>(0)[i] = double(i) / double(2*y_radius-1);
            }
            cv::Mat temp_map = cv::repeat(row_vec, height, 1);
            temp_map.copyTo(mass_overlap_1(cv::Range(0,mass_overlap_1.rows),cv::Range(y_median - y_radius,y_median + y_radius)));
            mass_overlap_1(cv::Range(0,mass_overlap_1.rows),cv::Range(y_median + y_radius,o_y_r))=1.0;
        }else{
            cv::Mat row_vec = cv::Mat::zeros(1,2*y_radius,CV_64FC1);
            for(int i=0;i<row_vec.cols;++i){
                row_vec.ptr<double>(0)[i] = 1.0 - (double)i / (double)(2*y_radius-1);
            }
            cv::Mat temp_map = cv::repeat(row_vec, height, 1);
            temp_map.copyTo(mass_overlap_1(cv::Range(0,mass_overlap_1.rows),cv::Range(y_median - y_radius,y_median + y_radius)));
            mass_overlap_1(cv::Range(0,mass_overlap_1.rows),cv::Range(o_y_l,y_median - y_radius))=1.0;
        }
    }
    
    mass_overlap_1 = mass_overlap_1.mul(mask_overlap);
    cv::Mat mass_overlap_2 = (-mass_overlap_1+1.0).mul(mask_overlap);
    mass_overlap_1 += mask1-mask_overlap;
    mass_overlap_2 += mask2-mask_overlap;

    return std::make_tuple(mass_overlap_1,mass_overlap_2,mask_super,mask_overlap);

}

cv::Mat loadMatFromFile(const std::string& filename) {
    cv::FileStorage file(filename, cv::FileStorage::READ);
    cv::Mat mat;
    file["matrix"] >> mat;
    file.release();
    return mat;
}
void saveMatToFile(const cv::Mat& mat, const std::string& filename) {
    cv::FileStorage file(filename, cv::FileStorage::WRITE);
    file << "matrix" << mat;
    file.release();
}

void createSIFTMask(cv::Mat& mask, double overlap_rate, char mode){
    int h = mask.rows;
    int w = mask.cols;
    int o_h = (int) h * overlap_rate;
    int o_w = (int) w * overlap_rate;
    int o_d_h = h - o_h;
    int o_r_w = w - o_w;
    switch (mode){
        case 'r':
            mask(cv::Range(0, h), cv::Range(o_r_w, w)) = 1;
            break;
        case 'l':
            mask(cv::Range(0, h), cv::Range(0, o_w)) = 1;
            break;
        case 'u':
            mask(cv::Range(0, o_h), cv::Range(0, w)) = 1;
            break;
        case 'd':
            mask(cv::Range(o_d_h, h), cv::Range(0, w)) = 1;
            break;
        default:
            break;
    }
}