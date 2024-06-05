#include"refinement.h"
#include"Utils.h"
#include"rigidtransform.h"
#include"elastictransform.h"


double calculateStdDev(cv::Mat& region) {

    cv::Mat mask = region != 0.0;
    cv::Scalar mean, stddev;
    cv::meanStdDev(region, mean, stddev,mask);
    return stddev[0];
}

std::tuple<cv::Mat,cv::Mat,cv::Mat,cv::Mat> fast_brief(cv::Mat& im1, cv::Mat& im2, cv::Mat& im1_mask, cv::Mat& im2_mask, cv::Mat& X1, cv::Mat& X2,
                int height, std::vector<int>& im1_region, std::vector<int>& im2_region, std::string& mode) {
    cv::Mat mask_1 = cv::Mat::zeros(im1.size(), CV_64FC1);
    cv::Mat mask_2 = cv::Mat::zeros(im2.size(), CV_64FC1);

    if (height != -1) {
        if (mode == "d") {
            mask_1(cv::Range(mask_1.rows-height,mask_1.rows),cv::Range(im1_region[0],im1_region[1])) = 1.0;
            mask_2(cv::Range(0,height),cv::Range(im2_region[0],im2_region[1])) = 1.0;
        } else if (mode == "r") {
            mask_1(cv::Range(im1_region[0],im1_region[1]),cv::Range(mask_1.cols-height,mask_1.cols)) = 1.0;
            mask_2(cv::Range(im2_region[0],im2_region[1]),cv::Range(0,height)) = 1.0;
        }
    } else {
        mask_1 = im1_mask;
        mask_2 = im2_mask;
    }
 
    cv::Mat new_img_1 = loG(im1).mul(im1_mask);
    cv::Mat new_img_2 = loG(im2).mul(im2_mask);
    new_img_1*=255.0;
    new_img_2*=255.0;
    new_img_1.convertTo(new_img_1, CV_8UC1);
    new_img_2.convertTo(new_img_2, CV_8UC1);

    double fusion_lambda = 0.3;
    cv::Mat gau_img1, gau_img2;
    cv::GaussianBlur(im1, gau_img1, cv::Size(7, 7), 0);
    cv::GaussianBlur(im2, gau_img2, cv::Size(7, 7), 0);
    gau_img1.convertTo(gau_img1,CV_64FC1);
    gau_img2.convertTo(gau_img2,CV_64FC1);
    new_img_1.convertTo(new_img_1,CV_64FC1);
    new_img_2.convertTo(new_img_2,CV_64FC1);
  
    cv::Mat fusion_1 = (gau_img1*fusion_lambda+new_img_1*(1-fusion_lambda)).mul(mask_1);
    fusion_1.convertTo(fusion_1, CV_8UC1);
    cv::Mat fusion_2 = (gau_img2*fusion_lambda+new_img_2*(1-fusion_lambda)).mul(mask_2);
    fusion_2.convertTo(fusion_2, CV_8UC1);




    gau_img1.release(),gau_img2.release(),new_img_2.release(),new_img_1.release();
    
    
    cv::Ptr<cv::FastFeatureDetector> fast = cv::FastFeatureDetector::create(10);
    std::vector<cv::KeyPoint> kp1, kp2;
    fast->detect(fusion_1, kp1);
    fast->detect(fusion_2, kp2);
   
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    cv::Mat dsp1, dsp2;
    orb->compute(im1, kp1, dsp1);
    orb->compute(im2, kp2, dsp2);
   


    cv::Ptr<cv::flann::IndexParams> indexParams = cv::makePtr<cv::flann::LshIndexParams>(12,20,2);
    cv::Ptr<cv::flann::SearchParams> searchParams = cv::makePtr<cv::flann::SearchParams>(50);
    cv::FlannBasedMatcher flann(indexParams, searchParams);
    std::vector<std::vector<cv::DMatch>> matches;
    flann.knnMatch(dsp1, dsp2, matches, 2);
    std::vector<std::vector<cv::Point2f>> good_matches;
    double ratio = 0.6;
    cv::Point2f im1_pt,im2_pt;
    for (const auto& match_pair : matches) {
        if(match_pair.size()==2){
            if (match_pair[0].distance < ratio * match_pair[1].distance) {
                im1_pt = kp1[match_pair[0].queryIdx].pt;
                im2_pt = kp2[match_pair[0].trainIdx].pt;
                good_matches.push_back({im1_pt, im2_pt});
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

    double dis = 0.0;
    if(mode=="d"){
        dis = im1_mask.rows;
    } else if(mode=="l"||mode=="r"){
        dis = im1_mask.cols;
    }
    std::pair<std::string,double> shifting = std::make_pair(mode,dis);
    std::vector<int> edge_ok;
    std::tie(std::ignore,std::ignore,edge_ok) = filterGeometry(srcdsp, tgtdsp, shifting, 3, true);
    cv::Mat srcdsp_ok,tgtdsp_ok;
    for(int i=0;i<edge_ok.size();i++){
        srcdsp_ok.push_back(srcdsp.row(edge_ok[i]));
        tgtdsp_ok.push_back(tgtdsp.row(edge_ok[i]));
    }
    if(X1.empty()){
        X1 = srcdsp_ok;
    } else{
        cv::vconcat(X1, srcdsp_ok, X1);
    }
    if(X2.empty()){
        X2 = tgtdsp_ok;
    } else{
        cv::vconcat(X2, tgtdsp_ok, X2);
    }

    cv::Mat X1_copy = X1.clone();
    cv::Mat X2_copy = X2.clone();
    cv::Mat ok;
    std::tie(std::ignore,ok)=RANSAC(X1_copy,X2_copy,2000,0.01);

    int point_num = X1.rows;
    double centroid_1_x = cv::mean(X1.col(0)).val[0];
    double centroid_1_y = cv::mean(X1.col(1)).val[0];
    double centroid_2_x = cv::mean(X2.col(0)).val[0];
    double centroid_2_y = cv::mean(X2.col(1)).val[0];
    cv::Mat X = X1.clone();
    cv::Mat Y = X2.clone();
    X.col(0) -=centroid_1_x;
    X.col(1) -=centroid_1_y;
    Y.col(0) -=centroid_2_x;
    Y.col(1) -=centroid_2_y;

    cv::Mat X_ok,Y_ok;
    for(int i=0;i<ok.rows;i++){
        if(ok.at<int>(i,0)){
            X_ok.push_back(X.row(i));
            Y_ok.push_back(Y.row(i));
        }
    }
    cv::Mat X_transpose;
    cv::transpose(X_ok, X_transpose);
    cv::Mat H = X_transpose * Y_ok;

    cv::Mat U, S, VT;
    cv::SVD::compute(H, S, U, VT,cv::SVD::FULL_UV);
    cv::Mat R = VT.t() * U.t();
    if (cv::determinant(R) < 0) {
        VT.row(1) *= -1;
        R = VT.t() * U.t();
    }
    cv::Mat centroid_1 = (cv::Mat_<double>(2, 1) << centroid_1_x, centroid_1_y);
    cv::Mat centroid_2 = (cv::Mat_<double>(2, 1) << centroid_2_x, centroid_2_y);
    cv::Mat t = -R * centroid_1 + centroid_2;

    H = cv::Mat::zeros(3, 3, CV_64F);
    H.at<double>(2, 2) = 1.0;
    for (int row = 0; row < 2; ++row) {
        H.at<double>(row, 2) = t.at<double>(row, 0);
    }
    for (int row = 0; row < 2; ++row) {
        for (int col = 0; col < 2; ++col) {
            H.at<double>(row, col) = R.at<double>(row, col);
        }
    }
    return std::make_tuple(H,ok,X1,X2);
}

void find_overlap(cv::Mat& im1_mask,cv::Mat& im2_mask,cv::Mat& H,cv::Mat& im_mask_overlap){
   
    
    cv::Mat box = (cv::Mat_<double>(3, 4) << 0, im2_mask.cols - 1, im2_mask.cols - 1, 0,
                                            0, 0, im2_mask.rows - 1, im2_mask.rows - 1,
                                            1, 1, 1, 1);
    cv::Mat box_ = H.inv() * box;
    box_.row(0) = box_.row(0)/box_.row(2);
    box_.row(1) = box_.row(1)/box_.row(2);


    double min_1,max_1,min_2,max_2;
    cv::minMaxLoc(box_.row(0), &min_1, &max_1,NULL, NULL);
    cv::minMaxLoc(box_.row(1), &min_2, &max_2,NULL, NULL);
    double u_left = std::min((double)0.0,(double)min_1);
    double u_right = std::max((double)im1_mask.cols - 1, (double)max_1);
    double v_up = std::min((double)0.0,(double)min_2);
    double v_down = std::max((double)im1_mask.rows - 1, (double)max_2);
    std::vector<double> v_h  = range(v_up,v_down,1);
    std::vector<double> u_w  = range(u_left,u_right,1);

    cv::Mat u = cv::Mat(u_w);
    cv::transpose(u,u);
    cv::Mat v = cv::Mat(v_h);
    cv::transpose(v,v);
    u = cv::repeat(u,v_h.size(),1);
    v = cv::repeat(v,u_w.size(),1);
    cv::transpose(v,v);



    im1_mask.convertTo(im1_mask,CV_32FC1);
    u.convertTo(u,CV_32FC1);
    v.convertTo(v,CV_32FC1);
    cv::Mat warped_mask1;
    warped_mask1.create(im1_mask.size(), im1_mask.type());
    cv::remap(im1_mask,warped_mask1,u,v,cv::INTER_CUBIC);
    warped_mask1.convertTo(warped_mask1,CV_64FC1);
    im1_mask.convertTo(im1_mask,CV_64FC1);
    u.convertTo(u,CV_64FC1);
    v.convertTo(v,CV_64FC1);
    
  
    cv::Mat z1_ = H.at<double>(2,0)*u + H.at<double>(2,1)*v + H.at<double>(2,2);
    cv::Mat u_ = (H.at<double>(0,0)*u + H.at<double>(0,1)*v + H.at<double>(0,2))/z1_;
    cv::Mat v_ = (H.at<double>(1,0)*u + H.at<double>(1,1)*v + H.at<double>(1,2))/z1_;
 
    
    im2_mask.convertTo(im2_mask,CV_32FC1);
    u_.convertTo(u_,CV_32FC1);
    v_.convertTo(v_,CV_32FC1);
    cv::Mat warped_mask2;
    warped_mask2.create(im2_mask.size(), im2_mask.type());
    cv::remap(im2_mask,warped_mask2,u_,v_,cv::INTER_CUBIC);

    warped_mask2.convertTo(warped_mask2,CV_64FC1);
    im2_mask.convertTo(im2_mask,CV_64FC1);
    u_.convertTo(u_,CV_64FC1);
    v_.convertTo(v_,CV_64FC1);

    im_mask_overlap  = warped_mask1 + warped_mask2;
    cv::threshold(im_mask_overlap,im_mask_overlap,1.5,1.0,cv::THRESH_BINARY);

    return ;
}

std::tuple<int,std::vector<int>,std::vector<int>>find_region(cv::Mat& im1,cv::Mat& im2,cv::Mat& X1,cv::Mat& X2,cv::Mat& ok,cv::Mat& im_mask_overlap){
    double o_x_u_1, o_x_d_1;
    cv::minMaxLoc(X1.col(1), &o_x_u_1, &o_x_d_1);
    double o_x_u_2, o_x_d_2;
    cv::minMaxLoc(X2.col(1), &o_x_u_2, &o_x_d_2);

    double height_1 = o_x_d_1 - o_x_u_1;
    double height_2 = o_x_d_1 - o_x_u_1;
    int height = int(std::max({height_1, height_2, im1.rows*0.15, im2.rows*0.15}));
    
    int stride = int(o_x_d_1 - o_x_u_1);
    std::map<int, int> feature_count;
    std::map<int, int> feature_count_2;
    int x_median = int((o_x_d_1 + o_x_u_1) / 2);
    int x_radius = int((o_x_d_1 - x_median) * 0.5);
    int x_median_2 = int((o_x_d_2 + o_x_u_2) / 2);
    int x_radius_2 = int((o_x_d_2 - x_median_2) * 0.5);

    cv::Mat o_y_map;
    cv::reduce(im_mask_overlap, o_y_map, 0, cv::REDUCE_SUM, CV_64FC1);

    int o_y_l = 0, o_y_r = 0;
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

    int total_num = (o_y_r - o_y_l) / stride;


    for (int i = 0; i < ok.rows; ++i) {
        if (ok.ptr<int>(i)[0]) {
            double y = X1.ptr<double>(i)[0];
            int n = int((y - o_y_l) / stride);
            feature_count[n] += 1;
            feature_count[n - 1] += 1;
        }
    }

    for (int i = 0; i < ok.rows; ++i) {
        if (ok.ptr<int>(i)[0]) {
            double y = X2.ptr<double>(i)[0];
            int n = int((y - o_y_l) / stride);
            feature_count_2[n] += 1;
            feature_count_2[n - 1] += 1;
        }
    }
    std::vector<int>im1_select_key;
    std::vector<int>im2_select_key;

    
    double ratio = 0.35;
    int select_range = int(total_num*ratio);
    std::vector<int>total_key,select_key;
    for(int i=0;i<total_num;i++) total_key.push_back(i);
    select_key.insert(select_key.end(), total_key.begin(), total_key.begin() + select_range);
    select_key.insert(select_key.end(), total_key.end() - select_range, total_key.end());


    for(int k=0;k<select_key.size();k++){
        int i=select_key[k];
        cv::Mat region1 = im1(cv::Range(std::max(0,x_median - x_radius),std::min(im1.rows,x_median + x_radius)),
                              cv::Range(std::max(0,o_y_l + int(i * stride)),std::min(im1.cols,o_y_l + int((i + 2) * stride))));
        cv::Mat region2 = im2(cv::Range(std::max(0,x_median_2 - x_radius_2),std::min(im2.rows,x_median_2 + x_radius_2)),
                              cv::Range(std::max(0,o_y_l + int(i * stride)),std::min(im2.cols,o_y_l + int((i + 2) * stride))));
        
        double stddev1 = calculateStdDev(region1);
        double stddev2 = calculateStdDev(region2);

        if (stddev1 >= 12.0 && feature_count[i] <= 3) {
            im1_select_key.push_back(i);
        }
        if (stddev2 >= 12.0 && feature_count_2[i] <= 3) {
            im2_select_key.push_back(i);
        }
    }
    if(im1_select_key.size()==0||im2_select_key.size()==0){
        return std::make_tuple(-1,std::vector<int>(),std::vector<int>());
    }
    std::vector<int> im1_select_region(2),im2_select_region(2);
    im1_select_region[0] = o_y_l + int(im1_select_key[0] * stride);
    im1_select_region[1] = o_y_l + int((im1_select_key[im1_select_key.size()-1] + 2) * stride);
    im2_select_region[0] = o_y_l + int(im2_select_key[0] * stride);
    im2_select_region[1] = o_y_l + int((im2_select_key[im2_select_key.size()-1] + 2) * stride);
    
    return std::make_tuple(height,im1_select_region,im2_select_region);
}

std::tuple<cv::Mat,cv::Mat,cv::Mat> refinement_local(cv::Mat& im1,cv::Mat& im2, cv::Mat& H,cv::Mat& X1,cv::Mat& X2,
                    cv::Mat& ok,cv::Mat& im1_mask,cv::Mat& im2_mask,std::string& mode){
    
    cv::Mat im_mask_overlap;
    find_overlap(im1_mask,im2_mask,H,im_mask_overlap);
    int height;
    std::vector<int>im1_region,im2_region;
    std::tie(height,im1_region,im2_region) = find_region(im1,im2,X1,X2,ok,im_mask_overlap);
    if(height==-1){
        return std::make_tuple(cv::Mat(),cv::Mat(),cv::Mat());
    }
    std::tie(H,ok,X1,X2) = fast_brief(im1,im2,im1_mask,im2_mask,X1,X2,height,im1_region,im2_region,mode);
    cv::Mat X1_ok,X2_ok;
    for(int i=0;i<ok.rows;i++){
        if(ok.ptr<int>(i)[0]){
            X1_ok.push_back(X1.row(i));
            X2_ok.push_back(X2.row(i));
        }
    }
    cv::transpose(X1_ok,X1_ok);
    cv::transpose(X2_ok,X2_ok);
    cv::Mat stitching_res,mass, overlap_mass;
    std::tie(stitching_res,std::ignore,std::ignore,
            std::ignore,std::ignore,mass,overlap_mass) = local_TPS(im1, im2, H, X1_ok, X2_ok, im1_mask, im2_mask, mode);
    
    return std::make_tuple(stitching_res,mass,overlap_mass);
}