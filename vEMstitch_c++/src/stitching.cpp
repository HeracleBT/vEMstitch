#include"stitching.h"
#include"Utils.h"
#include"refinement.h"
#include"elastictransform.h"
#include"rigidtransform.h"

std::tuple<cv::Mat, cv::Mat, cv::Mat> stitching_pair(cv::Mat& im1, cv::Mat& im2, cv::Mat& im1_mask, cv::Mat& im2_mask, std::string& mode) {
    cv::Mat stitching_res, mass, overlap_mass, H;//out
    cv::Mat X1, X2, ok;
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat dsp1, dsp2;
    std::tie(kp1, dsp1, kp2, dsp2) = SIFT_(im1, im2);
    std::tie(H, ok, X1, X2) = rigid_transform(kp1, dsp1, kp2, dsp2, im1_mask, im2_mask, mode);

    if (H.empty()) {
        X1=cv::Mat(), X2=cv::Mat();
        int height = int(im2.cols * 0.15);
        std::vector<int> im1_region = {0, im1.rows};
        std::vector<int> im2_region = {0, im2.rows};
        std::tie(H,ok,X1,X2)=fast_brief(im1, im2, im1_mask, im2_mask, X1, X2, height, im1_region, im2_region, mode);
        cv::Mat X1_ok,X2_ok;
        for(int i=0;i<ok.rows;i++){
            if(ok.ptr<int>(i)[0]){
                X1_ok.push_back(X1.row(i));
                X2_ok.push_back(X2.row(i));
            }
        }
        cv::transpose(X1_ok,X1_ok);
        cv::transpose(X2_ok,X2_ok);
        
        std::tie(stitching_res,std::ignore,std::ignore,
            std::ignore,std::ignore,mass,overlap_mass) = local_TPS(im1, im2, H, X1_ok, X2_ok, im1_mask, im2_mask, mode);
        return std::make_tuple(stitching_res, mass, overlap_mass);
    }
    cv::Mat X1_ok,X2_ok;
    for(int i=0;i<ok.rows;i++){
        if(ok.ptr<int>(i)[0]){
            X1_ok.push_back(X1.row(i));
            X2_ok.push_back(X2.row(i));
        }
    }

    cv::transpose(X1_ok,X1_ok);
    cv::transpose(X2_ok,X2_ok);
    
    std::tie(stitching_res,std::ignore,std::ignore,
            std::ignore,std::ignore,mass,overlap_mass) = local_TPS(im1, im2, H, X1_ok, X2_ok, im1_mask, im2_mask, mode);
    return std::make_tuple(stitching_res, mass, overlap_mass);
}

std::tuple<cv::Mat, cv::Mat, cv::Mat> stitching_pair(cv::Mat& im1, cv::Mat& im2, cv::Mat& im1_mask, cv::Mat& im2_mask, std::string& mode, double overlap) {
    cv::Mat stitching_res, mass, overlap_mass, H;//out
    cv::Mat X1, X2, ok;

    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat dsp1, dsp2;
    double SIFT_region = overlap * 1.5;
    cv::Mat im1_SIFT_mask, im2_SIFT_mask;
    im1_SIFT_mask = cv::Mat::zeros(im1.size(), CV_8UC1);
    im2_SIFT_mask = cv::Mat::zeros(im2.size(), CV_8UC1);
    createSIFTMask(im1_SIFT_mask, SIFT_region, 'r');
    createSIFTMask(im2_SIFT_mask, SIFT_region, 'l');
    std::tie(kp1, dsp1, kp2, dsp2) = SIFT_(im1, im1_SIFT_mask, im2, im2_SIFT_mask);
    std::tie(H, ok, X1, X2) = rigid_transform(kp1, dsp1, kp2, dsp2, im1_mask, im2_mask, mode);

    if (H.empty()) {
        X1=cv::Mat(), X2=cv::Mat();
        int height = int(im2.cols * overlap);
        std::vector<int> im1_region = {0, im1.rows};
        std::vector<int> im2_region = {0, im2.rows};
        std::tie(H,ok,X1,X2)=fast_brief(im1, im2, im1_mask, im2_mask, X1, X2, height, im1_region, im2_region, mode);
        cv::Mat X1_ok,X2_ok;
        for(int i=0;i<ok.rows;i++){
            if(ok.ptr<int>(i)[0]){
                X1_ok.push_back(X1.row(i));
                X2_ok.push_back(X2.row(i));
            }
        }
        cv::transpose(X1_ok,X1_ok);
        cv::transpose(X2_ok,X2_ok);
        
        std::tie(stitching_res,std::ignore,std::ignore,
            std::ignore,std::ignore,mass,overlap_mass) = local_TPS(im1, im2, H, X1_ok, X2_ok, im1_mask, im2_mask, mode);
        return std::make_tuple(stitching_res, mass, overlap_mass);
    }

    cv::Mat X1_ok,X2_ok;
    for(int i=0;i<ok.rows;i++){
        if(ok.ptr<int>(i)[0]){
            X1_ok.push_back(X1.row(i));
            X2_ok.push_back(X2.row(i));
        }
    }

    cv::transpose(X1_ok,X1_ok);
    cv::transpose(X2_ok,X2_ok);
    
    std::tie(stitching_res,std::ignore,std::ignore,
            std::ignore,std::ignore,mass,overlap_mass) = local_TPS(im1, im2, H, X1_ok, X2_ok, im1_mask, im2_mask, mode);
    return std::make_tuple(stitching_res, mass, overlap_mass);
}


std::tuple<cv::Mat,cv::Mat,cv::Mat>stitching_rows(cv::Mat& im1,cv::Mat& im2,cv::Mat& im1_mask,cv::Mat& im2_mask,std::string& mode,bool refine_flag){
    cv::Mat H,ok,X1,X2,stitching_res,mass,overlap_mass;
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat dsp1, dsp2;
    std::tie(kp1, dsp1, kp2, dsp2) = SIFT_(im1, im2);
    std::tie(H, ok, X1, X2) = rigid_transform(kp1, dsp1, kp2, dsp2, im1_mask, im2_mask, mode);
    if(refine_flag){
        std::tie(stitching_res,mass,overlap_mass) = refinement_local(im1, im2, H, X1, X2, ok, im1_mask, im2_mask, mode);
        if(stitching_res.empty()){
            cv::Mat X1_ok,X2_ok;
            for(int i=0;i<ok.rows;i++){
                if(ok.ptr<int>(i)[0]){
                    X1_ok.push_back(X1.row(i));
                    X2_ok.push_back(X2.row(i));
                }
            }
            cv::transpose(X1_ok,X1_ok);
            cv::transpose(X2_ok,X2_ok);
            std::tie(stitching_res,std::ignore,std::ignore,
                    std::ignore,std::ignore,mass,overlap_mass) = local_TPS(im1, im2, H, X1_ok, X2_ok, im1_mask, im2_mask, mode);
        }
    }
    else{
        cv::Mat X1_ok,X2_ok;
        for(int i=0;i<ok.rows;i++){
            if(ok.ptr<int>(i)[0]){
                X1_ok.push_back(X1.row(i));
                X2_ok.push_back(X2.row(i));
            }
        }
        cv::transpose(X1_ok,X1_ok);
        cv::transpose(X2_ok,X2_ok);
        std::tie(stitching_res,std::ignore,std::ignore,
                std::ignore,std::ignore,mass,overlap_mass) = local_TPS(im1, im2, H, X1_ok, X2_ok, im1_mask, im2_mask, mode);
    }
    return std::make_tuple(stitching_res, mass, overlap_mass);
}


std::tuple<cv::Mat,cv::Mat,cv::Mat> stitching_rows(cv::Mat& im1,cv::Mat& im2,cv::Mat& im1_mask,cv::Mat& im2_mask,std::string& mode,double overlap,bool refine_flag){
    cv::Mat H,ok,X1,X2,stitching_res,mass,overlap_mass;
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat dsp1, dsp2;
    double SIFT_region = overlap * 1.5;
    cv::Mat im1_SIFT_mask, im2_SIFT_mask;
    im1_SIFT_mask = cv::Mat::zeros(im1.size(), CV_8UC1);
    im2_SIFT_mask = cv::Mat::zeros(im2.size(), CV_8UC1);
    createSIFTMask(im1_SIFT_mask, SIFT_region, 'd');
    createSIFTMask(im2_SIFT_mask, SIFT_region, 'u');
    std::tie(kp1, dsp1, kp2, dsp2) = SIFT_(im1, im1_SIFT_mask, im2, im2_SIFT_mask);
    std::tie(H, ok, X1, X2) = rigid_transform(kp1, dsp1, kp2, dsp2, im1_mask, im2_mask, mode);

    if(refine_flag){
        std::tie(stitching_res,mass,overlap_mass) = refinement_local(im1, im2, H, X1, X2, ok, im1_mask, im2_mask, mode);
        if(stitching_res.empty()){
            cv::Mat X1_ok,X2_ok;
            for(int i=0;i<ok.rows;i++){
                if(ok.ptr<int>(i)[0]){
                    X1_ok.push_back(X1.row(i));
                    X2_ok.push_back(X2.row(i));
                }
            }
            cv::transpose(X1_ok,X1_ok);
            cv::transpose(X2_ok,X2_ok);
            std::tie(stitching_res,std::ignore,std::ignore,
                    std::ignore,std::ignore,mass,overlap_mass) = local_TPS(im1, im2, H, X1_ok, X2_ok, im1_mask, im2_mask, mode);
        }
    }
    else{
        cv::Mat X1_ok,X2_ok;
        for(int i=0;i<ok.rows;i++){
            if(ok.ptr<int>(i)[0]){
                X1_ok.push_back(X1.row(i));
                X2_ok.push_back(X2.row(i));
            }
        }
        cv::transpose(X1_ok,X1_ok);
        cv::transpose(X2_ok,X2_ok);
        std::tie(stitching_res,std::ignore,std::ignore,
                std::ignore,std::ignore,mass,overlap_mass) = local_TPS(im1, im2, H, X1_ok, X2_ok, im1_mask, im2_mask, mode);
    }
    return std::make_tuple(stitching_res, mass, overlap_mass);

}

std::tuple<cv::Mat, cv::Mat, bool> preprocess(cv::Mat& im1, cv::Mat& im2, cv::Mat& im1_mask, cv::Mat& im2_mask, std::string& mode) {

    if (mode == "r") {
        int half_w = im2.cols / 2;
        int half_h = im2.rows / 2;
        cv::Size im1_shape = im1.size();
        cv::Size im2_shape = im2.size();
        cv::Mat roi1 = im1(cv::Rect(im1.cols - half_w, 0, half_w, im1.rows));
        cv::Scalar mean1, stddev1;
        cv::meanStdDev(roi1, mean1, stddev1);
        double std_dev1 = stddev1.val[0];
        if (std_dev1<=12.0) {
            int h = im2_shape.height;
            int extra_w = int(im1_shape.width * 0.9);
            int w = im2_shape.width + extra_w;
            cv::Mat stitching_res = cv::Mat::zeros(h, w, im1.type());
            cv::Mat mass = cv::Mat::ones(h, w, CV_64FC1);
            im1(cv::Rect(0, 0, extra_w, im1.rows)).copyTo(stitching_res(cv::Rect(0, 0, extra_w, im1.rows)));
            im2.copyTo(stitching_res(cv::Rect(w - im2_shape.width, 0, im2_shape.width, im2_shape.height)));
            return std::make_tuple(stitching_res, mass, false);
        }

        cv::Mat roi2 = im2(cv::Rect(0, 0, half_w, im2.rows));
        cv::Scalar mean2, stddev2;
        cv::meanStdDev(roi2, mean2, stddev2);
        double std_dev2 = stddev2.val[0];
    
        if (std_dev2<=12.0) {
            return direct_stitch(im1, im2, im1_mask, im2_mask);
        }
        return std::make_tuple(cv::Mat(), cv::Mat(), true);
    } else {
        return std::make_tuple(cv::Mat(), cv::Mat(), true);
    }
}


void two_stitching(const std::string& data_path, const std::string& store_path, const std::string& top_num, bool refine_flag) {
    std::list<cv::Mat> tier_list;
    std::list<cv::Mat> tier_mask_list;


    #pragma omp parallel for
    for (int i = 0; i < 2; ++i) {
        cv::Mat im1_mask,im2_mask;
        cv::Mat img_1_mask, img_2_mask;
        cv::Mat img_1,img_2;
        cv::Mat stitching_res_temp, mass_temp;
        cv::Mat stitching_res, mass;

        img_1 = cv::imread(data_path + "/" + top_num + "_" + std::to_string(i + 1) + "_1.bmp");
        img_2 = cv::imread(data_path + "/" + top_num + "_" + std::to_string(i + 1) + "_2.bmp");
        if (!img_1.empty() && !img_2.empty()) {
            cv::cvtColor(img_1, img_1, cv::COLOR_BGR2GRAY);
            cv::cvtColor(img_2, img_2, cv::COLOR_BGR2GRAY);
            std::string mode = "r";
            bool process_flag;
            if(!im1_mask.empty()) im1_mask = cv::Mat();
            if(!im2_mask.empty()) im2_mask = cv::Mat();
            std::tie(stitching_res_temp, mass_temp, process_flag) = preprocess(img_1, img_2, im1_mask, im2_mask, mode);
            if (process_flag) {
                img_1_mask = cv::Mat::ones(img_1.size(), CV_64FC1);
                img_2_mask = cv::Mat::ones(img_2.size(), CV_64FC1);
                std::tie(stitching_res, mass, std::ignore) = stitching_pair(img_1, img_2, img_1_mask, img_2_mask, mode);
                stitching_res.convertTo(stitching_res, CV_8UC1);
            } else {
                stitching_res = stitching_res_temp;
                mass = mass_temp;
                stitching_res.convertTo(stitching_res, CV_8UC1);
            }
        }
        else if (img_1.empty()) {
            cv::cvtColor(img_2, img_2, cv::COLOR_BGR2GRAY);
            mass = cv::Mat::ones(img_2.size(), CV_64FC1);
        } else {
            cv::cvtColor(img_1, img_1, cv::COLOR_BGR2GRAY);
            mass = cv::Mat::ones(img_1.size(), CV_64FC1);
        }
        tier_list.push_back(stitching_res);
        tier_mask_list.push_back(mass);
    }
    std::list<cv::Mat>::iterator it_tier = tier_list.begin();
    cv::Mat im1 = *it_tier;
    cv::Mat im2 = *(++it_tier);

    cv::Mat im1_mask,im2_mask;
    cv::Mat stitching_res;
    std::list<cv::Mat>::iterator it_tier_mask = tier_mask_list.begin();
    im1_mask = *(it_tier_mask);
    im2_mask = *(++it_tier_mask);
    std::string mode = "d";
    std::tie(stitching_res,std::ignore,std::ignore) = stitching_rows(im1,im2,im1_mask,im2_mask,mode,refine_flag);
    stitching_res.convertTo(stitching_res,CV_8UC1);
    cv::imwrite(store_path + "/" + top_num + "-res.bmp",stitching_res);
}


void two_stitching_seq(const std::string& data_path, const std::string& store_path, const std::string& top_num, bool refine_flag) {
    std::list<cv::Mat> tier_list;
    std::list<cv::Mat> tier_mask_list;
    cv::Mat im1_mask,im2_mask;
    cv::Mat img_1_mask, img_2_mask;
    cv::Mat img_1,img_2;
    cv::Mat stitching_res_temp, mass_temp;
    cv::Mat stitching_res, mass;
    for (int i = 0; i < 2; ++i) {
        img_1 = cv::imread(data_path + "/" + top_num + "_" + std::to_string(i + 1) + "_1.bmp");
        img_2 = cv::imread(data_path + "/" + top_num + "_" + std::to_string(i + 1) + "_2.bmp");
        if (!img_1.empty() && !img_2.empty()) {
            cv::cvtColor(img_1, img_1, cv::COLOR_BGR2GRAY);
            cv::cvtColor(img_2, img_2, cv::COLOR_BGR2GRAY);
            std::string mode = "r";
            bool process_flag;
            if(!im1_mask.empty()) im1_mask.release();
            if(!im2_mask.empty()) im2_mask.release();
            std::tie(stitching_res_temp, mass_temp, process_flag) = preprocess(img_1, img_2, im1_mask, im2_mask, mode);
            if (process_flag) {
                img_1_mask = cv::Mat::ones(img_1.size(), CV_64FC1);
                img_2_mask = cv::Mat::ones(img_2.size(), CV_64FC1);
                std::tie(stitching_res, mass, std::ignore) = stitching_pair(img_1, img_2, img_1_mask, img_2_mask, mode);
                stitching_res.convertTo(stitching_res, CV_8UC1);
            } else {
                stitching_res = stitching_res_temp;
                mass = mass_temp;
                stitching_res.convertTo(stitching_res, CV_8UC1);
            }
        }
        else if (img_1.empty()) {
            cv::cvtColor(img_2, img_2, cv::COLOR_BGR2GRAY);
            mass = cv::Mat::ones(img_2.size(), CV_64FC1);
        } else {
            cv::cvtColor(img_1, img_1, cv::COLOR_BGR2GRAY);
            mass = cv::Mat::ones(img_1.size(), CV_64FC1);
        }

        tier_list.push_back(stitching_res);
        tier_mask_list.push_back(mass);

        im1_mask.release(),im2_mask.release();
        img_1_mask.release(),img_2_mask.release();
        img_1.release(),img_2.release();
        stitching_res_temp.release(), mass_temp.release();
        stitching_res.release(), mass.release();
    }
    std::list<cv::Mat>::iterator it_tier = tier_list.begin();
    cv::Mat im1 = *it_tier;
    cv::Mat im2 = *(++it_tier);

    std::list<cv::Mat>::iterator it_tier_mask = tier_mask_list.begin();
    im1_mask = *(it_tier_mask);
    im2_mask = *(++it_tier_mask);
    std::string mode = "d";
    std::tie(stitching_res,std::ignore,std::ignore) = stitching_rows(im1,im2,im1_mask,im2_mask,mode,refine_flag);
    // std::tie(stitching_res,std::ignore,std::ignore) = stitching_rows(im1,im2,im1_mask,im2_mask,mode,overlap_rate,refine_flag);
    stitching_res.convertTo(stitching_res,CV_8UC1);
    cv::imwrite(store_path + "/" + top_num + "-res.bmp",stitching_res);
}


void two_stitching(const std::string& data_path, const std::string& store_path, const std::string& top_num, double overlap_rate, bool refine_flag) {
    std::list<cv::Mat> tier_list;
    std::list<cv::Mat> tier_mask_list;

    #pragma omp parallel for ordered
    for (int i = 0; i < 2; ++i) {
        cv::Mat im1_mask,im2_mask;
        cv::Mat img_1_mask, img_2_mask;
        cv::Mat img_1,img_2;
        cv::Mat stitching_res_temp, mass_temp;
        cv::Mat stitching_res, mass;

        img_1 = cv::imread(data_path + "/" + top_num + "_" + std::to_string(i + 1) + "_1.bmp");
        img_2 = cv::imread(data_path + "/" + top_num + "_" + std::to_string(i + 1) + "_2.bmp");

        if (!img_1.empty() && !img_2.empty()) {
            cv::cvtColor(img_1, img_1, cv::COLOR_BGR2GRAY);
            cv::cvtColor(img_2, img_2, cv::COLOR_BGR2GRAY);
            std::string mode = "r";
            bool process_flag;
            if(!im1_mask.empty()) im1_mask = cv::Mat();
            if(!im2_mask.empty()) im2_mask = cv::Mat();
            std::tie(stitching_res_temp, mass_temp, process_flag) = preprocess(img_1, img_2, im1_mask, im2_mask, mode);
            if (process_flag) {
                img_1_mask = cv::Mat::ones(img_1.size(), CV_64FC1);
                img_2_mask = cv::Mat::ones(img_2.size(), CV_64FC1);
                std::tie(stitching_res, mass, std::ignore) = stitching_pair(img_1, img_2, img_1_mask, img_2_mask, mode, overlap_rate);
                stitching_res.convertTo(stitching_res, CV_8UC1);
            } else {
                stitching_res = stitching_res_temp;
                mass = mass_temp;
                stitching_res.convertTo(stitching_res, CV_8UC1);
            }
        }
        else if (img_1.empty()) {
            cv::cvtColor(img_2, img_2, cv::COLOR_BGR2GRAY);
            mass = cv::Mat::ones(img_2.size(), CV_64FC1);
        } else {
            cv::cvtColor(img_1, img_1, cv::COLOR_BGR2GRAY);
            mass = cv::Mat::ones(img_1.size(), CV_64FC1);
        }

        #pragma omp ordered
        {
            tier_list.push_back(stitching_res);
            tier_mask_list.push_back(mass);
        }
        
        // cv::imwrite(store_path + "/row-" + std::to_string(i+1) + "-res.bmp", stitching_res);
    }
    // std::cout << "rows completed" << std::endl;
    std::list<cv::Mat>::iterator it_tier = tier_list.begin();
    cv::Mat im1 = *it_tier;
    cv::Mat im2 = *(++it_tier);

    cv::Mat im1_mask,im2_mask;
    cv::Mat stitching_res;
    std::list<cv::Mat>::iterator it_tier_mask = tier_mask_list.begin();
    im1_mask = *(it_tier_mask);
    im2_mask = *(++it_tier_mask);
    std::string mode = "d";
    std::tie(stitching_res,std::ignore,std::ignore) = stitching_rows(im1,im2,im1_mask,im2_mask,mode,overlap_rate,refine_flag);
    stitching_res.convertTo(stitching_res,CV_8UC1);
    cv::imwrite(store_path + "/" + top_num + "-res.bmp",stitching_res);
}

void three_stitching(const std::string& data_path, const std::string& store_path, const std::string& top_num, bool refine_flag){
    std::vector<cv::Mat> tier_list;
    std::vector<cv::Mat> tier_mask_list;
    
        #pragma omp parallel for 
    for (int i = 0; i < 3; ++i) {
        cv::Mat img_1 = cv::imread(data_path + "/" + top_num + "_" + std::to_string(i + 1) + "_1.bmp");
        cv::Mat img_2 = cv::imread(data_path + "/" + top_num + "_" + std::to_string(i + 1) + "_2.bmp");
        cv::Mat stitching_res, mass;
        cv::Mat im1_mask,im2_mask;
        cv::Mat img_1_mask, img_2_mask;
        cv::Mat stitching_res_temp, mass_temp;
        if (!img_1.empty() && !img_2.empty()) {
            cv::cvtColor(img_1, img_1, cv::COLOR_BGR2GRAY);
            cv::cvtColor(img_2, img_2, cv::COLOR_BGR2GRAY);
            std::string mode = "r";
            bool process_flag;
            im1_mask = cv::Mat();
            im2_mask = cv::Mat();
            std::tie(stitching_res_temp, mass_temp, process_flag) = preprocess(img_1, img_2, im1_mask, im2_mask, mode);
            if (process_flag) {
                img_1_mask = cv::Mat::ones(img_1.size(), CV_64FC1);
                img_2_mask = cv::Mat::ones(img_2.size(), CV_64FC1);
                std::tie(stitching_res, mass, std::ignore) = stitching_pair(img_1, img_2, img_1_mask, img_2_mask, mode);
                stitching_res.convertTo(stitching_res, CV_8UC1);
            } else {
                stitching_res = stitching_res_temp;
                mass = mass_temp;
                stitching_res.convertTo(stitching_res, CV_8UC1);
            }
        }
        else if (img_1.empty()) {
            cv::cvtColor(img_2, img_2, cv::COLOR_BGR2GRAY);
            mass = cv::Mat::ones(img_2.size(), CV_64FC1);
        } else {
            cv::cvtColor(img_1, img_1, cv::COLOR_BGR2GRAY);
            mass = cv::Mat::ones(img_1.size(), CV_64FC1);
        }
        cv::Mat img_3 = cv::imread(data_path + "/" + top_num + "_" + std::to_string(i + 1) + "_3.bmp");
        if(img_3.empty()){
            tier_list.push_back(stitching_res);
            tier_mask_list.push_back(mass);
            continue;
        }
        cv::cvtColor(img_3, img_3, cv::COLOR_BGR2GRAY);
        cv::Mat img_3_mask = cv::Mat::ones(img_3.size(),img_3.type());
        std::string mode = "r";
        bool process_flag;
        cv::Mat none_mask = cv::Mat();
        std::tie(stitching_res_temp, mass_temp, process_flag) = preprocess(stitching_res, img_3, mass, none_mask, mode);
        if (process_flag) {
            std::tie(stitching_res, mass, std::ignore) = stitching_pair(stitching_res, img_3, mass, img_3_mask, mode);
            stitching_res.convertTo(stitching_res, CV_8UC1);
        } else {
            stitching_res = stitching_res_temp;
            mass = mass_temp;
            stitching_res.convertTo(stitching_res, CV_8UC1);
        }
        tier_list.push_back(stitching_res);
        tier_mask_list.push_back(mass);

    }

    while(tier_list.size()>=2){

        cv::Mat im1 = tier_list[0];
        cv::Mat im2 = tier_list[1];
        cv::Mat im1_mask = tier_mask_list[0];
        cv::Mat im2_mask = tier_mask_list[1];
        std::string mode = "d";
        cv::Mat stitching_res,mass, overlap_mass;
        std::tie(stitching_res, mass, overlap_mass) = stitching_rows(im1,im2,im1_mask,im2_mask,mode,refine_flag);
        stitching_res.convertTo(stitching_res,CV_8UC1);
        tier_list[1] = stitching_res;
        tier_mask_list[1] = mass;
        tier_list.erase(tier_list.begin());
        tier_mask_list.erase(tier_mask_list.begin());

        im1_mask.release(),im2_mask.release();
        im1.release(),im2.release();
        overlap_mass.release();
        stitching_res.release(), mass.release();
    }
    cv::Mat final_res = tier_list[0];
    final_res.convertTo(final_res,CV_8UC1);
    cv::imwrite(store_path + "/" + top_num + "-res.bmp",final_res);

}

