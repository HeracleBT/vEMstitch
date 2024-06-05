#include"rigidtransform.h"
#include"Utils.h"


std::vector<int> getRandomSubset(int point_num, int subset_size) {
    std::vector<int> nums(point_num);
    std::iota(nums.begin(), nums.end(), 0); 
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(nums.begin(), nums.end(), g); 
    std::vector<int> subset(nums.begin(), nums.begin() + subset_size);
    return subset;
}

void getSubsetMatrices(cv::Mat& x1, std::vector<int>& subset, cv::Mat& x1_subset) {
    int subset_size = subset.size();
    int rows = x1.rows;
    int cols = x1.cols;

    x1_subset = cv::Mat(subset_size, cols, x1.type());
    for (int i = 0; i < subset_size; ++i) {
        int index = subset[i];
        x1.row(index).copyTo(x1_subset.row(i));
    }
}

std::tuple<cv::Mat,cv::Mat> RANSAC(cv::Mat& ps1, cv::Mat& ps2, int iter_num, double min_dis) {
    int point_num = ps1.rows;

    cv::Mat x1 = ps1.col(0).reshape(1);
    cv::Mat y1 = ps1.col(1).reshape(1); 
    cv::Mat x2 = ps2.col(0).reshape(1);
    cv::Mat y2 = ps2.col(1).reshape(1); 

    cv::Mat concatenated;
    cv::vconcat(x1, y1, concatenated);
    cv::vconcat(concatenated, x2, concatenated);
    cv::vconcat(concatenated, y2, concatenated);
    cv::Scalar mean = cv::mean(concatenated);
    double scale = 1 / mean.val[0]; 
    x1*=scale;
    y1*=scale;
    x2*=scale;
    y2*=scale;

    cv::Mat X = cv::Mat::zeros(point_num, 7, CV_64FC1);
    cv::Mat Y = cv::Mat::zeros(point_num, 7, CV_64FC1);
    cv::Mat zeros = cv::Mat::zeros(point_num, 3, CV_64FC1);
    cv::Mat ones = cv::Mat::ones(point_num, 1, CV_64FC1);
    cv::hconcat(zeros, x1, X);
    cv::hconcat(X, y1, X);
    cv::hconcat(X, ones, X);
    cv::hconcat(X, -y2.mul(x1), X);
    cv::hconcat(X, -y2.mul(y1), X);
    cv::hconcat(X, -y2, X);
    cv::hconcat(x1, y1, Y);
    cv::hconcat(Y, ones, Y);
    cv::hconcat(Y, zeros, Y);
    cv::hconcat(Y, -x2.mul(x1), Y);
    cv::hconcat(Y, -x2.mul(y1), Y);
    cv::hconcat(Y, -x2, Y);

    std::vector<cv::Mat>ok(iter_num);
    std::vector<int>score(iter_num, 0);
    // std::vector<int> subset;

    #pragma omp parallel for
    for(int it=0;it<iter_num;it++){
        std::vector<int> subset;
        subset = getRandomSubset(point_num, 4);
        cv::Mat x1_subset, y1_subset, x2_subset, y2_subset;
        getSubsetMatrices(x1, subset, x1_subset);
        getSubsetMatrices(x2, subset, x2_subset);
        getSubsetMatrices(y1, subset, y1_subset);
        getSubsetMatrices(y2, subset, y2_subset);
        if(!rigidity_cons(x1_subset,y1_subset,x2_subset,y2_subset)){
            continue;
        }
        cv::Mat A_it,X_subset,Y_subset;
        getSubsetMatrices(X, subset, X_subset);
        getSubsetMatrices(Y, subset, Y_subset);
        A_it = X_subset;
        cv::vconcat(A_it,Y_subset,A_it);
        cv::Mat U, S, Vt;
        cv::SVD::compute(A_it, S, U, Vt,cv::SVD::FULL_UV);
        cv::Mat h = Vt.t().col(8);
        cv::Mat X_dot_h = X * h;
        cv::Mat Y_dot_h = Y * h;
        cv::Mat dis = X_dot_h.mul(X_dot_h) + Y_dot_h.mul(Y_dot_h);
        cv::Mat ok_it = dis < min_dis * min_dis;

        ok[it] = ok_it;
        score[it] = cv::countNonZero(ok_it);
        
        x1_subset.release(), y1_subset.release(), x2_subset.release(), y2_subset.release();
        A_it.release(),X_subset.release(),Y_subset.release();
        U.release(), S.release(), Vt.release();
        h.release(),X_dot_h.release(),Y_dot_h.release(),dis.release(),ok_it.release();
    }

    auto max_it = std::max_element(score.begin(), score.end());
    int max_score = *max_it;
    int best = std::distance(score.begin(), max_it);
    cv::Mat ok_best = ok[best];


    cv::Mat X_ok,Y_ok;
    for(int i=0;i<ok_best.rows;i++){
        if(ok_best.at<int>(i,0)){
            X_ok.push_back(X.row(i));
            Y_ok.push_back(Y.row(i));
        }
    }
    cv::Mat A = X_ok;
    cv::vconcat(A, Y_ok, A);
    cv::Mat U, S, Vt;
    cv::SVD::compute(A, S, U, Vt,cv::SVD::FULL_UV);
    cv::Mat h = Vt.t().col(8);
    cv::Mat H = cv::Mat::zeros(3, 3, CV_64F);
    int index = 0;
    for(int i = 0; i < H.rows; ++i) {
        for(int j = 0; j < H.cols; ++j) {
            H.at<double>(i, j) = h.at<double>(index++, 0);
        }
    }
    cv::Mat scale_matrix = (cv::Mat_<double>(3, 3) << 1/scale, 0.0, 0.0,
                                                     0.0, 1/scale, 0.0,
                                                     0.0, 0.0, 1.0);
    cv::Mat scale_matrix2 = (cv::Mat_<double>(3, 3) << scale, 0.0, 0.0,
                                                     0.0, scale, 0.0,
                                                     0.0, 0.0, 1.0);
    H = scale_matrix*H*scale_matrix2;
    
    return std::make_tuple(H,ok_best);

}

std::tuple<cv::Mat,cv::Mat,cv::Mat,cv::Mat>rigid_transform(std::vector<cv::KeyPoint>& kp1,cv::Mat& dsp1, 
                                                        std::vector<cv::KeyPoint>& kp2, cv::Mat& dsp2,
                                                        cv::Mat& im1_mask, cv::Mat& im2_mask,
                                                        std::string& mode){
    double dis = 0.0;
    if(mode == "d"){
        dis = im1_mask.rows;
    } else if(mode == "l"||mode == "r"){
        dis = im1_mask.cols;
    }
    std::pair<std::string,double> shifting = std::make_pair(mode,dis);
    cv::Mat X1,X2;
    std::tie(X1,X2) = flann_match(kp1,dsp1,kp2,dsp2,0.4,im1_mask,im2_mask,shifting);

    if (X1.empty()){
        return std::make_tuple(cv::Mat(),cv::Mat(),cv::Mat(),cv::Mat());
    }

    cv::Mat H;
    cv::Mat ok = cv::Mat::ones(X1.rows, 1, CV_8UC1);
    try {
        cv::Mat X1_copy = X1.clone();
        cv::Mat X2_copy = X2.clone();
        std::tie(H,ok) = RANSAC(X1_copy, X2_copy, 2000, 0.1);
    } catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return std::make_tuple(cv::Mat(),cv::Mat(),cv::Mat(),cv::Mat());
    }

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
    H = X_transpose * Y_ok;

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

std::tuple<cv::Mat,cv::Mat,cv::Mat,cv::Mat,cv::Mat,cv::Mat,cv::Mat> stitching_global(cv::Mat& im1,cv::Mat& im2, cv::Mat& H, cv::Mat& X1_ok,cv::Mat& X2_ok,
                cv::Mat& im1_mask,cv::Mat& im2_mask,std::string& mode){

    if (im1_mask.empty()) {
        im1_mask = cv::Mat::ones(im1.size(), CV_64FC1);
    }
    if (im2_mask.empty()) {
        im2_mask = cv::Mat::ones(im2.size(), CV_64FC1);
    }

    cv::Size imsize1 = im1.size();
    cv::Size imsize2 = im2.size();

    cv::Mat box1 = (cv::Mat_<double>(3, 4) <<
                    0, im1.cols - 1, im1.cols - 1, 0,
                    0, 0, im1.rows - 1, im1.rows - 1,
                    1, 1, 1, 1);

    cv::Mat box2 = (cv::Mat_<double>(3, 4) <<
                    0, im2.cols - 1, im2.cols - 1, 0,
                    0, 0, im2.rows - 1, im2.rows - 1,
                    1, 1, 1, 1);

    cv::Mat box2_ = H.inv() * box2;
    box2_.row(0) /= box2_.row(2);
    box2_.row(1) /= box2_.row(2);
    double min_1,max_1,min_2,max_2;
    cv::minMaxLoc(box2_.row(0), &min_1, &max_1,NULL, NULL);
    cv::minMaxLoc(box2_.row(1), &min_2, &max_2,NULL, NULL);
    double u0 = std::min((double)0.0, min_1);
    double u1 = std::max((double)(im1.cols - 1), max_1);
    std::vector<double> ur = range(u0, (u1+1), 1);
    double v0 = std::min((double)0.0, min_2);
    double v1 = std::max((double)(im1.rows - 1), max_2);
    std::vector<double> vr = range(v0, (v1+1), 1);
    int mosaicw = ur.size();
    int mosaich = vr.size();

    cv::Mat u = cv::Mat(ur);
    cv::transpose(u,u);
    cv::Mat v = cv::Mat(vr);
    cv::transpose(v,v);
    u = cv::repeat(u,vr.size(),1);
    v = cv::repeat(v,ur.size(),1);
    cv::transpose(v,v);

    im1.convertTo(im1,CV_32FC1);
    im1_mask.convertTo(im1_mask,CV_32FC1);
    u.convertTo(u,CV_32FC1);
    v.convertTo(v,CV_32FC1);
    cv::Mat im1_p;
    im1_p.create(im1.size(), im1.type());
    cv::Mat warped_mask1;
    warped_mask1.create(im1_mask.size(), im1_mask.type());

    cv::remap(im1,im1_p,u,v,cv::INTER_CUBIC);
    cv::remap(im1_mask,warped_mask1,u,v,cv::INTER_CUBIC);

    im1_p.convertTo(im1_p,CV_64FC1);
    warped_mask1.convertTo(warped_mask1,CV_64FC1);
    u.convertTo(u,CV_64FC1);
    v.convertTo(v,CV_64FC1);

    cv::Mat z_ = H.at<double>(2,0)*u + H.at<double>(2,1)*v + H.at<double>(2,2);
    cv::Mat u_ = (H.at<double>(0,0)*u + H.at<double>(0,1)*v + H.at<double>(0,2))/z_;
    cv::Mat v_ = (H.at<double>(1,0)*u + H.at<double>(1,1)*v + H.at<double>(1,2))/z_;

    im2.convertTo(im2,CV_32FC1);
    im2_mask.convertTo(im2_mask,CV_32FC1);
    u_.convertTo(u_,CV_32FC1);
    v_.convertTo(v_,CV_32FC1);
    cv::Mat im2_p;
    im2_p.create(im2.size(), im2.type());
    cv::Mat warped_mask2;
    warped_mask2.create(im2_mask.size(), im2_mask.type());
    cv::remap(im2,im2_p,u_,v_,cv::INTER_CUBIC);
    cv::remap(im2_mask,warped_mask2,u_,v_,cv::INTER_CUBIC);
    im2_p.convertTo(im2_p,CV_64FC1);
    warped_mask2.convertTo(warped_mask2,CV_64FC1);
    u_.convertTo(u_,CV_64FC1);
    v_.convertTo(v_,CV_64FC1);

    cv::threshold(warped_mask1, warped_mask1, 0.8, 1.0, cv::THRESH_BINARY);
    cv::threshold(warped_mask2, warped_mask2, 0.8, 1.0, cv::THRESH_BINARY);

    cv::Mat mass,overlap_mass,warped_mask11,warped_mask22;
    if(mode == "d"){
        std::tie(warped_mask11,warped_mask22,mass,overlap_mass) = stitch_add_mask_linear_per_border(warped_mask1,warped_mask2);
    }else{
        std::tie(warped_mask11,warped_mask22,mass,overlap_mass) = stitch_add_mask_linear_border(warped_mask1,warped_mask2,mode);
    }
    cv::Mat stitching_res = im1_p.mul(warped_mask11) + im2_p.mul(warped_mask22);

    return std::make_tuple(stitching_res,v,u,v_,u_,mass,overlap_mass);

}
