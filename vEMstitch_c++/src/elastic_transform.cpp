#include"elastictransform.h"
#include"Utils.h"
#define M_PI       3.14159265358979323846


std::tuple<cv::Mat,cv::Mat,cv::Mat,cv::Mat,cv::Mat,cv::Mat,cv::Mat> local_TPS(cv::Mat& im1,cv::Mat& im2, cv::Mat& H, cv::Mat& X1_ok,cv::Mat& X2_ok,
                cv::Mat& im1_mask,cv::Mat& im2_mask,std::string& mode){

    if (im1_mask.empty()) {
        im1_mask = cv::Mat::ones(im1.size(), CV_64FC1);
    }
    if (im2_mask.empty()) {
        im2_mask = cv::Mat::ones(im2.size(), CV_64FC1);
    }

    cv::Size imsize1 = im1.size();
    cv::Size imsize2 = im2.size();

    // Parameters
    double lambd;
    if (mode == "d") {
        lambd = 0.001 * imsize1.height; // weighting parameter to balance the fitting term and the smoothing term
    } else {
        lambd = 0.001 * imsize1.height * imsize1.width;
    }
    int intv_mesh = 3;
    int K_smooth = 5;

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



    //align the sub coordinates with the mosaic coordinates
    double margin = 0.1*std::min(imsize1.height,imsize1.width);
    double u0_im_ = std::max((min_1-margin),(u0));
    double u1_im_ = std::min((max_1+margin),(u1));
    double v0_im_ = std::max((min_2-margin),(v0));
    double v1_im_ = std::min((max_2+margin),(v1));
    int offset_u0_ = int(std::ceil(u0_im_ - u0));
    int offset_u1_ = int(std::floor(u1_im_ - u0));
    int offset_v0_ = int(std::ceil(v0_im_ - v0));
    int offset_v1_ = int(std::floor(v1_im_ - v0));
    int imw_ = int(std::floor(offset_u1_ - offset_u0_ + 1));
    int imh_ = int(std::floor(offset_v1_ - offset_v0_ + 1));


    cv::Mat box1_2 = H * box1;
    box1_2.row(0) = box1_2.row(0)/box1_2.row(2);
    box1_2.row(1) = box1_2.row(1)/box1_2.row(2);
    cv::minMaxLoc(box1_2.row(0), &min_1, &max_1,NULL, NULL);
    cv::minMaxLoc(box1_2.row(1), &min_2, &max_2,NULL, NULL);
    double sub_u0_ = std::max((double)0.0,min_1);
    double sub_u1_ = std::min((double)(imsize2.width-1),max_1);
    double sub_v0_ = std::max((double)(0.0),(min_2)) - margin;
    double sub_v1_ = std::min((double)(imsize2.height-1),max_2);

    //TPS
    //merge the coincided points（重合点）
    cv::Mat ok_nd1 = cv::Mat::zeros(1, X1_ok.cols, CV_8UC1);
    cv::Mat rounded_X1_ok;//2*n
    X1_ok.convertTo(rounded_X1_ok, CV_32S);
    std::set<std::pair<int,int>> unique_points_set;
    std::vector<int>idx1;
    for (int i = 0; i < rounded_X1_ok.cols; ++i) {
        std::pair<int,int> point(rounded_X1_ok.ptr<int>(0)[i], rounded_X1_ok.ptr<int>(1)[i]);
        if(unique_points_set.insert(point).second){
            idx1.push_back(i);
        }
    }
    rounded_X1_ok.release();
    for (auto& i : idx1) {
        ok_nd1.ptr<uchar>(0)[i] = 1;
    }
    cv::Mat ok_nd2 = cv::Mat::zeros(1, X2_ok.cols, CV_8UC1);
    cv::Mat rounded_X2_ok;//2*n
    X2_ok.convertTo(rounded_X2_ok, CV_32S);
    unique_points_set.clear();
    std::vector<int>idx2;
    for (int i = 0; i < rounded_X2_ok.cols; ++i) {
        std::pair<int,int> point(rounded_X2_ok.ptr<int>(0)[i], rounded_X2_ok.ptr<int>(1)[i]);
        if(unique_points_set.insert(point).second){
            idx2.push_back(i);
        }
    }
    for (auto& i : idx2) {
        ok_nd2.ptr<uchar>(0)[i] = 1;
    }
    cv::Mat ok_nd;
    cv::bitwise_and(ok_nd1,ok_nd2,ok_nd);
    cv::Mat X1_nd,X2_nd;
    for(int i=0;i<ok_nd.cols;i++){
        if(ok_nd.at<uchar>(0,i)){
            X1_nd.push_back(X1_ok.col(i).t());
            X2_nd.push_back(X2_ok.col(i).t());
        }
    }
    ok_nd.release(),ok_nd1.release(),ok_nd2.release();
    cv::transpose(X1_nd,X1_nd);
    cv::transpose(X2_nd,X2_nd);

    //form the linear system
    cv::Mat x1 = X1_nd.row(0);
    cv::Mat y1 = X1_nd.row(1);
    cv::Mat x2 = X2_nd.row(0);
    cv::Mat y2 = X2_nd.row(1);

    cv::Mat z1_ = H.at<double>(2,0)*x1 + H.at<double>(2,1)*y1 + H.at<double>(2,2);
    cv::Mat x1_ = (H.at<double>(0,0)*x1 + H.at<double>(0,1)*y1 + H.at<double>(0,2))/z1_;
    cv::Mat y1_ = (H.at<double>(1,0)*x1 + H.at<double>(1,1)*y1 + H.at<double>(1,2))/z1_;
    cv::Mat gxn = x1_-x2;
    cv::Mat hyn = y1_-y2;


    int n = x1_.cols;
    cv::Mat xx,yy;
    xx=cv::repeat(x1_,n,1);
    yy=cv::repeat(y1_,n,1);

    cv::Mat xx_t = xx.t();
    cv::Mat yy_t = yy.t();
    cv::Mat x_diff = xx-xx_t;
    cv::Mat y_diff = yy-yy_t;
    cv::pow(x_diff, 2, x_diff);
    cv::pow(y_diff, 2, y_diff);
    cv::Mat dist2 = x_diff + y_diff;
    dist2.diag(0) = 1.0;
    cv::Mat dis2_log;
    cv::log(dist2,dis2_log);
    cv::Mat K = 0.5*dist2.mul(dis2_log);
    
    K.diag(0) = lambd*8.0*M_PI;

    cv::Mat K_ = cv::Mat::zeros(n + 3, n + 3, CV_64FC1);
    K.copyTo(K_(cv::Range(0,n),cv::Range(0,n)));
    x1_.copyTo(K_(cv::Range(n,n+1),cv::Range(0,n)));
    y1_.copyTo(K_(cv::Range(n+1,n+2),cv::Range(0,n)));
    K_(cv::Range(n+2,n+3),cv::Range(0,n)) = 1.0;

    cv::Mat x1_t = x1_.t();
    x1_t.copyTo(K_(cv::Range(0,n),cv::Range(n,n+1)));
    cv::Mat y1_t = y1_.t();
    y1_t.copyTo(K_(cv::Range(0,n),cv::Range(n+1,n+2)));
    K_(cv::Range(0,n),cv::Range(n+2,n+3)) = 1.0;
    cv::Mat G_ = cv::Mat::zeros(n + 3, 2, CV_64FC1);
    cv::Mat gxn_t = gxn.t();
    cv::Mat hyn_t = hyn.t();
    gxn_t.copyTo(G_(cv::Range(0,n),cv::Range(0,1)));
    hyn_t.copyTo(G_(cv::Range(0,n),cv::Range(1,2)));

    //solve the linear system
    cv::Mat W_  = K_.inv() * G_;
    cv::Mat wx = W_(cv::Rect(0, 0, 1, n));
    cv::Mat wy = W_(cv::Rect(1, 0, 1, n));
    cv::Mat a = W_(cv::Rect(0, n, 1, 3));
    cv::Mat b = W_(cv::Rect(1, n, 1, 3));


    // remove outliers based on the distribution of weights
    cv::Mat std_wx_mat, std_wy_mat;
    cv::meanStdDev(wx, cv::noArray(), std_wx_mat);
    cv::meanStdDev(wy, cv::noArray(), std_wy_mat);
    double std_wx = (std_wx_mat.at<double>(0, 0));
    double std_wy = (std_wy_mat.at<double>(0, 0));

    cv::Mat abs_wx, abs_wy;
    abs_wx = cv::abs(wx);
    abs_wy = cv::abs(wy);
    cv::Mat outlier1,outlier2,outlier;
    cv::threshold(abs_wx, outlier1, 3*std_wx, 1.0, cv::THRESH_BINARY);
    cv::threshold(abs_wy, outlier2, 3*std_wy, 1.0, cv::THRESH_BINARY);
    cv::bitwise_or(outlier1, outlier2, outlier);

    std::vector<double> inlier_idx = range(0, x1_.cols, 1);
   
    for(int kiter=0;kiter<10;kiter++){
        if(cv::sum(outlier)[0]<0.0027*n) break;
        cv::Mat ok=outlier.clone();
        ok = 1 - ok;
        
        std::vector<double> inlier_idx_2;
        for(int i=0;i<ok.rows;i++){
            if(ok.ptr<double>(i)[0]){
                inlier_idx_2.push_back(inlier_idx[i]);
            }
        }
        inlier_idx.assign(inlier_idx_2.begin(), inlier_idx_2.end());
        ok.push_back(1.0),ok.push_back(1.0),ok.push_back(1.0);
        cv::Mat K_i,G_i_;
        for(int i=0;i<ok.rows;i++){
            if(ok.ptr<double>(i)[0]){
                if(K_i.empty()){
                    K_i.push_back(K_.row(i));
                    G_i_.push_back(G_.row(i));
                }
                else{
                    cv::vconcat(K_i,K_.row(i),K_i);
                    cv::vconcat(G_i_,G_.row(i),G_i_);
                }
            }
        }
        cv::Mat K_i_;
        for(int i=0;i<ok.rows;i++){
            if(ok.ptr<double>(i)[0]){
                if(K_i_.empty()){
                    K_i_.push_back(K_i.col(i));
                }
                else{
                    cv::hconcat(K_i_,K_i.col(i),K_i_);
                }
            }
        }
        K_.release(),G_.release();
        K_ = K_i_.clone(),G_ = G_i_.clone();
        W_ = K_.inv()*G_;

        n=inlier_idx.size();
        W_(cv::Range(0,n),cv::Range(0,1)).copyTo(wx);
        W_(cv::Range(0,n),cv::Range(1,2)).copyTo(wy);
        W_(cv::Range(n,n+3),cv::Range(0,1)).copyTo(a);
        W_(cv::Range(n,n+3),cv::Range(1,2)).copyTo(b);
        cv::meanStdDev(wx, cv::noArray(), std_wx_mat);
        cv::meanStdDev(wy, cv::noArray(), std_wy_mat);
        std_wx = std_wx_mat.at<double>(0, 0);
        std_wy = std_wy_mat.at<double>(0, 0);
        abs_wx = cv::abs(wx);
        abs_wy = cv::abs(wy);
        outlier1.release(),outlier2.release(),outlier.release();
        cv::threshold(abs_wx, outlier1, 3*std_wx, 1.0, cv::THRESH_BINARY);
        cv::threshold(abs_wy, outlier2, 3*std_wy, 1.0, cv::THRESH_BINARY);
        cv::bitwise_or(outlier1, outlier2, outlier);

        ok.release(),K_i.release(),G_i_.release(),K_i_.release();
        abs_wx.release(),abs_wy.release(),std_wx_mat.release(),std_wy_mat.release();
    }

    cv::Mat ok = cv::Mat::zeros(x1.rows,x1.cols,CV_64FC1);//1 * 
    #pragma omp parallel for
    for(int i=0;i<inlier_idx.size();i++){
        ok.ptr<double>(0)[int(inlier_idx[i])]=1.0;
    }
    cv::Mat x1_ok,y1_ok,x2_ok,y2_ok,x1_ok_,y1_ok_,gxn_ok,hyn_ok;
    for(int i=0;i<ok.cols;i++){
        if(ok.ptr<double>(0)[i]==1.0){
            x1_ok.push_back(x1.ptr<double>(0)[i]);
            y1_ok.push_back(y1.ptr<double>(0)[i]);
            x2_ok.push_back(x2.ptr<double>(0)[i]);
            y2_ok.push_back(y2.ptr<double>(0)[i]);
            x1_ok_.push_back(x1_.ptr<double>(0)[i]);
            y1_ok_.push_back(y1_.ptr<double>(0)[i]);
            gxn_ok.push_back(gxn.ptr<double>(0)[i]);
            hyn_ok.push_back(hyn.ptr<double>(0)[i]);
        }
    }
    cv::transpose(x1_ok,x1_ok);
    cv::transpose(y1_ok,y1_ok);
    cv::transpose(x2_ok,x2_ok);
    cv::transpose(y2_ok,y2_ok);
    cv::transpose(x1_ok_,x1_ok_);
    cv::transpose(y1_ok_,y1_ok_);
    cv::transpose(gxn_ok,gxn_ok);
    cv::transpose(hyn_ok,hyn_ok);
    x1=x1_ok,y1=y1_ok,x2=x2_ok,y2=y2_ok,x1_=x1_ok_,y1_=y1_ok_,gxn=gxn_ok,hyn=hyn_ok;


    // deform image
    cv::Mat gx = cv::Mat::zeros(mosaich,mosaicw,CV_64FC1);
    cv::Mat hy = cv::Mat::zeros(mosaich,mosaicw,CV_64FC1);
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

    


    cv::Mat u_im_;
    cv::Mat v_im_;
    for (int i = offset_v0_; i < offset_v1_+1; i += intv_mesh){
        if(u_im_.empty()){
            u_im_.push_back(u_.row(i));
            v_im_.push_back(v_.row(i));
        } else {
            cv::vconcat(u_im_,u_.row(i),u_im_);
            cv::vconcat(v_im_,v_.row(i),v_im_);
        }
    }
    cv::Mat u_im_2;
    cv::Mat v_im_2;
    for (int j = offset_u0_; j < offset_u1_+1; j += intv_mesh){
        if(u_im_2.empty()){
            u_im_2.push_back(u_im_.col(j));
            v_im_2.push_back(v_im_.col(j));
        } else {
            cv::hconcat(u_im_2,u_im_.col(j),u_im_2);
            cv::hconcat(v_im_2,v_im_.col(j),v_im_2);
        }
    }
    u_im_.release(),v_im_.release();
    u_im_ = u_im_2,v_im_ = v_im_2;

    int subImh_ = (int)std::ceil((double)imh_ / (double)intv_mesh);
    int subImw_ = (int)std::ceil((double)imw_ / (double)intv_mesh);
    
    cv::Mat gx_sub = cv::Mat::zeros(subImh_, subImw_, CV_64FC1);
    cv::Mat hy_sub = cv::Mat::zeros(subImh_, subImw_, CV_64FC1);

    #pragma omp parallel for
    for(int kf=0;kf<n;kf++){
        cv::Mat u_im_2 = u_im_ - x1_.ptr<double>(0)[kf];
        cv::Mat v_im_2 = v_im_ - y1_.ptr<double>(0)[kf];
        cv::Mat u_im_3,v_im_3;
        cv::pow(u_im_2,2,u_im_3);
        cv::pow(v_im_2,2,v_im_3);
        cv::Mat dist2 = u_im_3+v_im_3;
        cv::Mat dist2_log;
        cv::log(dist2,dist2_log);
        cv::Mat rbf = dist2.mul(dist2_log) * 0.5;
        #pragma omp critical
        {
            gx_sub = gx_sub + wx.ptr<double>(kf)[0]*rbf;
            hy_sub = hy_sub + wy.ptr<double>(kf)[0]*rbf;
        }
        u_im_2.release(),v_im_2.release(),u_im_3.release(),v_im_3.release(),dist2.release(),dist2_log.release(),rbf.release();
    }


    gx_sub= gx_sub+(a.ptr<double>(0)[0]*u_im_+a.ptr<double>(1)[0]*v_im_+a.ptr<double>(2)[0]);
    hy_sub= hy_sub+(b.ptr<double>(0)[0]*u_im_+b.ptr<double>(1)[0]*v_im_+b.ptr<double>(2)[0]);


    cv::resize(gx_sub,gx_sub,cv::Size(imw_,imh_));
    cv::resize(hy_sub,hy_sub,cv::Size(imw_,imh_));
    gx_sub.copyTo(gx(cv::Range(offset_v0_,offset_v1_ + 1),cv::Range(offset_u0_,offset_u1_ + 1)));
    hy_sub.copyTo(hy(cv::Range(offset_v0_,offset_v1_ + 1),cv::Range(offset_u0_,offset_u1_ + 1)));

    // smooth tansition to global transform
    double eta_d0 = 0;
    cv::minMaxLoc(gxn, &min_1, &max_1,NULL, NULL);
    cv::minMaxLoc(hyn, &min_2, &max_2,NULL, NULL);

    cv::Mat gxn_hyn;
    cv::vconcat(gxn,hyn,gxn_hyn);
    gxn_hyn = cv::abs(gxn_hyn);
    double max_gxn_hyn=0;
    cv::minMaxLoc(gxn_hyn, NULL, &max_gxn_hyn,NULL, NULL);
    double eta_d1 = K_smooth*max_gxn_hyn;


    sub_u0_+=min_1;
    sub_u1_+=max_1;
    sub_v0_+=min_2;
    sub_v1_+=max_2;
    cv::Mat dist_horizontal;
    cv::Mat sub_u0_u = sub_u0_ - u_;
    cv::Mat sub_u1_u = u_ - sub_u1_;
    cv::max(sub_u0_u ,sub_u1_u,dist_horizontal);
    cv::Mat dist_vertical;
    cv::Mat sub_v0_v = sub_v0_ - v_;
    cv::Mat sub_v1_v = v_ - sub_v1_;
    cv::max(sub_v0_v,sub_v1_v,dist_vertical);
    cv::Mat dist_sub;
    cv::max(dist_horizontal,dist_vertical,dist_sub);
    cv::Mat zeroMat = cv::Mat::zeros(dist_sub.size(),dist_sub.type());
    cv::max(zeroMat,dist_sub,dist_sub);
    cv::Mat eta = (eta_d1 - dist_sub)/(eta_d1 - eta_d0);
    
    #pragma omp parallel for
    for(int i=0;i<dist_sub.rows;i++){
        for(int j=0;j<dist_sub.cols;j++){
            if(dist_sub.ptr<double>(i)[j]<eta_d0){
                eta.ptr<double>(i)[j]=1.0;
            }
            if(dist_sub.ptr<double>(i)[j]>eta_d1){
                eta.ptr<double>(i)[j]=0.0;
            }
        }
    }
  

    gx = gx.mul(eta);
    hy = hy.mul(eta);
    u_ = u_ - gx;
    v_ = v_ - hy;

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