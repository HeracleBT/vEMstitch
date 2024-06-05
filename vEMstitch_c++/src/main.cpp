#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cstring>
#include <dirent.h>
#include <cstdlib> 
#include <list>
#include <fstream>
#include <chrono> 
#include"stitching.h"
#include"Utils.h"
#include"rigidtransform.h"
#include"refinement.h"
#include"elastictransform.h"

std::vector<std::string> listFiles(const std::string& path) {
    std::vector<std::string> files;
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(path.c_str())) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            if (ent->d_type == DT_REG) { 
                files.push_back(ent->d_name);
            }
        }
        closedir(dir);
    } else {
        std::cerr << "Error opening directory" << std::endl;
    }
    return files;
}

void showProgressBar(int progress, int total) {
    const int barWidth = 50;
    double fraction = static_cast<double>(progress) / total;
    int barLength = static_cast<int>(fraction * barWidth);

    std::cout << "[";
    for (int i = 0; i < barLength; ++i) {
        std::cout << "=";
    }
    for (int i = barLength; i < barWidth; ++i) {
        std::cout << " ";
    }
    std::cout << "] " << int(fraction * 100.0) << "%\r";
    std::cout.flush();
}


int main(int argc, char* argv[]) {
    std::string data_path;
    std::string store_path;
    std::string log_path;
    int pattern = 0;
    double overlap = 0.1;
    bool refine_flag = false;

    if (argc != 7) {
        std::cerr << "Usage: " << argv[0] << " <data_path> <store_path> <pattern> <overlapping_rate> <log_path> <refine_flag>" << std::endl;
        return 1;
    }
    data_path = argv[1];
    store_path = argv[2];
    pattern = std::atoi(argv[3]);
    overlap = std::atof(argv[4]);
    log_path = argv[5];
    refine_flag = (std::string(argv[6]) == "true");
    std::cout << "Data path: " << data_path << std::endl;
    std::cout << "Store path: " << store_path << std::endl;
    std::cout << "Pattern: " << pattern << std::endl;
    std::cout << "Log: " << log_path << std::endl;
    std::cout << "Refine flag: " << std::boolalpha << refine_flag << std::endl;

    //begin...
    std::vector<std::string> image_path = listFiles(data_path);
    std::vector<std::string> image_list;
    for (auto& filename : image_path) {
        std::string first = filename.substr(0, filename.find_first_of("."));
        first = first.substr(0, first.find_first_of("_"));
        image_list.push_back(first);
    }
    std::sort(image_list.begin(), image_list.end(), [](const std::string& a, const std::string& b) {
        return std::stoi(a) < std::stoi(b);
    });
    image_list.erase(std::unique(image_list.begin(), image_list.end()), image_list.end());
    if (std::system(("mkdir -p " + store_path).c_str()) != 0) {
        std::cout << "Error creating directory" << std::endl;
    }

    omp_set_num_threads(6);

    int progress = 0;
    int total = image_list.size();
    double total_time = 0.0; 
    std::vector<double> task_times; 

    for (const auto& top_num : image_list) {
        try {
            auto start_time = std::chrono::steady_clock::now(); 
            if (pattern == 3) {
                three_stitching(data_path, store_path, top_num, refine_flag);
            } else if (pattern == 2) {
                std::cout << "current section num: " << top_num << std::endl;
                // two_stitching(data_path, store_path, top_num, refine_flag);
                two_stitching(data_path, store_path, top_num, 0.1, refine_flag);
                // two_stitching_seq(data_path, store_path, top_num, refine_flag);
            }
            auto end_time = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed_seconds = end_time - start_time;
            double task_time = elapsed_seconds.count(); 
            task_times.push_back(task_time); 
            std::cout << "execution time: " << task_time << std::endl;
            total_time += task_time;
            
        } catch (const std::exception& e) {
            std::cout << "Exception occurred: " << e.what() << std::endl;
            cv::Mat final_res(1000, 1000, CV_8UC3, cv::Scalar(0, 0, 0));
            cv::imwrite(store_path + "/" + top_num + "-res.bmp", final_res);
        }

        ++progress;
        showProgressBar(progress, total);
    }
    std::cout << std::endl;

    double average_time = total_time / total;
    std::ofstream outfile(log_path, std::ios::app);
    if (outfile.is_open()) {
        outfile << data_path << " Average time: " << average_time << " seconds\n";
        outfile.close();
    } else {
        std::cerr << "Unable to open file for writing" << std::endl;
    }




}

