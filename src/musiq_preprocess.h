#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

namespace musiq {

struct PreprocessOptions {
    int patch_size = 32;
    int patch_stride = 32;
    int hse_grid_size = 10;
    std::vector<int> longer_side_lengths = {224, 384};
    int max_seq_len_from_original_res = -1;
};

class MUSIQPreprocessor {
public:
    MUSIQPreprocessor(const PreprocessOptions& options = PreprocessOptions());
    
    cv::Mat preprocess(const cv::Mat& image);
    
private:
    PreprocessOptions options_;
    
    cv::Mat extractImagePatches(const cv::Mat& image, int kernel, int stride);
    cv::Mat resizePreserveAspectRatio(const cv::Mat& image, int longer_side_length, int& rh, int& rw);
    cv::Mat getHashedSpatialPosEmbIndex(int grid_size, int count_h, int count_w);
    cv::Mat padOrCutToMaxSeqLen(const cv::Mat& x, int max_seq_len);
    cv::Mat extractPatchesAndPositions(const cv::Mat& image, int scale_id, int max_seq_len);
};

} 
