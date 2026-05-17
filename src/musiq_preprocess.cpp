#include "musiq_preprocess.h"
#include <cmath>
#include <algorithm>

namespace musiq {

MUSIQPreprocessor::MUSIQPreprocessor(const PreprocessOptions& options)
    : options_(options) {
}

cv::Mat MUSIQPreprocessor::extractImagePatches(const cv::Mat& image, int kernel, int stride) {
    int h = image.rows;
    int w = image.cols;
    
    // Calculate padding
    int h2 = static_cast<int>(std::ceil(static_cast<double>(h) / stride));
    int w2 = static_cast<int>(std::ceil(static_cast<double>(w) / stride));
    int pad_row = (h2 - 1) * stride + kernel - h;
    int pad_col = (w2 - 1) * stride + kernel - w;
    int pad_top = pad_row / 2;
    int pad_bottom = pad_row - pad_top;
    int pad_left = pad_col / 2;
    int pad_right = pad_col - pad_left;
    
    cv::Mat padded;
    cv::copyMakeBorder(image, padded, pad_top, pad_bottom, pad_left, pad_right, cv::BORDER_CONSTANT, cv::Scalar(0));
    
    int num_patches = h2 * w2;
    int patch_dim = kernel * kernel * 3;
    cv::Mat patches(num_patches, patch_dim, CV_32F);
    
    for (int y = 0; y < h2; ++y) {
        for (int x = 0; x < w2; ++x) {
            int patch_idx = y * w2 + x;
            
            // Extract patch from padded image
            int start_y = y * stride;
            int start_x = x * stride;
            cv::Rect roi(start_x, start_y, kernel, kernel);
            cv::Mat patch = padded(roi);
            
            // Flatten patch in the same order as F.unfold
            float* patch_ptr = patches.ptr<float>(patch_idx);
            int idx = 0;
            
            // First channel 0, then 1, then 2
            for (int c = 0; c < 3; ++c) {
                for (int py = 0; py < kernel; ++py) {
                    for (int px = 0; px < kernel; ++px) {
                        patch_ptr[idx] = patch.at<cv::Vec3f>(py, px)[c];
                        idx++;
                    }
                }
            }
        }
    }
    
    return patches;
}

cv::Mat MUSIQPreprocessor::resizePreserveAspectRatio(const cv::Mat& image, int longer_side_length, int& rh, int& rw) {
    int h = image.rows;
    int w = image.cols;
    
    double ratio = static_cast<double>(longer_side_length) / std::max(h, w);
    rh = static_cast<int>(std::round(h * ratio));
    rw = static_cast<int>(std::round(w * ratio));
    
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(rw, rh), 0, 0, cv::INTER_CUBIC);
    
    return resized;
}

cv::Mat MUSIQPreprocessor::getHashedSpatialPosEmbIndex(int grid_size, int count_h, int count_w) {
    cv::Mat pos_emb_hash(count_h, count_w, CV_32F);
    
    for (int y = 0; y < count_h; ++y) {
        for (int x = 0; x < count_w; ++x) {
            float src_y = static_cast<float>(y) / count_h * grid_size;
            float src_x = static_cast<float>(x) / count_w * grid_size;
            int y0 = static_cast<int>(std::floor(src_y));
            int x0 = static_cast<int>(std::floor(src_x));
            y0 = std::max(0, std::min(grid_size - 1, y0));
            x0 = std::max(0, std::min(grid_size - 1, x0));
            pos_emb_hash.at<float>(y, x) = static_cast<float>(y0 * grid_size + x0);
        }
    }
    
    // Flatten to 1 row, count_h*count_w cols
    cv::Mat result(1, count_h * count_w, CV_32F);
    for (int y = 0; y < count_h; ++y) {
        for (int x = 0; x < count_w; ++x) {
            result.at<float>(0, y * count_w + x) = pos_emb_hash.at<float>(y, x);
        }
    }
    return result;
}

cv::Mat MUSIQPreprocessor::padOrCutToMaxSeqLen(const cv::Mat& x, int max_seq_len) {
    int num_patches = x.rows;
    
    if (num_patches >= max_seq_len) {
        return x.rowRange(0, max_seq_len).clone();
    }
    
    cv::Mat result(max_seq_len, x.cols, CV_32F, cv::Scalar(0));
    x.copyTo(result.rowRange(0, num_patches));
    
    return result;
}

cv::Mat MUSIQPreprocessor::extractPatchesAndPositions(const cv::Mat& image, int scale_id, int max_seq_len) {
    int h = image.rows;
    int w = image.cols;
    
    cv::Mat patches = extractImagePatches(image, options_.patch_size, options_.patch_stride);
    
    int count_h = static_cast<int>(std::ceil(static_cast<double>(h) / options_.patch_stride));
    int count_w = static_cast<int>(std::ceil(static_cast<double>(w) / options_.patch_stride));
    
    cv::Mat spatial_p = getHashedSpatialPosEmbIndex(options_.hse_grid_size, count_h, count_w);
    
    int num_patches = patches.rows;
    int patch_dim = patches.cols;
    
    int out_dim = patch_dim + 3;
    cv::Mat result(num_patches, out_dim, CV_32F);
    
    for (int i = 0; i < num_patches; ++i) {
        float* result_row = result.ptr<float>(i);
        const float* patch_row = patches.ptr<float>(i);
        
        std::copy(patch_row, patch_row + patch_dim, result_row);
        
        result_row[patch_dim] = spatial_p.at<float>(0, i);
        result_row[patch_dim + 1] = static_cast<float>(scale_id);
        result_row[patch_dim + 2] = 1.0f;
    }
    
    if (max_seq_len >= 0) {
        result = padOrCutToMaxSeqLen(result, max_seq_len);
    }
    
    return result;
}

cv::Mat MUSIQPreprocessor::preprocess(const cv::Mat& image) {
    // image is uint8 HWC, convert to float32 HWC [0,1]
    cv::Mat img_float;
    image.convertTo(img_float, CV_32F, 1.0 / 255.0);
    
    // Normalize to [-1,1] as in PyTorch code
    img_float = (img_float - 0.5f) * 2.0f;
    
    std::vector<int> sorted_lengths = options_.longer_side_lengths;
    std::sort(sorted_lengths.begin(), sorted_lengths.end());
    
    std::vector<cv::Mat> outputs;
    
    for (size_t scale_id = 0; scale_id < sorted_lengths.size(); ++scale_id) {
        int longer_size = sorted_lengths[scale_id];
        int max_seq_len = static_cast<int>(std::ceil(static_cast<double>(longer_size) / options_.patch_stride));
        max_seq_len = max_seq_len * max_seq_len;
        
        int rh, rw;
        cv::Mat resized = resizePreserveAspectRatio(img_float, longer_size, rh, rw);
        cv::Mat out = extractPatchesAndPositions(resized, scale_id, max_seq_len);
        outputs.push_back(out);
    }
    
    if (options_.max_seq_len_from_original_res >= 0) {
        cv::Mat out = extractPatchesAndPositions(
            img_float, 
            sorted_lengths.size(), 
            options_.max_seq_len_from_original_res
        );
        outputs.push_back(out);
    }
    
    // Concatenate vertically (same as PyTorch)
    cv::Mat result;
    cv::vconcat(outputs, result);
    
    return result;
}

}
