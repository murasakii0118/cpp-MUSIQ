#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <algorithm>

#include <opencv2/opencv.hpp>

#include "musiq_preprocess.h"
#include "musiq_inference.h"
#include "musiq_postprocess.h"

namespace fs = std::filesystem;

struct ModelConfig {
    std::string name;
    int num_classes;
};

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n"
              << "Options:\n"
              << "  --models-dir <path>    Directory containing ONNX models (default: ../../onnx_models)\n"
              << "  --images-dir <path>    Directory containing images (default: ../../image)\n"
              << "  --output <path>        Output file path (default: musiq_scores.txt)\n"
              << "  -h, --help             Show this help message\n"
              << "\nExample:\n"
              << "  " << program_name << " --models-dir ./models --images-dir ./images --output results.txt\n"
              << std::endl;
}

fs::path resolvePath(const std::string& path_str) {
    fs::path p(path_str);
    if (p.is_absolute()) {
        return p;
    }
    return fs::absolute(p);
}

void savePreprocessedToFile(const cv::Mat& preprocessed, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    int rows = preprocessed.rows;
    int cols = preprocessed.cols;
    file.write(reinterpret_cast<const char*>(&rows), sizeof(int));
    file.write(reinterpret_cast<const char*>(&cols), sizeof(int));
    file.write(reinterpret_cast<const char*>(preprocessed.data), rows * cols * sizeof(float));
    file.close();
}

void processImage(
    const std::string& image_path,
    const std::vector<std::unique_ptr<musiq::MUSIQInference>>& inferencers,
    const std::vector<ModelConfig>& model_configs,
    std::ofstream& output_file,
    bool save_preprocessed = false
) {
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return;
    }
    
    cv::Mat image_rgb;
    cv::cvtColor(image, image_rgb, cv::COLOR_BGR2RGB);
    
    musiq::PreprocessOptions options;
    options.patch_size = 32;
    options.patch_stride = 32;
    options.hse_grid_size = 10;
    options.longer_side_lengths = {224, 384};
    options.max_seq_len_from_original_res = -1;
    
    musiq::MUSIQPreprocessor preprocessor(options);
    
    cv::Mat preprocessed = preprocessor.preprocess(image_rgb);
    
    if (save_preprocessed) {
        savePreprocessedToFile(preprocessed, "cpp_preprocessed.bin");
    }
    
    std::vector<float> input_data;
    input_data.reserve(preprocessed.rows * preprocessed.cols);
    
    for (int i = 0; i < preprocessed.rows; ++i) {
        const float* row = preprocessed.ptr<float>(i);
        for (int j = 0; j < preprocessed.cols; ++j) {
            input_data.push_back(row[j]);
        }
    }
    
    int batch_size = 1;
    int seq_len = preprocessed.rows;
    int dim = preprocessed.cols;
    
    fs::path p(image_path);
    std::string image_name = p.filename().string();
    
    output_file << std::left << std::setw(30) << image_name;
    std::cout << std::left << std::setw(30) << image_name;
    
    for (size_t i = 0; i < inferencers.size(); ++i) {
        std::vector<float> output = inferencers[i]->infer(input_data, batch_size, seq_len, dim);
        
        float score;
        if (model_configs[i].num_classes > 1) {
            score = musiq::distToMos(output);
        } else {
            score = output[0];
        }
        
        output_file << std::fixed << std::setprecision(6) << std::setw(16) << score;
        std::cout << std::fixed << std::setprecision(6) << std::setw(16) << score;
    }
    
    output_file << std::endl;
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    std::string models_dir = "../../onnx_models";
    std::string image_dir = "../../image";
    std::string output_file_path = "musiq_scores.txt";
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--models-dir" && i + 1 < argc) {
            models_dir = argv[++i];
        } else if (arg == "--images-dir" && i + 1 < argc) {
            image_dir = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_file_path = argv[++i];
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }
    
    fs::path exe_path(argv[0]);
    fs::path base_dir = exe_path.parent_path().parent_path();
    
    fs::path models_path(models_dir);
    fs::path full_models_dir;
    if (models_path.is_absolute()) {
        full_models_dir = models_path;
    } else {
        std::string rel_str = models_dir;
        if (rel_str.substr(0, 2) == "..") {
            fs::path result = base_dir;
            fs::path rel(models_dir);
            for (const auto& part : rel) {
                if (part == "..") {
                    result = result.parent_path();
                } else if (part != ".") {
                    result /= part;
                }
            }
            full_models_dir = result;
        } else {
            full_models_dir = fs::absolute(models_dir);
        }
    }
    
    fs::path images_path(image_dir);
    fs::path full_images_dir;
    if (images_path.is_absolute()) {
        full_images_dir = images_path;
    } else {
        std::string rel_str = image_dir;
        if (rel_str.substr(0, 2) == "..") {
            fs::path result = base_dir;
            fs::path rel(image_dir);
            for (const auto& part : rel) {
                if (part == "..") {
                    result = result.parent_path();
                } else if (part != ".") {
                    result /= part;
                }
            }
            full_images_dir = result;
        } else {
            full_images_dir = fs::absolute(image_dir);
        }
    }
    
    if (!fs::is_directory(full_models_dir)) {
        std::cerr << "Models directory does not exist: " << full_models_dir << std::endl;
        return 1;
    }
    
    if (!fs::is_directory(full_images_dir)) {
        std::cerr << "Images directory does not exist: " << full_images_dir << std::endl;
        return 1;
    }
    
    std::vector<ModelConfig> model_configs = {
        {"musiq-ava", 10},
        {"musiq-paq2piq", 1},
        {"musiq-spaq", 1},
        {"musiq", 1}
    };
    
    std::cout << "Loading models from: " << full_models_dir << std::endl;
    
    std::vector<std::unique_ptr<musiq::MUSIQInference>> inferencers;
    
    for (const auto& config : model_configs) {
        std::string model_path = (full_models_dir / (config.name + ".onnx")).string();
        try {
            auto inferencer = std::make_unique<musiq::MUSIQInference>(model_path, "CPU");
            inferencers.push_back(std::move(inferencer));
            std::cout << "  Loaded: " << config.name << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "  Failed to load model " << config.name << ": " << e.what() << std::endl;
        }
    }
    
    if (inferencers.empty()) {
        std::cerr << "No models loaded. Exiting." << std::endl;
        return 1;
    }
    
    std::ofstream output_file(output_file_path);
    if (!output_file.is_open()) {
        std::cerr << "Failed to open output file: " << output_file_path << std::endl;
        return 1;
    }
    
    output_file << std::left << std::setw(30) << "Image";
    std::cout << std::left << std::setw(30) << "Image";
    
    for (const auto& config : model_configs) {
        output_file << std::setw(16) << config.name;
        std::cout << std::setw(16) << config.name;
    }
    output_file << std::endl;
    std::cout << std::endl;
    
    output_file << std::string(30 + 16 * model_configs.size(), '-') << std::endl;
    std::cout << std::string(30 + 16 * model_configs.size(), '-') << std::endl;
    
    std::vector<std::string> image_files;
    for (const auto& entry : fs::directory_iterator(full_images_dir)) {
        if (entry.path().extension() == ".png" ||
            entry.path().extension() == ".jpg" ||
            entry.path().extension() == ".jpeg") {
            image_files.push_back(entry.path().string());
        }
    }
    
    std::sort(image_files.begin(), image_files.end());
    
    std::cout << "Processing " << image_files.size() << " images from: " << full_images_dir << std::endl;
    
    bool save_first = true;
    for (const auto& image_path : image_files) {
        processImage(image_path, inferencers, model_configs, output_file, save_first);
        save_first = false;
    }
    
    output_file.close();
    
    std::cout << "\nResults saved to: " << output_file_path << std::endl;
    
    return 0;
}
