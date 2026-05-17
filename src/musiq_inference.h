#pragma once

#include <string>
#include <vector>
#include <memory>
#include <stdexcept>

#define ORT_API_MANUAL_INIT
#include <onnxruntime/onnxruntime_c_api.h>

namespace musiq {

class MUSIQInference {
public:
    MUSIQInference(const std::string& model_path, 
                   const std::string& provider = "CPU");
    ~MUSIQInference();
    
    std::vector<float> infer(const std::vector<float>& input_data,
                            int batch_size,
                            int seq_len,
                            int dim);
    
private:
    OrtEnv* env_;
    OrtSessionOptions* session_options_;
    OrtSession* session_;
    OrtMemoryInfo* memory_info_;
    OrtAllocator* allocator_;
    std::string provider_;
    
    std::vector<char*> input_names_;
    std::vector<char*> output_names_;
};

} 
