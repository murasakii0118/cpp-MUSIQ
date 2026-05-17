#include "musiq_inference.h"
#include <iostream>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#endif

namespace musiq {

static const OrtApi* g_ort = nullptr;

static void OrtInit() {
    if (g_ort) return;
    
    // Try different API versions from 23 down to 1
    for (int version = 23; version >= 1; --version) {
        g_ort = OrtGetApiBase()->GetApi(version);
        if (g_ort != nullptr) {
            std::cout << "Using ONNX Runtime API version: " << version << std::endl;
            break;
        }
    }
    
    if (!g_ort) {
        throw std::runtime_error("Failed to get ONNX Runtime API");
    }
}

MUSIQInference::MUSIQInference(const std::string& model_path, const std::string& provider)
    : env_(nullptr), session_options_(nullptr), session_(nullptr), 
      memory_info_(nullptr), allocator_(nullptr), provider_(provider) {
    
    OrtInit();
    
    std::cout << "Loading model: " << model_path << std::endl;
    
    OrtStatus* status = g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "MUSIQ", &env_);
    if (status != nullptr) {
        const char* msg = g_ort->GetErrorMessage(status);
        g_ort->ReleaseStatus(status);
        throw std::runtime_error("Failed to create env: " + std::string(msg));
    }
    
    status = g_ort->CreateSessionOptions(&session_options_);
    if (status != nullptr) {
        const char* msg = g_ort->GetErrorMessage(status);
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseEnv(env_);
        throw std::runtime_error("Failed to create session options: " + std::string(msg));
    }
    
    g_ort->SetIntraOpNumThreads(session_options_, 4);
    g_ort->SetSessionGraphOptimizationLevel(session_options_, ORT_ENABLE_EXTENDED);
    
    status = g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info_);
    if (status != nullptr) {
        const char* msg = g_ort->GetErrorMessage(status);
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseSessionOptions(session_options_);
        g_ort->ReleaseEnv(env_);
        throw std::runtime_error("Failed to create memory info: " + std::string(msg));
    }
    
#ifdef _WIN32
    int len = MultiByteToWideChar(CP_UTF8, 0, model_path.c_str(), -1, nullptr, 0);
    std::wstring model_path_w(len, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, model_path.c_str(), -1, &model_path_w[0], len);
    status = g_ort->CreateSession(env_, model_path_w.c_str(), session_options_, &session_);
#else
    status = g_ort->CreateSession(env_, model_path.c_str(), session_options_, &session_);
#endif
    
    if (status != nullptr) {
        const char* msg = g_ort->GetErrorMessage(status);
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseMemoryInfo(memory_info_);
        g_ort->ReleaseSessionOptions(session_options_);
        g_ort->ReleaseEnv(env_);
        throw std::runtime_error("Failed to create session: " + std::string(msg));
    }
    
    size_t num_inputs = 0;
    status = g_ort->SessionGetInputCount(session_, &num_inputs);
    if (status != nullptr) {
        const char* msg = g_ort->GetErrorMessage(status);
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseSession(session_);
        g_ort->ReleaseMemoryInfo(memory_info_);
        g_ort->ReleaseSessionOptions(session_options_);
        g_ort->ReleaseEnv(env_);
        throw std::runtime_error("Failed to get input count: " + std::string(msg));
    }
    
    status = g_ort->GetAllocatorWithDefaultOptions(&allocator_);
    if (status != nullptr) {
        const char* msg = g_ort->GetErrorMessage(status);
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseSession(session_);
        g_ort->ReleaseMemoryInfo(memory_info_);
        g_ort->ReleaseSessionOptions(session_options_);
        g_ort->ReleaseEnv(env_);
        throw std::runtime_error("Failed to get allocator: " + std::string(msg));
    }
    
    input_names_.resize(num_inputs);
    for (size_t i = 0; i < num_inputs; ++i) {
        char* name = nullptr;
        status = g_ort->SessionGetInputName(session_, i, allocator_, &name);
        if (status != nullptr) {
            const char* msg = g_ort->GetErrorMessage(status);
            g_ort->ReleaseStatus(status);
            g_ort->ReleaseAllocator(allocator_);
            g_ort->ReleaseSession(session_);
            g_ort->ReleaseMemoryInfo(memory_info_);
            g_ort->ReleaseSessionOptions(session_options_);
            g_ort->ReleaseEnv(env_);
            throw std::runtime_error("Failed to get input name: " + std::string(msg));
        }
        input_names_[i] = name;
    }
    
    size_t num_outputs = 0;
    status = g_ort->SessionGetOutputCount(session_, &num_outputs);
    if (status != nullptr) {
        const char* msg = g_ort->GetErrorMessage(status);
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseAllocator(allocator_);
        g_ort->ReleaseSession(session_);
        g_ort->ReleaseMemoryInfo(memory_info_);
        g_ort->ReleaseSessionOptions(session_options_);
        g_ort->ReleaseEnv(env_);
        throw std::runtime_error("Failed to get output count: " + std::string(msg));
    }
    
    output_names_.resize(num_outputs);
    for (size_t i = 0; i < num_outputs; ++i) {
        char* name = nullptr;
        status = g_ort->SessionGetOutputName(session_, i, allocator_, &name);
        if (status != nullptr) {
            const char* msg = g_ort->GetErrorMessage(status);
            g_ort->ReleaseStatus(status);
            g_ort->ReleaseAllocator(allocator_);
            g_ort->ReleaseSession(session_);
            g_ort->ReleaseMemoryInfo(memory_info_);
            g_ort->ReleaseSessionOptions(session_options_);
            g_ort->ReleaseEnv(env_);
            throw std::runtime_error("Failed to get output name: " + std::string(msg));
        }
        output_names_[i] = name;
    }
    
    std::cout << "Model loaded successfully" << std::endl;
    std::cout << "Input names: ";
    for (const auto& name : input_names_) {
        std::cout << name << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Output names: ";
    for (const auto& name : output_names_) {
        std::cout << name << " ";
    }
    std::cout << std::endl;
}

MUSIQInference::~MUSIQInference() {
    if (allocator_) {
        for (auto& name : input_names_) {
            if (name) g_ort->AllocatorFree(allocator_, name);
        }
        for (auto& name : output_names_) {
            if (name) g_ort->AllocatorFree(allocator_, name);
        }
        g_ort->ReleaseAllocator(allocator_);
    }
    
    if (memory_info_) g_ort->ReleaseMemoryInfo(memory_info_);
    if (session_) g_ort->ReleaseSession(session_);
    if (session_options_) g_ort->ReleaseSessionOptions(session_options_);
    if (env_) g_ort->ReleaseEnv(env_);
}

std::vector<float> MUSIQInference::infer(const std::vector<float>& input_data,
                                         int batch_size,
                                         int seq_len,
                                         int dim) {
    std::vector<int64_t> input_shape = {batch_size, seq_len, dim};
    
    OrtValue* input_tensor = nullptr;
    OrtStatus* status = g_ort->CreateTensorWithDataAsOrtValue(
        memory_info_,
        const_cast<float*>(input_data.data()),
        input_data.size() * sizeof(float),
        input_shape.data(),
        input_shape.size(),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &input_tensor
    );
    
    if (status != nullptr) {
        const char* msg = g_ort->GetErrorMessage(status);
        g_ort->ReleaseStatus(status);
        throw std::runtime_error("Failed to create input tensor: " + std::string(msg));
    }
    
    const char* input_names_ptr[] = {input_names_[0]};
    const char* output_names_ptr[] = {output_names_[0]};
    
    OrtValue* output_tensor = nullptr;
    status = g_ort->Run(
        session_, nullptr, input_names_ptr, (const OrtValue* const*)&input_tensor, 1, 
        output_names_ptr, 1, &output_tensor
    );
    
    if (status != nullptr) {
        const char* msg = g_ort->GetErrorMessage(status);
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseValue(input_tensor);
        throw std::runtime_error("Failed to run inference: " + std::string(msg));
    }
    
    float* output_data = nullptr;
    status = g_ort->GetTensorMutableData(output_tensor, (void**)&output_data);
    if (status != nullptr) {
        const char* msg = g_ort->GetErrorMessage(status);
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseValue(output_tensor);
        g_ort->ReleaseValue(input_tensor);
        throw std::runtime_error("Failed to get tensor data: " + std::string(msg));
    }
    
    OrtTensorTypeAndShapeInfo* output_info = nullptr;
    status = g_ort->GetTensorTypeAndShape(output_tensor, &output_info);
    if (status != nullptr) {
        const char* msg = g_ort->GetErrorMessage(status);
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseValue(output_tensor);
        g_ort->ReleaseValue(input_tensor);
        throw std::runtime_error("Failed to get tensor shape: " + std::string(msg));
    }
    
    size_t dims_count = 0;
    status = g_ort->GetDimensionsCount(output_info, &dims_count);
    if (status != nullptr) {
        const char* msg = g_ort->GetErrorMessage(status);
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseTensorTypeAndShapeInfo(output_info);
        g_ort->ReleaseValue(output_tensor);
        g_ort->ReleaseValue(input_tensor);
        throw std::runtime_error("Failed to get dimensions count: " + std::string(msg));
    }
    
    std::vector<int64_t> output_shape(dims_count);
    status = g_ort->GetDimensions(output_info, output_shape.data(), dims_count);
    if (status != nullptr) {
        const char* msg = g_ort->GetErrorMessage(status);
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseTensorTypeAndShapeInfo(output_info);
        g_ort->ReleaseValue(output_tensor);
        g_ort->ReleaseValue(input_tensor);
        throw std::runtime_error("Failed to get dimensions: " + std::string(msg));
    }
    
    size_t output_size = 1;
    for (auto d : output_shape) {
        output_size *= d;
    }
    
    std::vector<float> result(output_data, output_data + output_size);
    
    g_ort->ReleaseTensorTypeAndShapeInfo(output_info);
    g_ort->ReleaseValue(output_tensor);
    g_ort->ReleaseValue(input_tensor);
    
    return result;
}

} 
