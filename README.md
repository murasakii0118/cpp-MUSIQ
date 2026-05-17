# MUSIQ C++ Inference

该项目部分代码由AI辅助生成。

This project was partially generated with AI assistance.

部分参考自 IQA-PyTorch 项目。

Partially referenced from the IQA-PyTorch project.

---

## 项目简介 / Project Description

MUSIQ模型的C++推理实现，使用ONNX Runtime进行模型推理。

C++ inference implementation for the MUSIQ model, using ONNX Runtime for model inference.

## 项目结构 / Project Structure

```
cpp-musiq/
├── src/
│   ├── CMakeLists.txt      # CMake构建配置 / CMake build configuration
│   ├── main.cpp            # 主程序入口 / Main program entry
│   ├── musiq_inference.cpp # ONNX Runtime推理 / ONNX Runtime inference
│   ├── musiq_inference.h
│   ├── musiq_preprocess.cpp # 图像预处理 / Image preprocessing
│   ├── musiq_preprocess.h
│   ├── musiq_postprocess.cpp # 后处理 / Post-processing
│   └── musiq_postprocess.h
├── .gitignore              # Git忽略文件配置 / Git ignore configuration
└── README.md
```

## 依赖项 / Dependencies

本项目依赖以下库（需要提前安装）：

This project depends on the following libraries (need to be installed in advance):

1. **ONNX Runtime** (MinGW预编译版本 / MinGW precompiled version)
   - 下载地址 / Download: https://github.com/microsoft/onnxruntime/releases
   - 或使用预编译版本 / Or use precompiled version: onnx-mingw64

2. **OpenCV** (MinGW预编译版本 / MinGW precompiled version)
   - 下载地址 / Download: https://github.com/opencv/opencv/releases
   - 或使用预编译版本 / Or use precompiled version: opencv4120-extra

3. **MUSIQ ONNX模型 / MUSIQ ONNX Models**
   - 需要从Python版本的MUSIQ模型导出为ONNX格式
   - Need to export from Python version of MUSIQ model to ONNX format
   - 参考导出脚本 / Reference export script: export_onnx_dynamic.py

4. **CMake** (>= 3.10)
   - 下载地址 / Download: https://cmake.org/download/

5. **MinGW-w64** (GCC编译器 / GCC compiler)
   - 下载地址 / Download: https://www.mingw-w64.org/

## 构建步骤 / Build Steps

### 1. 准备依赖项 / Prepare Dependencies

确保以下目录存在（相对于项目根目录）：

Make sure the following directories exist (relative to project root):

```
../onnx-mingw64/       # ONNX Runtime头文件和库 / ONNX Runtime headers and libraries
../opencv4120-extra/   # OpenCV头文件和库 / OpenCV headers and libraries
../onnx_models/        # MUSIQ ONNX模型文件 / MUSIQ ONNX model files
../image/              # 测试图像 / Test images
```

### 2. 配置CMake / Configure CMake

```bash
cd cpp-musiq
mkdir build
cd build
cmake -G "MinGW Makefiles" ^
      -DCMAKE_C_COMPILER=gcc ^
      -DCMAKE_CXX_COMPILER=g++ ^
      -DONNXRUNTIME_DIR=../../onnx-mingw64 ^
      -DOpenCV_DIR=../../opencv4120-extra ^
      ..
```

### 3. 编译 / Compile

```bash
mingw32-make
```

### 4. 运行 / Run

编译完成后，可执行文件位于 `build/bin/musiq_inference.exe`

After compilation, the executable is located at `build/bin/musiq_inference.exe`

```bash
# 默认参数运行（从默认目录加载模型和图像）
# Run with default parameters (load models and images from default directories)
./bin/musiq_inference.exe

# 指定模型目录、图像目录和输出文件
# Specify models directory, images directory and output file
./bin/musiq_inference.exe --models-dir ./models --images-dir ./images --output results.txt
```

## 命令行参数 / Command Line Arguments

| 参数 / Argument | 描述 / Description | 默认值 / Default |
|----------------|-------------------|-----------------|
| `--models-dir <path>` | ONNX模型目录 / ONNX models directory | `../../onnx_models` |
| `--images-dir <path>` | 图像目录 / Images directory | `../../image` |
| `--output <path>` | 输出文件路径 / Output file path | `musiq_scores.txt` |
| `-h, --help` | 显示帮助信息 / Show help message | - |

## 支持的模型 / Supported Models

本项目支持以下4个MUSIQ模型：

This project supports the following 4 MUSIQ models:

- `musiq-ava` - AVA数据集模型（输出为分布，需要转换）/ AVA dataset model (output is distribution, needs conversion)
- `musiq-paq2piq` - PAQ2PIQ数据集模型 / PAQ2PIQ dataset model
- `musiq-spaq` - SPAQ数据集模型 / SPAQ dataset model
- `musiq` - Koniq10k数据集模型 / Koniq10k dataset model

## 输入要求 / Input Requirements

- 图像格式：PNG、JPG、JPEG / Image formats: PNG, JPG, JPEG
- 图像会被预处理为多尺度补丁表示 / Images are preprocessed into multi-scale patch representations
- 预处理包括：归一化、多尺度裁剪、补丁提取、位置嵌入生成
  / Preprocessing includes: normalization, multi-scale cropping, patch extraction, positional embedding generation

## 输出 / Output

程序会输出每个图像的4个质量分数，格式为：

The program outputs 4 quality scores for each image, in the following format:

```
Image                         musiq-ava       musiq-paq2piq   musiq-spaq      musiq
----------------------------------------------------------------------------------------------
0_83.050000.png               3.971556        72.844574       61.253693       60.093494
...
```

## 常见问题 / FAQ

### Q: CMake找不到MinGW编译器 / CMake cannot find MinGW compiler
A: 确保MinGW已正确安装并添加到PATH环境变量。在cmake命令中显式指定编译器：
   / Make sure MinGW is properly installed and added to PATH environment variable. Specify compiler explicitly in cmake command:
```bash
cmake -G "MinGW Makefiles" -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ ..
```

### Q: ONNX Runtime加载失败 / ONNX Runtime loading failed
A: 确保onnxruntime.dll在可执行文件同一目录或系统PATH中。
   / Make sure onnxruntime.dll is in the same directory as the executable or in system PATH.

### Q: OpenCV加载失败 / OpenCV loading failed
A: 确保OpenCV的DLL文件在可执行文件同一目录或系统PATH中。
   / Make sure OpenCV DLL files are in the same directory as the executable or in system PATH.

### Q: 模型或图像目录找不到 / Models or images directory not found
A: 检查路径是否正确，或使用绝对路径指定目录。
   / Check if the path is correct, or use absolute path to specify directories.

## License

本项目仅供研究学习使用。

This project is for research and study purposes only.
