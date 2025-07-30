# OnnxLibrary

这是一个 .NET 应用程序，支持 YOLO v11 图像分类和目标检测模型的统一处理。程序能够自动识别模型类型（分类或检测），并相应地处理图像。

## ✨ 核心特性

- **🤖 智能模型识别**: 自动检测 YOLO v11 分类或检测模型类型
- **🎯 统一处理接口**: 一套代码同时支持分类和检测
- **⚡ 高性能处理**: 支持 CPU/GPU 加速和批量处理
- **🎨 可视化结果**: 检测模型绘制边界框，分类模型显示预测结果
- **🔧 高度可配置**: 灵活的参数配置和阈值调整
- **📊 详细分析**: 自动分析模型输入输出格式

## 🚀 功能对比

| 功能 | 分类模式 | 检测模式 |
|------|----------|----------|
| 输入尺寸 | 自动检测 (通常 224x224) | 自动检测 (通常 640x640) |
| 输出格式 | 类别标签 + 置信度 | 边界框 + 类别 + 置信度 |
| 结果保存 | 控制台显示 | 绘制边界框图像 |
| NMS处理 | 不适用 | 自动应用 |

## 📋 系统要求

- **.NET Framework 4.7.2** 或更高版本
- **ONNX Runtime**: Microsoft.ML.OnnxRuntime.Gpu.Windows (1.18.0)
- **OpenCV Sharp**: 图像处理和绘制 (4.10.0)
- **YAML 解析**: YamlDotNet (16.3.0)
- **GPU 支持** (可选): CUDA 12.x + cuDNN 8.x

---

## 📦 安装步骤

### 1. 克隆仓库

```bash
git clone https://github.com/Timmy0618/Sharp-Onnx-.git
cd Sharp-Onnx-
```

### 2. 安装依赖

```bash
dotnet restore
```

### 3. 配置模型和类别

#### 模型文件
将您的 ONNX 模型文件放置在 `models/` 目录中：
```
models/
  yolov11/
    250318/
      best.onnx  # 您的模型文件
```

#### 类别配置
在项目根目录创建 `classes.yaml` 文件：
```yaml
classes:
  - id: 0
    name: E-type
  - id: 1
    name: other
```

## 🎮 使用方法

### 方法一：直接运行

```bash
# 构建项目
dotnet build

# 运行程序 (自动识别模型类型)
dotnet run

# 运行测试模式
dotnet run test
```

### 方法二：使用批处理脚本 (Windows)

```cmd
# 运行交互式菜单
run.bat
```

### 方法三：程序内配置

修改 `Program.cs` 中的配置：
```csharp
string modelPath = "models/yolov11/250318/best.onnx";
string yamlPath = "classes.yaml";
string directoryPath = "./valid/others";    // 输入图片目录
string outputPath = "./output";             // 输出目录
string device = "cpu";                      // "cpu" 或 "gpu"
```

## 📊 输出示例

### 自动模型识别
```
🔍 模型分析结果:
   类型: Classification
   输入尺寸: 224x224
   类别数量: 2
   描述: YOLO v11 分类模型，输出 2 个类别
```

### 分类结果
```
处理图像: image1.tif
  分类结果: E-type (置信度: 0.876)

处理图像: image2.tif
  分类结果: other (置信度: 0.654)
```

### 检测结果 (当使用检测模型时)
```
处理图像: image1.tif
  检测到 2 个目标:
    - E-type: 0.876 [245, 123, 150, 200]
    - other: 0.654 [400, 250, 100, 120]
    结果已保存到: ./output/result_image1.tif
```

---

## Model Requirements

### For Classification
- Input shape: `[1, 3, 224, 224]`
- Output shape: `[1, num_classes]`
- YOLO v11 classification model

### For Detection
- Input shape: `[1, 3, 640, 640]`
- Output shape: `[1, 84, 8400]` (for 2 classes: 4 bbox coords + 2 class scores)
- YOLO v11 detection model

---

## Troubleshooting

1. **CUDA/GPU Issues**: Ensure you have CUDA 12.x and cuDNN 8.x installed
2. **Model Format**: Ensure your ONNX models are compatible with YOLO v11 format
3. **Class Mapping**: Verify that class IDs in YAML match your model's output classes
4. **Image Format**: Supported formats include `.tif`, `.jpg`, `.png`, etc.
5. **Memory Issues**: For large batches, consider processing images individually

---

## Dependencies

- Microsoft.ML.OnnxRuntime.Gpu.Windows (1.18.0)
- OpenCvSharp4 (4.10.0.20241108)
- OpenCvSharp4.runtime.win (4.10.0.20241108)
- System.Drawing.Common (9.0.0)
- YamlDotNet (16.3.0)
string device = "cpu"; // 可选择 "cpu" 或 "gpu"
```

Run the application:

```bash
dotnet run
```

### 3. Output

The application will output the predicted class name and scores for each class. Example:

```
Predicted Class: Dog
Scores:
Class 0: 0.123
Class 1: 0.872
Class 2: 0.005
```

---

## Error Handling

If an error occurs during execution, it will be displayed in the console. Example:

```
Error: File not found: classes.yaml
```

Ensure the file paths are correct and all required files are available.

---

## Contributing

Feel free to fork the repository and submit pull requests. Issues and feature requests are welcome!

---

## License

This project is licensed under the MIT License.
