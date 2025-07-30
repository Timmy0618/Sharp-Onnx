# OnnxLibrary

A .NET library for unified processing of YOLO v11 image classification and object detection models. The library automatically identifies model types (classification or detection) and processes images accordingly.

## ✨ Core Features

- **🤖 Smart Model Recognition**: Automatically detects YOLO v11 classification or detection model types
- **🎯 Unified Processing Interface**: Single codebase supports both classification and detection
- **⚡ High Performance**: Supports CPU/GPU acceleration and batch processing
- **🎨 Visual Results**: Detection models draw bounding boxes, classification models show predictions
- **🔧 Highly Configurable**: Flexible parameter configuration and threshold adjustment
- **📊 Detailed Analysis**: Automatic analysis of model input/output formats
- **📦 Library Support**: Can be packaged as DLL for integration into other projects

## 🚀 Feature Comparison

| Feature | Classification Mode | Detection Mode |
|---------|-------------------|----------------|
| Input Size | Auto-detect (usually 224x224) | Auto-detect (usually 640x640) |
| Output Format | Class label + confidence | Bounding box + class + confidence |
| Result Saving | Console display | Draw bounding box images |
| NMS Processing | Not applicable | Automatically applied |

## 📋 System Requirements

- **.NET Framework 4.7.2** or higher
- **ONNX Runtime**: Microsoft.ML.OnnxRuntime (1.18.0)
- **OpenCV Sharp**: Image processing and drawing (4.10.0)
- **YAML Parser**: YamlDotNet (16.3.0)
- **GPU Support** (optional): CUDA 12.x + cuDNN 8.x

---

## 📦 Installation and Usage

### Method 1: Direct Use (Executable)

#### 1. Clone Repository

```bash
git clone https://github.com/Timmy0618/Sharp-Onnx-.git
cd Sharp-Onnx-
```

#### 2. Install Dependencies

```bash
dotnet restore
```

#### 3. Build and Run

```bash
dotnet build --configuration Release --framework net472
./bin/Release/net472/OnnxLibrary.exe
```

### Method 2: Build as DLL Library

#### 1. Modify Project File

Update `OnnxLibrary.csproj` to generate a library:

```xml
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>net472</TargetFramework>
    <OutputType>Library</OutputType>  <!-- Change from Exe to Library -->
    <LangVersion>7.3</LangVersion>
  </PropertyGroup>
  
  <ItemGroup>
    <PackageReference Include="Microsoft.ML.OnnxRuntime" Version="1.18.0" />
    <PackageReference Include="OpenCvSharp4" Version="4.10.0.20241108" />
    <PackageReference Include="OpenCvSharp4.runtime.win" Version="4.10.0.20241108" />
    <PackageReference Include="System.Drawing.Common" Version="9.0.0" />
    <PackageReference Include="YamlDotNet" Version="16.3.0" />
  </ItemGroup>
</Project>
```

#### 2. Build DLL

```bash
dotnet build --configuration Release --framework net472
```

This will generate `OnnxLibrary.dll` in `bin/Release/net472/` folder.

#### 3. Use DLL in Your Project

##### Add Reference

```xml
<ProjectReference Include="path/to/OnnxLibrary.csproj" />
<!-- OR -->
<Reference Include="OnnxLibrary">
  <HintPath>path/to/OnnxLibrary.dll</HintPath>
</Reference>
```

##### Code Example

```csharp
using OnnxLibrary;
using System;
using System.Collections.Generic;

class Program
{
    static void Main()
    {
        // Configuration
        string modelPath = "models/yolov11/best.onnx";
        string yamlPath = "classes.yaml";
        string imagePath = "test_image.jpg";
        string device = "cpu"; // or "gpu"
        
        // Initialize processor
        using (var processor = new UniversalYoloProcessor(modelPath, yamlPath, device, 0.5f, 0.45f))
        {
            Console.WriteLine($"Model Type: {processor.ModelInfo.Type}");
            Console.WriteLine($"Input Size: {processor.ModelInfo.InputWidth}x{processor.ModelInfo.InputHeight}");
            
            // Process image
            List<Detection> results = processor.ProcessImage(imagePath);
            
            if (processor.ModelInfo.Type == ModelType.Classification)
            {
                // Classification result
                if (results.Count > 0)
                {
                    var result = results[0];
                    Console.WriteLine($"Class: {result.ClassName}, Confidence: {result.Confidence:F3}");
                }
            }
            else if (processor.ModelInfo.Type == ModelType.Detection)
            {
                // Detection results
                Console.WriteLine($"Detected {results.Count} objects:");
                foreach (var detection in results)
                {
                    Console.WriteLine($"- {detection.ClassName}: {detection.Confidence:F3} " +
                                    $"[{detection.X:F0}, {detection.Y:F0}, {detection.Width:F0}, {detection.Height:F0}]");
                }
                
                // Save visualization
                if (results.Count > 0)
                {
                    ImageDrawer.DrawDetections(imagePath, results, "output_result.jpg");
                }
            }
        }
    }
}
```

#### 4. Advanced Usage

##### Batch Processing

```csharp
using (var processor = new UniversalYoloProcessor(modelPath, yamlPath, "cpu"))
{
    string[] imageFiles = Directory.GetFiles("input_folder", "*.jpg");
    
    foreach (string imagePath in imageFiles)
    {
        var results = processor.ProcessImage(imagePath);
        Console.WriteLine($"Processed {Path.GetFileName(imagePath)}: {results.Count} detections");
    }
}
```

##### Custom Thresholds

```csharp
// For detection models
using (var processor = new UniversalYoloProcessor(modelPath, yamlPath, "gpu", 
    confidenceThreshold: 0.3f,  // Lower confidence threshold
    nmsThreshold: 0.5f))        // Higher NMS threshold
{
    var results = processor.ProcessImage(imagePath);
}
```

### Method 3: NuGet Package (Advanced)

#### 1. Create NuGet Package

```bash
dotnet pack --configuration Release --output ./nupkg
```

#### 2. Install Package

```bash
dotnet add package OnnxLibrary --source ./nupkg
```
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

---

## � Configuration Files

### Model Files
Place your ONNX models in the `models/` directory:
```
models/
├── yolov11/
│   └── 250318/
│       └── best.onnx          # Classification or Detection model
└── particle_v2.onnx           # Alternative model
```

### Class Configuration (classes.yaml)
```yaml
classes:
  - id: 0
    name: E-type
  - id: 1
    name: other
```

### Detection Class Configuration (classes_detection.yaml)
```yaml
classes:
  - id: 0
    name: defect
```

---

## �📊 Output Examples

### Automatic Model Recognition
```
🔍 Model Analysis Result:
   Type: Classification
   Input Size: 224x224
   Class Count: 2
   Description: YOLO v11 classification model, outputs 2 classes
```

### Classification Results
```
Processing image: image1.tif
  Classification result: E-type (Confidence: 0.876)

Processing image: image2.tif
  Classification result: other (Confidence: 0.654)
```

### Detection Results (when using detection model)
```
Processing image: image1.tif
  Detected 2 objects:
    - E-type: 0.876 [245, 123, 150, 200]
    - other: 0.654 [400, 250, 100, 120]
    Results saved to: ./output/result_image1.tif
```

---

## 🏗️ API Reference

### Core Classes

#### UniversalYoloProcessor
Main processing class that handles both classification and detection models.

```csharp
public class UniversalYoloProcessor : IDisposable
{
    // Constructor
    public UniversalYoloProcessor(string onnxModelPath, string yamlPath, 
        string device = "cpu", float confidenceThreshold = 0.5f, float nmsThreshold = 0.45f)
    
    // Properties
    public ModelInfo ModelInfo { get; }
    
    // Methods
    public List<Detection> ProcessImage(string imagePath)
}
```

#### Detection
Result data structure for both classification and detection.

```csharp
public class Detection
{
    public float X { get; set; }           // Bounding box X (detection only)
    public float Y { get; set; }           // Bounding box Y (detection only)
    public float Width { get; set; }       // Bounding box Width (detection only)
    public float Height { get; set; }      // Bounding box Height (detection only)
    public float Confidence { get; set; }  // Confidence score
    public int ClassId { get; set; }       // Class ID
    public string ClassName { get; set; }  // Class name
}
```

#### ImageDrawer
Utility class for drawing detection results.

```csharp
public static class ImageDrawer
{
    public static void DrawDetections(string imagePath, List<Detection> detections, string outputPath)
}
```

### Supported Image Formats
- `.bmp` - Windows Bitmap
- `.jpg`, `.jpeg` - JPEG
- `.png` - Portable Network Graphics
- `.tif`, `.tiff` - Tagged Image File Format
- `.gif` - Graphics Interchange Format

---

## Model Requirements

### For Classification
- Input shape: `[1, 3, 224, 224]`
- Output shape: `[1, num_classes]`
- YOLO v11 classification model

### For Detection
- Input shape: `[1, 3, 640, 640]`
- Output shape: `[1, features, 8400]` where features = 4 (bbox coords) + num_classes
- YOLO v11 detection model

---

## 🚨 Troubleshooting

### Common Issues

1. **CUDA/GPU Issues**: Ensure you have CUDA 12.x and cuDNN 8.x installed
2. **Model Format**: Ensure your ONNX models are compatible with YOLO v11 format
3. **Class Mapping**: Verify that class IDs in YAML match your model's output classes
4. **Image Format**: Supported formats include `.tif`, `.jpg`, `.png`, etc.
5. **Memory Issues**: For large batches, consider processing images individually

### Error Messages

#### "Model file not found"
```
❌ Model file not found: models/yolov11/best.onnx
```
**Solution**: Check model file path and ensure the file exists.

#### "Class file not found"
```
❌ Class file not found: classes.yaml
```
**Solution**: Create the classes.yaml file with proper format.

#### "No supported image files found"
```
❌ No supported image files found in ./valid/images
```
**Solution**: Check image directory path and file formats.

---

## 📚 Dependencies

- **Microsoft.ML.OnnxRuntime** (1.18.0) - ONNX model inference
- **OpenCvSharp4** (4.10.0.20241108) - Image processing
- **OpenCvSharp4.runtime.win** (4.10.0.20241108) - OpenCV runtime
- **System.Drawing.Common** (9.0.0) - System drawing capabilities
- **YamlDotNet** (16.3.0) - YAML configuration parsing

---

## 🤝 Contributing

Feel free to fork the repository and submit pull requests. Issues and feature requests are welcome!

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

---

## 📄 License

This project is licensed under the MIT License.
