# OnnxLibrary

This project is a .NET application that utilizes an ONNX model to classify images. It processes an image, performs inference using ONNX Runtime, and outputs the predicted class name along with scores for each class, using a YAML configuration file for class mapping.

## Prerequisites

- **.NET SDK** (version 5.0 or later, recommended: [Download here](https://dotnet.microsoft.com/download))
- **ONNX Runtime NuGet Package**: Ensure ONNX Runtime is installed as a dependency.
- **YAML configuration file**: A `.yaml` file mapping class indices to class names.
- **ONNX model file**: The ONNX model used for inference.
- **ONNX CUDA compatible**: [See here](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
- An image file for prediction.
- **ONNX Runtime Version Compatibility**: Currently using **ONNX Runtime 1.18.0**, which requires:
  - **CUDA 12.x**
  - **cuDNN 8.x**

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/OnnxLibrary.git
cd OnnxLibrary
```

### 2. Install dependencies

Run the following command to restore NuGet packages:

```bash
dotnet restore
```

### 3. Configuration

- Place your ONNX model file (e.g., `best_model.onnx`) in the project root directory.
- Add your YAML configuration file (e.g., `classes.yaml`) in the project root directory. The YAML structure should be similar to:

```yaml
classes:
  - id: 0
    name: Cat
  - id: 1
    name: Dog
  - id: 2
    name: Bird
```

- Ensure the image file you want to predict (e.g., `2025-01-03-14-51-29-1321.tif`) is accessible.

---

## Usage

### 1. Build the project

```bash
dotnet build
```

### 2. Run the application

Provide the paths to the ONNX model, YAML file, and the image for prediction by modifying the code or passing arguments. Ensure the file paths in `Program.cs` are accurate:

```csharp
string modelPath = "your_onnx_path"; // Path to ONNX model
string yamlPath = "your_classes_yaml_path";     // Path to YAML file
string imagePath = "your_image_path"; // Path to the image file
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
