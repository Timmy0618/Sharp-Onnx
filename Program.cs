using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace OnnxLibrary
{
    class Program
    {
        static void Main(string[] args)
        {
            string modelPath = "best_model.onnx"; // 模型路径
            string yamlPath = "classes.yaml";
            string directoryPath = "./test/E-type"; // 图片文件夹路径
            string device = "cpu"; // 可选择 "cpu" 或 "gpu"

            try
            {
                // 检查目录是否存在
                if (!Directory.Exists(directoryPath))
                {
                    Console.WriteLine($"Directory not found: {directoryPath}");
                    return;
                }

                // 获取所有图片文件路径
                string[] imagePaths = Directory.GetFiles(directoryPath, "*.tif");

                if (imagePaths.Length == 0)
                {
                    Console.WriteLine("No images found in the specified directory.");
                    return;
                }

                // 计时器用于记录预处理时间
                Stopwatch preprocessTimer = new Stopwatch();
                preprocessTimer.Start();

                Console.WriteLine("Preprocessing images...");
                var preprocessedImages = new List<(string fileName, DenseTensor<float> tensor)>();
                foreach (var path in imagePaths)
                {
                    try
                    {
                        var bitmap = new Bitmap(path);
                        var tensor = ImagePreprocessor.PreprocessImage(bitmap, 224, 224);
                        preprocessedImages.Add((Path.GetFileName(path), tensor));
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Failed to preprocess image {path}: {ex.Message}");
                        Console.WriteLine($"Stack Trace: {ex.StackTrace}");
                    }
                }

                preprocessTimer.Stop();
                Console.WriteLine($"Preprocessing Time: {preprocessTimer.Elapsed.TotalSeconds:F2} seconds");

                // 如果没有加载成功的图片，直接退出
                if (preprocessedImages.Count == 0)
                {
                    Console.WriteLine("No valid images preprocessed.");
                    return;
                }

                // 计时器用于记录预测时间
                Stopwatch predictTimer = new Stopwatch();
                predictTimer.Start();

                using (var predictor = new OnnxImagePredictor(modelPath, yamlPath, device))
                {
                    foreach (var (fileName, tensor) in preprocessedImages)
                    {
                        try
                        {
                            var (scores, className) = predictor.PredictFromTensor(tensor);

                            Console.WriteLine($"Image: {fileName}");
                            Console.WriteLine($"Predicted Class: {className}");
                            Console.WriteLine(new string('-', 50)); // 分隔线
                        }
                        catch (Exception ex)
                        {
                            Console.WriteLine($"Failed to process image {fileName}: {ex.Message}");
                        }
                    }
                }

                predictTimer.Stop();
                Console.WriteLine($"Prediction Time: {predictTimer.Elapsed.TotalSeconds:F2} seconds");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
            }
        }
    }
}
