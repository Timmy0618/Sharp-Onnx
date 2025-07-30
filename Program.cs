using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace OnnxLibrary
{
    class Program
    {
        // 支持的图片格式
        private static readonly string[] SupportedImageExtensions = 
        {
            "*.bmp", "*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.gif"
        };

        static void Main(string[] args)
        {
            // 检查是否为测试模式
            if (args.Length > 0 && args[0].ToLower() == "test")
            {
                DetectionTest.PrintSystemInfo();
                DetectionTest.RunTest();
                return;
            }

            // 配置参数
            string modelPath = "models/yolov11/250318/best.onnx";
            string yamlPath = "classes.yaml";  // 检测模型使用检测类别文件
            string directoryPath = "./valid/E-type"; // 图片文件夹路径
            string outputPath = "./output"; // 检测结果输出路径
            string device = "cpu"; // 可选择 "cpu" 或 "gpu"

            Console.WriteLine("=== ONNX Library - Universal YOLO v11 处理器 ===");
            Console.WriteLine($"模型路径: {modelPath}");
            Console.WriteLine($"图片目录: {directoryPath}");
            Console.WriteLine($"输出目录: {outputPath}");
            Console.WriteLine($"支持格式: {string.Join(", ", SupportedImageExtensions.Select(ext => ext.Replace("*", "")))}");
            Console.WriteLine();

            try
            {
                // 检查目录是否存在
                if (!Directory.Exists(directoryPath))
                {
                    Console.WriteLine($"Directory not found: {directoryPath}");
                    return;
                }

                // 获取所有支持格式的图片文件路径
                string[] imagePaths = GetSupportedImageFiles(directoryPath);

                if (imagePaths.Length == 0)
                {
                    Console.WriteLine("No supported image files found in the specified directory.");
                    Console.WriteLine($"Supported formats: {string.Join(", ", SupportedImageExtensions.Select(ext => ext.Replace("*", "")))}");
                    return;
                }

                Console.WriteLine($"找到 {imagePaths.Length} 张图片");

                RunUniversalProcessing(imagePaths, modelPath, yamlPath, device, outputPath);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
                Console.WriteLine($"Stack Trace: {ex.StackTrace}");
            }
        }

        /// <summary>
        /// 获取目录中所有支持格式的图片文件
        /// </summary>
        /// <param name="directoryPath">目录路径</param>
        /// <returns>图片文件路径数组</returns>
        static string[] GetSupportedImageFiles(string directoryPath)
        {
            var imageFiles = new List<string>();
            var formatCount = new Dictionary<string, int>();
            
            foreach (string extension in SupportedImageExtensions)
            {
                var files = Directory.GetFiles(directoryPath, extension, SearchOption.TopDirectoryOnly);
                imageFiles.AddRange(files);
                
                if (files.Length > 0)
                {
                    string ext = extension.Replace("*", "").ToUpper();
                    formatCount[ext] = files.Length;
                }
            }
            
            // 显示找到的格式统计
            if (formatCount.Count > 0)
            {
                Console.WriteLine("找到的图片格式统计:");
                foreach (var kvp in formatCount.OrderBy(x => x.Key))
                {
                    Console.WriteLine($"  {kvp.Key}: {kvp.Value} 张");
                }
            }
            
            // 按文件名排序
            return imageFiles.OrderBy(Path.GetFileName).ToArray();
        }

        static void RunUniversalProcessing(string[] imagePaths, string modelPath, string yamlPath, string device, string outputPath)
        {
            // 创建输出目录
            Directory.CreateDirectory(outputPath);

            // 计时器用于记录处理时间
            Stopwatch processTimer = new Stopwatch();
            processTimer.Start();

            Console.WriteLine("初始化统一 YOLO 处理器...");
            
            using (var processor = new UniversalYoloProcessor(modelPath, yamlPath, device, 0.1f, 0.1f))
            {
                int processedCount = 0;
                int successCount = 0;

                foreach (var imagePath in imagePaths)
                {
                    try
                    {
                        string fileName = Path.GetFileName(imagePath);
                        Console.WriteLine($"\n处理图像: {fileName}");
                        
                        var detections = processor.ProcessImage(imagePath);
                        
                        if (processor.ModelInfo.Type == ModelType.Classification)
                        {
                            // 分类结果
                            if (detections.Count > 0)
                            {
                                var result = detections[0];
                                Console.WriteLine($"  分类结果: {result.ClassName} (置信度: {result.Confidence:F3})");
                            }
                            else
                            {
                                Console.WriteLine("  未找到有效分类结果");
                            }
                        }
                        else if (processor.ModelInfo.Type == ModelType.Detection)
                        {
                            // 检测结果
                            Console.WriteLine($"  检测到 {detections.Count} 个目标:");
                            
                            foreach (var detection in detections)
                            {
                                Console.WriteLine($"    - {detection.ClassName}: {detection.Confidence:F3} " +
                                                $"[{detection.X:F0}, {detection.Y:F0}, {detection.Width:F0}, {detection.Height:F0}]");
                            }

                            // 绘制并保存检测结果
                            if (detections.Count > 0)
                            {
                                string outputImagePath = Path.Combine(outputPath, $"result_{fileName}");
                                ImageDrawer.DrawDetections(imagePath, detections, outputImagePath);
                                Console.WriteLine($"    结果已保存到: {outputImagePath}");
                            }
                        }

                        successCount++;
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"  处理失败: {ex.Message}");
                    }
                    
                    processedCount++;
                }

                Console.WriteLine($"\n=== 处理完成 ===");
                Console.WriteLine($"总计处理: {processedCount} 张图片");
                Console.WriteLine($"成功处理: {successCount} 张图片");
                Console.WriteLine($"模型类型: {processor.ModelInfo.Type}");
            }

            processTimer.Stop();
            Console.WriteLine($"总处理时间: {processTimer.Elapsed.TotalSeconds:F2} 秒");
        }
    }
}
