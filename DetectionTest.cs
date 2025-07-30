using System;
using System.IO;
using System.Linq;

namespace OnnxLibrary
{
    /// <summary>
    /// 简单的测试类，用于验证检测功能
    /// </summary>
    public static class DetectionTest
    {
        // 支持的图片格式
        private static readonly string[] SupportedImageExtensions = 
        {
            "*.bmp", "*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.gif"
        };

        /// <summary>
        /// 获取目录中所有支持格式的图片文件
        /// </summary>
        /// <param name="directoryPath">目录路径</param>
        /// <returns>图片文件路径数组</returns>
        private static string[] GetSupportedImageFiles(string directoryPath)
        {
            var imageFiles = new System.Collections.Generic.List<string>();
            
            foreach (string extension in SupportedImageExtensions)
            {
                var files = Directory.GetFiles(directoryPath, extension, SearchOption.TopDirectoryOnly);
                imageFiles.AddRange(files);
            }
            
            // 按文件名排序
            return imageFiles.OrderBy(Path.GetFileName).ToArray();
        }

        public static void RunTest()
        {
            Console.WriteLine("=== OnnxLibrary Universal YOLO Test ===");
            
            // 测试配置
            string testImagePath = "./valid/others"; // 测试图片路径
            string modelPath = "models/yolov11/250318/best.onnx";
            string yamlPath = "classes.yaml";
            string outputPath = "./test_output";
            
            try
            {
                // 检查文件是否存在
                if (!File.Exists(modelPath))
                {
                    Console.WriteLine($"❌ Model file not found: {modelPath}");
                    return;
                }
                
                if (!File.Exists(yamlPath))
                {
                    Console.WriteLine($"❌ Class file not found: {yamlPath}");
                    return;
                }
                
                if (!Directory.Exists(testImagePath))
                {
                    Console.WriteLine($"❌ Test image directory not found: {testImagePath}");
                    return;
                }
                
                // 获取一张测试图片
                string[] imageFiles = GetSupportedImageFiles(testImagePath);
                if (imageFiles.Length == 0)
                {
                    Console.WriteLine($"❌ No supported image files found in {testImagePath}");
                    Console.WriteLine($"   Supported formats: {string.Join(", ", SupportedImageExtensions.Select(ext => ext.Replace("*", "")))}");
                    return;
                }
                
                string testImage = imageFiles[0];
                Console.WriteLine($"✅ Using test image: {Path.GetFileName(testImage)}");
                
                // 创建输出目录
                Directory.CreateDirectory(outputPath);
                
                // Test unified processor
                Console.WriteLine("🔄 Initializing Universal YOLO Processor...");
                using (var processor = new UniversalYoloProcessor(modelPath, yamlPath, "cpu", 0.5f, 0.45f))
                {
                    Console.WriteLine("🔄 Processing image...");
                    var detections = processor.ProcessImage(testImage);
                    Console.WriteLine($"✅ Processing completed, found {detections.Count} objects");
                    
                    foreach (var detection in detections)
                    {
                        Console.WriteLine($"   - {detection.ClassName}: {detection.Confidence:F3} " +
                                        $"[{detection.X:F0}, {detection.Y:F0}, {detection.Width:F0}, {detection.Height:F0}]");
                    }
                    
                    // Test drawing results
                    if (detections.Count > 0)
                    {
                        Console.WriteLine("🔄 Drawing results...");
                        string outputImagePath = Path.Combine(outputPath, $"result_{Path.GetFileName(testImage)}");
                        
                        // For classification models, don't draw bounding boxes, just show results
                        if (processor.ModelInfo.Type == ModelType.Classification)
                        {
                            Console.WriteLine($"   Classification result: {detections[0].ClassName} (Confidence: {detections[0].Confidence:F3})");
                            Console.WriteLine($"   Note: Classification model does not draw bounding boxes");
                        }
                        else
                        {
                            ImageDrawer.DrawDetections(testImage, detections, outputImagePath);
                            Console.WriteLine($"✅ Results saved to: {outputImagePath}");
                        }
                    }
                }
                
                Console.WriteLine("🎉 All tests passed!");
                
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Test failed: {ex.Message}");
                Console.WriteLine($"   Details: {ex.StackTrace}");
            }
        }
        
        public static void PrintSystemInfo()
        {
            Console.WriteLine("=== System Information ===");
            Console.WriteLine($"Operating System: {Environment.OSVersion}");
            Console.WriteLine($".NET Version: {Environment.Version}");
            Console.WriteLine($"Processor Count: {Environment.ProcessorCount}");
            Console.WriteLine($"Working Directory: {Environment.CurrentDirectory}");
            Console.WriteLine();
        }
    }
}
