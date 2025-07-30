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
                    Console.WriteLine($"❌ 模型文件不存在: {modelPath}");
                    return;
                }
                
                if (!File.Exists(yamlPath))
                {
                    Console.WriteLine($"❌ 类别文件不存在: {yamlPath}");
                    return;
                }
                
                if (!Directory.Exists(testImagePath))
                {
                    Console.WriteLine($"❌ 测试图片目录不存在: {testImagePath}");
                    return;
                }
                
                // 获取一张测试图片
                string[] imageFiles = GetSupportedImageFiles(testImagePath);
                if (imageFiles.Length == 0)
                {
                    Console.WriteLine($"❌ 在 {testImagePath} 中没有找到支持的图片文件");
                    Console.WriteLine($"   支持格式: {string.Join(", ", SupportedImageExtensions.Select(ext => ext.Replace("*", "")))}");
                    return;
                }
                
                string testImage = imageFiles[0];
                Console.WriteLine($"✅ 使用测试图片: {Path.GetFileName(testImage)}");
                
                // 创建输出目录
                Directory.CreateDirectory(outputPath);
                
                // 测试统一处理器
                Console.WriteLine("🔄 初始化统一 YOLO 处理器...");
                using (var processor = new UniversalYoloProcessor(modelPath, yamlPath, "cpu", 0.5f, 0.45f))
                {
                    Console.WriteLine("🔄 处理图像...");
                    var detections = processor.ProcessImage(testImage);
                    Console.WriteLine($"✅ 处理完成，发现 {detections.Count} 个目标");
                    
                    foreach (var detection in detections)
                    {
                        Console.WriteLine($"   - {detection.ClassName}: {detection.Confidence:F3} " +
                                        $"[{detection.X:F0}, {detection.Y:F0}, {detection.Width:F0}, {detection.Height:F0}]");
                    }
                    
                    // 测试绘制结果
                    if (detections.Count > 0)
                    {
                        Console.WriteLine("🔄 绘制结果...");
                        string outputImagePath = Path.Combine(outputPath, $"result_{Path.GetFileName(testImage)}");
                        
                        // 对于分类模型，不绘制边界框，只显示结果
                        if (processor.ModelInfo.Type == ModelType.Classification)
                        {
                            Console.WriteLine($"   分类结果: {detections[0].ClassName} (置信度: {detections[0].Confidence:F3})");
                            Console.WriteLine($"   注意: 分类模型不绘制边界框");
                        }
                        else
                        {
                            ImageDrawer.DrawDetections(testImage, detections, outputImagePath);
                            Console.WriteLine($"✅ 结果已保存到: {outputImagePath}");
                        }
                    }
                }
                
                Console.WriteLine("🎉 所有测试通过！");
                
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ 测试失败: {ex.Message}");
                Console.WriteLine($"   详细信息: {ex.StackTrace}");
            }
        }
        
        public static void PrintSystemInfo()
        {
            Console.WriteLine("=== 系统信息 ===");
            Console.WriteLine($"操作系统: {Environment.OSVersion}");
            Console.WriteLine($".NET 版本: {Environment.Version}");
            Console.WriteLine($"处理器数量: {Environment.ProcessorCount}");
            Console.WriteLine($"工作目录: {Environment.CurrentDirectory}");
            Console.WriteLine();
        }
    }
}
