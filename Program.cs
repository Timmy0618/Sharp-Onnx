using System;
using System.Diagnostics;
using System.IO;

namespace OnnxLibrary
{
    class Program
    {
        static void Main(string[] args)
        {
            string modelPath = "best_model.onnx"; // 模型路徑
            string yamlPath = "classes.yaml";
            string directoryPath = "./test/E-type"; // 圖片資料夾路徑
            string device = "cpu"; // 可选择 "cpu" 或 "gpu"

            try
            {
                // 計時器
                Stopwatch stopwatch = new Stopwatch();
                stopwatch.Start();

                // 確認目錄是否存在
                if (!Directory.Exists(directoryPath))
                {
                    Console.WriteLine($"Directory not found: {directoryPath}");
                    return;
                }

                // 獲取所有圖片檔案
                string[] imagePaths = Directory.GetFiles(directoryPath, "*.tif");

                if (imagePaths.Length == 0)
                {
                    Console.WriteLine("No images found in the specified directory.");
                    return;
                }

                using (var predictor = new OnnxImagePredictor(modelPath, yamlPath, device))
                {
                    foreach (var imagePath in imagePaths)
                    {
                        var (scores, className) = predictor.Predict(imagePath);

                        Console.WriteLine($"Image: {Path.GetFileName(imagePath)}");
                        Console.WriteLine($"Predicted Class: {className}");
                        Console.WriteLine("Scores:");
                        for (int i = 0; i < scores.Length; i++)
                        {
                            Console.WriteLine($"Class {i}: {scores[i]}");
                        }
                        Console.WriteLine(new string('-', 50)); // 分隔線
                    }
                }

                // 停止計時
                stopwatch.Stop();
                Console.WriteLine($"Total Processing Time: {stopwatch.Elapsed.TotalSeconds:F2} seconds");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
            }
        }
    }
}
