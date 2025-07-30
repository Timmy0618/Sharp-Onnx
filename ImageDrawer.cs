using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.IO;

namespace OnnxLibrary
{
    public static class ImageDrawer
    {
        private static readonly Scalar[] Colors = new Scalar[]
        {
            new Scalar(255, 0, 0),     // 红色
            new Scalar(0, 255, 0),     // 绿色
            new Scalar(0, 0, 255),     // 蓝色
            new Scalar(255, 255, 0),   // 黄色
            new Scalar(255, 0, 255),   // 洋红
            new Scalar(0, 255, 255),   // 青色
            new Scalar(128, 0, 128),   // 紫色
            new Scalar(255, 165, 0),   // 橙色
            new Scalar(0, 128, 0),     // 深绿色
            new Scalar(128, 128, 0)    // 橄榄色
        };

        public static void DrawDetections(string imagePath, List<Detection> detections, string outputPath)
        {
            using (var image = new Mat(imagePath))
            {
                foreach (var detection in detections)
                {
                    // 选择颜色
                    var color = Colors[detection.ClassId % Colors.Length];
                    
                    // 绘制边界框
                    var rect = new Rect((int)detection.X, (int)detection.Y, 
                                       (int)detection.Width, (int)detection.Height);
                    Cv2.Rectangle(image, rect, color, 2);
                    
                    // 准备标签文本
                    string label = $"{detection.ClassName}: {detection.Confidence:F2}";
                    
                    // 计算文本大小
                    var textSize = Cv2.GetTextSize(label, HersheyFonts.HersheySimplex, 0.6, 2, out int baseline);
                    
                    // 绘制标签背景
                    var labelRect = new Rect((int)detection.X, (int)detection.Y - textSize.Height - baseline - 5,
                                           textSize.Width + 10, textSize.Height + baseline + 10);
                    Cv2.Rectangle(image, labelRect, color, -1);
                    
                    // 绘制标签文本
                    var textPoint = new Point((int)detection.X + 5, (int)detection.Y - 5);
                    Cv2.PutText(image, label, textPoint, HersheyFonts.HersheySimplex, 0.6, 
                               new Scalar(255, 255, 255), 2);
                }
                
                // 保存结果图像
                Directory.CreateDirectory(Path.GetDirectoryName(outputPath));
                Cv2.ImWrite(outputPath, image);
            }
        }

        public static void ShowDetections(string imagePath, List<Detection> detections, string windowName = "Detections")
        {
            using (var image = new Mat(imagePath))
            {
                foreach (var detection in detections)
                {
                    // 选择颜色
                    var color = Colors[detection.ClassId % Colors.Length];
                    
                    // 绘制边界框
                    var rect = new Rect((int)detection.X, (int)detection.Y, 
                                       (int)detection.Width, (int)detection.Height);
                    Cv2.Rectangle(image, rect, color, 2);
                    
                    // 准备标签文本
                    string label = $"{detection.ClassName}: {detection.Confidence:F2}";
                    
                    // 计算文本大小
                    var textSize = Cv2.GetTextSize(label, HersheyFonts.HersheySimplex, 0.6, 2, out int baseline);
                    
                    // 绘制标签背景
                    var labelRect = new Rect((int)detection.X, (int)detection.Y - textSize.Height - baseline - 5,
                                           textSize.Width + 10, textSize.Height + baseline + 10);
                    Cv2.Rectangle(image, labelRect, color, -1);
                    
                    // 绘制标签文本
                    var textPoint = new Point((int)detection.X + 5, (int)detection.Y - 5);
                    Cv2.PutText(image, label, textPoint, HersheyFonts.HersheySimplex, 0.6, 
                               new Scalar(255, 255, 255), 2);
                }
                
                // 显示图像
                Cv2.ImShow(windowName, image);
                Cv2.WaitKey(0);
                Cv2.DestroyAllWindows();
            }
        }
    }
}
