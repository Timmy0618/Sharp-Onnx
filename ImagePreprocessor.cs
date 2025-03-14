using OpenCvSharp;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Threading.Tasks;
using System;

namespace OnnxLibrary
{
    public static class ImagePreprocessor
    {
        public static DenseTensor<float> PreprocessImage(string imagePath, int inputWidth, int inputHeight)
        {
            // 讀取影像
            using (var mat = new Mat(imagePath))
            {
                // 取得原始影像的尺寸
                int originalWidth = mat.Width;
                int originalHeight = mat.Height;

                // 計算縮放比例
                float scale = System.Math.Min((float)inputWidth / originalWidth, (float)inputHeight / originalHeight);

                // 計算縮放後的尺寸
                int newWidth = (int)(originalWidth * scale);
                int newHeight = (int)(originalHeight * scale);

                // 調整影像大小
                var resized = new Mat();
                Cv2.Resize(mat, resized, new Size(newWidth, newHeight));

                // 創建一個新的影像，填充顏色為 (114, 114, 114)
                var padded = new Mat(new Size(inputWidth, inputHeight), MatType.CV_8UC3, new Scalar(114, 114, 114));

                // 計算填充位置
                int top = (inputHeight - newHeight) / 2;
                int left = (inputWidth - newWidth) / 2;

                // 將調整大小後的影像放置到填充影像上
                var roi = new Rect(left, top, newWidth, newHeight);
                resized.CopyTo(new Mat(padded, roi));

                // 初始化存放影像資料的陣列
                float[] data = new float[1 * 3 * inputHeight * inputWidth];

                // 使用平行處理迭代每個像素，並進行標準化處理
                Parallel.For(0, inputHeight, y =>
                {
                    for (int x = 0; x < inputWidth; x++)
                    {
                        var pixel = padded.At<Vec3b>(y, x);

                        // 取得 RGB 值並進行標準化
                        float r = pixel.Item2 / 255.0f;
                        float g = pixel.Item1 / 255.0f;
                        float b = pixel.Item0 / 255.0f;

                        // 將標準化後的值存入資料陣列
                        data[0 * (inputHeight * inputWidth) + y * inputWidth + x] = r;
                        data[1 * (inputHeight * inputWidth) + y * inputWidth + x] = g;
                        data[2 * (inputHeight * inputWidth) + y * inputWidth + x] = b;
                    }
                });

                // 定義張量的形狀
                var shape = new int[] { 1, 3, inputHeight, inputWidth };
                // 建立並返回 DenseTensor
                var inputTensor = new DenseTensor<float>(data, shape);
                return inputTensor;
            }
        }
    }
}
