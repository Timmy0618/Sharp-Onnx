using OpenCvSharp;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Threading.Tasks;

namespace OnnxLibrary
{
    public static class ImagePreprocessor
    {
        public static DenseTensor<float> PreprocessImage(string imagePath, int width, int height)
        {
            // 讀取影像
            using (var mat = new Mat(imagePath))
            {
                // 調整影像大小
                var resized = new Mat();
                Cv2.Resize(mat, resized, new Size(width, height));

                // 定義影像標準化所需的均值和標準差
                float[] mean = { 0.485f, 0.456f, 0.406f };
                float[] std = { 0.229f, 0.224f, 0.225f };

                // 初始化存放影像資料的陣列
                float[] data = new float[1 * 3 * height * width];

                // 使用平行處理迭代每個像素，並進行標準化處理
                Parallel.For(0, height, y =>
                {
                    for (int x = 0; x < width; x++)
                    {
                        var pixel = resized.At<Vec3b>(y, x);

                        // 取得 RGB 值並進行標準化
                        float r = (pixel.Item2 / 255.0f - mean[0]) / std[0];
                        float g = (pixel.Item1 / 255.0f - mean[1]) / std[1];
                        float b = (pixel.Item0 / 255.0f - mean[2]) / std[2];

                        // 將標準化後的值存入資料陣列
                        data[0 * (height * width) + y * width + x] = r;
                        data[1 * (height * width) + y * width + x] = g;
                        data[2 * (height * width) + y * width + x] = b;
                    }
                });

                // 定義張量的形狀
                var shape = new int[] { 1, 3, height, width };
                // 建立並返回 DenseTensor
                var inputTensor = new DenseTensor<float>(data, shape);
                return inputTensor;
            }
        }
    }
}
