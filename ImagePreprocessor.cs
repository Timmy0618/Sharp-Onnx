using System.Drawing;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace OnnxLibrary
{
    public static class ImagePreprocessor
    {
        public static DenseTensor<float> PreprocessImage(Bitmap image, int width, int height)
        {
            float[] mean = { 0.485f, 0.456f, 0.406f };
            float[] std = { 0.229f, 0.224f, 0.225f };

            float[] data = new float[1 * 3 * height * width];

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    Color pixel = image.GetPixel(x, y);

                    float r = (pixel.R / 255.0f - mean[0]) / std[0];
                    float g = (pixel.G / 255.0f - mean[1]) / std[1];
                    float b = (pixel.B / 255.0f - mean[2]) / std[2];

                    data[0 * (height * width) + y * width + x] = r;
                    data[1 * (height * width) + y * width + x] = g;
                    data[2 * (height * width) + y * width + x] = b;
                }
            }

            var shape = new int[] { 1, 3, height, width };
            var inputTensor = new DenseTensor<float>(data, shape);
            return inputTensor;
        }
    }
}
