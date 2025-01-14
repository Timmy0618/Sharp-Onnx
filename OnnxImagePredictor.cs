using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

namespace OnnxLibrary
{
    public class OnnxImagePredictor : IDisposable
    {
        private InferenceSession _session;
        private Dictionary<int, string> _classMapping;

        public OnnxImagePredictor(string onnxModelPath, string yamlPath, string device = "cpu")
        {
            var options = new SessionOptions();

            if (device.ToLower() == "gpu")
            {
                // Enable GPU acceleration
                options.AppendExecutionProvider_CUDA();
            }
            else
            {
                // Default to CPU
                options.AppendExecutionProvider_CPU();
            }

            _session = new InferenceSession(onnxModelPath, options);

            // Initialize class mapping
            _classMapping = LoadClassMapping(yamlPath);
        }

        public (float[] scores, string predictedClassName) Predict(
            string imagePath,
            int targetWidth = 224,
            int targetHeight = 224)
        {
            var inputTensor = PreprocessImage(imagePath, targetWidth, targetHeight);

            var inputName = _session.InputMetadata.Keys.First();
            var outputName = _session.OutputMetadata.Keys.First();

            using (var results = _session.Run(new[]
            {
                NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
            }))
            {
                var outputValue = results.First(x => x.Name == outputName);
                var outputTensor = outputValue.AsTensor<float>();
                if (outputTensor == null)
                    return (null, null);

                float[] scores = TensorToArray(outputTensor);

                // Apply Softmax to scores
                scores = ApplySoftmax(scores);

                int predictedClassIndex = ArgMax(scores);

                // Find class name
                string className = _classMapping.ContainsKey(predictedClassIndex)
                    ? _classMapping[predictedClassIndex]
                    : "Unknown";

                return (scores, className);
            }
        }

        private float[] ApplySoftmax(float[] scores)
        {
            float max = scores.Max();
            float sumExp = 0.0f;
            for (int i = 0; i < scores.Length; i++)
            {
                scores[i] = (float)Math.Exp(scores[i] - max);
                sumExp += scores[i];
            }
            for (int i = 0; i < scores.Length; i++)
            {
                scores[i] /= sumExp;
            }
            return scores;
        }

        private DenseTensor<float> PreprocessImage(string imagePath, int width, int height)
        {
            using (var original = new Bitmap(imagePath))
            using (var resized = new Bitmap(original, new Size(width, height)))
            {
                float[] mean = { 0.485f, 0.456f, 0.406f };
                float[] std = { 0.229f, 0.224f, 0.225f };

                float[] data = new float[1 * 3 * height * width];

                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        Color pixel = resized.GetPixel(x, y);

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

        private float[] TensorToArray(Tensor<float> tensor)
        {
            float[] array = new float[tensor.Length];
            int idx = 0;
            foreach (float val in tensor)
            {
                array[idx++] = val;
            }
            return array;
        }

        private int ArgMax(float[] values)
        {
            if (values == null || values.Length == 0)
                return -1;

            int maxIndex = 0;
            float maxVal = values[0];
            for (int i = 1; i < values.Length; i++)
            {
                if (values[i] > maxVal)
                {
                    maxVal = values[i];
                    maxIndex = i;
                }
            }
            return maxIndex;
        }

        private Dictionary<int, string> LoadClassMapping(string yamlPath)
        {
            if (!File.Exists(yamlPath))
                throw new FileNotFoundException("YAML file not found.", yamlPath);

            var deserializer = new DeserializerBuilder()
                .WithNamingConvention(CamelCaseNamingConvention.Instance)
                .Build();

            var yamlContent = File.ReadAllText(yamlPath);
            var parsed = deserializer.Deserialize<YamlClasses>(yamlContent);

            return parsed.Classes.ToDictionary(c => c.Id, c => c.Name);
        }

        public void Dispose()
        {
            _session?.Dispose();
        }
    }

    public class YamlClasses
    {
        public List<ClassInfo> Classes { get; set; }
    }

    public class ClassInfo
    {
        public int Id { get; set; }
        public string Name { get; set; }
    }
}
