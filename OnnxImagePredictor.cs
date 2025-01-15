using System;
using System.Collections.Generic;
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

        public (float[] scores, string predictedClassName) PredictFromTensor(DenseTensor<float> inputTensor)
        {
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
