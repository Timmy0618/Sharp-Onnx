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
    public enum ModelType
    {
        Classification,
        Detection,
        Unknown
    }

    public class ModelInfo
    {
        public ModelType Type { get; set; }
        public int InputWidth { get; set; }
        public int InputHeight { get; set; }
        public int NumClasses { get; set; }
        public string Description { get; set; }
    }

    public class UniversalYoloProcessor : IDisposable
    {
        private InferenceSession _session;
        private Dictionary<int, string> _classMapping;
        private readonly float _confidenceThreshold;
        private readonly float _nmsThreshold;
        private readonly ModelInfo _modelInfo;

        public ModelInfo ModelInfo => _modelInfo;

        public UniversalYoloProcessor(string onnxModelPath, string yamlPath, string device = "cpu", 
            float confidenceThreshold = 0.5f, float nmsThreshold = 0.45f)
        {
            var options = new SessionOptions();

            if (device.ToLower() == "gpu")
            {
                options.AppendExecutionProvider_CUDA();
            }
            else
            {
                options.AppendExecutionProvider_CPU();
            }

            _session = new InferenceSession(onnxModelPath, options);
            _classMapping = LoadClassMapping(yamlPath);
            _confidenceThreshold = confidenceThreshold;
            _nmsThreshold = nmsThreshold;
            
            // 自动分析模型类型
            _modelInfo = AnalyzeModel();
            
            Console.WriteLine($"🔍 Model Analysis Result:");
            Console.WriteLine($"   Type: {_modelInfo.Type}");
            Console.WriteLine($"   Input Size: {_modelInfo.InputWidth}x{_modelInfo.InputHeight}");
            Console.WriteLine($"   Class Count: {_modelInfo.NumClasses}");
            Console.WriteLine($"   Description: {_modelInfo.Description}");
        }

        private ModelInfo AnalyzeModel()
        {
            var inputMeta = _session.InputMetadata.First().Value;
            var outputMeta = _session.OutputMetadata.First().Value;
            
            var inputShape = inputMeta.Dimensions.ToArray();
            var outputShape = outputMeta.Dimensions.ToArray();
            
            var modelInfo = new ModelInfo();
            
            // 分析输入形状
            if (inputShape.Length >= 4)
            {
                modelInfo.InputHeight = inputShape[2];
                modelInfo.InputWidth = inputShape[3];
            }
            
            // 分析输出形状来判断模型类型
            if (outputShape.Length == 2)
            {
                // 形状为 [batch, classes] -> 分类模型
                modelInfo.Type = ModelType.Classification;
                modelInfo.NumClasses = outputShape[1];
                modelInfo.Description = $"YOLO v11 classification model, outputs {modelInfo.NumClasses} classes";
            }
            else if (outputShape.Length == 3)
            {
                // 形状为 [batch, features, predictions] -> 检测模型
                // For YOLO v11: features = 4 (bbox) + num_classes
                modelInfo.Type = ModelType.Detection;
                modelInfo.NumClasses = outputShape[1] - 4; // 减去 4 个边界框坐标
                modelInfo.Description = $"YOLO v11 detection model, outputs {outputShape[2]} predictions, {modelInfo.NumClasses} classes";
            }
            else
            {
                modelInfo.Type = ModelType.Unknown;
                modelInfo.NumClasses = _classMapping.Count;
                modelInfo.Description = $"未知模型类型，输出形状: [{string.Join(", ", outputShape)}]";
            }
            
            return modelInfo;
        }

        public List<Detection> ProcessImage(string imagePath)
        {
            if (_modelInfo.Type == ModelType.Classification)
            {
                return ProcessClassification(imagePath);
            }
            else if (_modelInfo.Type == ModelType.Detection)
            {
                return ProcessDetection(imagePath);
            }
            else
            {
                throw new NotSupportedException($"不支持的模型类型: {_modelInfo.Type}");
            }
        }

        private List<Detection> ProcessClassification(string imagePath)
        {
            // 使用分类预处理
            var tensor = ImagePreprocessor.PreprocessImage(imagePath, _modelInfo.InputWidth, _modelInfo.InputHeight);
            
            var inputName = _session.InputMetadata.Keys.First();
            var outputName = _session.OutputMetadata.Keys.First();

            using (var results = _session.Run(new[]
            {
                NamedOnnxValue.CreateFromTensor(inputName, tensor)
            }))
            {
                var outputValue = results.First(x => x.Name == outputName);
                var outputTensor = outputValue.AsTensor<float>();
                
                if (outputTensor == null)
                    return new List<Detection>();

                float[] scores = TensorToArray(outputTensor);
                
                // 检查输出是否已经是概率值 (和接近1.0)
                float sum = scores.Sum();
                bool isAlreadyProbability = Math.Abs(sum - 1.0f) < 0.01f; // 允许小误差
                
                if (!isAlreadyProbability)
                {
                    // 只有当输出不是概率时才应用 Softmax
                    scores = ApplySoftmax(scores);
                }
                
                int predictedClassIndex = ArgMax(scores);
                float confidence = scores[predictedClassIndex];
                
                // 将分类结果转换为检测格式（整个图像作为一个检测框）
                using (var image = OpenCvSharp.Cv2.ImRead(imagePath))
                {
                    string className = _classMapping.ContainsKey(predictedClassIndex) 
                        ? _classMapping[predictedClassIndex] 
                        : $"Class_{predictedClassIndex}";

                    return new List<Detection>
                    {
                        new Detection
                        {
                            X = 0,
                            Y = 0,
                            Width = image.Width,
                            Height = image.Height,
                            Confidence = confidence,
                            ClassId = predictedClassIndex,
                            ClassName = className
                        }
                    };
                }
            }
        }

        private List<Detection> ProcessDetection(string imagePath)
        {
            // 获取原始图像尺寸
            int originalWidth, originalHeight;
            using (var image = OpenCvSharp.Cv2.ImRead(imagePath))
            {
                originalWidth = image.Width;
                originalHeight = image.Height;
            }

            // 使用检测预处理
            var (tensor, _, _) = ImagePreprocessor.PreprocessImageForDetection(imagePath, _modelInfo.InputWidth, _modelInfo.InputHeight);
            
            var inputName = _session.InputMetadata.Keys.First();
            var outputName = _session.OutputMetadata.Keys.First();

            using (var results = _session.Run(new[]
            {
                NamedOnnxValue.CreateFromTensor(inputName, tensor)
            }))
            {
                var outputValue = results.First(x => x.Name == outputName);
                var outputTensor = outputValue.AsTensor<float>();
                
                if (outputTensor == null)
                    return new List<Detection>();

                var detections = ParseDetections(outputTensor, originalWidth, originalHeight);
                return ApplyNMS(detections);
            }
        }

        private List<Detection> ParseDetections(Tensor<float> outputTensor, int originalWidth, int originalHeight)
        {
            var detections = new List<Detection>();
            
            var shape = outputTensor.Dimensions.ToArray();
            int numPredictions = shape[2];
            int numClasses = _modelInfo.NumClasses;
            
            // 计算缩放比例
            float scaleX = (float)originalWidth / _modelInfo.InputWidth;
            float scaleY = (float)originalHeight / _modelInfo.InputHeight;

            for (int i = 0; i < numPredictions; i++)
            {
                // 获取边界框坐标 (center_x, center_y, width, height)
                float centerX = outputTensor[0, 0, i];
                float centerY = outputTensor[0, 1, i];
                float width = outputTensor[0, 2, i];
                float height = outputTensor[0, 3, i];

                // 获取最高置信度的类别
                float maxScore = 0f;
                int maxClassId = -1;
                
                for (int c = 0; c < numClasses; c++)
                {
                    float score = outputTensor[0, 4 + c, i];
                    if (score > maxScore)
                    {
                        maxScore = score;
                        maxClassId = c;
                    }
                }

                // 过滤低置信度检测
                if (maxScore < _confidenceThreshold)
                    continue;

                // 转换为左上角坐标
                float x = (centerX - width / 2) * scaleX;
                float y = (centerY - height / 2) * scaleY;
                
                // 确保坐标在图像范围内
                x = Math.Max(0, Math.Min(x, originalWidth - 1));
                y = Math.Max(0, Math.Min(y, originalHeight - 1));
                width = Math.Max(0, Math.Min(width * scaleX, originalWidth - x));
                height = Math.Max(0, Math.Min(height * scaleY, originalHeight - y));

                string className = _classMapping.ContainsKey(maxClassId) 
                    ? _classMapping[maxClassId] 
                    : $"Class_{maxClassId}";

                detections.Add(new Detection
                {
                    X = x,
                    Y = y,
                    Width = width,
                    Height = height,
                    Confidence = maxScore,
                    ClassId = maxClassId,
                    ClassName = className
                });
            }

            return detections;
        }

        private List<Detection> ApplyNMS(List<Detection> detections)
        {
            if (detections.Count == 0)
                return detections;

            detections = detections.OrderByDescending(d => d.Confidence).ToList();
            var results = new List<Detection>();

            while (detections.Count > 0)
            {
                var current = detections[0];
                results.Add(current);
                detections.RemoveAt(0);

                for (int i = detections.Count - 1; i >= 0; i--)
                {
                    if (CalculateIoU(current, detections[i]) > _nmsThreshold)
                    {
                        detections.RemoveAt(i);
                    }
                }
            }

            return results;
        }

        private float CalculateIoU(Detection a, Detection b)
        {
            float x1 = Math.Max(a.X, b.X);
            float y1 = Math.Max(a.Y, b.Y);
            float x2 = Math.Min(a.X + a.Width, b.X + b.Width);
            float y2 = Math.Min(a.Y + a.Height, b.Y + b.Height);

            if (x2 <= x1 || y2 <= y1)
                return 0f;

            float intersection = (x2 - x1) * (y2 - y1);
            float areaA = a.Width * a.Height;
            float areaB = b.Width * b.Height;
            float union = areaA + areaB - intersection;

            return intersection / union;
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
}
