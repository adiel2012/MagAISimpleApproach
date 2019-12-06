﻿using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace xoronnxCSharp
{
    class Program
    {
        static void Main(string[] args)
        {
            var session = new InferenceSession("model.onnx");
            string name = "XR:0";
            var inputMeta = session.InputMetadata;
            float[][] inputData = {
                new[] { 0.0f, 0.0f },
                new[] { 0.0f, 1.0f },
                new[] { 1.0f, 0.0f },
                new[] { 1.0f, 1.0f } };
            foreach (float[] row in inputData)
            {
                Tensor<float> t1 = new DenseTensor<float>(row, inputMeta[name].Dimensions);
                var inputs = new List<NamedOnnxValue>() { NamedOnnxValue.CreateFromTensor<float>(name, t1) };
                using (var results = session.Run(inputs))
                {
                    var output = results.ToArray()[0].AsEnumerable<float>().First();
                    Console.WriteLine($"{row[0]} {row[1]}  --> {output}");
                }
            }
            Console.ReadLine();
        }
    }
}
