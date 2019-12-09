using System;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Drawing;
using System.IO;
using System.Collections.Generic;
using System.Linq;

namespace xoronnxCSharp
{
    // dotnet add package System.Drawing.Common
    // dotnet add package Microsoft.ML.OnnxRuntime
    class Program
    {
        static void Main(string[] args)
        {            
            var session = new InferenceSession(Path.GetFullPath("../pythonexamples/modelkerasimg.onnx"));
            string name = "conv2d_1_input";
            var inputMeta = session.InputMetadata; 
            string[] classes = {"cars", "planes"};
            int[] dimentions =  inputMeta[name].Dimensions.Select(dim => dim == -1 ? 1 : dim).ToArray();
            int[,] confussion_matrix = new int[2,2];

            for(int class_index = 0 ; class_index < 2; class_index++)
            {
                foreach(string file in Directory.GetFiles(Path.GetFullPath($"../pythonexamples/v_data/train/{classes[class_index]}"), "*.jpg"))
                {
                    float[][] inputData = {loadimage(file)};
                                    
                        foreach(float[] row in inputData)
                        {
                            Tensor<float> t1 = new DenseTensor<float>(row,dimentions);
                            var inputs = new List<NamedOnnxValue>(){NamedOnnxValue.CreateFromTensor<float>(name, t1)};
                            using (var results = session.Run(inputs))
                            {
                                float[] output = results.ToArray()[0].AsEnumerable<float>().ToArray();
                                int index = output[0] > 0.5 ? 1 : 0;                                
                                confussion_matrix[class_index, index]++;
                            }
                        }     
                }
            } 
            DisplayConfussionMatrix(confussion_matrix, classes);         
        
        }

        private static void DisplayConfussionMatrix(int[,] confussion_matrix, string[] classes)
        {
           
        }

        private static float[] loadimage(string img_file)
        {
            var image = new Bitmap(img_file, false);
            float[] result = new float[3 * image.Height * image.Width];
            int pos = 0;
            for(int x=0; x<image.Width; x++)
            {
                for(int y=0; y<image.Height; y++)
                {
                    Color pixelColor = image.GetPixel(x, y);
                    result[pos++] = (float)pixelColor.R/255;
                    result[pos++] = (float)pixelColor.G/255;
                    result[pos++] = (float)pixelColor.B/255;
                }
            }

            return result;
        }

    }
}


