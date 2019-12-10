using System;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Drawing;
using System.IO;
using System.Collections.Generic;
using System.Linq;

namespace csharpsentimentanalysis
{
    class Program
    {
        static void Main(string[] args)
        {
            var session = new InferenceSession(Path.GetFullPath("../pythonexamples/modelkerassentimentanalisis.onnx"));
            string name = "embedding_1_input";
            var inputMeta = session.InputMetadata; 
            int[] dimentions =  inputMeta[name].Dimensions.Select(dim => dim == -1 ? 1 : dim).ToArray();
        
            foreach(string file in GetSentenses())
            {
                /*float[][] inputData = {loadimage(file)};
                                
                    foreach(float[] row in inputData)
                    {
                        Tensor<float> t1 = new DenseTensor<float>(row,dimentions);
                        var inputs = new List<NamedOnnxValue>(){NamedOnnxValue.CreateFromTensor<float>(name, t1)};
                        using (var results = session.Run(inputs))
                        {
                            float[] output = results.ToArray()[0].AsEnumerable<float>().ToArray();                                
                            int maxIndex = output.ToList().IndexOf(output.Max()); 
                        }
                    }  */   
            }
            
        }

        private static IEnumerable<string> GetSentenses()
        {
            yield return "";
            yield return "";
            yield return "";
            yield return "";
            yield return "";
        }
    }
}
