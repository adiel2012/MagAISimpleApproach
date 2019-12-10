using Microsoft.ML.OnnxRuntime;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using Newtonsoft.Json.Linq;
using System;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace csharpsentimentanalysis
{
    class Program
    {
        static Dictionary<string, int> word_index = new Dictionary<string, int>();
        // dotnet add package Json.Net
        // dotnet add package Newtonsoft.Json
        static void Main(string[] args)
        {
            LoadIMDBWordIndex();
            var session = new InferenceSession(Path.GetFullPath("../pythonexamples/modelkerassentimentanalisis.onnx"));
            string name = "embedding_1_input";
            var inputMeta = session.InputMetadata; 
            int[] dimentions =  inputMeta[name].Dimensions.Select(dim => dim == -1 ? 1 : dim).ToArray();
        
            foreach(string sentence in GetSentenses())
            {                
                float[][] inputData = {ConvertSentence(sentence)};
                                
                foreach(float[] row in inputData)
                {
                    Tensor<float> t1 = new DenseTensor<float>(row,dimentions);
                    var inputs = new List<NamedOnnxValue>(){NamedOnnxValue.CreateFromTensor<float>(name, t1)};
                    using (var results = session.Run(inputs))
                    {
                        float[] output = results.ToArray()[0].AsEnumerable<float>().ToArray();                                
                        int maxIndex = output.ToList().IndexOf(output.Max()); 
                    }
                }    
            }
            
        }

        private static float[] ConvertSentence(string sentence)
        {
            throw new NotImplementedException();
        }

        private static IEnumerable<string> GetSentenses()
        {
            yield return "";
            yield return "";
            yield return "";
            yield return "";
            yield return "";
        }


        private static void LoadIMDBWordIndex()
        {
            JObject o1 = JObject.Parse(File.ReadAllText("imdb_word_index.json"));

            foreach (var x in o1)
            {
                string name = x.Key;
                var value =  x.Value.ToObject<int>();
                word_index[name] = value;
            }
        }
    }

            
       
}
