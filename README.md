"# MagAISimpleApproach" 

<label>Prerequisites</label>
<ul>  
    <li>
    <a href="#">Python 3.6.8</a>
     <ul>
        <li>
            <a href="#">pip install tensorflow==1.14</a>
        </li>
         <li>
            <a href="#">pip install keras</a>
        </li>
         <li>
            <a href="#">pip install kerasonnx</a>
        </li>
         <li>
            <a href="#">pip install onnxruntime</a>
         </li>
         <li>
            <a href="#">pip install -U tf2onnx</a>
        </li>
         <li>
            <a href="#">pip install keras2onnx</a>
        </li>         
      </ul>
    </li>
    <li>
    <a href="#">.netcore 3.1</a>
     <ul>
        <li>
           <a href="#">dotnet add package Microsoft.ML.OnnxRuntime</a>
        </li>
        <li>
           <a href="#">dotnet add package System.Drawing.Common (in examples with images)</a>
        </li>
      </ul>
    </li>
 </ul>
 <br/>
 <b>Run XOR python project</b><br/>
 cd pythonexamples<br/>
 python ./xor.py<br/>
 Convert the freozen model to an onnx model<br/>
 python -m tf2onnx.convert     --input frozen_model.pb   --inputs XR:0  --outputs MyOutput:0    --output modelxor.onnx    --verbose<br/> 
<br/>
 <b>Run XOR C# project</b><br/> 
 cd ../csharpxor<br/>
 dotnet add package Microsoft.ML.OnnxRuntime<br/>
 dotnet restore<br/>
 dotnet build<br/> 
 dotnet run<br/>
 
 <br/><b>Run XOR python project</b><br/>
 cd ../pythonexamples<br/>
 python ./kerasimgclassif.py<br/>
  python ./kerasimgclassif_save_onnx_best_model.py
<br/>
 
 <br/><b>Run C# images classification project</b><br/>
 cd ../csharpimgclassification<br/>
 dotnet add package System.Drawing.Common<br/>
 dotnet add package Microsoft.ML.OnnxRuntime<br/>
 dotnet restore<br/>
 dotnet build<br/> 
 dotnet run<br/>
