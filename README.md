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
            <a href="#">pip install onnxruntime</a>
         </li>
         <li>
            <a href="#">pip install -U tf2onnx</a>
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
 python -m tf2onnx.convert     --input frozen_model.pb   --inputs XR:0  --outputs MyOutput:0    --output model.onnx    --verbose<br/> 
<br/>
 <b>Run XOR C# project</b><br/>
