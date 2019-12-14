# https://github.com/StephenOman/TensorFlowExamples/blob/master/xor%20nn/xor_nn.py
import tensorflow as tf
import os

X = tf.placeholder(tf.float32, shape=[None,2], name = 'X')
Y = tf.placeholder(tf.float32, shape=[None,1], name = 'Y')
W = tf.Variable(tf.random_uniform([2,2], -1, 1), name = "W")
w = tf.Variable(tf.random_uniform([2,1], -1, 1), name = "w")
W_bias = tf.Variable(tf.zeros([2]), name = "c")
w_bias = tf.Variable(tf.zeros([1]), name = "b")

h = tf.nn.sigmoid(tf.add(tf.matmul(X, W),W_bias))
y_estimated = tf.sigmoid(tf.add(tf.matmul(h,w),w_bias), name = "MyOutput")
loss = tf.reduce_mean(( (Y * tf.log(y_estimated)) + 
       ((1 - Y) * tf.log(1.0 - y_estimated)) ) * -1)
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

INPUT_XOR = [[0,0],[0,1],[1,0],[1,1]]
OUTPUT_XOR = [[0],[1],[1],[0]]

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for epoch in range(10000):
   sess.run(train_step, feed_dict={X: INPUT_XOR, Y: OUTPUT_XOR})
   if epoch % 1000 == 0:
       print(sess.run(y_estimated, feed_dict={X: INPUT_XOR, Y: OUTPUT_XOR}))

output_names = 'MyOutput'
output_graph_def = tf.graph_util.convert_variables_to_constants(
           sess, # The session is used to retrieve the weights
           tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes
           output_names.split(",") # The output node names are used to select the usefull nodes
       )

absolute_model_dir = os.getcwd()+"/"
output_graph = absolute_model_dir + "frozen_model.pb"

with tf.gfile.GFile(output_graph, "wb") as f:
       f.write(output_graph_def.SerializeToString())

# https://github.com/onnx/tensorflow-onnx
# python -m tf2onnx.convert     --input frozen_model.pb   --inputs X:0  --outputs MyOutput:0    --output modelxor.onnx    --verbose




