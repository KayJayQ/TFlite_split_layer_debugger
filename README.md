# TFlite_split_layer_debugger
A tflite forward tools that can get and modify the intermediate tensor to verify the output of tflite model.


I have a tflite model. Now I want to print the quantization information (scale and zero point) in some nodes inside this model. This is a simple LSTM network, with 3 lstm cell and each has 31 timesteps. You can use following lambda expression to get wanted node names: lambda u,ts:f"generator/static-rnn-{u+1}/rnn{'' if u == 0 else '_'+str(u)}/lstm_cell_{u+1}/lstm_cell/mul_{3*t+1}". The u is in the range of 0 to 2 and ts means timestep is in the range of 0 to 30. Can you write a python program to display the quantization information of wanted nodes?
