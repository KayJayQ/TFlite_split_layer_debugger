from tflite_runtime import interpreter
import numpy as np
import sys
import ctypes

model_path, index, tensor, outputs = sys.argv[1:5]


inputs = np.load("inputs.npy", allow_pickle=True)

model = interpreter.Interpreter(model_path, experimental_preserve_all_tensors=True)
if outputs == 'None':
    outputs = [detail['name'] for detail in model.get_output_details()]
else:
    outputs = outputs.split(',')

model.allocate_tensors()

input_info = model.get_input_details()
output_info = model.get_output_details()
assert len(inputs) == len(input_info), "Number of inputs mismatchs"

for i in range(len(inputs)):
    model.set_tensor(input_info[i]['index'], inputs[i])

model.invoke()

if int(index) == -1:
    for details in model.get_tensor_details():
        if details['name'] in outputs:
            res_tensor = model.get_tensor(details['index'])
            np.save(f"dump/{details['name'].replace('/','_')}.npy", res_tensor)
    quit()

alt_tensor = np.load(tensor).flatten()

index = int(index)
tensor_need_to_replace = model.tensor(index)
addr, flag = tensor_need_to_replace().__array_interface__['data']
tensor_need_to_replace = None
p = ctypes.c_void_p(addr)
pf = ctypes.cast(p, ctypes.POINTER(np.ctypeslib.as_ctypes_type(alt_tensor.dtype)))
for i in range(alt_tensor.shape[0]):
    pf[i] = alt_tensor[i]

for i in range(len(inputs)):
    model.set_tensor(input_info[i]['index'], inputs[i])
    
model.invoke()

for details in model.get_tensor_details():
    if details['name'] in outputs:
        res_tensor = model.get_tensor(details['index'])
        np.save(f"dump/{details['name'].replace('/','_')}.npy", res_tensor)
    
