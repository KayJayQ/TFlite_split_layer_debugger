import argparse
from tflite_runtime import interpreter
import numpy as np
import os

model = None
path = ""
inputs = []

def load_model(model_path):
    global model
    global path
    path = model_path
    model = interpreter.Interpreter(model_path, experimental_preserve_all_tensors=True)
    
def export_input_tensors(input_tensors):
    global inputs
    if input_tensors is None:
        input_info = [[t['shape'], t['dtype']] for t in model.get_input_details()]
        for shape, dtype in input_info:
            inputs.append(np.random.random(shape).astype(dtype))
    else:
        for input_tensor in input_tensors.split(','):
            inputs.append(np.load(input_tensor))
    np.save("inputs.npy", inputs, allow_pickle=True)
            
            
def invoke(execution = 0, index = None, tensor = None, outputs=None, dumpall=False):
    with open("in.txt",'w') as f:
        f.write(str(execution))
    if index is None:
        index = -1
        tensor = '_'
    if outputs is None:
        outputs = 'None'
    os.system(f"python3 run.py {path} {index} {tensor} {outputs} < in.txt > out.txt")
    
def parse_args(assign, tensor):
    if assign is None or tensor is None:
        return (None, None, 0)
    
    index = None
    if not assign is None and not tensor is None:
        for details in model.get_tensor_details():
            if details['name'] == assign:
                t = np.load(tensor)
                index = details['index']
                break
            
    execution = None
    with open("out.txt",'r') as f:
        current_index = -1
        for line in f.readlines():
            if "Execution" in line:
                current_index += 1
            else:
                if assign in line:
                    break
    if current_index >= 0:
        execution = current_index
    
    return (index, tensor, execution)
    
    

def run():
    parser = argparse.ArgumentParser(description='This script allows user to insert\
        a custom tensor into tflite model and start inferencing from that layer.')
    parser.add_argument("model", type=str, metavar='g', nargs=1, help="tflite model path")
    parser.add_argument("-i", action="store", help = "Input tensors, use delimiter ','")
    parser.add_argument("-l", action="store_true", help = "List all tensors writtenble")
    parser.add_argument("--assign", action="store", help = "Tensor name needed to be assigned")
    parser.add_argument("--tensor", action="store", help = "Assigned tensor path")
    parser.add_argument("--outputs", action="store", help = "Select which tensors will be dumped, use delimeter ','")
    parser.add_argument("--dumpall", action="store_true", help = "Dump all tensors after inferencing")
    args = parser.parse_args()

    load_model(args.model[0])
    export_input_tensors(args.i)
    
    invoke()
    
    if args.l:
        with open("out.txt",'r') as f:
            result = f.readlines()
        print(''.join(result))
        for details in model.get_tensor_details():
            print(details['name'], '\t', details['index'])
        quit()
        
    index, tensor, execution = parse_args(args.assign, args.tensor)
    invoke(execution, index, tensor, args.outputs, False)
    print("Done, please check 'dump' directory")
    

if __name__ == '__main__':
    run()