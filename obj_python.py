import tflite_runtime.interpreter as tflite
from PIL import Image
import numpy as np
import os

"""
Index 0: Locations
Index 1: Classes
Index 2: Scores
Index 3: Number and detections 
"""




def prep_image(pic_path):
    im = Image.open(pic_path)
    resized = im.resize((300, 300))
    a = np.asarray(resized)
    a = np.asarray([a])
    return a


def load_labels(label_path):
    fp = open(label_path)
    label_dictionary = {}
    for line in fp:
        label_parts = line.split()
        label_index = int(label_parts[0])
        label_name = ""
        for i in range(1, len(label_parts)):
            label_name += label_parts[i] + " "
        label_name = label_name[:-1]
        label_dictionary[label_index] = label_name
    fp.close()
    return load_labels

# Load TFLite model and allocate tensors.
interpreter = tflite.Interpreter(model_path='models/ssd_mobilenet/detect.tflite')
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
input_shape = input_details[0]['shape']
input_data = prep_image('pics/friends.jpg')
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print('printing locations')
for obj_ind, obj_val in enumerate(output_data):
    for coord_ind, coord_val in enumerate(obj_val):
        output_data[obj_ind][coord_ind] *= 300

store_classes = interpreter.get_tensor(output_details[1]['index'])[0]
store_scores = interpreter.get_tensor(output_details[2]['index'])[0]


length = len(store_classes)
counter = 0

for i in range(length):
    if(store_classes[i] == 0 and (store_scores[i] > 60)):
        counter += 1


print(counter)


print(output_data[0])
print("\n")

#output_data = interpreter.get_tensor(output_details[1]['index'])
#print('printing classes')
#print(output_data[0])
#print("\n")
#
#output_data = interpreter.get_tensor(output_details[2]['index'])
#print('printing scores')
#print(output_data[0])
#print("\n")
#
#output_data = interpreter.get_tensor(output_details[3]['index'])
#print('num of detections')
#print(output_data[0])
#print("\n")

load_labels('models/ssd_mobilenet/coco_labels.txt')
