import tflite_runtime.interpreter as tflite
import numpy as np
from PIL import Image, ImageDraw
import os

image = Image.open("test3.png")
input_data = np.asarray(image)
r, g, b, a = np.rollaxis(input_data, axis=-1)
input_data = np.dstack([r, g, b])
input_data = np.array([input_data])
print(input_data.shape)

interpreter = tflite.Interpreter("detect.tflite")

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.resize_tensor_input(input_details[0]['index'],
                                [1, 300, 300, 3])
interpreter.allocate_tensors()

# input_data = np.array(np.random.random_sample((1, 300, 300, 3)),
#                       dtype=np.uint8)

interpreter.set_tensor(input_details[0]['index'],
                       input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)

image = Image.fromarray(input_data[0])
draw = ImageDraw.Draw(image)
for i in output_data[0]:
    draw.line((int(i[1]*300), int(i[0]*300), int(i[1]*300), 300-int(i[2]*300)),
              fill=0xff0000, width=1)
    '''
    draw.line((int(i[1]*300), int(i[0]*300), 300-int(i[3]*300), int(i[0]*300)),
              fill=0x00ff00, width=1)
    draw.line((300-int(i[3]*300), 300-int(i[2]*300),
               int(i[1]*300), 300-int(i[2]*300)),
              fill=0x0000ff, width=1)
    draw.line((300-int(i[3]*300), 300-int(i[2]*300),
               int(i[1]*300), 300-int(i[2]*300)),
              fill=0xffff00, width=1)
    '''

# image.show()
image.save("test4.png")
os.system("feh test4.png")
