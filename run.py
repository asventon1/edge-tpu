import tensorflow as tf
import tflite_runtime.interpreter as tflite
import numpy as np

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]


train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(100)

interpreter = tflite.Interpreter("my_model.tflite", experimental_delegates=[
    tflite.load_delegate('libedgetpu.so.1')])

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.resize_tensor_input(input_details[0]['index'],
                                [100, 28, 28, 1])
interpreter.allocate_tensors()

input_shape = input_details[0]['shape']
good_counter = 0
print(input_shape)
for image, label in test_ds:
    image2 = tf.cast(image, tf.float32)
    interpreter.set_tensor(input_details[0]['index'],
                           image2)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    for i in range(len(output_data)):
        # print(tf.math.argmax(output_data[i]).numpy(), label[i].numpy())
        if tf.math.argmax(output_data[i]).numpy() == label[i].numpy():
            good_counter += 1

print(good_counter/100)
