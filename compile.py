import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

converter = tf.lite.TFLiteConverter.from_saved_model("my_model")
tflite_model = converter.convert()
open("my_model.tflite", "wb").write(tflite_model)
