import tensorflow as tf
import tensorflow_datasets as tfds

# Construct a tf.data.Dataset
ds = tfds.load('open_images_v4', split='train', shuffle_files=True,
               try_gcs=True)

# Build your input pipeline
ds = ds.shuffle(10000000).batch(512)
