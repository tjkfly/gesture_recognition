import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import os
import pathlib
import random
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
test_root = pathlib.Path('D:\\code\\PYTHON\\gesture_recognition\\testdata')
print(test_root)

for item in test_root.iterdir():
  print(item)

test_image_paths = list(test_root.glob('*/*'))
test_image_paths = [str(path) for path in test_image_paths]
random.shuffle(test_image_paths)

image_count = len(test_image_paths)
print(image_count)
print(test_image_paths[:10])

test_names = sorted(item.name for item in test_root.glob('*/') if item.is_dir())
print(test_names)
label_test_index = dict((name, index) for index, name in enumerate(test_names))
print(label_test_index)
test_image_labels = [label_test_index[pathlib.Path(path).parent.name]
                    for path in test_image_paths]

print("First 10 labels indices: ", test_image_labels[:10])




data_root = pathlib.Path('D:\code\PYTHON\gesture_recognition\Dataset')
print(data_root)

for item in data_root.iterdir():
  print(item)

all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)

image_count = len(all_image_paths)
print(image_count)
for i in range(10):
  print(all_image_paths[i])

label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
print(label_names)
label_to_index = dict((name, index) for index, name in enumerate(label_names))
print(label_to_index)
all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                    for path in all_image_paths]

print("First 10 labels indices: ", all_image_labels[:10])

# #加载和格式化图片
# img_path = all_image_paths[0]
# print(img_path)

# img_raw = tf.io.read_file(img_path)
# print(repr(img_raw)[:100]+"...")
# img_tensor = tf.image.decode_image(img_raw)

# print(img_tensor.shape)
# print(img_tensor.dtype)

def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [100, 100])
  image /= 255.0  # normalize to [0,1] range
  # image = tf.reshape(image,[100*100*3])
  return image

def load_and_preprocess_image(path,label):
  image = tf.io.read_file(path)
  return preprocess_image(image),label

# image_path = all_image_paths[0]
# label = all_image_labels[0]

# # plt.imshow(load_and_preprocess_image(image_path))
# # plt.grid(False)
# # # plt.xlabel(caption_image(image_path))
# # plt.title(label_names[label].title())
# # plt.show()

# #构建一个 tf.data.Dataset
# ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))
# train_data = ds.map(load_and_preprocess_image).batch(16)

# ts = tf.data.Dataset.from_tensor_slices((test_image_paths, test_image_labels))
# test_data = ts.map(load_and_preprocess_image).batch(16)


# sample = next(iter(train_data))
# print(sample[0].shape, sample[1])
# # print(sample[0][1].shape)
# # plt.imshow(sample[0][1])
# # print(sample[1][1])
# # plt.show()
# # # print(sample[1][1])

# network = keras.Sequential([
#             keras.layers.Conv2D(32,kernel_size=[5,5],padding="same",activation=tf.nn.relu),
#             keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
#             keras.layers.Conv2D(64,kernel_size=[3,3],padding="same",activation=tf.nn.relu),
#             keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
#             keras.layers.Conv2D(64,kernel_size=[3,3],padding="same",activation=tf.nn.relu),
#             keras.layers.Flatten(),
#             keras.layers.Dense(512,activation='relu'),
#             keras.layers.Dropout(0.5),
#             keras.layers.Dense(128,activation='relu'),
#             keras.layers.Dense(10),
# ])
# network.build(input_shape=(None,100,100,3))
# network.summary()

# network.compile(optimizer=optimizers.SGD(lr=0.001),
# 		loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
# 		metrics=['accuracy']
# )
# network.fit(train_data, epochs = 100,validation_data=test_data,validation_freq=2)  
# network.evaluate(test_data)

# tf.saved_model.save(network,'D:\\code\\PYTHON\\gesture_recognition\\model\\')
# print("保存模型成功")



# # Convert Keras model to ConcreteFunction
# full_model = tf.function(lambda x: network(x))
# full_model = full_model.get_concrete_function(
# tf.TensorSpec(network.inputs[0].shape, network.inputs[0].dtype))

# # Get frozen ConcreteFunction
# frozen_func = convert_variables_to_constants_v2(full_model)
# frozen_func.graph.as_graph_def()

# layers = [op.name for op in frozen_func.graph.get_operations()]
# print("-" * 50)
# print("Frozen model layers: ")
# for layer in layers:
#   print(layer)

# print("-" * 50)
# print("Frozen model inputs: ")
# print(frozen_func.inputs)
# print("Frozen model outputs: ")
# print(frozen_func.outputs)

# # Save frozen graph from frozen ConcreteFunction to hard drive
# tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
# 		logdir="D:\\code\\PYTHON\\gesture_recognition\\model\\frozen_model\\",
# 		name="frozen_graph.pb",
# 		as_text=False)
