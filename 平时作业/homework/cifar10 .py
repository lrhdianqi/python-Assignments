# %%
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

# %%
cifar_10 = tf.keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = cifar_10.load_data()
# %%
# 训练集包含50000张32*32*3的图片
print(train_images.shape)
print(test_images.shape)
# %%
# 标签0-9，50000张图一共50000个没啥说的
len(train_labels)
# %%
train_labels
# %%
df_train_images = train_images[:40000]
df_valid_images = train_images[40000:]
df_train_labels = train_labels[:40000]
df_valid_labels = train_labels[40000:]
# %%
# Code here!
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
# %%
# 画张图看看，像素的范围为[0, 255]
plt.figure()
plt.imshow(df_train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
# %%
# 像素归一化，归一化后的像素范围为[0, 1]
df_train_images = df_train_images / 255.0
df_valid_images = df_valid_images / 255.0
test_images = test_images / 255.0
# %%
# 画出来
for i in range(9):
	# define subplot
	plt.subplot(330+1+i) 
	# plot raw pixel data
	plt.imshow(df_train_images[i])
# show the figure
plt.show()
print(df_train_labels[:8])
# %%
from keras import callbacks
learning_rate_reduction = callbacks.ReduceLROnPlateau(monitor='val_loss', patience=4, factor=0.7,
                                                               min_lr=0.0000000005)  # 这里定义batch_size和epochs
# %%
# 进行一个极其简单的神经网络的搭建
# Flatten层把二维的输入转为一维向量
# Dense层即全连接层，第一层Dense作为隐藏层，激活函数为relu
# 最后一层Dense作为输出，输出每个类别的概率
cnn_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax'),
])
# %%
# 编译模型
# 优化器选择adam（不知道选啥的时候用adam就完事了）
# 损失函数选用SparseCategoricalCrossentropy
cnn_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
# %%
# fit模型，开始训练啦！
cnn_model.fit(df_train_images, df_train_labels, 
            validation_data=(df_valid_images, df_valid_labels),
            epochs=50, batch_size=64,
            callbacks=[learning_rate_reduction], # put your callbacks in a list
            verbose=0)  # turn off training log)
# %%
# 测试模型
test_loss, test_acc = cnn_model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)
# %%
# 这里直接接一个softmax把输出转化成概率
probability_model = tf.keras.Sequential([cnn_model, 
                                         tf.keras.layers.Softmax()])
# %%
predictions = probability_model.predict(test_images)
# %%
predictions[0]
# %%
# argmax把输出最大概率的元素，得到结果
np.argmax(predictions[0])
# %%
test_labels[0]
# %%
# 下面都是可视化
def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                np.array(class_names)[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  #np.array(thisplot)[true_label].set_color('blue')
# %%
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()
# %%
i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()
# %%
# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

# %%
