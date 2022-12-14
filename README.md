# Workout Exercise Classification with MobileNet

Welcome to this project, where you'll be using transfer learning on a pre-trained CNN to build a workout exercise classifier!



A pre-trained model is a network that's already been trained on a large dataset and saved, which allows you to use it to customize your own model cheaply and efficiently. The one you'll be using, MobileNet, was designed to provide fast and computationally efficient performance. It's been pre-trained on ImageNet, a dataset containing over 14 million images and 1000 classes.

By the end of this assignment, you will be able to:

- Create a dataset from a directory
- Preprocess and augment data using the Sequential API
- Adapt a pretrained model to new data and train a classifier using the Functional API and MobileNet
- Fine-tune a classifier's final layers to improve accuracy

## Table of Contents

- [1 - Packages](#1)
    - [1.1 Create the Dataset and Split it into Training and Validation Sets] (#1-1)
- [2 - Preprocess and Augment Training Data](#2)
    - [Exercise 1 - data_augmenter](#ex-1)
- [3 - Using MobileNetV2 for Transfer Learning](#3)
    - [3.1 - Inside a MobileNetV2 Convolutional Building Block](#3-1)
    - [3.2 - Layer Freezing with the Functional API] (#3-2)
        - [Exercise 2 - alpaca_model](#ex-2)
    - [3.3 - Fine-tuning the Model] (#3-3)
        - [Exercise 3] (#ex-3)

<a name='1'></a>
## 1 - Packages


```python
import os
import cv2
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from collections import deque
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import MobileNet


from tensorflow.keras.preprocessing.image import ImageDataGenerator
```

<a name='1-1'></a>
### 1.1 Create the Dataset and Split it into Training and Validation Sets

When training and evaluating deep learning models in Keras, generating a dataset from image files stored on disk is simple and fast. Call `image_data_set_from_directory()` to read from the directory and create both training and validation datasets.

If you're specifying a validation split, you'll also need to specify the subset for each portion. Just set the training set to `subset='training'` and the validation set to `subset='validation'`.

You'll also set your seeds to match each other, so your training and validation sets don't overlap. :)


```python
# hyperparameter
height = 256
width = 256
channels = 3
batch_size = 64
img_shape = (height, width, channels)
img_size = (height, width)

train_ds = tf.keras.utils.image_dataset_from_directory(DATA_DIR,
                                                    labels = 'inferred',
                                                    label_mode = 'categorical',
                                                    validation_split = 0.1,
                                                    subset = 'training',
                                                    image_size = img_size,
                                                    shuffle = True,
                                                    batch_size = batch_size,
                                                    seed = 127
                                                    )

val_ds = tf.keras.utils.image_dataset_from_directory(DATA_DIR,
                                                    labels = 'inferred',
                                                    label_mode = 'categorical',
                                                    validation_split = 0.1,
                                                    subset = 'validation',
                                                    image_size = img_size,
                                                    shuffle = True,
                                                    batch_size = batch_size,
                                                    seed = 127
```

Output: 

Found 1942 files belonging to 8 classes.

Using 1748 files for training.

Found 1942 files belonging to 8 classes.

Using 194 files for validation.

['bench press', 'biceps curl', 'chest fly machine', 'deadlift', 'incline bench press', 'lat pulldown', 'push-up', 'tricep pushdown']


Now let's take a look at some of the images from the training set:


```python
#Defing a function to see images
def show_img(data):
    plt.figure(figsize=(10,10))
    for images, labels in data.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            ax.imshow(images[i].numpy().astype("uint8"))
            ax.axis("off")
```

Output:

![image](https://user-images.githubusercontent.com/86894225/190180930-6b645428-e807-4c9e-9b3b-e7ff88846878.png)





## 2 - Preprocess and Augment Training Data


 your model learn the data better, it's standard practice to augment the images by transforming them, i.e., randomly flipping and rotating them. Keras' Sequential API offers a straightforward method for these kinds of data augmentations, with built-in, customizable preprocessing layers. These layers are saved with the rest of your model and can be re-used later.
 
 
 
 As always, you're invited to read the official docs, which you can find for data augmentation [here](https://www.tensorflow.org/tutorials/images/data_augmentation).

Implement data augmentation. Use a `Sequential` keras model composed of 4 layers.

```python
data_augmentation = tf.keras.Sequential([tf.keras.layers.RandomFlip("horizontal"),
                                         tf.keras.layers.GaussianNoise(10),
                                         tf.keras.layers.RandomContrast(0.1),
                                         tf.keras.layers.RandomZoom(0.2)
                                        ])

train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y))
```



`
```python

#Plotting the images in dataset
show_img(train_ds)
```
 
 
 
Output:

 
 ![image](https://user-images.githubusercontent.com/86894225/190185605-de301608-abfd-4caf-828c-4b845ca23901.png)




##3 - Using MobileNet for Transfer Learning

MobileNet was trained on ImageNet and is optimized to run on mobile and other low-power applications. It's very efficient for object detection and image segmentation tasks, as well as classification tasks like this one. The architecture has three defining characteristics:

*   Depthwise separable convolutions
*   Thin input and output bottlenecks between layers
*   Shortcut connections between bottleneck layers

MobileNet uses depthwise separable convolutions as efficient building blocks. Traditional convolutions are often very resource-intensive, and  depthwise separable convolutions are able to reduce the number of trainable parameters and operations and also speed up convolutions in two steps:

1. The first step calculates an intermediate result by convolving on each of the channels independently. This is the depthwise convolution.
2. In the second step, another convolution merges the outputs of the previous step into one. This gets a single result from a single feature at a time, and then is applied to all the filters in the output layer. This is the pointwise convolution, or: Shape of the depthwise convolution X Number of filters.


```python

pre_trained = MobileNet(weights='imagenet', include_top=False, input_shape=img_shape, pooling='avg')

for layer in pre_trained.layers:
    layer.trainable = False
```


output:

Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/mobilenet/mobilenet_1_0_224_tf_no_top.h5
17227776/17225924 [==============================] - 0s 0us/step
17235968/17225924 [==============================] - 0s 0us/step



```python

x = pre_trained.output
x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
predictions = tf.keras.layers.Dense(len(labels), activation='softmax')(x)

workout_model = tf.keras.models.Model(inputs = pre_trained.input, 
                                      outputs = predictions
                                     )

workout_model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy']
                     )

workout_model.summary()
```


![image](https://user-images.githubusercontent.com/86894225/193568811-55c1e054-c19f-43be-a3f1-05f45b1bcba4.png)


## 4 - Early Stoping and history 

Early stopping is a form of regularization used to avoid overfitting on the training dataset. Early stopping keeps track of the validation loss, if the loss stops decreasing for several epochs in a row the training stops.



```python
early_stopping_callback = EarlyStopping(monitor = 'val_loss', 
                                        patience = 5, 
                                        mode = 'min', 
                                        restore_best_weights = True
                                       )

history = workout_model.fit(train_ds,
                            validation_data = val_ds,
                            epochs = 100,
                            callbacks = [early_stopping_callback]
                           )
```


In the next sections, you'll see how you can use a pretrained model to modify the classifier task so that it's able to recognize workout exercises. You can achieve this in three steps:

1. Delete the top layer (the classification layer).
    - Set include_top in base_model as False.
2. Add a new classifier layer.
    - Train only one layer by freezing the rest of the network.
    - As mentioned before, a single neuron is enough to solve a binary classification problem.
3. Freeze the base model and train the newly-created classifier layer.
    - Set base model.trainable=False to avoid changing the weights and train only the new layer.
    - Set training in base_model to False to avoid keeping track of statistics in the batch norm layer.


# Plot the training and validation accuracy:


```python
evaluate = workout_model.evaluate(val_ds)

epoch = range(len(history.history["loss"]))
plt.figure()
plt.plot(epoch, history.history['loss'], 'red', label = 'train_loss')
plt.plot(epoch, history.history['val_loss'], 'blue', label = 'val_loss')
plt.plot(epoch, history.history['accuracy'], 'orange', label = 'train_acc')
plt.plot(epoch, history.history['val_accuracy'], 'green', label = 'val_acc')
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
```


Output:


4/4 [==============================] - 1s 72ms/step - loss: 0.1749 - accuracy: 0.9536
<matplotlib.legend.Legend at 0x7f17879d2210>

![image](https://user-images.githubusercontent.com/86894225/191490823-a65a646a-f5c6-4954-aa43-9648f409f4f6.png)



# Save the result of the trained model


```python

# Save Model
workout_model.save('workout_model')

# Save .h5 model
workout_model.save('workout_model.h5')

# Convert the model to tflite
converter = tf.lite.TFLiteConverter.from_saved_model('./workout_model')
tflite_model = converter.convert()

# Save the tflite model
with open('workout_model.tflite', 'wb') as f:
    f.write(tflite_model)
```


