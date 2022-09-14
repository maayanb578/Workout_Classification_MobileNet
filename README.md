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
class_names = train_dataset.class_names

plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
```

Output:

![image](https://user-images.githubusercontent.com/86894225/190180930-6b645428-e807-4c9e-9b3b-e7ff88846878.png)













