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