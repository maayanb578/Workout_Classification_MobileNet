from imports import *
from functions import *

# hyperparameter
height = 256
width = 256
channels = 3
batch_size = 64
img_shape = (height, width, channels)
img_size = (height, width)


DATA_DIR = 'workout_exercises_recognition/workoutexercises-dataset_images'

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
                                                    )

labels = train_ds.class_names
print(labels)

with open('workout_label.txt', 'w') as f:
    for workout_class in labels:
        f.write(f'{workout_class}\n')
    

data_augmentation = tf.keras.Sequential([tf.keras.layers.RandomFlip("horizontal"),
                                         tf.keras.layers.GaussianNoise(10),
                                         tf.keras.layers.RandomContrast(0.1),
                                         tf.keras.layers.RandomZoom(0.2)
                                        ])

train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y))

#Plotting the images in dataset
show_img(train_ds)

# load pre-trained MobileNet
pre_trained = MobileNet(weights='imagenet', include_top=False, input_shape=img_shape, pooling='avg')

for layer in pre_trained.layers:
    layer.trainable = False

workout_model = creat_model(pre_trained, labels)

model_compile(workout_model)

history = model_training(workout_model, train_ds, val_ds)

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


random_classes_names = random.choice(os.listdir('workout_exercises_recognition/workoutexercises-dataset_videos/workout'))
random_file = random.choice(os.listdir(f'workout_exercises_recognition/workoutexercises-dataset_videos/workout/{random_classes_names}'))
print(f'{random_classes_names}/{random_file}')

# Construct the input video file path
input_video_file_path = f'/workout_exercises_recognition/workoutexercises-dataset_videos/workout/{random_classes_names}/{random_file}'

video_frame_predict(input_video_file_path, workout_model, img_size, labels)

image_pred = cv2.imread('mypic.jpeg')

image_pred(workout_model,img_size, labels)
