from imports import *

#Defing a function to see images
def show_img(data):
    plt.figure(figsize=(10,10))
    for images, labels in data.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            ax.imshow(images[i].numpy().astype("uint8"))
            ax.axis("off")

def creat_model(pre_trained, labels):
    x = pre_trained.output
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    predictions = tf.keras.layers.Dense(len(labels), activation='softmax')(x)
    
    return tf.keras.models.Model(inputs = pre_trained.input, 
                                    outputs = predictions
                                    )
    
def model_compile(workout_model):
    workout_model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy']
                     )

    workout_model.summary()
    
    return workout_model

def model_training(workout_model, train_ds, val_ds):
    early_stopping_callback = EarlyStopping(monitor = 'val_loss', 
                                    patience = 5, 
                                    mode = 'min', 
                                    restore_best_weights = True
                                    )

    return workout_model.fit(train_ds,
                            validation_data = val_ds,
                            epochs = 100,
                            callbacks = [early_stopping_callback]
                           )


def video_frame_predict(input_video_file_path, workout_model, img_size, labels):
    
    # Initialize video input
    video_capture = cv2.VideoCapture(input_video_file_path)
    writer = None
    H, W = None, None
    Q = deque(maxlen=128)
    n = 0

    # Loop through each frame in the video
    while True:
        # count the frame
        n += 1
        
        # predict every 5 frame (1, 6, 11, ... etc)
        if n % 5 != 1:
            continue
        
        # read a frame
        success, frame = video_capture.read()
        
        # if frame not read properly then break the loop
        if not success:
            break
        
        # get frame dimensions
        if W is None or H is None:
            (H, W) = frame.shape[:2]
        
        # clone the frame for the output then preprocess the frame for the prediction
        output = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, img_size).astype("float32")
            
        predictions = workout_model.predict(np.expand_dims(frame, axis=0))[0]
        Q.append(predictions)
        
        results = np.array(Q).mean(axis=0)
        i = np.argmax(results)
        label = labels[i]
        
        text = 'activity:{}'.format(label)
        cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 10)
        
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter('output.avi', fourcc, 30, (W, H), True)
            
        writer.write(output)
        
        plt.imshow(output)
        plt.axis('off')
        
        # break the loop if prediction > 90% and video already more than 2 seconds (60 frame)
        if results[i] >= 0.9 and n >= 60:
            break

    print(text)
    print(f'confidence: {results[i]}')
    writer.release()
    video_capture.release()
    

def image_pred(workout_model,img_size, labels):
    Q_image = deque(maxlen=128)

    output_image = image_pred.copy()
    image_pred = cv2.cvtColor(image_pred, cv2.COLOR_BGR2RGB)
    image_pred = cv2.resize(image_pred, img_size).astype("float32")
        
    predictions = workout_model.predict(np.expand_dims(image_pred, axis=0))[0]
    Q_image.append(predictions)

    results = np.array(Q_image).mean(axis=0)
    i = np.argmax(results)
    label = labels[i]

    text = 'activity:{}'.format(label)
    cv2.putText(output_image, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 10)

    plt.imshow(output_image)
    plt.axis('off')

    print(text)
    print(f'confidence: {results[i]}')