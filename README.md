# IADAI201-1000066-Dipika_Patra
To create this **Pose Detection Project**, I researched how pose detection works, explored tools like MediaPipe and TensorFlow, and then coded in Python using Visual Studio Code (VS Code). I installed the necessary Python libraries—like NumPy, Pandas, OpenCV, and TensorFlow—via command prompt. To organize everything, I set up Jupyter Notebook and Python extensions in VS Code. I collected 12 videos (4 for each gesture: clap, walk, and run) from YouTube, converted them to AVI format, and used these to train and test my model. The final model achieved 100% accuracy during testing, and the project can be run and terminated by pressing the "Q" key.

Here’s a breakdown of the project’s code and workflow, including an explanation of each step, code snippet, and the future scope.



### Step 1: Import Libraries and Initialize Pose Detection
First, we load the required libraries—OpenCV for video processing, MediaPipe for pose detection, and CSV for saving data. We also initialize the MediaPipe Pose object for pose detection.

```python
import cv2
import mediapipe as mp
import csv

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils
```

This initialization enables us to use MediaPipe's pose model, which can detect 33 landmarks on the body.

---

### Step 2: Process Video Frames and Extract Pose Landmarks
We open the training video file, "walk3.avi", and set up a CSV file to store pose landmarks data for each frame.

```python
cap = cv2.VideoCapture('walk3.avi')
with open('pose_landmarks.csv', mode='a', newline='') as f:
    csv_writer = csv.writer(f)
    headers = ['frame', 'label'] + [f'{axis}_{i}' for i in range(33) for axis in ['x', 'y', 'z', 'visibility']]
    csv_writer.writerow(headers)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        # ...
```

For each frame, we convert it to RGB for MediaPipe, extract pose landmarks, and write them to `pose_landmarks.csv`. Each landmark has x, y, z, and visibility values, and each row in the CSV file represents a frame with all landmarks for that frame.

---

### Step 3: Normalize Landmarks
After saving raw landmarks data, we normalize it by anchoring each coordinate relative to the left hip (landmark 23). This step helps create consistency regardless of the person's position.

```python
import pandas as pd
df = pd.read_csv('pose_landmarks.csv')
for i in range(33):
    df[f'x_{i}'] -= df['x_23']
    df[f'y_{i}'] -= df['y_23']
df.to_csv('pose_landmarks_normalized.csv', index=False)
```



### Step 4: Split Data into Training and Testing Sets
Now, we split our normalized dataset into training and testing sets, reserving 20% for testing. This split is essential for evaluating the model's accuracy.

```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=['frame', 'label'])
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

### Step 5: Train a Simple Neural Network Model
We define and train a neural network model using TensorFlow and Keras. This model learns to classify poses based on the extracted landmarks.

```python
import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(len(set(y)), activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

model.save('my_model.keras')
```

The model is designed to predict the type of pose, with layers optimized for detecting patterns in the input data. We save the model to be used later for testing.

---

### Step 6: Real-Time Pose Detection and Testing
In this step, we load the trained model and use it on a test video to classify poses in real-time. We open a new video and process each frame with MediaPipe Pose, extracting landmarks and using the model to predict the pose.

```python
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('my_model.keras')
label_map = {0: "smile", 1: "clap", 2: "walk"}

video_path = 'walk4.avi'
cap = cv2.VideoCapture(video_path)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        # ...
```

For each detected pose, we display the predicted label on the video frame. This real-time feedback is key to evaluating how well the model generalizes.

---

### Future Scope
- **Expand Gesture Library:** Add more gestures and activities, improving the model's versatility.
- **Integrate Multi-Person Detection:** Adapt the system to track multiple people, potentially useful in group exercises or sports.
- **Enhanced Real-Time Performance:** Optimize processing speed, so it works on mobile devices and real-time applications with minimal latency.
- **Transfer Learning with Pretrained Models:** Fine-tune with larger pre-trained models for better accuracy, especially useful in complex pose classification tasks.

---

This project demonstrates how to extract pose data, train a model, and classify gestures in real-time. Future work could turn this into a powerful tool for applications like fitness tracking, sign language recognition, or sports analytics.

Output:
![Screenshot 2024-11-14 210957](https://github.com/user-attachments/assets/45578847-84b8-4929-929d-b7c2561af1b8)
![Screenshot 2024-11-14 211312](https://github.com/user-attachments/assets/edaa126c-eda2-43e3-957c-2eac2f3fcade)

Link to the github repositary:https://github.com/dipikapatra14/IADAI201-1000066-Dipika_Patra

Screeshots of my confusion_matrices and my F1 score :-
![WhatsApp Image 2024-11-12 at 22 05 12_307bfa7f](https://github.com/user-attachments/assets/4339c078-e99b-4b16-9cc0-65387d969d1f)

