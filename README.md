# yolov5
yolov5 learning report

# 1. Introduction
## 1.1. Project Background
In mountainous regions, encountering wild animals such as bears and wild boars can be dangerous. In order to enhance safety and respond quickly to these encounters, it is important to have a real-time animal detection system. This project aims to use a deep learning-based object detection model to quickly identify bears and wild boars when encountered in the wild.

The YOLOv5 (You Only Look Once) model is chosen for this task due to its high speed and accuracy, making it ideal for real-time applications. The goal of this project is to train a model that can accurately detect and label bears and wild boars in images, helping to alert individuals to potential threats in a fast and efficient manner.
![image](https://github.com/user-attachments/assets/3cb6a036-550c-41e7-8c18-b3d9614a8cad)
resource : https://youtu.be/3xTDKG5wGVI(MBCNEWS)

## 1.2. Objective
The objective of this project is to develop an object detection system using YOLOv5 to identify bears and wild boars quickly in images. By training the model on a labeled dataset containing images of these animals, we aim to achieve real-time detection and alerting capabilities.



# 2. Related Work
Object detection is a well-established field in computer vision, and the YOLO (You Only Look Once) model is known for its fast and efficient performance. YOLOv5, the latest version of YOLO, offers improvements in speed, accuracy, and model size, making it suitable for applications requiring real-time object detection.

Previous research in wildlife monitoring and animal detection has shown the utility of machine learning models like YOLO in detecting and identifying animals in various environments. This project builds on these methods to focus specifically on bears and wild boars, animals that can pose significant risks to humans in certain environments.

# 3. Dataset Preparation
## 3.1. Dataset Description
For this project, a dataset consisting of images containing bears and wild boars was gathered. These images were collected from various online sources and wildlife datasets. Each image was annotated with bounding boxes around the animals, specifically labeling the bears and wild boars.

The dataset contains a balanced number of images for each class, ensuring that the model receives adequate exposure to both animals. The number of images and the variety of scenarios (different backgrounds, lighting conditions, and animal poses) are essential for improving model robustness.

## 3.2. Data Preprocessing
Before training the model, all images were resized to meet YOLOv5’s input requirements. Each image was labeled using annotation tools such as blacklabel. The data was then split into training, validation, typically allocating 70% for training, 30% for validation.

Additionally, data augmentation techniques were applied to increase the diversity of the dataset. This included random transformations like rotation, cropping, and brightness adjustments to ensure the model could generalize better to different scenarios.
![00000904](https://github.com/user-attachments/assets/68d4ca4f-83d7-42b6-b8da-c6e00022f769)
![00000000](https://github.com/user-attachments/assets/9a88b474-2214-4d41-bd2d-c6c6e11ece4d)


# 4. Model Training
## 4.1. Model Selection
The YOLOv5 model was selected due to its proven performance in object detection tasks, particularly in real-time applications. YOLOv5 provides a good balance between speed and accuracy, which is essential for quick decision-making in real-world situations.

The model works by dividing an image into a grid and predicting bounding boxes and class probabilities for each grid cell, making it highly efficient for detecting multiple objects in one image.

## 4.2. Hyperparameter Configuration
During the training process, various hyperparameters were tuned to optimize the model's performance. These included the learning rate, batch size, and number of epochs. A batch size of 16 was used, with an initial learning rate of 0.01. The model was trained for 300 epochs to ensure convergence.

Additional strategies, such as adjusting the confidence threshold for detection and experimenting with different loss functions, were also explored to improve the model's accuracy and minimize false positives and negatives.

## 4.3. Training Process
The model was trained on a GPU for faster computation. Training logs were monitored using tools like TensorBoard, which helped track the loss and accuracy metrics over time. Any issues, such as overfitting or underfitting, were addressed by adjusting hyperparameters and using techniques like early stopping and regularization.

The total training time was approximately 12 hours, and the model's performance was evaluated after each epoch using validation data to ensure continuous improvement.

![image](https://github.com/user-attachments/assets/93a8af9b-06fb-4eef-80d8-f6e795819c80)
![image](https://github.com/user-attachments/assets/2842a819-38c9-45d7-9009-48505a0a09df)

![image](https://github.com/user-attachments/assets/2c24f6ef-9892-4e50-992e-7898c93fc5db)
![image](https://github.com/user-attachments/assets/dffaa119-cb1d-4f8b-9100-5eb79c7972e3)
![image](https://github.com/user-attachments/assets/e4795f34-1042-47cf-89db-6691988c5d8c)


# 5. Results
## 5.1. Model Evaluation
The performance of the YOLOv5 model was evaluated using metrics such as mAP (mean Average Precision), Precision, Recall, and F1-score. The final model achieved the following performance:



mAP (mean Average Precision): 99.3%
Precision: 100% (The ratio of correctly predicted objects to all predicted objects for bears and wild boars)
Recall: 99.7% (The ratio of correctly predicted objects to all actual objects for bears and wild boars)
F1-score: 99.6% (A combined metric considering both Precision and Recall)
These results demonstrate that the model can accurately detect bears and wild boars. Specifically, the bear class achieved 100% Precision and 100% Recall, while the wild boar class recorded 99.3% Precision and 99.5% Recall.

Such high performance indicates that the model generalizes well to unseen data and is highly reliable in real-world applications.
![image](https://github.com/user-attachments/assets/d29e28ab-895a-43b8-8f14-73385fbc0a02)

## 5.2. Result Analysis
Upon testing the model on unseen images, it successfully detected bears and wild boars with high accuracy, even in challenging scenarios like low lighting and dense vegetation. Some misclassifications occurred in cases where the animals were partially obscured or at a great distance, but overall, the model's performance was reliable.

Visualization of the results shows the bounding boxes correctly placed around the animals in most cases, along with class labels (bear or wild boar), demonstrating the model's ability to quickly identify these animals.

https://github.com/user-attachments/assets/3bba6aac-e895-40cf-89fe-b00721572fc2

resource : https://www.youtube.com/shorts/AyaYtN7WoLo (Travel Ninja)

https://github.com/user-attachments/assets/f3aa4ec4-4d47-43f4-8fe2-d092d0a00475

resource : https://www.youtube.com/shorts/wojf4S3N8l8 (호랑상식)

## 5.3. Comparative Analysis
Compared to other object detection models, YOLOv5 outperformed alternatives like Faster R-CNN and SSD in terms of speed and real-time detection. Although these models achieved similar accuracy, YOLOv5's faster inference time made it more suitable for real-time applications, such as field monitoring of wildlife.


# 6. Conclusion
## 6.1. Project Summary
This project successfully developed an object detection system using YOLOv5 to identify bears and wild boars in images. The trained model can now detect and label these animals with high accuracy, which is valuable for real-time monitoring in mountainous regions where such encounters may pose a danger to humans.

## 6.2. Future Work
Future improvements could involve expanding the dataset to include other wildlife species and testing the model under different environmental conditions (e.g., different seasons, weather conditions). Additionally, the model could be deployed in real-time monitoring systems using cameras or drones to provide alerts in case of wildlife encounters.

## 6.3. Practical Applications
This system could be used in various real-world applications, such as wildlife monitoring, forest safety, and outdoor activities. By accurately detecting dangerous animals like bears and wild boars, it can help reduce risks to humans and allow for faster response in potentially hazardous situations.


https://github.com/user-attachments/assets/3bba6aac-e895-40cf-89fe-b00721572fc2


https://github.com/user-attachments/assets/f3aa4ec4-4d47-43f4-8fe2-d092d0a00475


