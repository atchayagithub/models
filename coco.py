pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu


pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# -------------------- Paths --------------------
dataDir = 'coco'
train_img_dir = os.path.join(dataDir, 'images', 'train2017')
train_ann_file = os.path.join(dataDir, 'annotations', 'instances_train2017.json')
val_img_dir = os.path.join(dataDir, 'images', 'val2017')
val_ann_file = os.path.join(dataDir, 'annotations', 'instances_val2017.json')

# -------------------- Load COCO Dataset --------------------
from torchvision.datasets import CocoDetection

train_coco = CocoDetection(root=train_img_dir, annFile=train_ann_file)
val_coco = CocoDetection(root=val_img_dir, annFile=val_ann_file)

print(f"Number of training images: {len(train_coco)}")
print(f"Number of validation images: {len(val_coco)}")

# -------------------- Show some images --------------------
def show_images(dataset, num=5):
    plt.figure(figsize=(15,5))
    for i in range(num):
        img, ann = dataset[i]
        plt.subplot(1, num, i+1)
        plt.imshow(np.array(img))
        plt.axis('off')
    plt.show()

show_images(train_coco, num=5)

# -------------------- Augmentation --------------------
augment_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.5, contrast=0.5),
    transforms.ToTensor()
])

val_transform = transforms.Compose([transforms.ToTensor()])

class CocoDatasetWithAug(CocoDetection):
    def __init__(self, root, annFile, transform=None):
        super().__init__(root=root, annFile=annFile)
        self.transform = transform

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        if self.transform:
            img = self.transform(img)
        return img, target

train_dataset = CocoDatasetWithAug(train_img_dir, train_ann_file, transform=augment_transform)
val_dataset = CocoDatasetWithAug(val_img_dir, val_ann_file, transform=val_transform)

print(f"Training images after augmentation: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")

# -------------------- Normalize Images --------------------
normalize_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

train_dataset_norm = CocoDatasetWithAug(train_img_dir, train_ann_file, transform=normalize_transform)
val_dataset_norm = CocoDatasetWithAug(val_img_dir, val_ann_file, transform=normalize_transform)

# -------------------- Use a subset for faster training --------------------
train_dataset_norm = Subset(train_dataset_norm, range(500))
val_dataset_norm = Subset(val_dataset_norm, range(100))

# -------------------- DataLoader --------------------
train_loader = DataLoader(train_dataset_norm, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset_norm, batch_size=8, shuffle=False)

# -------------------- Simple CNN for Classification --------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=80):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.MaxPool2d(2,2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*32*32, 256),  # Adjust input size if image size changes
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    def forward(self,x):
        x = self.features(x)
        x = self.classifier(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model = SimpleCNN(num_classes=80).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

# -------------------- Train CNN --------------------
num_epochs = 2
for epoch in range(num_epochs):
    cnn_model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, targets in train_loader:
        imgs = imgs.to(device)
        labels = torch.tensor([t[0]['category_id'] for t in targets], dtype=torch.long).to(device)
        optimizer.zero_grad()
        outputs = cnn_model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted==labels).sum().item()
    train_acc = 100*correct/total
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")

# -------------------- Evaluate CNN --------------------
cnn_model.eval()
correct, total = 0,0
with torch.no_grad():
    for imgs, targets in val_loader:
        imgs = imgs.to(device)
        labels = torch.tensor([t[0]['category_id'] for t in targets], dtype=torch.long).to(device)
        outputs = cnn_model(imgs)
        _, predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted==labels).sum().item()
val_acc = 100*correct/total
print(f"Validation Accuracy: {val_acc:.2f}%")

# -------------------- Faster R-CNN --------------------
faster_rcnn = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 80 + 1  # 80 classes + background
in_features = faster_rcnn.roi_heads.box_predictor.cls_score.in_features
faster_rcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
faster_rcnn = faster_rcnn.to(device)
optimizer_rcnn = torch.optim.Adam(faster_rcnn.parameters(), lr=0.0001)

print("Faster R-CNN ready to train (training loop omitted for brevity)")



-------------------------------------------------------------------------------------------------------------------------------------

pip install tf-slim
pip install tensorflow-object-detection-api


import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from pycocotools.coco import COCO
import random

# -------------------- Paths --------------------
dataDir = 'coco'
train_img_dir = os.path.join(dataDir, 'images', 'train2017')
val_img_dir = os.path.join(dataDir, 'images', 'val2017')
train_ann_file = os.path.join(dataDir, 'annotations', 'instances_train2017.json')
val_ann_file = os.path.join(dataDir, 'annotations', 'instances_val2017.json')

# -------------------- Load COCO annotations --------------------
train_coco = COCO(train_ann_file)
val_coco = COCO(val_ann_file)

train_ids = list(train_coco.imgs.keys())[:500]  # subset for speed
val_ids = list(val_coco.imgs.keys())[:100]

print(f"Number of training images: {len(train_ids)}")
print(f"Number of validation images: {len(val_ids)}")

# -------------------- Helper Functions --------------------
def load_image_and_label(coco, img_id, img_dir):
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    label = anns[0]['category_id']  # first category
    img_info = coco.loadImgs(img_id)[0]
    path = os.path.join(img_dir, img_info['file_name'])
    img = Image.open(path).convert('RGB')
    img = img.resize((128,128))
    return np.array(img), label

def prepare_dataset(coco, img_ids, img_dir):
    images = []
    labels = []
    for img_id in img_ids:
        img, label = load_image_and_label(coco, img_id, img_dir)
        images.append(img)
        labels.append(label)
    images = np.array(images, dtype='float32') / 255.0  # normalize
    labels = np.array(labels)
    return images, labels

X_train, y_train = prepare_dataset(train_coco, train_ids, train_img_dir)
X_val, y_val = prepare_dataset(val_coco, val_ids, val_img_dir)

print(f"Training data shape: {X_train.shape}, labels shape: {y_train.shape}")

# -------------------- Plot sample images --------------------
plt.figure(figsize=(15,5))
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(X_train[i])
    plt.axis('off')
plt.show()

# -------------------- Data Augmentation --------------------
data_augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomContrast(0.2),
])

X_train_aug = data_augment(X_train)

# -------------------- CNN Classification --------------------
num_classes = 80

cnn_model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(128,128,3)),
    tf.keras.layers.Conv2D(32,3,activation='relu',padding='same'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,3,activation='relu',padding='same'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128,3,activation='relu',padding='same'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

cnn_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

cnn_model.summary()

history = cnn_model.fit(X_train_aug, y_train,
                        validation_data=(X_val, y_val),
                        epochs=5,
                        batch_size=16)

# -------------------- Plot CNN Accuracy --------------------
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# -------------------- Faster R-CNN Object Detection --------------------
# Using TensorFlow Model Zoo pre-trained Faster R-CNN
# Download a pre-trained model, e.g., 'faster_rcnn_resnet50_v2_640x640_coco17_tpu-8'
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md

import tensorflow_hub as hub

print("Loading pre-trained Faster R-CNN from TensorFlow Hub...")
detector = hub.load("https://tfhub.dev/tensorflow/faster_rcnn/openimages_v4/inception_resnet_v2/1")

# -------------------- Run detection on a sample image --------------------
sample_img = X_val[0]
sample_input = tf.convert_to_tensor(sample_img[tf.newaxis, ...], dtype=tf.float32)

result = detector(sample_input)
print("Detection result keys:", result.keys())

# Draw boxes on image
def draw_boxes(image, boxes, scores, classes, max_boxes=5, min_score=0.3):
    plt.figure(figsize=(8,8))
    plt.imshow(image)
    for i in range(min(boxes.shape[0], max_boxes)):
        if scores[i] >= min_score:
            y1,x1,y2,x2 = boxes[i]
            plt.gca().add_patch(plt.Rectangle((x1*128,y1*128),
                                              (x2-x1)*128, (y2-y1)*128,
                                              fill=False, color='red', linewidth=2))
            plt.text(x1*128, y1*128, str(int(classes[i])), color='yellow', fontsize=12)
    plt.axis('off')
    plt.show()

draw_boxes(sample_img, result['detection_boxes'][0].numpy(),
           result['detection_scores'][0].numpy(),
           result['detection_class_entities'][0].numpy())
