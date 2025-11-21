==============================================================

# Brain Tumor Classification Using Deep Learning (ResNet50)

==============================================================

**Project Overview**
---

This project aims to classify kidney images into four categories: Cyst, Normal, Stone, Tumor using deep learning techniques. A pretrained ResNet50 model is fine-tuned to achieve high accuracy on the task. The model can assist radiologists and hospitals with automated image diagnosis.

**Dataset**
---

The dataset contains images categorized into four classes: Cyst, Normal, Stone, Tumor.

Images are organized into training, validation, and test sets, with subfolders for each class.

Image sizes vary but are normalized for the model.

**Preprocessing**
---

ConvertToRGB ensures all images are in RGB format.

Training Augmentations include resizing, random crop, horizontal flip, rotation, color jitter, Gaussian blur, and normalization.

Validation/Test Transformations include resizing, center crop, and normalization only.

Data Loaders ensure batch loading with deterministic seeding.

**Model Architecture**
---

Base model: ResNet50 pretrained on ImageNet.

Fully connected layer modified to output 4 classes.

Loss function: CrossEntropyLoss with class weights.

Optimizer: Adam with learning rate 0.0001.

Scheduler: StepLR to reduce learning rate every 5 epochs.

**Training**
---
Epochs: 30

Early Stopping: patience of 7 epochs.

Metrics tracked: training and validation loss, accuracy.

Best model: saved based on minimum validation loss.

Progress monitoring: real-time using progress bars.

**Evaluation**
---

Evaluated on the test set using:

Accuracy

Loss

Precision

Recall

F1-Score

Confusion matrix visualized as percentages per class.

Sample predictions include true label, predicted label, and confidence scores.

**Results**
---

Test Loss: ~0.0510

Test Accuracy: 0.98%

Precision (Macro): ~0.97%

Recall (Macro): ~0.98%

F1 Score (Macro): ~0.97%

The confusion matrix highlights class-wise performance and misclassifications.

**Sample Predictions**
---

Random test images were visualized with true label, predicted label, and confidence score.

Correct predictions are highlighted in green, incorrect in red.

**Usage**
---

Clone the repository and install dependencies.

Update dataset paths as needed.

Train the model or load pretrained weights.

Evaluate on the test set or visualize predictions.

**Requirements**
---

Python >= 3.8

PyTorch >= 2.0

torchvision

numpy

pandas

matplotlib

seaborn

scikit-learn

PIL (Pillow)

tqdm
