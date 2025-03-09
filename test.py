
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Ensure these variables are defined
# true_labels: True labels for the images
# predicted_labels: Predicted labels by the ResNet model
# classes: List of class names
# resnet_history: History object returned by the ResNet model training

# Confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for ResNet')
plt.savefig("results/resnet/confusion_matrix.png")
plt.close()

# Classification report
class_report = classification_report(true_labels, predicted_labels, target_names=classes, zero_division=0)
with open("results/resnet/classification_report.txt", "w") as f:
    f.write(class_report)

# Loss and accuracy plots for ResNet
plt.plot(resnet_history.history['loss'], label='Training Loss')
plt.plot(resnet_history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig("results/resnet/loss_plot.png")
plt.close()

plt.plot(resnet_history.history['accuracy'], label='Training Accuracy')
plt.plot(resnet_history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.savefig("results/resnet/accuracy_plot.png")
plt.close()