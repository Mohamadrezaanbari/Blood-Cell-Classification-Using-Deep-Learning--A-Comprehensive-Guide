import os
import numpy as np
import tensorflow as tf
from utils.data_loader import load_images
from utils.mask_generator import generate_mask
from utils.preprocessing import normalize_images, normalize_masks
from models.modified_unet import build_dual_unet
from models.modified_resnet import build_dual_resnet
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# --- Functions for saving processing steps ---
def save_processing_steps(image, path, prefix, step_name, idx):
    """Save images at different processing stages"""
    output_dir = os.path.join(path, prefix)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = os.path.join(output_dir, f"{step_name}_{idx}.png")
    if len(image.shape) == 2:  # If the image is grayscale
        cv2.imwrite(filename, image)
    else:  # If the image is RGB
        cv2.imwrite(filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

# --- Image segmentation and saving ---
def segment_image(image):
    """Segment image based on color range"""
    lower_purple = np.array([120, 50, 50])  # Lower bound for purple color
    upper_purple = np.array([160, 255, 255])
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv_image, lower_purple, upper_purple)
    segmented_image = cv2.bitwise_and(image, image, mask=mask)
    return segmented_image

def segment_and_save(image, idx):
    segmented = segment_image(image)
    save_processing_steps(image, "processing_steps", "original", "original", idx)
    save_processing_steps(segmented, "processing_steps", "segmented", "segmented", idx)
    return segmented

# --- Mask improvement and overlay functions ---
def improve_mask(mask):
    """Improve mask using morphological operations"""
    mask = (mask > 0.5).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Remove noise
    return mask

def overlay_image_on_mask(image, mask):
    """Overlay image on mask"""
    mask_3channel = cv2.merge([mask, mask, mask])
    overlay = cv2.bitwise_and(image, mask_3channel)
    return overlay

# --- Data loading and preprocessing ---
data_dir = 'data/images'
images_bgr = load_images(data_dir)

# Resize images and save steps
images_resized = []
for i, img in enumerate(images_bgr):
    resized = cv2.resize(img, (128, 128))
    images_resized.append(resized)
    save_processing_steps(resized, "processing_steps", "resized", "resized", i)

# Convert to RGB and segment
images = []
segmented_images = []
for i, img in enumerate(images_resized):
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    images.append(rgb_img)
    segmented = segment_and_save(rgb_img, i)
    segmented_images.append(segmented)

# Normalize images
images_norm = normalize_images(images)
segmented_norm = normalize_images(segmented_images)

# Save normalized images
for i, (img_norm, seg_norm) in enumerate(zip(images_norm, segmented_norm)):
    img_norm_uint8 = (img_norm * 255).astype(np.uint8)
    seg_norm_uint8 = (seg_norm * 255).astype(np.uint8)
    save_processing_steps(img_norm_uint8, "processing_steps", "normalized", "image_norm", i)
    save_processing_steps(seg_norm_uint8, "processing_steps", "normalized", "seg_norm", i)

# Generate and resize masks
masks = [generate_mask(img) for img in images_bgr]
masks_resized = [cv2.resize(mask, (128, 128)) for mask in masks]
masks = normalize_masks(masks_resized)

# Print data shapes
print("Images shape:", np.array(images_norm).shape)  # Should be (num_images, 128, 128, 3)
print("Masks shape:", np.array(masks).shape)  # Should be (num_images, 128, 128, 1)

# --- U-Net model training ---
dual_unet = build_dual_unet(input_shape=(128, 128, 3))
dual_unet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train U-Net
unet_history = dual_unet.fit(
    [np.array(images_norm), np.array(segmented_norm)],
    np.array(masks),
    batch_size=4,
    epochs=5,
    validation_split=0.2
)

# Predict masks using U-Net
unet_predicted_masks = dual_unet.predict([np.array(images_norm), np.array(segmented_norm)])

# Debugging: Check unique values of masks and predicted masks
print("Unique values in true masks:", np.unique(masks))
print("Unique values in predicted masks:", np.unique(unet_predicted_masks))

# Convert masks and predicted masks to binary (0 or 1)
masks_binary = (np.array(masks) > 0.5).astype(int)
unet_predicted_masks_binary = (unet_predicted_masks > 0.5).astype(int)

# Debugging: Check shapes and unique values after conversion
print("True masks shape:", masks_binary.shape)
print("Predicted masks shape:", unet_predicted_masks_binary.shape)
print("Unique values in true masks (binary):", np.unique(masks_binary))
print("Unique values in predicted masks (binary):", np.unique(unet_predicted_masks_binary))

# Save U-Net results
for i, (img, true_mask, pred_mask) in enumerate(zip(images_norm, masks_binary, unet_predicted_masks_binary)):
    save_processing_steps(true_mask * 255, "results/unet", "true_masks", "true_mask", i)
    improved_mask = improve_mask(pred_mask[:, :, 0])
    save_processing_steps(improved_mask, "results/unet", "pred_masks", "pred_mask", i)
    overlay = overlay_image_on_mask((img * 255).astype(np.uint8), improved_mask)
    save_processing_steps(overlay, "results/unet", "overlays", "overlay", i)

# --- ResNet model training ---
labels = [name.split('_')[0] for name in os.listdir(data_dir) if name.endswith(('.jpg', '.png'))]
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
label_binarizer = LabelBinarizer()
labels = label_binarizer.fit_transform(labels)  # Ensure labels are one-hot encoded

# Ensure classes are strings
classes = label_binarizer.classes_.astype(str)  # Convert classes to strings

dual_resnet = build_dual_resnet(input_shape=(128, 128, 3), num_classes=len(classes))
dual_resnet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train ResNet
resnet_history = dual_resnet.fit(
    [np.array(images_norm), np.array(segmented_norm)],
    np.array(labels),
    batch_size=4,
    epochs=5,
    validation_split=0.2
)

# Save ResNet results
resnet_predicted_probs = dual_resnet.predict([np.array(images_norm), np.array(segmented_norm)])
resnet_predicted_labels = np.argmax(resnet_predicted_probs, axis=1)
true_labels = np.argmax(labels, axis=1)

# Confusion matrix for ResNet
conf_matrix = confusion_matrix(true_labels, resnet_predicted_labels)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for ResNet')
plt.savefig("results/resnet/confusion_matrix.png")
plt.close()

# Classification report for ResNet
class_report = classification_report(true_labels, resnet_predicted_labels, target_names=classes, zero_division=0)
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

# --- U-Net confusion matrix and classification report ---
# Compute confusion matrix for U-Net
conf_matrix = confusion_matrix(masks_binary.flatten(), unet_predicted_masks_binary.flatten())
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for U-Net')
plt.savefig("results/unet/confusion_matrix.png")
plt.close()

# Compute classification report for U-Net
class_report = classification_report(masks_binary.flatten(), unet_predicted_masks_binary.flatten(), zero_division=0)
with open("results/unet/classification_report.txt", "w") as f:
    f.write(class_report)

# Loss and accuracy plots for U-Net
plt.plot(unet_history.history['loss'], label='Training Loss')
plt.plot(unet_history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig("results/unet/loss_plot.png")
plt.close()

plt.plot(unet_history.history['accuracy'], label='Training Accuracy')
plt.plot(unet_history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.savefig("results/unet/accuracy_plot.png")
plt.close()