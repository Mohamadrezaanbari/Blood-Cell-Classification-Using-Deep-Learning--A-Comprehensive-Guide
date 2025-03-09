import os
import cv2
import numpy as np

def load_images(data_dir, target_size=(256, 256)):
    """
    لود کردن تصاویر از یک پوشه.
    
    پارامترها:
        data_dir (str): مسیر پوشه‌ی تصاویر.
        target_size (tuple): ابعاد هدف تصاویر.
    
    بازگشت:
        images (list): لیست تصاویر.
    """
    images = []
    
    for image_name in os.listdir(data_dir):
        if image_name.endswith('.jpg') or image_name.endswith('.png'):
            image_path = os.path.join(data_dir, image_name)
            image = cv2.imread(image_path)
            image = cv2.resize(image, target_size)
            images.append(image)
    
    return np.array(images)