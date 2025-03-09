import cv2
import numpy as np
import os

def generate_mask(image):
    """
    ساخت ماسک به صورت خودکار از تصویر با تمرکز بر رنگ‌های بنفش پررنگ.
    """
    # تبدیل تصویر به فضای رنگی HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # محدوده‌ی رنگ‌های بنفش پررنگ در فضای HSV
    lower_purple = np.array([120, 50, 50])  # حد پایین رنگ بنفش
    upper_purple = np.array([160, 255, 255])  # حد بالا رنگ بنفش

    # ایجاد ماسک بر اساس محدوده‌ی رنگ
    mask = cv2.inRange(hsv, lower_purple, upper_purple)

    # اعمال Morphological Operations برای بهبود ماسک
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # حذف نویز
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # پر کردن حفره‌ها

    return mask

def check_color_range(image):
    """
    بررسی محدوده‌ی رنگ‌ها در تصویر و ذخیره نتایج.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    lower_purple = np.array([120, 50, 50])
    upper_purple = np.array([160, 255, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    purple_mask = cv2.inRange(hsv, lower_purple, upper_purple)
    return white_mask, purple_mask

def load_images_from_directory(data_dir):
    """
    بارگذاری تمام تصاویر از یک پوشه.
    """
    image_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg') or f.endswith('.png')]
    images = []
    for image_file in image_files:
        image_path = os.path.join(data_dir, image_file)
        image = cv2.imread(image_path)
        if image is not None:
            images.append(image)
        else:
            print(f"خطا: تصویر '{image_path}' بارگذاری نشد.")
    return images