import matplotlib.pyplot as plt

def display_results(image, true_mask, predicted_mask, predicted_class):
    """
    نمایش تصویر، ماسک واقعی، ماسک پیش‌بینی شده و کلاس پیش‌بینی شده.
    """
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title('Input Image')
    plt.imshow(image)
    
    plt.subplot(1, 3, 2)
    plt.title('True Mask')
    plt.imshow(true_mask, cmap='gray')
    
    plt.subplot(1, 3, 3)
    plt.title(f'Predicted Mask ({predicted_class})')
    plt.imshow(predicted_mask, cmap='gray')
    
    plt.show()