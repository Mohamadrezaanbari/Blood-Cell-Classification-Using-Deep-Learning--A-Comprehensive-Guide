from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model


def build_resnet(input_shape=(128, 128, 3), num_classes=2):
    """
    ساخت مدل ResNet برای کلاس‌بندی تصاویر.

    پارامترها:
        input_shape (tuple): ابعاد ورودی تصویر.
        num_classes (int): تعداد کلاس‌ها.

    بازگشت:
        model (tf.keras.Model): مدل ResNet.
    """
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model