from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, concatenate
from tensorflow.keras.models import Model


def build_dual_resnet(input_shape=(128, 128, 3), num_classes=4):
    """
    ساخت مدل ResNet دوگانه برای کلاس‌بندی تصاویر.

    پارامترها:
        input_shape (tuple): ابعاد ورودی تصویر (پیش‌فرض: (128, 128, 3)).
        num_classes (int): تعداد کلاس‌های خروجی.

    بازگشت:
        model (tf.keras.Model): مدل ResNet دوگانه.
    """
    # ورودی‌های دوگانه
    input_main = Input(input_shape, name='main_input')  # ورودی اصلی
    input_seg = Input(input_shape, name='seg_input')  # ورودی سگمنت‌شده

    # پایه ResNet برای ورودی اصلی
    base_model_main = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape,
        name='resnet50_main'  # نام منحصر به فرد برای مدل اصلی
    )
    x_main = base_model_main(input_main)
    x_main = GlobalAveragePooling2D(name='global_avg_pool_main')(x_main)

    # پایه ResNet برای ورودی سگمنت‌شده
    base_model_seg = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape,
        name='resnet50_seg'  # نام منحصر به فرد برای مدل سگمنت‌شده
    )
    x_seg = base_model_seg(input_seg)
    x_seg = GlobalAveragePooling2D(name='global_avg_pool_seg')(x_seg)

    # ادغام ویژگی‌های استخراج‌شده از دو ورودی
    merged = concatenate([x_main, x_seg], axis=-1, name='merge_features')

    # لایه‌های fully connected برای طبقه‌بندی
    x = Dense(1024, activation='relu', name='fc1')(merged)
    outputs = Dense(num_classes, activation='softmax', name='output')(x)

    # ساخت مدل نهایی
    model = Model(inputs=[input_main, input_seg], outputs=outputs, name='Dual_ResNet')
    return model