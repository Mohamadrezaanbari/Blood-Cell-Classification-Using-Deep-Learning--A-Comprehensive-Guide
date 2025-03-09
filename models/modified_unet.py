from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model


def build_dual_unet(input_shape=(128, 128, 3)):
    # ورودی‌های دوگانه
    input_main = Input(input_shape, name='main_input')
    input_seg = Input(input_shape, name='seg_input')

    # --- انکودر برای ورودی اصلی ---
    # Block 1
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(input_main)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    # Block 2
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    # Block 3
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)

    # --- انکودر برای ورودی سگمنت‌شده ---
    # Block 1 (Segmented)
    conv1_seg = Conv2D(64, 3, activation='relu', padding='same')(input_seg)
    conv1_seg = Conv2D(64, 3, activation='relu', padding='same')(conv1_seg)
    pool1_seg = MaxPooling2D((2, 2))(conv1_seg)

    # Block 2 (Segmented)
    conv2_seg = Conv2D(128, 3, activation='relu', padding='same')(pool1_seg)
    conv2_seg = Conv2D(128, 3, activation='relu', padding='same')(conv2_seg)
    pool2_seg = MaxPooling2D((2, 2))(conv2_seg)

    # Block 3 (Segmented)
    conv3_seg = Conv2D(256, 3, activation='relu', padding='same')(pool2_seg)
    conv3_seg = Conv2D(256, 3, activation='relu', padding='same')(conv3_seg)
    pool3_seg = MaxPooling2D((2, 2))(conv3_seg)

    # --- ادغام ویژگی‌ها ---
    merged = concatenate([pool3, pool3_seg], axis=-1)

    # --- میانی (Bottleneck) ---
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(merged)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)

    # --- دیکودر ---
    # Block 5
    up5 = UpSampling2D((2, 2))(conv4)
    up5 = Conv2D(256, 2, activation='relu', padding='same')(up5)
    merge5 = concatenate([conv3, up5], axis=-1)
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(merge5)
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(conv5)

    # Block 6
    up6 = UpSampling2D((2, 2))(conv5)
    up6 = Conv2D(128, 2, activation='relu', padding='same')(up6)
    merge6 = concatenate([conv2, up6], axis=-1)
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(conv6)

    # Block 7
    up7 = UpSampling2D((2, 2))(conv6)
    up7 = Conv2D(64, 2, activation='relu', padding='same')(up7)
    merge7 = concatenate([conv1, up7], axis=-1)
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(conv7)

    # --- لایه خروجی ---
    outputs = Conv2D(1, 1, activation='sigmoid')(conv7)

    return Model(inputs=[input_main, input_seg], outputs=outputs, name='Dual_U-Net')