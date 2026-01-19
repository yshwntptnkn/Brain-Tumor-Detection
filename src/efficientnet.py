from tensorflow.keras.applications import EfficientNetV2B1
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

def build_model(img_size, num_classes):
    base_model = EfficientNetB1(
        include_top=False,
        weights="imagenet",
        input_shape=(240, 240, 3),
    )

    base_model.name = "EfficientNetB1"
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(4, activation="softmax")(x)

    model = Model(base_model.input, outputs)

    return model, base_model
