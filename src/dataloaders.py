from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

def get_train_val_generators(train_dir, img_size, batch_size):
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=10,
        zoom_range=0.05,
        validation_split=0.2
    )

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True
    )

    val_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        classes=list(train_gen.class_indices.keys())
    )

    return train_gen, val_gen


def get_test_generator(test_dir, img_size, batch_size, class_indices):
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    return test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
        classes=list(class_indices.keys())
    )
