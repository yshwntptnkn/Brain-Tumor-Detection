import tensorflow as tf
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def compute_class_weights(train_generator):
    """
    Computes class weights based on training data distribution.
    """
    y = train_generator.classes

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y),
        y=y
    )

    return dict(enumerate(class_weights))


def train_frozen(model, train_gen, val_gen, lr, epochs, class_weights=None):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                patience=3,
                restore_best_weights=True
            )
        ]
    )


def finetune(model, base_model, train_gen, val_gen, lr, epochs, class_weights=None):
    # Unfreeze top 20% of backbone
    for layer in base_model.layers[int(0.8 * len(base_model.layers)):]:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                patience=3,
                restore_best_weights=True
            )
        ]
    )
