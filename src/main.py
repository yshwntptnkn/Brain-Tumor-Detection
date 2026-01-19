from config import *
from dataloaders import get_train_val_generators, get_test_generator
from model import build_model
from train import train_frozen, finetune, compute_class_weights
from evaluate import evaluate_model

def main():
    # 1. Data
    train_gen, val_gen = get_train_val_generators(
        TRAIN_DIR, IMG_SIZE, BATCH_SIZE
    )

    test_gen = get_test_generator(
        TEST_DIR, IMG_SIZE, BATCH_SIZE, train_gen.class_indices
    )

    # 2. Model
    model, base_model = build_model(IMG_SIZE, NUM_CLASSES)

    # 3. Class weights
    class_weights = compute_class_weights(train_gen)
    print("Class weights:", class_weights)

    # 4. Frozen training
    train_frozen(
        model,
        train_gen,
        val_gen,
        LR_FROZEN,
        EPOCHS_FROZEN,
        class_weights=class_weights
    )

    # 5. Fine-tuning
    finetune(
        model,
        base_model,
        train_gen,
        val_gen,
        LR_FINETUNE,
        EPOCHS_FINETUNE,
        class_weights=class_weights
    )

    # 6. Evaluation
    cm, report = evaluate_model(model, test_gen)
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", cm)

    # 7. Save model
    model.save("brain_tumor_classifier.h5")


if __name__ == "__main__":
    main()
