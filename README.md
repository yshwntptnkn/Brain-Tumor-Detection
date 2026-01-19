# Brain Tumor Classification using EfficientNet

## 📌 Overview
This project implements a **multi-class brain tumor classification system** using deep learning on MRI images. The model classifies brain MRI scans into four categories:
- **Glioma**
- **Meningioma**
- **Pituitary tumor**
- **No tumor**

The work is based on **transfer learning with EfficientNet**, fine-tuned on a curated MRI dataset. The emphasis of this project is on **correct preprocessing, reliable evaluation, and class-wise performance**, rather than accuracy alone.

---

## 🎯 Key Results
- **Test Accuracy:** ~91%
- **Strong diagonal dominance in confusion matrix**
- **Very low false negatives for tumor classes**

> ⚠️ Grad-CAM–based interpretability was planned but is not included in the current version due to time constraints.

---

## 🧠 Model Architecture
- Backbone: **EfficientNet (ImageNet pretrained)**
- Input size: **240 × 240 × 3**
- Classification head:
  - Global Average Pooling
  - Fully connected layers
  - Softmax output (4 classes)

### Training Strategy
The model was trained in two stages:
1. **Frozen backbone training** – only the classifier head was trained
2. **Fine-tuning** – higher-level convolutional layers were unfrozen and trained with a reduced learning rate

Class imbalance was addressed using **class-weighted loss**, which improved recall for tumor classes without significantly affecting overall accuracy.

---

## 🗂 Dataset Structure
```
Dataset/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

---

## ⚙️ Preprocessing
- Model-specific preprocessing using `preprocess_input` (EfficientNet-compatible)
- Mild augmentation applied **only to training data**
- Validation and test sets kept free of augmentation

Ensuring consistent preprocessing across training, validation, and testing was critical to achieving reliable performance.

---

## 📊 Evaluation
Model performance was evaluated using:
- Overall accuracy
- Confusion matrix (raw and normalized)
- Class-wise error analysis

The confusion matrix shows that most errors occur **between tumor subtypes**, rather than tumor vs non-tumor, which is desirable in a medical imaging context.

---

## 🧪 Legacy Work (Background)
This project builds upon an **earlier binary classification model** developed as an initial exploration into brain tumor detection.

The earlier model focused on:
- Binary classification (tumor vs no tumor)
- Simpler CNN-based architecture
- Understanding MRI preprocessing and dataset handling

While that model is **not part of the final system**, it served as an important learning step and directly influenced:
- Dataset organization
- Preprocessing decisions
- Evaluation practices

The legacy notebook is included in the repository for reference and historical context.

---

## 🚀 How to Run
1. Clone the repository
2. Open `notebooks/multiclass_model.ipynb`
3. Update dataset paths as required
4. Run cells sequentially

---

## 🔮 Future Improvements
- Add Grad-CAM visualizations for interpretability
- Experiment with alternative backbones (EfficientNetV2, ResNet)
- Hyperparameter tuning
- Model deployment (API or web interface)

---

## 📜 Disclaimer
This project is intended for **academic and research purposes only** and should not be used for real-world medical diagnosis.

---

## 🙌 Acknowledgements
- TensorFlow / Keras
- Publicly available brain MRI datasets

---

## 👤 Author
Yashwant Patnaikuni

📧 yashwantpatnaikuni@gmail.com
ℹ️ www.linkedin.com/in/yashwant-patnaikuni
