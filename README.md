## 🧠 Brain Tumor Detection Using Deep Learning & Computer Vision

A deep learning-based medical imaging application for **automated brain tumor classification** using MRI scans. This project uses **transfer learning (VGG16)** and powerful visualization techniques to build a real-time brain tumor detection system capable of identifying tumor types with high accuracy.

### 📁 Dataset

* Source: [Kaggle Brain Tumor Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
* Classes:

  * **Glioma**
  * **Meningioma**
  * **Pituitary**
  * **No Tumor**

### 🛠️ Tech Stack

| Category            | Tools / Libraries                                  |
| ------------------- | -------------------------------------------------- |
| Data Preprocessing  | `numpy`, `pandas`, `PIL`, `matplotlib`, `seaborn`  |
| Deep Learning Model | `TensorFlow`, `Keras`, `VGG16` (Transfer Learning) |
| Image Augmentation  | `PIL.ImageEnhance`, `random`, `resize`, etc.       |
| Evaluation Metrics  | `classification_report`, `confusion_matrix`, `ROC` |
| Interface / Demo    | `Matplotlib`, `Jupyter Notebook`, `Google Colab`   |

 🚀 Features

* ✅ Augments MRI images with brightness/contrast for robust training
* ✅ Uses **VGG16** pretrained model for powerful feature extraction
* ✅ Custom top layers trained for tumor classification
* ✅ Evaluates model using:

  * Confusion matrix
  * Classification report
  * ROC Curve (per class)
* ✅ Predicts and displays tumor status from user-supplied image
* ✅ Clearly detects "**No Tumor**" cases
  
### 🧪 Model Training

```python
model.fit(datagen(train_paths, train_labels, batch_size=20, epochs=5),
          steps_per_epoch=len(train_paths) // 20,
          epochs=5)
```

* Image size: `128 x 128`
* Batch size: `20`
* Epochs: `5`
* Optimizer: `Adam (lr = 0.0001)`
* Loss: `sparse_categorical_crossentropy`

### 📊 Evaluation

* `Classification Report`: Precision, Recall, F1-score
* `Confusion Matrix`: Visual comparison of prediction vs truth
* `ROC Curve`: AUC scores for each tumor type

### 🧠 Inference Function

```python
def detect_and_display(img_path, model):
    # Predicts tumor type from given image path
```

✅ Shows:

* Predicted tumor class (or **No Tumor**)
* Confidence score
* Original image with overlayed prediction

### 📂 Project Structure

```
📁 BrainTumorDetection
├── 📁 data/
├── 📁 models/
├── 📁 notebook/
├── model.h5
├── main.py
├── utils.py
├── inference.py
└── README.md

### ▶️ How to Run

1. Clone the repository

   ```bash
   git clone https://github.com/yourusername/brain-tumor-detection.git
   cd brain-tumor-detection
   ```

2. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

3. Run training (or load `model.h5` directly)

   ```bash
   python main.py
   ```

4. Run predictions:

   ```python
   detect_and_display('/path/to/image.jpg', model)
   ```

### 📌 Future Work

* Add Streamlit or Gradio-based Web Interface
* Deploy model on Hugging Face Spaces / Heroku / Render
* Extend to CT Scan modality
* Hyperparameter tuning and early stopping

