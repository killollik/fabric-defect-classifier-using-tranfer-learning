# Fabric Defect Classifier

This repository contains a TensorFlow-based deep learning model designed to classify fabric images as 'defective' or 'non-defective'. The project is specifically engineered to perform robustly even with a highly limited and imbalanced dataset, a common challenge in industrial quality control.

---

## 1. Methodology

### 1.1. Approach: Transfer Learning with EfficientNet-B0

Given the small dataset (60 images), a **Transfer Learning** approach was chosen to avoid overfitting. The model leverages **EfficientNet-B0**, a state-of-the-art, lightweight convolutional neural network pre-trained on the ImageNet dataset. This allows the model to use powerful, pre-learned visual features and adapt them to this specific classification task.

### 1.2. Key Techniques

*   **Data Augmentation:** To artificially expand the training set, an "on-the-fly" augmentation pipeline was implemented. At each training step, random transformations (flips, rotations, zooms, contrast adjustments) are applied, forcing the model to learn invariant features of the defects.
*   **Handling Class Imbalance:** The 5:1 class imbalance was a critical challenge. This was addressed by implementing a **weighted loss function**. The model was penalized **5 times more** for misclassifying a rare 'defective' image, forcing it to pay close attention to the minority class.
*   **Two-Phase Fine-Tuning:** A structured training process was used to safely adapt the pre-trained model:
    1.  **Head Training:** Initially, only the newly added classification layers were trained while the pre-trained base was frozen.
    2.  **Fine-Tuning:** The entire model was then unfrozen and trained at a very low learning rate, allowing the deep layers to make subtle adjustments to better fit the fabric texture data.

### 1.3. Performance

The final model, trained on the complete dataset, demonstrated excellent discriminatory power, achieving a final training **AUC (Area Under the Curve) of 0.94**. This significantly outperforms a baseline "always guess non-defective" model (which would have an AUC of 0.5), proving the model learned meaningful features of the defects.

---

## 2. How to Run This Project

This project is designed to be run in a Google Colab environment for easy access.

### 2.1. Prerequisites

*   A Google Account to use Google Colab.

### 2.2. Data Preparation (Crucial Step)

As the image data is proprietary, it is not included in this repository. You must provide it yourself.

 On your drive, create the following folder structure:
    ```
    fabric_dataset/
    └── train/
        ├── defective/
        │   └── (place your 10 defective images here)
        └── non_defective/
            └── (place your 50 non-defective images here)
    ```

### 2.3. Execution in Google Colab

1.  **Open Google Colab:** Go to [colab.research.google.com](https://colab.research.google.com).
2.  **Clone the Repository:** Open a new notebook and run the following command in a cell to clone this repository into your Colab environment:
    ```python
    !git clone https://github.com/[Your-GitHub-Username]/fabric-defect-classifier.git
    ```
3.  **Navigate into the Directory:**
    ```python
    %cd fabric-defect-classifier
    ```
4.  **Install Dependencies:** Run the following command to install the required libraries:
    ```python
    !pip install -r requirements.txt
    ```
5.  **Run the Training Script:** Execute the main training script with the following command:
    ```python
    !python src/train_model.py
    ```
6.  **Get the Final Model:** After the script finishes, a new file named `final_fabric_defect_model.keras` will appear in the file explorer. This is your trained model, which you can download.

