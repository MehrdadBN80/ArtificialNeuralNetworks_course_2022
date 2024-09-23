# ResNet Intel image classification

#### Overview
This repository presents an implementation for image classification using a modified ResNet-101 architecture integrated with the Leaky ReLU activation function. The main objective is to highlight how Leaky ReLU offers improvements over the standard ReLU, particularly in addressing issues like the "dying ReLU" and enhancing the speed of training.

#### Benefits of Leaky ReLU:
- **Mitigating the Dying ReLU Problem**: Leaky ReLU allows a small gradient for negative inputs, keeping neurons active during training.
- **Faster Training**: With near-zero mean activation, Leaky ReLU can accelerate convergence compared to traditional ReLU.

---

### Setup Instructions

#### Prerequisites
Ensure the following dependencies are installed:
- Python 3.x
- PyTorch
- torchvision
- numpy
- matplotlib
- tqdm

You can install these libraries via:
```bash
pip install torch torchvision numpy matplotlib tqdm
```

#### Data
The project uses the Intel Image Classification dataset. For users on Kaggle, the dataset will automatically be available in the correct directories. For local execution, download and unzip the dataset:

```bash
wget https://example.com/intel-image-classification.zip
unzip intel-image-classification.zip
```

Configure the dataset paths in the notebook:
```python
train_data = '/path/to/seg_train/seg_train'
test_data = '/path/to/seg_test/seg_test'
```

---

### Implementation Breakdown

The Jupyter notebook **`Intel_image_classification.ipynb`** details the following steps:

1. **Data Preparation**:
   - Data augmentation and normalization techniques for both training and test datasets.
   - Loading datasets via `torchvision.datasets.ImageFolder` and `DataLoader`.

2. **Model Architecture**:
   - Utilizes a pre-trained ResNet-101 model.
   - Modifies the final fully connected layer to integrate Leaky ReLU activation.

3. **Training**:
   - The model is trained using the Adam optimizer, with a learning rate scheduler.
   - Training and test loss are monitored during the process.

4. **Evaluation**:
   - Loss plots for both training and testing phases.
   - Calculation and display of accuracy metrics for training and test datasets.

---

### Results and Accuracy

The model delivers the following performance:
- **Training Accuracy**: 94.22%
- **Test Accuracy**: 89.00%

Additionally, the notebook provides loss curves that visualize the model’s performance over training epochs.

---

### Usage Instructions
To replicate the results:
1. Open the **`Intel_image_classification.ipynb`** file in Jupyter Notebook or JupyterLab.
2. Run all the cells to perform data loading, model training, and evaluation.

For any questions or further assistance, feel free to reach out.

---

### Report Summary: Comparison of Architectures

This report compares the performance of multiple deep learning models on the **Intel Image Classification Dataset** (25k images across six categories: buildings, forest, glacier, mountain, sea, street). The models evaluated include ResNet-152, WRN (Wide Residual Networks), and Inception-v3.

#### **ResNet-152**:
- **Total Parameters**: 60M
- **Training Observations**: The model showed stable convergence and provided the best accuracy.
- **Memory Usage**: ~1.3GB

#### **WRN**:
- **Total Parameters**: 126M (double that of ResNet-152)
- **Training Observations**: Despite having more parameters, WRN struggled with unstable loss behavior and higher computation cost. 
- **Memory Usage**: ~1.5GB

#### **Inception-v3**:
- **Total Parameters**: 27M (smaller and lighter)
- **Training Observations**: The model demonstrated instability during training, providing inconsistent predictions across epochs.
- **Memory Usage**: ~333MB

#### Confusion Matrix:
The confusion matrix for the best-performing model (ResNet-152) was generated to identify areas where misclassifications occurred, helping refine model weaknesses.

