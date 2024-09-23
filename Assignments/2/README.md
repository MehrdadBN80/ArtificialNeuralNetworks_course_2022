# Artificial Neural Networks - Basic CNN for CIFAR10

## Overview
This repository contains the second assignment for the Artificial Neural Networks course at Shahid Beheshti University. The assignment focuses on Convolutional Neural Networks (CNNs) and involves both theoretical questions and practical implementation using the CIFAR-10 dataset.

## Exercises
### 1. Backpropagation in Convolutional Layers
- **Task**: Describe the backpropagation process in convolutional layers.
- **Reference**: [Link to relevant resource]

### 2. Symmetry Breaking Phenomenon
- **Task**: Explain symmetry breaking in neural networks. How can it be avoided?

### 3. Pooling Layers
- **Task**: Discuss the advantages and disadvantages of pooling layers. Would you use them frequently in your models?

### 4. Cross-Entropy vs. Quadratic Cost Functions
- **Task**: Compare cross-entropy and quadratic cost functions. Which one is better suited for classification problems?

### 5. Leaky ReLU vs. Standard ReLU
- **Task**: Compare Leaky ReLU with standard ReLU. Which one is faster? Which prevents the gradient vanishing problem better?

### 6. CNN Implementation on CIFAR-10 Dataset
- **Task**: Implement a CNN to classify images from the CIFAR-10 dataset and analyze:
  - The effect of different hidden layer depths.
  - The impact of batch normalization on performance.
  - Optimal architectures for achieving high accuracy.
  - Confusion matrix and performance analysis.

## CNN Implementation Details
### 1. Convolutional Neural Network Architecture
The CNN is implemented using TensorFlow and Keras. It explores multiple architectures with variations in:
- **Depth of hidden layers**: The network begins with 3 convolutional layers and experiments with adding more layers.
- **Batch Normalization and Dropout**: These techniques are applied to evaluate their effect on training stability and overfitting.
- **Global Average Pooling**: A variant of the model utilizes global average pooling to reduce parameters while maintaining performance.

### 2. Model Training and Evaluation
- **Initial CNN Model**: A 3-layer CNN with max pooling and ReLU activations. 
- **Extended CNN Model**: Deeper architectures are tested by adding convolutional layers and global average pooling.
- **Early Stopping**: Implemented to halt training when the validation accuracy stagnates or starts to decrease, preventing overfitting.
- **Batch Normalization and Dropout**: Improved performance by stabilizing training and preventing overfitting.
- **Confusion Matrix & Classification Report**: Used to evaluate and compare model performance across classes.

### 3. Results Summary
- **Initial Model Performance**: Achieved ~72% accuracy on test data with a basic 3-layer CNN.
- **Extended Model**: Adding more layers didn't necessarily improve performance due to overfitting and slower training times.
- **Batch Normalization**: Significantly improved accuracy, achieving ~76.8% on the test set. It helped in better distinguishing similar classes (e.g., cats vs. dogs).
- **Confusion Matrix Analysis**: Detailed class-level accuracy and error rates revealed key insights into model strengths and weaknesses.

For detailed results and graphs, refer to `Report of question 6.pdf`.

## How to Run
1. **Clone the repository**:
    ```bash
    git clone <repository-url>
    ```
2. **Install required packages**:
    ```bash
    pip install tensorflow pandas numpy matplotlib scikit-learn
    ```
3. **Run the Jupyter notebook**:
    Open `Homework2.ipynb` in Jupyter Notebook or Google Colab, then execute all cells to run the CNN implementation and view the results.

## Contact
For any questions or issues, please contact [Instructor's Contact Information] or reach out in the course Telegram group.

---
