# Artificial Neural Networks - Assignment 1  
### Student: Mehrdad Baradaran (99222020)  
### Project: Fashion MNIST Classification

## Overview
This assignment includes both theoretical and practical tasks related to neural networks and regularization techniques. The first task focuses on understanding L1 and L2 regularization, while the second involves building a Multi-Layer Perceptron (MLP) model using PyTorch to classify images from the Fashion MNIST dataset.

---

## Task 1: Regularization Techniques

### **L1 Regularization**  
- **Concept**: L1 regularization adds a penalty to the loss function equal to the absolute values of the model's weights. This encourages sparsity, meaning that some weights are pushed to zero, effectively performing feature selection by reducing the number of active weights.
  
- **Effect on Weights**: L1 regularization forces less important feature weights towards zero, creating a simpler and more interpretable model.

- **Use Case**: L1 is preferred when feature selection is important, and a sparse, simpler model is desired.

---

### **L2 Regularization**  
- **Concept**: L2 regularization adds a penalty equal to the squared values of the weights to the loss function. Unlike L1, it does not push weights to zero but rather reduces their magnitude to prevent overfitting.

- **Effect on Weights**: L2 regularization reduces the size of weights, helping in controlling the model's complexity without eliminating features entirely.

- **Use Case**: L2 is preferred when all features are expected to contribute to the model, but there is a risk of overfitting.

---

### **Comparison of L1 and L2 Regularization**
- **L1 Regularization**:
  - Encourages sparsity (forces some weights to zero).
  - Useful for feature selection.
  - Better for simpler models with fewer active features.

- **L2 Regularization**:
  - Penalizes large weights without setting them to zero.
  - Better for cases where most features are expected to contribute to the outcome.
  - More computationally efficient but adds complexity.

---

## Task 2: Fashion MNIST Classification Using MLP

### **Implementation Overview**  
The second task involved implementing an MLP neural network using PyTorch to classify images from the Fashion MNIST dataset. Due to data access issues, the implementation process included several challenges and observations.

### **Steps Involved**:

#### **1. Data Handling**
- **Dataset**: Fashion MNIST dataset.
- **Challenges**: Encountered HTTP errors while attempting to load the dataset, which affected the ability to fully train and evaluate models.

#### **2. Model Architectures**
Three different MLP architectures were tested to observe their performance:

- **Basic MLP**: A model with 5 hidden layers was implemented and evaluated. However, it faced challenges due to missing data.
- **3-Layered MLP**: Reduced the number of hidden layers to investigate its effect on model performance.
- **7-Layered MLP**: Increased the number of layers to assess improvements in classification accuracy and potential risks of overfitting.

#### **3. Regularization and Overfitting Control**
- **Dropout**: Applied dropout layers to reduce overfitting and improve the learning process by randomly deactivating certain neurons during training.
- **Early Stopping**: Implemented early stopping to halt training when validation performance plateaued, thus preventing overfitting.

#### **4. Error Handling**
Due to issues with accessing the Fashion MNIST dataset and various code execution errors, the models could not be fully tested and evaluated as planned.

---

## Key Findings

- **Effect of Depth**: Increasing the number of hidden layers has the potential to improve accuracy but also increases the risk of overfitting.
- **Dropout**: Dropout layers can help prevent overfitting and improve the model's generalization by reducing the reliance on any particular neurons.
- **Early Stopping**: Early stopping is a useful technique to avoid unnecessary training once the model's performance plateaus, thus saving computational resources.

---

## Technologies Used
- **Programming Language**: Python  
- **Libraries**: PyTorch, NumPy, Matplotlib  
- **Dataset**: Fashion MNIST  

---

## References
- **Fashion MNIST dataset**: [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist)  
- **PyTorch Documentation**: [PyTorch](https://pytorch.org)  

--- 
