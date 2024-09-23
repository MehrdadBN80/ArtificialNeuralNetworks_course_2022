### AutoEncode_Cifar10

This repository contains both theoretical answers and the practical implementation for the 4th assignment of the **Artificial Neural Networks** course at **Shahid Beheshti University (Bachelor’s Program)**. The assignment covers three exercises, with theoretical questions focused on **Variational Autoencoders (VAE)** and a practical task involving image reconstruction using the **CIFAR-10** dataset.

---

### Exercises Overview

#### Exercise 1: Variational Autoencoder (VAE) vs. Traditional Autoencoders
This exercise explains the VAE's ability to generate new data points, unlike a traditional Associative Autoencoder. The answer is available in **Exercises 1,2.pdf**.

#### Exercise 2: Optimization in VAE
Focused on VAE optimization, this exercise addresses:
- The role of the **KL-divergence** term.
- The advantages of using normal distributions with diagonal covariance matrices in modeling **p_θ(z)** and **q_ϕ(z|x)**
- The impact of the first term on the latent space.

Answers are provided in **Exercises 1,2.pdf**.

#### Exercise 3: Image Reconstruction using CIFAR-10
The practical part of the assignment involves building a neural network that reconstructs two images, **x₁** and **x₂**, using only their average as input. The architecture is open-ended, allowing experimentation. Extra credit is awarded for achieving the lowest loss on test data.


---

### Setup Instructions

#### Prerequisites
Ensure the following Python libraries are installed:
- Python 3.x
- PyTorch
- torchvision
- numpy
- matplotlib

To install them, run:
```bash
pip install torch torchvision numpy matplotlib
```


### Dataset
The **CIFAR-10** dataset contains 60,000 32x32 color images across 10 classes (e.g., dogs, cats, planes). It is loaded using the `torchvision.datasets` module in the notebook.

---

### Results
The notebook presents:
- Training and test loss.
- Reconstructed images.
  
Detailed reports of the model’s architecture, training process, and results are available in the **Report_Assignment4.pdf** file.

---

### Architecture Experimentation Report

#### Task Overview
The goal was to select 1000 images from CIFAR-10 and generate 45,000 training samples by averaging random image pairs. The neural network was trained to reconstruct the two original images using only their average as input. Below are three main architectures explored during the task:

1. **VAE (Variational Autoencoder) with 2 Decoders**:
   - The network includes an encoder and two decoders.
   - The loss function combines the reconstruction losses of both decoders with their respective labels using **MSE**.
   - Despite being trained over 30 epochs, the model suffered from increasing **val_loss**, leading to poor performance.
   
   **Model Summary**:
   - Total Parameters: 161,190
   - Training details were tracked using TensorBoard for further analysis.

2. **Autoencoder with 2 Decoders**:
   - Similar to VAE, but without the variational components.
   - Although the model trained successfully, it showed high dependence on input rather than labels, with the outputs being overly similar to the inputs. This indicates the network struggled to generalize beyond simply reproducing the average input.
   
   **Model Summary**:
   - Total Parameters: 161,190 (Trainable: 160,422, Non-trainable: 768)
   - Loss and val_loss remained flat, indicating overfitting.

3. **U-Net Architecture**:
   - **U-Net** performed significantly better than both VAE and Autoencoder. This model's encoder-decoder structure was effective in reconstructing both images from their averaged input.
   - The loss curve behaved as expected, with **val_loss** correlating with the output labels rather than the average input.
   
   **Model Summary**:
   - Total Parameters: 10,809,638
   - **Performance**: Successfully reconstructed the dog image clearly, while the cat image remained somewhat blurry due to the lack of prominent features in the input.

---

### Conclusion
The U-Net architecture delivered the best results, reconstructing images close to their original labels, whereas the VAE and Autoencoder models struggled with generalization and produced less accurate outputs.
