# Generative PointNet: Deep Energy-Based Learning on Unordered Point Sets

## Project Overview

This repository contains our presentation and the implementation of **Generative PointNet**, a novel approach for learning deep energy-based models (EBMs) on 3D point clouds, along with presentation slides. The project is based on the paper:

**[Generative PointNet: Deep Energy-Based Learning on Unordered Point Sets for 3D Generation, Reconstruction, and Classification](https://arxiv.org/abs/2004.01301)**

The model introduced in this paper utilizes an innovative energy-based framework to learn generative models for unordered 3D point sets, enabling tasks such as generation, reconstruction, and classification of point clouds. The energy-based model facilitates explicit density estimation and is trained using Maximum Likelihood Estimation (MLE) with MCMC-based Langevin dynamics for sampling.

### Key Features

- **Energy-Based Modeling**: The discriminator functions as an energy function, assigning low energy to data near the real data manifold and high energy to unrealistic data.
- **3D Point Cloud Processing**: The model effectively processes unordered collections of 3D points, supporting tasks such as classification, segmentation, and reconstruction.
- **Langevin Dynamics Sampling**: Utilizes MCMC-based Langevin dynamics to generate 3D point clouds.
- **Competitive Performance**: Demonstrates competitive results in 3D point cloud generation, reconstruction, and classification compared to other state-of-the-art methods.

## Project Contents

- **2004.01301.pdf**: Original paper detailing the Generative PointNet model.
- **PointNet GAN.pdf**: Presentation slides summarizing key concepts of GANs and the Generative PointNet model.
- **deep-energy-based-model.ipynb**: Jupyter notebook with the implementation of the Generative PointNet model, including training and testing code.
- **گزارش پیاده سازی.pdf**: Detailed report (in Persian) documenting the implementation, methodology, and results of the Generative PointNet model.

### File Descriptions

1. **deep-energy-based-model.ipynb**
   - Contains the implementation of the Generative PointNet model.
   - Includes model architecture, MLE training, Langevin dynamics sampling, and evaluation metrics.

2. **PointNet GAN.pdf**
   - Slide presentation covering:
     - Introduction to GANs.
     - Point cloud data applications.
     - Energy-based learning concepts.
     - Details of the Generative PointNet model, including experiments and results.

3. **گزارش پیاده سازی.pdf**
   - A comprehensive report documenting:
     - Technical details of the implementation.
     - Design choices made during model development.
     - Results and evaluations on various point cloud datasets.

4. **2004.01301.pdf**
   - Paper explaining the theory behind the Generative PointNet model, its architecture, and applications in point cloud tasks.

## Installation and Dependencies

### Requirements

- Python 3.x
- Jupyter Notebook
- NumPy
- PyTorch
- Matplotlib
- Scikit-learn


## Running the Code

- **Training the Model**: Execute code blocks in the Jupyter notebook to train the Generative PointNet model on point cloud datasets like ModelNet10.
- **Generating Point Clouds**: The notebook includes code for sampling 3D point clouds using Langevin dynamics.
- **Reconstruction and Classification**: Evaluate the model's performance on reconstruction and classification tasks, with visualizations of the results provided.

## Results

- **Point Cloud Generation**: The model generates realistic 3D point clouds from noise using Langevin dynamics sampling.
- **Point Cloud Reconstruction**: High accuracy in reconstructing 3D point clouds by minimizing reconstruction errors.
- **Classification**: Competitive performance in point cloud classification tasks, even with partial or noisy input.

## Conclusion

Generative PointNet showcases the effectiveness of energy-based models in processing 3D point clouds. By integrating generation, reconstruction, and classification tasks within a single framework, this model serves as a versatile tool for working with 3D point cloud data.

## References

- [Generative PointNet: Deep Energy-Based Learning on Unordered Point Sets for 3D Generation, Reconstruction, and Classification](https://arxiv.org/abs/2004.01301)
- [GAN Lab - Visualize GANs](https://poloclub.github.io/ganlab/)
