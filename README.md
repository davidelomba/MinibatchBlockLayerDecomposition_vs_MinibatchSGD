# MinibatchBlockLayerDecomposition_vs_MinibatchSGD

## Introduction
This project aims to analyze the performance of the **Minibatch Block Layer Decomposition (MBLD)** algorithm and compare it against **Minibatch Stochastic Gradient Descent (Minibatch SGD)**. A series of experiments were conducted to evaluate the best parameter configurations, focusing on the impact of the learning rate and network depth on training neural networks.

## Theoretical Background
The MBLD algorithm differs from Minibatch SGD by updating only a subset of weights within each batch, using a decomposition technique. Instead of updating all layers during each iteration, MBLD selects a subset of layers to update.

## Experiments
### Neural Network Structure
A fully-connected neural network with the following architecture was used:
- **Input Layer**: 784 neurons (28x28 pixels)
- **First Hidden Layer**: 128 neurons
- **Second Hidden Layer**: 64 neurons
- **Third Hidden Layer**: 32 neurons
- **Output Layer**: 10 neurons

### Dataset
The **MNIST dataset** was used for training, validation, and testing:
- **Training Set**: 60,000 images
- **Testing Set**: 10,000 images
- Normalization applied using Z-score.
- Training and testing data were divided into batches of **64 images each**.

### Training Methods
Four different MBLD models were trained:
1. MBLD with Incremental Rule, updating all layers individually from last to first.
2. MBLD with Random without replacement rule, updating all layers individually from last to first.
3. MBLD with Incremental Rule, updating layers in pairs from last to first.
4. MBLD with Incremental Rule, using Adagrad for different stepsizes per weight.

Additionally, a model using **Minibatch SGD** was trained for comparison.

### Hyperparameter Tuning
A **Grid Search** approach was used to identify optimal hyperparameters, evaluated using validation loss. The parameters examined were:
- **Learning Rate (lr)**: [0.5, 0.1, 0.01]
- **Eps (ϵ)**: [0.001, 0.0001]
- **Rho (ρ)**: [0.001, 0.0001, 0.00001, 0.0]

The training process was limited to **20 epochs** for faster hyperparameter tuning.

## Results
The models were trained for **150 epochs** and evaluated on metrics such as:
- **Training Loss**
- **Training Time**
- **Test Loss**
- **Test Accuracy**

### Summary of Findings
- **MBLD 1** (Incremental Rule) achieved the highest accuracy of **97.54%**, comparable to **SGD** with **98.10%**.
- The performance of **MBLD 4 (Adagrad)** was slightly superior to other MBLD variants in terms of accuracy.
- **SGD** achieved the lowest test loss but required less training time compared to MBLD.

## Conclusion
The results demonstrate that the **Minibatch SGD** algorithm has better generalization capabilities, achieving the lowest test loss despite higher training loss. However, the **MBLD algorithm with Adagrad (MBLD 4)** performed competitively, suggesting that it can be a viable alternative depending on specific requirements.


