# CNN Model Documentation

## 1. What is a CNN?
A **Convolutional Neural Network (CNN)** is a type of deep learning algorithm specifically designed to process structured arrays of data such as images. Unlike traditional neural networks, CNNs can automatically learn spatial hierarchies of features, from low-level edges to high-level complex patterns.

## 2. Model Architecture & Layer Roles

The model implemented in this project follows a classic CNN architecture:

| Layer Type | Role | Details in this Model |
| :--- | :--- | :--- |
| **Conv2D** | Feature Extraction | Uses kernels (filters) to scan the image and detect patterns (edges, textures). |
| **ReLU** | Activation Function | Introduces non-linearity, allowing the model to learn complex relationships. |
| **MaxPooling2D** | Downsampling | Reduces the spatial dimensions (width/height), decreasing computation and preventing overfitting. |
| **Flatten** | Dimensionality Shift | Converts the 2D feature maps into a 1D vector to feed into the dense layers. |
| **Dense** | Classification | Fully connected layers that use the extracted features to vote on the final class. |
| **Dropout** | Regularization | Randomly shuts off neurons during training to prevent the model from memorizing specific training data (overfitting). |
| **Softmax** | Output Activation | Converts raw output scores (logits) into probabilities that sum to 1. |

## 3. How the Forward Pass Works
1. **Input**: A 32x32 pixel image with 3 color channels (RGB) enters the network.
2. **Feature Extraction**: Convolutional layers apply filters to create feature maps.
3. **Reduction**: Pooling layers shrink these maps while keeping the most important information.
4. **Reasoning**: The flattened vector passes through dense layers where the "thinking" happens.
5. **Output**: The softmax layer provides a probability distribution across the 10 possible classes.

## 4. Training and Evaluation
- **Optimizer**: `Adam` (adaptive learning rate).
- **Loss Function**: `Categorical Crossentropy` (ideal for multi-class classification).
- **Metrics**: `Accuracy`.
- **Epochs**: 10.

### Expected Results
By focusing on **Animals Only**, the model avoids confusion between organic shapes and mechanical ones (like planes vs birds). This typically results in a higher validation accuracy (**75-82%**) compared to the general-purpose model.
