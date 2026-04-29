import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import matplotlib.pyplot as plt
import numpy as np
import os

# 1. Load and Preprocess Dataset (CIFAR-10 - Animals Only)
print("Loading CIFAR-10 dataset and filtering for animals...")
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# CIFAR-10 Indices for animals: 2:bird, 3:cat, 4:deer, 5:dog, 6:frog, 7:horse
ANIMAL_INDICES = [2, 3, 4, 5, 6, 7]
ANIMAL_NAMES = ['Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse']

def filter_animals(images, labels):
    mask = np.isin(labels, ANIMAL_INDICES).flatten()
    filtered_images = images[mask]
    filtered_labels = labels[mask]
    
    # Map old indices to new indices (0-5)
    for i, idx in enumerate(ANIMAL_INDICES):
        filtered_labels[filtered_labels == idx] = i
        
    return filtered_images, filtered_labels

train_images, train_labels = filter_animals(train_images, train_labels)
test_images, test_labels = filter_animals(test_images, test_labels)

print(f"Filtered Training set size: {len(train_images)}")
print(f"Filtered Test set size: {len(test_images)}")

# Normalize pixel values
train_images, test_images = train_images / 255.0, test_images / 255.0

# Convert labels to categorical
train_labels = tf.keras.utils.to_categorical(train_labels, 6)
test_labels = tf.keras.utils.to_categorical(test_labels, 6)

# 2. Build CNN Model
model = models.Sequential([
    # Feature Extraction
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    # Classification Head
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(6, activation='softmax') # 6 classes now
])

# 3. Compile Model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# 4. Train Model
epochs = 15 # Increased epochs slightly as the task is more focused
print(f"Training specialized Animal Classifier for {epochs} epochs...")
history = model.fit(train_images, train_labels, epochs=epochs, 
                    validation_data=(test_images, test_labels))

# 5. Evaluate Model
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(f'\nTest accuracy: {test_acc:.4f}')

# 6. Save Model
if not os.path.exists('model'):
    os.makedirs('model')
    
model.save('model/model.h5')
print("Specialized Animal Model saved to model/model.h5")

# Save training history plot
def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.title('Animal Model Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label = 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title('Animal Model Loss')
    
    plt.savefig('docs/training_history.png')
    print("Plot saved to docs/training_history.png")

plot_history(history)
