# ====================================
# 1. Import required libraries
# ====================================
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


# ================================================
# 2. Import MNIST dataset
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()


# 3. Explore data format
print("Training data shape:", x_train.shape)   # (60000, 28, 28)
print("Test data shape:", x_test.shape)         # (10000, 28, 28)

# Visualize one image
sample = x_train[2]
plt.imshow(sample, cmap="gray")
plt.title("MNIST Sample Image")
plt.show()


# 4. Data Preprocessing
# Normalize pixel values (0–255 -> 0–1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# One-hot encoding of labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# =======================================================
# 5. Define the Neural Network (MLP)
model = tf.keras.models.Sequential([

    # Convert 28x28 image into 1D vector (784)
    tf.keras.layers.Flatten(input_shape=(28, 28)),

    # Hidden Layer 1
    tf.keras.layers.Dense(256, activation='relu'),

    # Hidden Layer 2
    tf.keras.layers.Dense(256, activation='relu'),

    # Output Layer
    tf.keras.layers.Dense(10, activation='softmax')
])

# =======================================================
# 6. Compile the Model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
# =======================================================
# 7. Train the Model
model.fit(
    x_train,
    y_train,
    epochs=15,
    batch_size=100,
    validation_data=(x_test, y_test)
)

print("Training Finished")

# =======================================================
# 8. Evaluate the Model
loss, accuracy = model.evaluate(x_test, y_test)
print("Test Accuracy:", accuracy)

# ========================================================
# 9. Prediction Example
prediction = model.predict(x_test[:1])
print("Predicted Digit:", np.argmax(prediction))
print("Actual Digit:", np.argmax(y_test[0]))
