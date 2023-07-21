import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = x_train.reshape(x_train.shape[0], 28 * 28)
x_test = x_test.reshape(x_test.shape[0], 28 * 28)

# Split the dataset to get a validation set
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

# Create the MLP model
def create_mlp_model():
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(28 * 28,)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Train the model
model = create_mlp_model()
model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_val, y_val))

def preprocess_user_input(user_input):
    user_input = user_input / 255.0  # Normalize the input
    user_input = user_input.reshape(1, 28 * 28)  # Flatten the input
    return user_input

def main():
    st.title("MNIST Digit Classification with MLP")

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    st.write(f"Test Accuracy: {test_accuracy:.4f}")
    st.write(f"Test Loss: {test_loss:.4f}")

    # Allow user to input a digit for prediction
    st.subheader("Input a Digit for Prediction:")
    user_input = st.number_input("Enter a number (0 to 9)", min_value=0, max_value=9, step=1, key="user_input")
    user_input = int(user_input)

    # Show data for the digit in the dataset
    st.subheader(f"Data for Digit {user_input}:")
    digit_indices = np.where(y_train == user_input)[0]
    st.write(f"Number of instances of digit {user_input} in the dataset: {len(digit_indices)}")
    st.write("Sample images of the digit from the dataset:")
    for i in range(min(5, len(digit_indices))):
        st.image(x_train[digit_indices[i]].reshape(28, 28), caption=f"Sample {i+1}", width=100)

    

if __name__ == "__main__":
    main()
