import tensorflow as tf
from tensorflow.keras.datasets import mnist


# Best practice solution combining explicit references and PyCharm compatibility
def build_and_train_model():
    # Load dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Preprocess data
    X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

    # Build model with explicit tensorflow.keras references
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Compile and train
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train,
                        epochs=5,
                        validation_data=(X_test, y_test))

    return model, history


# Run the model
if __name__ == "__main__":
    model, history = build_and_train_model()
    print("\nTraining complete!")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.2%}")
