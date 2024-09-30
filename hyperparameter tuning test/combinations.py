import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import time
import itertools

def hex_string_to_bytes(hex_string):
    bytes_list = [int(hex_string[i:i+2], 16) for i in range(0, len(hex_string), 2)]
    return bytes_list

input_file = 'train-classes-no-duplicates.csv'

# Load original file
print("Reading training file...")
time.sleep(1)
train_df = pd.read_csv(input_file, sep=';')
print(f"Data rows: {len(train_df)}")
time.sleep(1)

train_df['features'] = train_df['Data'].apply(hex_string_to_bytes)
X_train = np.array(train_df['features'].tolist())
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)  # Reshape for Conv1D input
y_train = train_df['Type'].values

num_classes = 3
y_train_adjusted = y_train - 1  # Adjust for categorical encoding
y_train_categorical = to_categorical(y_train_adjusted, num_classes)

# Define hyperparameter space
dropouts = [0.4, 0.5, 0.6]
batch_sizes = [8, 16, 32, 64]
validation_splits = [0.1, 0.2, 0.3]
activations = ['relu', 'tanh']
neurons = [16, 32, 64]

# Output file for results
output_file = 'hyperparameter_tuning_results.txt'
best_result = {'val_accuracy': 0}

with open(output_file, 'w') as f:
    f.write('Hyperparameter Tuning Results\n\n')

    # Generate all combinations of hyperparameters
    combinations = list(itertools.product(dropouts, batch_sizes, validation_splits, activations, neurons))

    for combo in combinations:
        dropout, batch_size, val_split, activation, neuron = combo
        print(f"Testing combination: Dropout={dropout}, Batch size={batch_size}, Validation split={val_split}, Activation={activation}, Neurons={neuron}")

        # Build the model
        model = Sequential()
        model.add(Conv1D(32, kernel_size=3, activation=activation, input_shape=(8, 1), padding='same'))  # Padding to maintain input size
        model.add(MaxPooling1D(pool_size=2))  # Max pooling layer
        model.add(Dropout(dropout))  # Dropout for regularization

        model.add(Conv1D(64, kernel_size=3, activation=activation, padding='same'))  # Additional convolutional layer with padding
        model.add(MaxPooling1D(pool_size=2))  # Max pooling layer
        model.add(Dropout(dropout))

        model.add(Flatten())  # Flatten to prepare for dense layers
        model.add(Dense(neuron, activation=activation))
        model.add(Dropout(dropout))

        model.add(Dense(num_classes, activation='softmax'))  # Output layer

        # Compile the model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Train the model
        history = model.fit(X_train, y_train_categorical, epochs=50, batch_size=batch_size, validation_split=val_split, verbose=0)

        # Evaluate the model on the validation set
        val_loss = history.history['val_loss'][-1]
        val_accuracy = history.history['val_accuracy'][-1]

        # Save results to file
        f.write(f"Combination: Dropout={dropout}, Batch size={batch_size}, Validation split={val_split}, Activation={activation}, Neurons={neuron}\n")
        f.write(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}\n")

        # Update best result
        if val_accuracy > best_result['val_accuracy']:
            best_result.update({
                'val_accuracy': val_accuracy,
                'dropout': dropout,
                'batch_size': batch_size,
                'val_split': val_split,
                'activation': activation,
                'neurons': neuron,
                'history': history
            })

        # Confusion matrix
        y_train_pred = model.predict(X_train)
        y_train_pred_classes = np.argmax(y_train_pred, axis=1) + 1  # Convert back to original class labels (adjusting by +1)

        cm = confusion_matrix(y_train, y_train_pred_classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1, 2, 3])

        # Save confusion matrix figure
        plt.figure()
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix: Dropout={dropout}, Batch size={batch_size}, Val split={val_split}, Activation={activation}, Neurons={neuron}")
        plt.savefig(f"confusion_matrix_dropout{dropout}_batch{batch_size}_val{val_split}_activation{activation}_neurons{neuron}.png")
        plt.close()

# Output the best result
print("Best hyperparameter combination based on validation accuracy:")
print(best_result)

# Save the best confusion matrix and training history plots
best_history = best_result['history']

# Plot training history for the best model
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(best_history.history['accuracy'], label='Train Accuracy')
plt.plot(best_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Best Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(best_history.history['loss'], label='Train Loss')
plt.plot(best_history.history['val_loss'], label='Validation Loss')
plt.title('Best Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig('best_model_training_history.png')
plt.show()
