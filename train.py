import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import time

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

print("Training model....")
time.sleep(3)

train_df['features'] = train_df['Data'].apply(hex_string_to_bytes)
X_train = np.array(train_df['features'].tolist())
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)  # Reshape for Conv1D input
y_train = train_df['Type'].values

num_classes = 3
y_train_adjusted = y_train - 1  # Adjust for categorical encoding
y_train_categorical = to_categorical(y_train_adjusted, num_classes)

# CNN model definition with padding
model = Sequential()
model.add(Conv1D(64, kernel_size=3, activation='tanh', input_shape=(8, 1), padding='same'))  # Padding to maintain input size
model.add(MaxPooling1D(pool_size=2))  # Max pooling layer
model.add(Dropout(0.4))  # Dropout for regularization

model.add(Conv1D(64, kernel_size=3, activation='tanh', padding='same'))  # Additional convolutional layer with padding
model.add(MaxPooling1D(pool_size=2))  # Max pooling layer
model.add(Dropout(0.4))

model.add(Flatten())  # Flatten to prepare for dense layers
model.add(Dense(32, activation='tanh'))
model.add(Dropout(0.4))

model.add(Dense(num_classes, activation='softmax'))  # Output layer

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# EarlyStopping callback
#early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train_categorical, epochs=50, batch_size=16, validation_split=0.1) #, callbacks=[early_stopping])

# Save the trained model
model.save('hex_classifier_model_cnn.h5')
print("CNN model with kernel size 3 and padding saved as 'hex_classifier_model_cnn.h5'")

# Predict on the training data to create confusion matrix
y_train_pred = model.predict(X_train)
y_train_pred_classes = np.argmax(y_train_pred, axis=1) + 1  # Convert back to original class labels (adjusting by +1)

# Confusion matrix
cm = confusion_matrix(y_train, y_train_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1, 2, 3])

# Plot confusion matrix
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Plot training history
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()
