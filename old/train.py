import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import save_model
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
y_train = train_df['Type'].values

num_classes = 3
y_train_adjusted = y_train - 1  # Adjust for categorical encoding
y_train_categorical = to_categorical(y_train_adjusted, num_classes)

# Model definition
model = Sequential()
model.add(Dense(16, input_dim=8, activation='relu'))
model.add(Dropout(0.1))  
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.1))  
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=60, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train_categorical, epochs=200, batch_size=8, validation_split=0.1, callbacks=[early_stopping])

# Save the trained model
model.save('hex_classifier_model.h5')
print("Model saved as 'hex_classifier_model.h5'")

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
