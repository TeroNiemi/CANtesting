import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping


def hex_string_to_bytes(hex_string):
    bytes_list = [int(hex_string[i:i+2], 16) for i in range(0, len(hex_string), 2)]
    return bytes_list

train_df = pd.read_csv('train-classes.csv', sep=';')

train_df['features'] = train_df['Data'].apply(hex_string_to_bytes)
X_train = np.array(train_df['features'].tolist())
y_train = train_df['Type'].values

num_classes = 6
y_train_adjusted = y_train - 1
y_train_categorical = to_categorical(y_train_adjusted, num_classes)

model = Sequential()
model.add(Dense(16, input_dim=8, activation='relu'))
model.add(Dropout(0.5))  # Dropout 
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))  # Dropout 
model.add(Dense(num_classes, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)


history = model.fit(X_train, y_train_categorical, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Read test data
test_df = pd.read_csv('test.csv', sep=';')
test_hex_data = test_df['Data'][0]
test_features = np.array(hex_string_to_bytes(test_hex_data)).reshape(1, -1)

# Predict the type
pred_probs = model.predict(test_features)
pred_class = np.argmax(pred_probs, axis=1)
pred_type = pred_class[0] + 1  

print("Predicted type:", pred_type)

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
