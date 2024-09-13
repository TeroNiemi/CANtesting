import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam





model_filepath = 'classification_model.h5'

def create_model(input_dim, output_dim):
    model = Sequential([
        Dense(64, input_dim=input_dim, activation='relu'),
        Dense(64, activation='relu'),
        Dense(output_dim, activation='sigmoid')  
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

train_df = pd.read_csv('train-classes.csv', delimiter=';')

X = train_df.iloc[:, :8].values  
y = train_df.iloc[:, 8:].values  

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

if os.path.exists(model_filepath):
    user_input = input("A trained model exists. Do you want to load it (y) or train a new one (n)? ").strip().lower()
    if user_input == 'y':
        model = load_model(model_filepath)
        print("Model loaded successfully.")
    else:
        model = create_model(X_train.shape[1], y_train.shape[1])
        print("Training a new model...")
        
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32)
        
        model.save(model_filepath)
        print(f"Model trained and saved as {model_filepath}.")
else:
    model = create_model(X_train.shape[1], y_train.shape[1])
    print("No previous model found. Training a new model...")
   
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=32)
    
    model.save(model_filepath)
    print(f"Model trained and saved as {model_filepath}.")

test_df = pd.read_csv('test.csv', delimiter=';')
X_test = test_df.values

X_test = scaler.transform(X_test)

predictions = model.predict(X_test)

percentages = predictions * 100

# 50% treshold for others class
threshold = 50.0


classes = ['Current', 'Voltage', 'Service/Warnings', 'Overtemp', 'SOC', 
           'Temperature', 'Charge/discharge limits', 'Cell voltage data', 'Other/Unknown']

for i, prediction in enumerate(percentages):
    print("Predicted content of data:")
    # predicted category or other (probs < 50 of all)
    if all(prob < threshold for prob in prediction):
        print("DATA CANNOT BE CLASSIFIED CORRECTLY, POSSIBLE UNKNOWN DATA")
    else:
        for j, percent in enumerate(prediction):
            print(f"{classes[j]}: {percent:.2f} %")
    print()

# Plot 
if 'history' in locals():
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.tight_layout()
    plt.show()
