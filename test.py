import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

def hex_string_to_bytes(hex_string):
    bytes_list = [int(hex_string[i:i+2], 16) for i in range(0, len(hex_string), 2)]
    return bytes_list

# Load the model
model = load_model('hex_classifier_model_cnn.h5')
print("Model loaded successfully")

# Read test data with 'Data' column forced as string
test_df = pd.read_csv('test.csv', sep=';', dtype={'Data': str})
test_hex_data = test_df['Data'][0]

# Convert the hex string to bytes
test_features = np.array(hex_string_to_bytes(test_hex_data)).reshape(1, -1)

# Predict the type
pred_probs = model.predict(test_features)
pred_class = np.argmax(pred_probs, axis=1)
pred_type = pred_class[0] + 1  # Adjust back to original class

print("Predicted type:", pred_type)
