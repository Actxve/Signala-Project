import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

#---Load data from CSV---#
data = pd.read_csv('landmark_data.csv')

X = data.drop(columns=['label']) # 63 coordinate columns (features)
y = data['label'] # 1 label column (targets)
#X is your input data which describes the hand pose
#y is your target which is the sign label

# --- ENCODE LABELS ---#
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("Classes:", label_encoder.classes_)

#---Convert to numpy arrays for training---#
X = X.to_numpy()
Y = y_encoded

#---Train/Test Split---#
X_train, X_test, Y_train, Y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
    )

print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

#---NN Model Architecture---#

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(63,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax'),
])

#---Compile the model---#
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],

)

model.summary()

#---Train the model---#
history = model.fit(
    X_train, Y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1

)

#---EVALUATE---#
test_loss, test_accuracy = model.evaluate(X_test, Y_test, verbose=0)
print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

#---SAVE MODEL---#
model.save('signala_model.keras')
print("Model saved as signala_model.keras")