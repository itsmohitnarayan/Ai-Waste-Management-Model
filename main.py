import pandas as pd
import tensorflow as tf
from keras.src.layers import LSTM, GRU, Bidirectional, Dense, Input
from keras.src.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt
from keras.src.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# -------------------------------
# Load and preprocess dataset
# -------------------------------
# Load data from dataset.csv
data = pd.read_csv(r"D:\2024\Ai-Waste-Management-Model\dataset.csv")

# Encode labels
label_encoder = LabelEncoder()
data["Condition"] = label_encoder.fit_transform(data["Condition"])

# Split features and labels
X = data.drop("Condition", axis=1)
y = data["Condition"]

# Normalize features
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# Define AI Model
# -------------------------------
input_layer = Input(shape=(X_train.shape[1], 1), name="sensor_input")

# For LSTM
x = LSTM(64, activation="relu")(input_layer)

# For GRU
# x = GRU(64, activation="relu")(input_layer)

# For Bidirectional LSTM
# x = Bidirectional(LSTM(64, activation="relu"))(input_layer)

x = Dense(32, activation="relu")(x)
output_layer = Dense(3, activation="softmax", name="condition_output")(x)

model = Model(inputs=input_layer, outputs=output_layer)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Reshape data for RNN
X_train_rnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_rnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
history = model.fit(X_train_rnn, y_train, validation_data=(X_test_rnn, y_test), 
                    epochs=100, batch_size=32, callbacks=[early_stopping])

loss, accuracy = model.evaluate(X_test_rnn, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save the trained model
model.save("waste_management_model.keras")
print("Model saved as 'waste_management_model.keras'")


# Plot training & validation accuracy values
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

y_pred = model.predict(X_test_rnn)
y_pred_classes = y_pred.argmax(axis=1)
cm = confusion_matrix(y_test, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()