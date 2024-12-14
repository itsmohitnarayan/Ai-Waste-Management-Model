import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import tkinter as tk
from tkinter import messagebox

# Load the trained model
model = tf.keras.models.load_model("waste_management_model.keras")

# Load the dataset to fit the scaler and label encoder
data = pd.read_csv(r"D:\2024\Ai-Waste-Management-Model\dataset.csv")

# Encode labels
label_encoder = LabelEncoder()
data["Condition"] = label_encoder.fit_transform(data["Condition"])

# Split features and labels
X = data.drop("Condition", axis=1)

# Normalize features
scaler = MinMaxScaler()
scaler.fit(X)  # Fit the scaler on the dataset

# -------------------------------
# Predict Single Input
# -------------------------------
def predict_condition(input_data):
    # Preprocess the input data
    input_data = scaler.transform([input_data])
    # Predict the condition
    prediction = model.predict(input_data)
    # Decode the prediction
    predicted_class = label_encoder.inverse_transform([prediction.argmax()])[0]
    return predicted_class

# Get input from the user
user_input = input("Enter sensor data ('Temperature,Humidity,Gas_Level,Vibration'): ")
user_input = list(map(float, user_input.split(',')))

# Predict and display the condition
predicted_condition = predict_condition(user_input)
print(f"The predicted condition of the dustbin is: {predicted_condition}")
# Function to handle prediction and display result
def on_predict():
    try:
        user_input = [float(entry.get()) for entry in entries]
        predicted_condition = predict_condition(user_input)
        messagebox.showinfo("Prediction Result", f"The predicted condition of the dustbin is: {predicted_condition}")
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter valid numeric values.")

# Create the main window
root = tk.Tk()
root.title("Waste Management Model Prediction")

# Create and place labels and entry widgets for user input
labels = ["Temperature", "Humidity", "Gas Level", "Vibration"]
entries = []

for i, label in enumerate(labels):
    tk.Label(root, text=label).grid(row=i, column=0, padx=10, pady=5)
    entry = tk.Entry(root)
    entry.grid(row=i, column=1, padx=10, pady=5)
    entries.append(entry)

# Create and place the Predict button
predict_button = tk.Button(root, text="Predict", command=on_predict)
predict_button.grid(row=len(labels), columnspan=2, pady=10)

# Run the GUI event loop
root.mainloop()