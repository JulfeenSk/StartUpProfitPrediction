
import tkinter as tk
from tkinter import ttk
import joblib


# load models and transformer
ct = joblib.load("C:\\Users\\SHAIK JULFEEN AHMADH\\Downloads\\project\\temp\\col_transformer.joblib")
linear_regressor = joblib.load("C:\\Users\\SHAIK JULFEEN AHMADH\\Downloads\\project\\temp\\linear_regressor.joblib")
rf_regressor = joblib.load("C:\\Users\\SHAIK JULFEEN AHMADH\\Downloads\\project\\temp\\rf_regressor.joblib")

# Prediction function
def predict():
    rd_spending = float(rd_entry.get())
    administration = float(admin_entry.get())
    marketing_spending = float(marketing_entry.get())
    state = state_combobox.get()
    model = model_combobox.get()

    inputs = ct.transform([[rd_spending, administration, marketing_spending, state]])
    if model == "Randomforest Regressor":
        output = rf_regressor.predict(inputs)
    else:
        output = linear_regressor.predict(inputs)

    output_label.config(text=f"Predicted Profit: ${output[0]:,.2f}")

# Create main window
root = tk.Tk()
root.title("Profit Prediction")

# Create input fields
rd_label = tk.Label(root, text="R&D Spending:")
rd_label.grid(row=0, column=0, padx=10, pady=5)
rd_entry = tk.Entry(root)
rd_entry.grid(row=0, column=1, padx=10, pady=5)

admin_label = tk.Label(root, text="Administration:")
admin_label.grid(row=1, column=0, padx=10, pady=5)
admin_entry = tk.Entry(root)
admin_entry.grid(row=1, column=1, padx=10, pady=5)

marketing_label = tk.Label(root, text="Marketing Spending:")
marketing_label.grid(row=2, column=0, padx=10, pady=5)
marketing_entry = tk.Entry(root)
marketing_entry.grid(row=2, column=1, padx=10, pady=5)

state_label = tk.Label(root, text="State:")
state_label.grid(row=3, column=0, padx=10, pady=5)
state_combobox = ttk.Combobox(root, values=["New York", "California", "Florida"])
state_combobox.grid(row=3, column=1, padx=10, pady=5)

model_label = tk.Label(root, text="Model:")
model_label.grid(row=4, column=0, padx=10, pady=5)
model_combobox = ttk.Combobox(root, values=["Randomforest Regressor", "Linear Regressor"])
model_combobox.grid(row=4, column=1, padx=10, pady=5)

# Create button for prediction
predict_button = tk.Button(root, text="Predict", command=predict)
predict_button.grid(row=5, columnspan=2, padx=10, pady=10)

# Output label for displaying predicted profit
output_label = tk.Label(root, text="")
output_label.grid(row=6, columnspan=2, padx=10, pady=5)

# Run the main event loop
root.mainloop()
