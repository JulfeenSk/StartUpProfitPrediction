import gradio as gr
import joblib

# load models and transformer
ct = joblib.load("C:\\Users\\SHAIK JULFEEN AHMADH\\Downloads\\project\\temp\\col_transformer.joblib")
linear_regressor = joblib.load("C:\\Users\\SHAIK JULFEEN AHMADH\\Downloads\\project\\temp\\linear_regressor.joblib")
rf_regressor = joblib.load("C:\\Users\\SHAIK JULFEEN AHMADH\\Downloads\\project\\temp\\rf_regressor.joblib")



def predict(R_and_D_Spending,	Administration,	Marketing_spending,	State, model = "Randomforest Regressor"):
    rdspending = int(R_and_D_Spending)
    admin = int(Administration)
    mspending = int(Marketing_spending)
    s = State

    inputs = ct.transform([[rdspending, admin, mspending, s]])
    print(inputs)

    if model == "Randomforest Regressor":
        print("Using Randomforest Regression")
        model = rf_regressor
    else:
        print("Using Linear Regression")
        model = linear_regressor
    output = model.predict(inputs)

    return output



demo = gr.Interface(
    title="Profit Prediction",
    fn=predict,
    inputs=[gr.Text(label='R_and_D_Spending'),
            gr.Text(label='Administration'),
            gr.Text(label='Marketing_spending'),
            gr.Dropdown(['New York', 'California', 'Florida']),
            gr.Dropdown(["Randomforest Regressor", "LInear Regrssor"])],
    outputs="number")



if __name__ == "__main__":
    demo.launch(share=True)



