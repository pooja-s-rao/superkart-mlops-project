
import gradio as gr
import pandas as pd
import joblib

# Load trained model
model = joblib.load("sales_forecast_model.pkl")

# IMPORTANT: define expected columns (based on training)
# We will simulate input with correct structure

def predict(product_weight, product_mrp, store_age):
    
    # Create dataframe with required columns
    data = {
        'Product_Weight': product_weight,
        'Product_MRP': product_mrp,
        'Store_Age': store_age
    }
    
    df = pd.DataFrame([data])
    
    # NOTE: model was trained on many encoded columns
    # So we align structure by reindexing
    model_features = model.feature_names_in_
    
    df = pd.get_dummies(df)
    
    df = df.reindex(columns=model_features, fill_value=0)
    
    prediction = model.predict(df)
    
    return float(prediction[0])


interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Product Weight"),
        gr.Number(label="Product MRP"),
        gr.Number(label="Store Age")
    ],
    outputs="number",
    title="Sales Forecast Prediction App"
)

interface.launch()
