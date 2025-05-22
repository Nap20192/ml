from model import WineModel
import gradio as gr

# Initialize models for red and white wine
red_model = WineModel('red')
white_model = WineModel('white')

# Define a prediction function for Gradio
def predict_wine_quality(wine_type, fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                         free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol):
    model = red_model if wine_type == "Red" else white_model
    prediction = model.predict_quality(fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                                        free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol)
    return f"Predicted Quality: {prediction}"

# Create a Gradio interface
inputs = [
    gr.Radio(choices=["Red", "White"], label="Wine Type"),
    gr.Number(label="Fixed Acidity"),
    gr.Number(label="Volatile Acidity"),
    gr.Number(label="Citric Acid"),
    gr.Number(label="Residual Sugar"),
    gr.Number(label="Chlorides"),
    gr.Number(label="Free Sulfur Dioxide"),
    gr.Number(label="Total Sulfur Dioxide"),
    gr.Number(label="Density"),
    gr.Number(label="pH"),
    gr.Number(label="Sulphates"),
    gr.Number(label="Alcohol"),
]

outputs = gr.Textbox(label="Wine Quality Prediction")

gr.Interface(fn=predict_wine_quality, inputs=inputs, outputs=outputs, title="Wine Quality Predictor").launch()