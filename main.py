import gradio as gr
import matplotlib.pyplot as plt
import seaborn as sns
from model import WineModel
from wine_analyzer import WineDataAnalyzer

red_model = WineModel('red')
white_model = WineModel('white')
red_analyzer = WineDataAnalyzer('red')
white_analyzer = WineDataAnalyzer('white')

class GradioWineDataAnalyzer(WineDataAnalyzer):
    def plot_quality_distribution(self, figsize=(10, 6)):
      plt.figure(figsize=figsize)
      sns.countplot(x='quality', data=self.data)
      plt.title(f'{self.wine_type.title()} Wine Quality Distribution')
      return plt.gcf()

    def plot_feature_distributions(self, figsize=(15, 20)):
      plt.figure(figsize=figsize)
      self.data.drop(columns=['quality']).hist(bins=20)
      plt.suptitle(f'{self.wine_type.title()} Feature Distributions', y=1.02)
      plt.tight_layout()
      return plt.gcf()

def predict_quality(wine_type, *args):
    model = red_model if wine_type == "Red" else white_model
    prediction = model.predict_quality(*args)
    return f"Predicted Quality: {prediction}"

def update_plots(wine_type):
    analyzer = red_analyzer if wine_type == "Red" else white_analyzer
    return (
        analyzer.plot_quality_distribution(),
        analyzer.plot_feature_distributions()
    )

with gr.Blocks(title="Wine Quality App") as app:
    gr.Markdown("# 🍷 Wine Quality Prediction & Analysis")
    with gr.Row():
        wine_type = gr.Radio(
            choices=["Red", "White"],
            value="Red",
            label="Wine Type"
        )
    
    with gr.Tabs():
        with gr.Tab("Prediction"):
            with gr.Row():
                with gr.Column():
                    inputs = [
                        gr.Number(label="Fixed Acidity", value=7.4, step=0.01),
                        gr.Number(label="Volatile Acidity", value=0.7,step=0.01),
                        gr.Number(label="Citric Acid", value=0.0, step=0.01),
                        gr.Number(label="Residual Sugar", value=1.9, step=0.1),
                        gr.Number(label="Chlorides", value=0.076, step=0.001),
                        gr.Number(label="Free Sulfur Dioxide", value=11.0, step=1.0),
                        gr.Number(label="Total Sulfur Dioxide", value=34.0, step=1.0),
                        gr.Number(label="Density", value=0.9978, step=0.001),
                        gr.Number(label="pH", value=3.51, step=0.01),
                        gr.Number(label="Sulphates", value=0.56, step=0.01),
                        gr.Number(label="Alcohol", value=9.4, step=0.1),
                    ]
                with gr.Column():
                    output = gr.Textbox(label="Prediction Result")
                    predict_btn = gr.Button("Predict Quality")
        
        with gr.Tab("Data Analysis"):
            with gr.Tabs():
                with gr.Tab("Quality Distribution"):
                    quality_plot = gr.Plot()
                
                with gr.Tab("Feature Distributions"):
                    feature_plot = gr.Plot()

    predict_btn.click(
        predict_quality,
        inputs=[wine_type] + inputs,
        outputs=output
    )
    
    wine_type.change(
        update_plots,
        inputs=wine_type,
        outputs=[quality_plot, feature_plot]
    )

if __name__ == "__main__":
  app.launch(share=True)