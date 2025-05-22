import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

class WineDataAnalyzer:
    def __init__(self, wine_type):
        self.wine_type = wine_type
        self.data = self._load_data()
        self._prepare_data()
        
    def _load_data(self):
        if self.wine_type == 'red':
            return pd.read_csv("./wine/winequality-red.csv", sep=";")
        elif self.wine_type == 'white':
            return pd.read_csv("./wine/winequality-white.csv", sep=";")
        else:
            raise ValueError("Invalid wine type. Use 'red' or 'white'")
    
    def _prepare_data(self):
        self.data = self.data.drop(columns=['quality_category'], errors='ignore')
    
    def show_summary_statistics(self):
        return self.data.describe()
    
    def plot_quality_distribution(self, figsize=(10, 6)):
        sns.set_style("darkgrid")

        plt.figure(figsize=figsize)
        ax = sns.countplot(
            x='quality',
            data=self.data,
            order=sorted(self.data['quality'].unique()),
            hue='quality',
            palette='plasma' 
        )
        ax.set_facecolor("#222222")
        ax.figure.set_facecolor("#222222")
        ax.set_title(f'{self.wine_type.title()} Wine Quality Distribution', fontsize=16, weight='bold', color='white')
        ax.set_xlabel('Quality Score', fontsize=14, color='white')
        ax.set_ylabel('Count', fontsize=14, color='white')
        ax.tick_params(axis='x', colors='white', labelsize=12)
        ax.tick_params(axis='y', colors='white', labelsize=12)
        ax.grid(True, color='gray', linestyle='--', alpha=0.5)

        plt.tight_layout()
        return plt.gcf()
    
    def plot_feature_distributions(self, figsize=(15, 20)):
        sns.set(style="darkgrid")
        features = self.data.drop(columns=['quality'])

        fig, axes = plt.subplots(
            nrows=(len(features.columns) + 2) // 3,
            ncols=3,
            figsize=figsize,
            facecolor="#222222"
        )

        axes = axes.flatten()
        for idx, col in enumerate(features.columns):
            ax = axes[idx]
            ax.hist(features[col], bins=20, color='skyblue', edgecolor='white')
            ax.set_title(col, color='white')
            ax.set_facecolor("#222222")
            ax.tick_params(colors='white')


        for ax in axes[len(features.columns):]:
            ax.set_visible(False)

        fig.suptitle(f'{self.wine_type.title()} Feature Distributions', y=0.98, fontsize=18, weight='bold', color='white')
        fig.tight_layout()
        return fig


    
    def plot_feature_vs_quality(self, feature, figsize=(10, 6)):
        plt.figure(figsize=figsize)
        sns.boxplot(x='quality', y=feature, data=self.data)
        plt.title(f'{feature} vs Quality for {self.wine_type.title()} Wine')
        return plt.gcf()
    
    def plot_confusion_matrix(self, y_true, y_pred, figsize=(10, 8)):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        return plt.gcf()
    
    def generate_full_report(self):
        self.plot_quality_distribution()
        self.plot_feature_distributions()
        for feature in self.data.columns.drop(['quality']):
            self.plot_feature_vs_quality(feature)
        return self.show_summary_statistics()

if __name__ == "__main__":
    red_analyzer = WineDataAnalyzer('red')
    white_analyzer = WineDataAnalyzer('white')
    
    print("Red Wine Summary Statistics:")
    print(red_analyzer.show_summary_statistics())
    
    print("\nWhite Wine Summary Statistics:")
    print(white_analyzer.show_summary_statistics())
    
    red_analyzer.generate_full_report()
    white_analyzer.generate_full_report()