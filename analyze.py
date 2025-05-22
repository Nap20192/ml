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
        """Загрузка данных из CSV-файла"""
        if self.wine_type == 'red':
            return pd.read_csv("./wine/winequality-red.csv", sep=";")
        elif self.wine_type == 'white':
            return pd.read_csv("./wine/winequality-white.csv", sep=";")
        else:
            raise ValueError("Invalid wine type. Use 'red' or 'white'")
    
    def _prepare_data(self):
        """Предварительная обработка данных"""
        self.data['quality_category'] = pd.cut(self.data['quality'],
                                             bins=[0, 4, 6, 10],
                                             labels=['Low', 'Medium', 'High'])
    
    def show_summary_statistics(self):
        """Возвращает основные статистические показатели"""
        return self.data.describe()
    
    def plot_quality_distribution(self, figsize=(10, 6)):
        """Визуализация распределения качества вина"""
        plt.figure(figsize=figsize)
        sns.countplot(x='quality', data=self.data)
        plt.title(f'{self.wine_type.title()} Wine Quality Distribution')
        plt.xlabel('Quality Score')
        plt.ylabel('Count')
        plt.savefig(f'{self.wine_type}_quality_distribution.png')
        plt.close()
    
    def plot_feature_distributions(self, figsize=(15, 20)):
        """Визуализация распределения всех характеристик"""
        plt.figure(figsize=figsize)
        self.data.drop(columns=['quality', 'quality_category']).hist(bins=20)
        plt.suptitle(f'{self.wine_type.title()} Wine Feature Distributions', y=1.02)
        plt.tight_layout()
        plt.savefig(f'{self.wine_type}_feature_distributions.png')
        plt.close()
    
    def plot_correlation_matrix(self, figsize=(12, 10)):
        """Визуализация матрицы корреляций"""
        plt.figure(figsize=figsize)
        corr = self.data.corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm',
                   annot_kws={"size": 10}, linewidths=0.5)
        plt.title(f'{self.wine_type.title()} Wine Correlation Matrix')
        plt.savefig(f'{self.wine_type}_correlation_matrix.png')
        plt.close()
    
    def plot_feature_vs_quality(self, feature, figsize=(10, 6)):
        """Визуализация зависимости качества от конкретной характеристики"""
        plt.figure(figsize=figsize)
        sns.boxplot(x='quality', y=feature, data=self.data)
        plt.title(f'{feature} vs Quality for {self.wine_type.title()} Wine')
        plt.savefig(f'{self.wine_type}_{feature}_vs_quality.png')
        plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred, figsize=(10, 8)):
        """Визуализация матрицы ошибок"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'{self.wine_type}_confusion_matrix.png')
        plt.close()
    
    def generate_full_report(self):
        """Генерация полного отчета с графиками"""
        self.plot_quality_distribution()
        self.plot_feature_distributions()
        self.plot_correlation_matrix()
        for feature in self.data.columns.drop(['quality', 'quality_category']):
            self.plot_feature_vs_quality(feature)
        return self.show_summary_statistics()

if __name__ == "__main__":
    # Пример использования
    red_analyzer = WineDataAnalyzer('red')
    white_analyzer = WineDataAnalyzer('white')
    
    # Генерация отчетов
    print("Red Wine Summary Statistics:")
    print(red_analyzer.show_summary_statistics())
    
    print("\nWhite Wine Summary Statistics:")
    print(white_analyzer.show_summary_statistics())
    
    # Генерация графиков
    red_analyzer.generate_full_report()
    white_analyzer.generate_full_report()