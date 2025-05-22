import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix
import numpy as np
import pandas as pd

class WineModel:
    def __init__(self, wine_type):
        self.wine_type = wine_type
        self.model = None
        self.X_test = None
        self.y_test = None
        self.min_quality = None
        self.accuracy = None
        self.loss = None
        self.classification_report = None
        self.confusion_matrix = None
        self._load_data()
        self._train_model()
        self._evaluate_model()
        
    def _load_data(self):
        if self.wine_type == 'red':
            data = pd.read_csv("./wine/winequality-red.csv", sep=";")
        elif self.wine_type == 'white':
            data = pd.read_csv("./wine/winequality-white.csv", sep=";")
        else:
            raise ValueError("wine_type must be 'red' or 'white'")
        
        self.min_quality = data["quality"].min()
        self.X = data.drop(columns=["quality"])
        self.y = data["quality"] - self.min_quality  
        
    def _train_model(self):
        X_train, self.X_test, y_train, self.y_test = train_test_split(
            self.X, self.y, 
            test_size=0.1, 
            random_state=42, 
            stratify=self.y  
        )
        
        self.model = xgb.XGBClassifier(eval_metric="mlogloss")
        self.model.fit(X_train, y_train)
        
    def _evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        y_proba = self.model.predict_proba(self.X_test)
        
        self.accuracy = accuracy_score(self.y_test, y_pred)
        self.loss = log_loss(self.y_test, y_proba, labels=self.model.classes_)
        self.classification_report = classification_report(
            self.y_test, y_pred, 
            output_dict=True, 
            zero_division=0  
        )
        self.confusion_matrix = confusion_matrix(self.y_test, y_pred)
        
    def predict_quality(self, fixed_acidity, volatile_acidity, citric_acid, 
                       residual_sugar, chlorides, free_sulfur_dioxide, 
                       total_sulfur_dioxide, density, pH, sulphates, alcohol):
        input_data = np.array([[
            fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
            chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
            pH, sulphates, alcohol
        ]])
        prediction = self.model.predict(input_data)
        return prediction[0] + self.min_quality  

if __name__ == "__main__":
    red_model = WineModel('red')
    white_model = WineModel('white')
    print("Red Wine Model Metrics:")
    print(f"Accuracy: {red_model.accuracy:.4f}")
    print(f"Log Loss: {red_model.loss:.4f}")
    print("Classification Report:")
    print(pd.DataFrame(red_model.classification_report).transpose())
    print("Confusion Matrix:")
    print(red_model.confusion_matrix)
    print("\nWhite Wine Model Metrics:")
    print(f"Accuracy: {white_model.accuracy:.4f}")
    print(f"Log Loss: {white_model.loss:.4f}")
    print("Classification Report:")
    print(pd.DataFrame(white_model.classification_report).transpose())
    print("Confusion Matrix:")
    print(white_model.confusion_matrix)