import os

from dotenv import load_dotenv
from joblib import dump
import random
import numpy as np
from fastapi import FastAPI, UploadFile, File, Depends
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, auc, f1_score, ConfusionMatrixDisplay, precision_score, recall_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVC
from xgboost import XGBClassifier

load_dotenv()
FILE_PATH = os.getenv("FILE_PATH")

class BaseDataProcessor:
    def __init__(self, csv_file: str, target_class: str, excluded_col: str):
        self.preprocessor = None
        self.y = None
        self.X = None
        self.column_transformer = None
        self.numerical_columns = None
        self.categorical_columns = None
        self.target_class = target_class
        self.df = self.load_data(csv_file, excluded_col)

    def load_data(self, csv_file: str, excluded_col: str):
        # Load CSV into a DataFrame
        return pd.read_csv(csv_file, index_col = excluded_col)

    def get_datatypes(self):
        self.numerical_columns = self.df.select_dtypes(include=['int64', 'float64']).columns
        self. categorical_columns = self.df.select_dtypes(include=['object']).columns
        num_list = self.numerical_columns.tolist()
        cat_list = self.categorical_columns.tolist()
        print(num_list, cat_list)
        if self.target_class in num_list:
            num_list.remove(self.target_class)
        if self.target_class in cat_list:
            cat_list.remove(self.target_class)
        return num_list, cat_list
        pass

    def change_datatypes(self, columns: list[str]):
        for col in columns:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        self.categorical_columns = self.df.select_dtypes(include=['object']).columns
        self.numerical_columns = self.df.select_dtypes(include=['int64', 'float64']).columns

        return self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    def map_categorical_columns(self, columns: list[str], map_list: list[dict]):
        for col, dtype_map in zip(columns, map_list):
            int_dtype_map = {int(k): v for k, v in dtype_map.items()}
            self.df[col] = self.df[col].replace(int_dtype_map)

        self.categorical_columns = self.df.select_dtypes(include=['object']).columns
        self.numerical_columns = self.df.select_dtypes(include=['int64', 'float64']).columns

        return self.df.select_dtypes(include=['object']).columns.tolist()



    def find_nulls(self):
        null_dict = {}
        for col in self.df.columns:
            if np.True_ == self.df[col].isnull().values.any():
                null_dict[col] =  True
            else:
                null_dict[col] = False
        return null_dict

    def fix_nulls(self):
        updated_columns = []
        for col in self.df.columns:
            if self.df[col].isnull().values.any() == np.True_:
                imputer = SimpleImputer(strategy='median')
                self.df[[col]] = imputer.fit_transform(self.df[[col]])
                updated_columns.append(col)
        return updated_columns

    def log_transform(self, X):
        return np.log1p(X.clip(lower=0))

    def preprocessing_pipeline(self):

        numerical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        categorical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        log_tr_list = [col for col in self.numerical_columns if self.df[col].skew() >= 0.65]
        cat_list = [col for col in self.categorical_columns if self.target_class not in col]
        self.column_transformer = ColumnTransformer(
            transformers=[
                ('num', numerical_pipeline, self.numerical_columns),
                ('cat', categorical_pipeline, cat_list),  # One-hot encode categorical feature
                ('log', FunctionTransformer(self.log_transform), log_tr_list)
            ]
        )
        self.preprocessor = Pipeline(steps=[('preprocessor', self.column_transformer)])

        self.X = self.df.drop(columns=self.target_class)

        # self.df["Churn"] = self.df["Churn"].map({"Yes": 1, "No": 0})
        self.y = self.df[self.target_class]


class ModelTrainer:
    def __init__(self, model_name: str, data_processor :BaseDataProcessor):
        self.predictions = None
        self.model_name = model_name
        self.target = data_processor.target_class
        self.model_pipeline = Pipeline([
            ('preprocessor', data_processor.preprocessor),
            ('classifier', self.initialize_model())
        ])
        self.model = None
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data_processor.X, data_processor.y, test_size=0.2, random_state=42)

    def initialize_model(self):
            if self.model_name == "LR":
                return LogisticRegression(max_iter=1000)
            elif self.model_name == 'RF':
                return RandomForestClassifier()
            elif self.model_name == 'SVM':
                return SVC()
            elif self.model_name == 'XGB':
                return XGBClassifier(eval_metric='logloss')

    def train_model(self):
        self.model = self.model_pipeline.fit(self.X_train, self.y_train)

    def model_prediction(self):
        self.predictions = self.model.predict(self.X_test)
        print(set(self.y_test))
        print(set(self.predictions))

        return self.evaluate_model()

    def evaluate_model(self):
        return {
            "accuracy": accuracy_score(self.y_test, self.predictions),
            "precision": precision_score(self.y_test, self.predictions, pos_label='Yes'),
            "recall": recall_score(self.y_test, self.predictions, pos_label='Yes'),
            "f1": f1_score(self.y_test, self.predictions, pos_label='Yes'),
        }

        pass

class ModelPersistence:
    def __init__(self, model_name, hash):
        self.model_name = f"{model_name}_{hash}"

    def save_model(self, model):
        file_name = f"{self.model_name}.joblib"
        output_path = FILE_PATH
        dump(model, f"{output_path}/{file_name}")
