import pickle

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

df_train = pd.read_csv("https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv")
df_train.to_csv()

df_train.drop_duplicates(subset=df_train.columns.drop("selling_price"), keep="first", inplace=True)
df_train.reset_index(drop=True, inplace=True)

X_train = df_train.drop(columns=["selling_price"])
y_train = df_train["selling_price"]


class NanFiller:
    def fit(self, X, y=None):
        self.medians = self.calculate_medians(X)
        return self

    def calculate_medians(self, X):
        r = {}
        for column in X.select_dtypes(np.number).columns:
            r[column] = X[column].median()
        return r

    def fill_with_median(self, X, medians):
        for column in X.select_dtypes(np.number).columns:
            X.fillna({column: medians[column]}, inplace=True)

    def transform(self, X):
        self.fill_with_median(X, self.medians)
        return X


class StringFieldsParser:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X["mileage"] = X["mileage"].map(lambda x: float(x.split(" ")[0]) if isinstance(x, str) else x)
        X["engine"] = X["engine"].map(lambda x: float(x.split(" ")[0]) if isinstance(x, str) else x)
        X["max_power"] = X["max_power"].map(lambda x: x.split(" ")[0] if isinstance(x, str) else x).map(
            lambda x: None if x == "" else float(x))
        return X.drop(columns=["name", "torque"])


class DropObjectColumns:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.select_dtypes(exclude=["object"])


ct = ColumnTransformer([
    ("scaler", StandardScaler(), ["year", "km_driven", "mileage", "engine", "max_power", "seats"]),
    ("categories_encoder", OneHotEncoder(sparse_output=False), ["fuel", "seller_type", "transmission", "owner"]),
])

ct.set_output(transform="pandas")

with open('best_params.pickle', 'rb') as file:
    best_params = pickle.load(file)

model = Pipeline(steps=[
    ("string_field_parser", StringFieldsParser()),
    ("nan_filler", NanFiller()),
    ("column_transformer", ct),
    ("drop_objects", DropObjectColumns()),

    ("elastic_net", ElasticNet(alpha=best_params["alpha"], l1_ratio=best_params["l1_ratio"])),
])

model.fit(X_train, y_train)

joblib.dump(model, "pipeline.joblib")


with open("pipeline.joblib", "rb") as file:
    pipeline = joblib.load(file)
