import io
from dataclasses import dataclass
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from starlette.responses import StreamingResponse
from pipeline import StringFieldsParser

app = FastAPI()

with open("pipeline.joblib", "rb") as file:
    pipeline = joblib.load(file)


class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


@dataclass
class PricedItem:
    item: Item
    price: int


class Items(BaseModel):
    objects: List[Item]


@app.post("/predict_item")
def predict_item(item: Item) -> PricedItem:
    price = pipeline.predict(pd.DataFrame([vars(item)]))
    return PricedItem(item, int(price[0]))


@app.post("/predict_items")
def predict_items(items_file: UploadFile):
    items = pd.read_csv(items_file.file)[[
        "name",
        "year",
        "km_driven",
        "fuel",
        "seller_type",
        "transmission",
        "owner",
        "mileage",
        "engine",
        "max_power",
        "torque",
        "seats",
    ]]

    items["predicted_price"] = pipeline.predict(items)
    stream = io.StringIO()
    items.to_csv(stream, index=False)

    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=result.csv"
    return response
