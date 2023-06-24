# -*- encoding: utf-8 -*-

from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI

from pydantic import BaseModel
import statsmodels.api as sm
from datetime import date
import pandas as pd
import numpy as np
import functools
import json

app = FastAPI()
origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
    "*"
]
train = pd.read_excel("data/train.xlsx").set_index("dt")
test = pd.read_excel("data/test.xlsx").set_index("dt")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])


class PredictRequest(BaseModel):
    start: date
    end: date


def sarimax(data, step):
    mod = sm.tsa.statespace.SARIMAX(data,
                                    order=(0,2,1),
                                    seasonal_order=(1,1,2,12),
                                    freq="W-MON",
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    results = mod.fit()
    return results.forecast(steps=step)


@functools.lru_cache()
def predict():
    preds = np.array([])
    choices = []
    step = 5
    for i in range(0, len(test)):
        data = pd.concat([train['Цена на арматуру'], test['Цена на арматуру'][:i]])
        fore = sarimax(data, step)
        preds = np.append(preds, fore[2])

        if data[-1] < np.min(fore):
            choices.append(max(0, step - (sum(choices) - i)))
        else:
            choices.append(max(0, 1 - (sum(choices) - i)))
    df = test.copy().reset_index().rename(columns={'dt': 'date', 'Цена на арматуру': 'price'})
    df['preds'] = preds
    df['quantity'] = choices
    return df


@app.post("/predict/")
async def predict_handler(request: PredictRequest):
    df = predict()
    res = df.to_json(orient='records')

    return json.loads(res)


@app.get("/excel/")
def read_root():
    df = predict()
    df.to_excel('data/result.xlsx', index=False)

    return FileResponse(path='data/result.xlsx', filename='result.xlsx', media_type='multipart/form-data')
