from __future__ import annotations
import os
from typing import Union
from click import format_filename
from dotenv import load_dotenv
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, Depends
from starlette.responses import FileResponse
from models.ml_model_processor import BaseDataProcessor, ModelTrainer, ModelPersistence
from fastapi.responses import JSONResponse
import random


load_dotenv()
FILE_PATH = os.getenv("FILE_PATH")

router = APIRouter()

class Change_DataType(BaseModel):
    columns: list[str]
    map_cat_datatype: Union[list[dict], None] = None

data_processor_map = dict()
model_trainer_map = dict()


def save_model_background(model_persistence: ModelPersistence, model_trainer: ModelTrainer ):
    model_persistence.save_model(model_trainer.model)

@router.post("/upload_csv")
async def upload_csv(target_class: str, excluded_col: str, file: UploadFile = File(...)):
    hash = random.getrandbits(128)
    print(hash,file.filename, target_class, excluded_col)
    file_path = f'/{FILE_PATH}/{hash}_csv.csv'
    with open(file_path, "wb") as f:
        f.write(await file.read())
    dataprocessor = BaseDataProcessor(file_path, target_class, excluded_col)
    data_processor_map[f"{hash}"] = dataprocessor
    return {"message": "Data has been loaded", "hash_key": f"{hash}"}

@router.get("/get_datatypes/")
async def get_data_types(hash: str):
    print(hash, data_processor_map)
    data_processor = data_processor_map[hash]
    num_cols, cat_cols = data_processor.get_datatypes()
    return {'numerical_columns': num_cols, 'categorical_columns': cat_cols}

@router.post("/change_obj_numeric/")
async def change_datatypes(hash: str, body: Change_DataType):
    li =data_processor_map[hash].change_datatypes(body.columns)
    return JSONResponse(status_code=200, content={'msg':'data types changed','updated_numeric_columns':li})
    # return {'msg':'data types changed','updated_columns':li}

@router.post("/map_obj_category/")
async def map_cat_datatype(hash: str, body: Change_DataType):
    li =data_processor_map[hash].map_categorical_columns(body.columns, body.map_cat_datatype)
    return JSONResponse(status_code=200, content={'msg':'data types changed','updated_categorical_columns':li})
@router.get("/find_nulls/")
async def find_nulls(hash: str):
    resp_dict = data_processor_map[hash].find_nulls()
    return JSONResponse(status_code=200, content=resp_dict)

@router.post("/fix_nulls/")
async def fix_nulls(hash: str):
    resp_list = data_processor_map[hash].fix_nulls()
    return JSONResponse(status_code=200, content={'effected columns':resp_list})

@router.get("/preprocessing_pipeline/{hash}")
async def preprocessing_pipeline(hash: str):
    data_processor_map[hash].preprocessing_pipeline()
    return JSONResponse(status_code=200, content={"message": "Data has been preprocessed. Use hashId to train the model", "hash_key": hash})

@router.get("/train_model/")
async def train_model(hash: str, model_name: str):
   model_trainer = ModelTrainer(model_name, data_processor_map[hash])
   model_trainer.train_model()
   model_trainer_map[hash] = model_trainer
   return {"message": "Model has been trained", "hash_key": hash}



@router.get("/predict_model")
async def predict_model(hash: str):
    model_trainer = model_trainer_map[hash]
    resp_dict = model_trainer.model_prediction()
    return JSONResponse(status_code=200, content=resp_dict)

@router.get("/evaluate_model/")
async def evaluate_model(hash: str):
    model = model_trainer_map[hash]
    resp_dict = model.evaluate_model()
    return JSONResponse(status_code=200, content=resp_dict)


@router.post("/save_model")
async def save_model(hash: str, model_name: str, background_tasks:BackgroundTasks):
    model_trainer = model_trainer_map[hash]
    model_persistence = ModelPersistence(model_name, hash)

    background_tasks.add_task(model_persistence.save_model, model_trainer.model)

    return JSONResponse(status_code=200, content={"message": "model is being saved.", "file_name": f"{model_name}_{hash}.joblib"})
    
@router.get("/download_model")
async def download_model(filename: str):
    path = FILE_PATH
    if not os.path.exists(f"{path}/{filename}"):
        raise HTTPException(status_code=404, detail="Model file not found")

    return FileResponse(f"{path}/{filename}", media_type='application/octet-stream', filename=filename)


@router.get("/get_models")
async def root():
    return {"message": "Hello World"}
