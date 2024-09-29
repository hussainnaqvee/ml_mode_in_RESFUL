import os

from fastapi import FastAPI, UploadFile, File, Depends
from controllers.controller import router
from dotenv import load_dotenv

load_dotenv()
FILE_PATH = os.getenv("FILE_PATH")
app = FastAPI()
app.include_router(router)

