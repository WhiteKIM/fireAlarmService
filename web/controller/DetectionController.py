"""
# Detection RestAPI Controller
# 2024.06.26 - Init
"""
from fastapi import FastAPI

from web.model.InputSource import InputSource

# import My Modules
import init

app = FastAPI()

@app.get("/")
async def main():
    return {"message" : "test"}

@app.get("/DetectionList")
async def getDetectionList():
    return init.sharedData

@app.post("/addSource")
async def addInputSource(InputSource : InputSource):
    return