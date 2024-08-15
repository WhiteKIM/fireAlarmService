"""
# Detection RestAPI Controller
# 2024.06.26 - Init
"""
from fastapi import FastAPI
import uvicorn

from web.model.InputSource import InputSource

# import My Modules
import init

def initApiService():
    app = FastAPI()

    @app.get("/")
    async def main():
        return {"message" : "test"}

    @app.get("/DetectionList")
    async def getDetectionList():
        return init.sharedData

    @app.post("/addSource")
    async def addInputSource(InputSource : InputSource):
        return None
    
    # Uvicorn 서버를 실행합니다.
    uvicorn.run(app, host="127.0.0.1", port=8000)