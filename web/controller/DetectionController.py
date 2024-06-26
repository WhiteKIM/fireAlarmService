"""
# Detection RestAPI Controller
# 2024.06.26 - Init
"""
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def main():
    return {"message" : "test"}

@app.get("/DetectionList")
async def getDetectionList():
    return []