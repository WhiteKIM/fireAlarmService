from pydantic import BaseModel

class DetectionInfo(BaseModel):
    modelList : list
    def addModel(self, detectionModel):
        self.modelList.append(detectionModel)

    def getModelList(self):
        return self.modelList
        
        
class DetectionModel(BaseModel):
    name : str
    cam : str
    minX : int
    minY : int
    maxX : int
    maxY : int

    def getLocation(self):
        return [self.minX, self.minY, self.maxX, self.maxY]

    def getName(self):
        return self.name
    
    def updateLocation(self, minX, minY, maxX, maxY):
        self.minX = minX
        self.minY = minY
        self.maxX = maxX
        self.maxY = maxY

    def getJsonInfo(self):
        return self