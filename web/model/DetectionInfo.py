import json

class DetectionInfo:
    def __init__(self):
        self.modelList = []

    def addModel(self, detectionModel):
        self.modelList.append(detectionModel)

    def getModelList(self):
        return self.modelList
    
    def getModelListByJson(self):
        return json.dumps([model.getJsonInfo() for model in self.modelList], indent=4)
        
        
class DetectionModel:
    def __init__(self, name, cam, x, y, z, w):
        self.name = name
        self.cam = cam
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def getLocation(self):
        return [self.x, self.y, self.z, self.w]

    def getName(self):
        return self.name
    
    def updateLocation(self, x, y, z, w):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def printLocation(self):
        print('name '+str(self.name)+' '+'Location XYZW : '+str(self.x)+' '+str(self.y)+' '+str(self.z)+' '+str(self.w))

    def find(self, cam):
        if(self.cam == cam):
            return DetectionModel(self.cam, self.x, self.y, self.z, self.w)

    def getJsonInfo(self):
        jsonify = {
            'name' : str(self.name),
            'camInfo' : str(self.cam),
            'Points' : {
                'X':int(self.x),
                'Y':int(self.y),
                'Z':int(self.z),
                'W':int(self.w)
            }
        }
        return json.dumps(jsonify)