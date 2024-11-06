from pydantic import BaseModel

class Item(BaseModel):
    data: list[list[float]]

class ModelItem(BaseModel):
    modelname: str
    modelversion: str
