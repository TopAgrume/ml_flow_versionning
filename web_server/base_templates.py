from pydantic import BaseModel

class Item(BaseModel):
    data: list[list[float]]

class ModelItem(BaseModel):
    modelname: str
    modelversion: str


model_name = "sk-learn-log-reg-model"
model_version = "latest"
model_uri = f"models:/{model_name}/{model_version}"

next_model_name = "sk-learn-random-forest-model"
next_model_version = "latest"
next_model_uri = f"models:/{next_model_name}/{next_model_version}"