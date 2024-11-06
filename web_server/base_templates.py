from pydantic import BaseModel
from fastapi import HTTPException

import mlflow.sklearn

model_name = "sk-learn-log-reg-model"
model_version = "latest"
model_uri = f"models:/{model_name}/{model_version}"

next_model_name = "sk-learn-random-forest-model"
next_model_version = "latest"
next_model_uri = f"models:/{next_model_name}/{next_model_version}"

class Item(BaseModel):
    data: list[list[float]]

class ModelItem(BaseModel):
    modelname: str
    modelversion: str


class ModelManager:
    def __init__(self, model_uri: str, next_model_uri: str):
        self.current_model = self.__load_model(model_uri)
        self.next_model = self.__load_model(next_model_uri)
        self.next_model_uri = next_model_uri

    def __load_model(self, model_uri: str):
        return mlflow.sklearn.load_model(model_uri)

    def update_next_model(self, next_model_uri: str):
        try:
            self.next_model = self.__load_model(next_model_uri)
            self.next_model_uri = next_model_uri
        except mlflow.exceptions.MlflowException:
            raise HTTPException(status_code=400, detail="The given model uri does not exist")

    def switch_to_next_model(self):
        if self.next_model is None:
            raise HTTPException(status_code=400, detail="Next model does not exist")

        self.current_model = self.next_model
        self.next_model = None

        save_uri = self.next_model_uri
        self.next_model_uri = None

        return save_uri