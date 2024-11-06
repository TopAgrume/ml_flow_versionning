from fastapi import FastAPI, HTTPException
import mlflow.exceptions
from base_templates import Item, ModelItem
import uvicorn

import mlflow.sklearn
import numpy as np

# Set the MLflow tracking URI
mlflow.set_tracking_uri("http://mlflow-server:8080")

model_name = "sk-learn-log-reg-model"
model_version = "latest"
model_uri = f"models:/{model_name}/{model_version}"

def load_model(model_uri: str):
    return mlflow.sklearn.load_model(model_uri)


model = load_model(model_uri)
app = FastAPI()


@app.post("/predict")
async def read_root(item: Item):
    """API endpoint to handle POST requests for predictions.

    Args:
        item (Item): an instance of Item, containing input data for the prediction

    Returns:
        dict: a dictionary containing the prediction result
    """
    X = np.array(item.data)
    prediction = model.predict(X).tolist()
    return {"prediction": prediction}


@app.post("/update-model")
async def read_root(item: ModelItem):
    """Creates an API endpoint to handle model updates dynamically.

    Args:
        item (ModelItem): an instance of ModelItem containing the model name and version to load

    Returns:
        _type_: a message confirming the model update
    """
    global model
    model_uri = f"models:/{item.modelname}/{item.modelversion}"
    try:
        model = load_model(model_uri)
        return {"message": "New model loaded", "model": model_uri}
    except mlflow.exceptions.MlflowException as e:
        return HTTPException(status_code=400, detail="Model does not exist")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)