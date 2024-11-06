from fastapi import FastAPI, HTTPException
import mlflow.exceptions
from base_templates import Item, ModelItem, model_uri, next_model_uri
import uvicorn

import mlflow.sklearn
import numpy as np

# Set the MLflow tracking URI
mlflow.set_tracking_uri("http://mlflow-server:8080")

def load_model(model_uri: str):
    return mlflow.sklearn.load_model(model_uri)

current_model = load_model(model_uri)
next_model = load_model(next_model_uri)

app = FastAPI()
canary_p = 0.1

@app.post("/predict")
async def predict(item: Item):
    """API endpoint to handle POST requests for predictions.

    Args:
        item (Item): an instance of Item, containing input data for the prediction

    Returns:
        dict: a dictionary containing the prediction result
    """
    X = np.array(item.data)
    if np.random.rand() < canary_p:
        prediction = next_model.predict(X).tolist()
        print("Called model: next model")
    else:
        prediction = current_model.predict(X).tolist()
        print("Called model: current model")
    return {"prediction": prediction}


@app.post("/update-model")
async def update_model(item: ModelItem):
    """Creates an API endpoint to handle model updates dynamically.

    Args:
        item (ModelItem): an instance of ModelItem containing the model name and version to load

    Returns:
        _type_: a message confirming the model update
    """
    global next_model
    model_uri = f"models:/{item.modelname}/{item.modelversion}"
    try:
        next_model = load_model(model_uri)
        return {"message": "New next model loaded", "next model": model_uri}
    except mlflow.exceptions.MlflowException as e:
        return HTTPException(status_code=400, detail="Model does not exist")


@app.get("/accept-next-model")
async def accept_next_model():
    global current_model, next_model, next_model_uri

    if next_model is None:
        return HTTPException(status_code=400, detail="Next model does not exist")
    else:
        current_model = next_model
        next_model = None
        return {"message": "New model accepted", "model": next_model_uri}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)