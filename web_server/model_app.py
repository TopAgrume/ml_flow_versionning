from fastapi import FastAPI
import mlflow.exceptions
from base_templates import Item, ModelItem, ModelManager, model_uri, next_model_uri
import uvicorn
import logging
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Set the MLflow tracking URI
mlflow.set_tracking_uri("http://mlflow-server:8080")

model_manager = ModelManager(model_uri, next_model_uri)
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
        prediction = model_manager.next_model.predict(X).tolist()
        logger.info("Called model: next model")
    else:
        prediction = model_manager.current_model.predict(X).tolist()
        logger.info("Called model: current model")
    return {"prediction": prediction}


@app.post("/update-model")
async def update_model(item: ModelItem):
    """Creates an API endpoint to handle model updates dynamically.

    Args:
        item (ModelItem): an instance of ModelItem containing the model name and version to load

    Returns:
        dict: a message confirming the model update

    Raises:
        HTTPException: If a wrong next model path is given to `load_model`.
    """
    model_uri = f"models:/{item.modelname}/{item.modelversion}"
    model_manager.update_next_model(model_uri)
    return {"message": "New next model loaded", "next model": model_uri}


@app.get("/accept-next-model")
async def accept_next_model():
    """Accepts the next available model and sets it as the current model.

    Returns:
        dict: A response containing a success message and the URI of the accepted model if successful.

    Raises:
        HTTPException: If `next_model` is `None`. In this case, we cannot set the next model as current model
    """
    next_model_uri = model_manager.switch_to_next_model()
    return {"message": "New model accepted", "model": next_model_uri}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)