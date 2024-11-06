# Project usage guide

This guide provides instructions on how to use and test the project, including setting up the MLflow server, pushing the model, and testing the web service.

## 1. Launching the MLflow dockerized server

To start the MLflow server inside a Docker container, use the following command:

```sh
docker compose up mlflow-server -d
```

This will:
- Start the MLflow server in detached mode (`-d`).
- Make the MLflow UI available at `http://localhost:8080`.

## 2. Create and push the default model to MLflow

After launching the MLflow server, you need to create and register the default model and next model. Run the following Python script to train and push the model to the MLflow model registry:

```sh
python run_and_push_model.py
```

This script:
- Trains a simple scikit-learn logistic regression model.
- Trains a simple scikit-learn random forest model.
- Pushes the models to the MLflow server for future use.

## 3. Launch the web service

To start the web service (FastAPI), use this command:

```sh
docker compose up
```

This will:
- Launch the FastAPI application.
- Make the web service available at `http://localhost:8000`.

## How to test the project

### 1. Test predictions

To test the prediction functionality of the web service, run the following script:

```sh
./web_server/tests/test_predict.sh
```

This script:
- Sends a POST request with sample input data to the `/predict` endpoint of the web service.
- Prints out the prediction response from the model.

### 2. Test model updates

To test dynamic model updates (e.g., updating the model used by the web service), run:

```sh
./web_server/tests/test_update.sh
```

This script:
- Sends a POST request to the `/update-model` endpoint of the web service to update the model version.
- Confirms that the model has been successfully updated and is being used for future predictions.