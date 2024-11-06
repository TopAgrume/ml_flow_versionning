#!/bin/sh

curl -X POST "http://localhost:8000/update-model" \
     -H "Content-Type: application/json" \
     -d '{"modelname": "sk-learn-log-reg-model", "modelversion": "latest"}'

# You should get:
#   {"message":"New model loaded","model":"models:/sk-learn-log-reg-model/latest"}