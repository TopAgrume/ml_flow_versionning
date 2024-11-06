#!/bin/sh

curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"data": [[5.1, 3.5, 1.4, 0.2]]}'

# The prediction should be:
#    {"prediction":[0]}