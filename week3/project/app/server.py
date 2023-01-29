from fastapi import FastAPI
from pydantic import BaseModel
from loguru import logger
import time

from classifier import NewsCategoryClassifier

LOGS_OUTPUT_PATH = "../data/logs.out"
logger.add(LOGS_OUTPUT_PATH)


class PredictRequest(BaseModel):
    source: str
    url: str
    title: str
    description: str


class PredictResponse(BaseModel):
    scores: dict
    label: str


MODEL_PATH = "../data/news_classifier.joblib"

app = FastAPI()
my_classifier = None


@app.on_event("startup")
def startup_event():
    """
    [TO BE IMPLEMENTED]
    1. Initialize an instance of `NewsCategoryClassifier`.
    2. Load the serialized trained model parameters (pointed to by `MODEL_PATH`) into the NewsCategoryClassifier you initialized.
    3. Open an output file to write logs, at the destimation specififed by `LOGS_OUTPUT_PATH`

    Access to the model instance and log file will be needed in /predict endpoint, make sure you
    store them as global variables
    """
    print("Loading NewsCategoryClassifier model")
    global my_classifier
    my_classifier = NewsCategoryClassifier()
    my_classifier.load(MODEL_PATH)
    print("Setup completed")


@app.on_event("shutdown")
def shutdown_event():
    # clean up
    """
    [TO BE IMPLEMENTED]
    1. Make sure to flush the log file and close any file pointers to avoid corruption
    2. Any other cleanups
    """
    print("Shutting down application")


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    # get model prediction for the input request
    # construct the data to be logged
    # construct response
    """
    [TO BE IMPLEMENTED]
    1. run model inference and get model predictions for model inputs specified in `request`
    2. Log the following data to the log file (the data should be logged to the file that was opened in `startup_event`)
    {
        'timestamp': <YYYY:MM:DD HH:MM:SS> format, when the request was received,
        'request': dictionary representation of the input request,
        'prediction': dictionary representation of the response,
        'latency': time it took to serve the request, in millisec
    }
    3. Construct an instance of `PredictResponse` and return
    """
    logger.info("PredictRequest:")
    start = time.time()
    
    logger.info(f"Request: {request}")
    global my_classifier
    if my_classifier is None:
        startup_event()

    preds = my_classifier.predict_proba(request)
    logger.info(f"(Server)Preds: {preds}")

    sorted_preds = dict(sorted(preds.items(), key=lambda item: item[1], reverse=True))
    logger.info(f"(Server)Sorted Preds: {sorted_preds}")

    label = list(sorted_preds)[0]
    logger.info(f"(Server)Label: {label}")

    response = PredictResponse(scores=sorted_preds, label=label)
    logger.info(f"(Server)Response: {response}")
    end = time.time()
    
    print(f"Request Latency: {start} -> {end} {end - start} seconds")
    
    return response


@app.get("/")
def read_root():
    logger.info("Request to /")
    return {"Hello": "World"}
