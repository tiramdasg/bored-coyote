import datetime
import warnings

from sklearn.exceptions import InconsistentVersionWarning
from torch.onnx.errors import OnnxExporterWarning

warnings.filterwarnings("ignore", category=OnnxExporterWarning)
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
import json
import logging
import os

import sqlite3

import time
from threading import Thread

import numpy as np
import requests
import torch
import uvicorn
from fastapi import FastAPI, Request

from general.TorchSerializer import TorchSerializer
from general.SentimentAnalysisClassifier import SentimentAnalysisClassifier

from general.TextPreProcessor import TextPreProcessor


def setup_sqlite(folder):
    conn = sqlite3.connect(folder + 'sentiment.db')

    # Create a cursor object to execute SQL queries
    cursor = conn.cursor()

    # Define the SQL query to create the table
    create_table_query = '''
    CREATE TABLE IF NOT EXISTS sent (
        timestamp INT,
        biden_sentiment FLOAT,
        trump_sentiment FLOAT
    )
    '''
    cursor.execute(create_table_query)
    conn.commit()
    conn.close()


def insert_sentiment(folder, ts, biden_sent, trump_sent):

    conn = sqlite3.connect(folder + 'sentiment.db')
    cursor = conn.cursor()

    insert_query = '''
    INSERT INTO sent (timestamp, biden_sentiment, trump_sentiment)
    VALUES (?, ?, ?)
    '''
    # execute sql insertion
    cursor.execute(insert_query, (ts, biden_sent, trump_sent))
    # commit and close
    conn.commit()
    conn.close()


class AggregatorServer:
    def __init__(self):

        self.data_dir = "src/data/"  # for easy access of data
        self.models_dir = "src/models/"  # for easy access of model
        setup_sqlite(self.data_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # for model

        self.model = self.initialize_model()  # the aggregator model

        client_list = os.getenv("CLIENTS", "http://localhost:1337")  # is comma-separated string of clients
        self.clients = client_list.split(",")

        self.query_interval = int(os.getenv("QUERY_INTERVAL", 15))  # Query interval in seconds

        self.app = define_app(self)  # fastapi app for api calls
        # for one-source-of-truth-preprocessing of text
        self.pre_processor = TextPreProcessor(self.data_dir)
        # for easy sentiment collection on trump and biden
        self.trump_biden_embed = self.pre_processor.words_to_vec(["trump", "biden"])

        self.current_timestamp = None

    def initialize_model(self):
        mdl = SentimentAnalysisClassifier(input_length=1,
                                          model_path=self.models_dir + "model_weights.pth")
        logging.info(self.device)
        try:
            mdl.load_model(self.device)
        except Exception as e:
            logging.exception(f"Client:::Exception {e}")

        return mdl

    def predict_sentiment(self, words):
        embedding = self.pre_processor.pre_process_pipe(words)
        sentiment_prediction = self.model.predict(embedding, device=self.device)
        return sentiment_prediction

    def update_cycle(self):
        while True:
            time.sleep(self.query_interval)
            logging.info("Querying and updating client weights.")

            try:
                self.query_and_average_models()
                self.update_client_models()
            except Exception as e:
                logging.exception(e)

            try:
                self.get_current_biden_trump()

            except Exception as e:
                logging.exception(e)


    def average_model_weights(self, models):
        _ = self
        avg_weights = {}

        for name, param in models[0].named_parameters():
            avg_weights[name] = param.data.clone()

        # weights of all models get added
        for model in models[1:]:
            for name, param in model.named_parameters():
                avg_weights[name] += param.data

        # then averaged
        num_models = len(models)
        for name in avg_weights:
            avg_weights[name] /= num_models

        # create and return new model, having loaded the averaged weights

        avg_model = self.make_model(avg_weights, optimizer=self.model.optimizer)

        return avg_model

    def run(self):
        # update cycle thread
        update_cycle_thread = Thread(target=self.update_cycle)
        update_cycle_thread.start()

        # FastAPI app with Uvicorn
        uvicorn.run(self.app, host="0.0.0.0", port=int(os.getenv("PORT", 8081)))

    def make_model(self, state_dict, optimizer):
        mdl = SentimentAnalysisClassifier(input_length=1,
                                          model_path=self.models_dir + "model_weights.pth",
                                          optimizer=optimizer)
        if state_dict:
            mdl.load_state_dict(state_dict)
        mdl.to(self.device)
        return mdl

    def query_and_average_models(self):
        client_models = []

        for client in self.clients:
            try:
                response = requests.get(client + "/query_weights")

                if response.status_code == 200:
                    # extract weights from response
                    state_dict = TorchSerializer.deserialize(response.json(), self.device)

                    cli_model = self.make_model(state_dict, optimizer=self.model.optimizer)

                    client_models += [cli_model]
                else:
                    # something go wrong, give some feedback and skip to next client
                    logging.error(f"Error in query client {client}: {response.status_code}, {response.text}")
                    pass
            except Exception as e:
                logging.exception(f"{e}")

        if client_models:
            self.model = self.average_model_weights(client_models)
            self.model.store(self.models_dir + "model_weights.pth")

    def update_client_models(self):
        weights_str = TorchSerializer.serialize(self.model.state_dict())

        for client in self.clients:
            response = requests.post(client + "/update_weights", json=weights_str)
            if response.status_code != 200:
                # something go wrong, give some feedback and skip to next client
                logging.error(f"Error in query client {client}: {response.status_code}, {response.text}")

    def get_current_biden_trump(self):

        sent = self.model.predict(self.trump_biden_embed, self.device)

        timestamp = int(datetime.datetime.now().timestamp())
        insert_sentiment(
            self.data_dir,
            timestamp,
            sent[1],  # biden sentiment
            sent[0]  # trump sentiment
        )

    def get_sentiments(self):
        conn = sqlite3.connect(self.data_dir + 'sentiment.db')
        cursor = conn.cursor()

        # get all sentiment
        select_query = f'''
         SELECT * FROM sent ORDER BY timestamp DESC LIMIT 5000;
         '''

        cursor.execute(select_query)
        rows = cursor.fetchall()
        # get list of records
        data_list = []
        for row in rows:
            data_dict = {
                'timestamp': row[0],
                'biden_sentiment': row[1],
                'trump_sentiment': row[2]
            }
            data_list.append(data_dict)

        # for easy transmission: make it into json string
        json_data = json.dumps(data_list)
        return json_data


def define_app(aggregator):
    app = FastAPI()
    aggregator = aggregator

    # endpoints go here
    @app.get("/")
    def read_root():
        return "Hello World"

    @app.post("/train_params")
    async def train_params(request: Request):
        """Updates the models' train params with the weights from the request.
        Supports epochs (epochs) and learning rate (lr)."""

        data = await request.json()
        for client in aggregator.clients:
            resp = requests.post(client + "/train_params", json=data)
            if resp.status_code != 200:
                logging.error(f"Problem in train_params: {client}: {resp.json()}")

        return {"Status": "Success"}
    @app.post("/predict_keywords")
    async def predict_keyword(request: Request):
        data = await request.json()

        text = data["text"]

        # proba is probability for positive sentiment, 1-proba is for negative sentiment
        proba = aggregator.predict_sentiment(text)

        return json.dumps(proba)

    @app.get("/fetch_time_series")
    def fetch_time_series():
        return aggregator.get_sentiments()

    return app
