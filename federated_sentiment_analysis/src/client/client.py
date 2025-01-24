"""Client implements the client side of the federated learning. The clients listen to kafka-2, which distributes
data to topics as assigned by data-splitter pyflink job. There are four of these subreddit groups (topics) and
four clients. The clients implement a server, housing a flask-api and a kafka consumer. The kafka is consumed
and utilized for federated training of the models, which can be queried and updated using the api."""

import json
import logging
import os
import sys
import time

from threading import Lock, Thread

import numpy as np
import torch

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from kafka import KafkaConsumer

from general.TorchSerializer import TorchSerializer
from general.SentimentAnalysisClassifier import SentimentAnalysisClassifier


def define_app(weights_lock, client):
    """FastAPI app-building and endpoint definition. Defines query_weights, query_current_ts and
    update_weights endpoints."""

    app = FastAPI()
    client = client

    @app.get("/")
    def read_root():
        """For feedback on API availability."""
        return "Hi There :))"

    @app.get("/query_weights")
    async def query_weights():
        """Serializes and sends the state of the weights to the aggregator."""
        with weights_lock:
            # get state dict and return
            weights = TorchSerializer.serialize(client.model.state_dict())
        return weights

    @app.post("/train_params")
    async def train_params(request: Request):
        """Updates the models' train params with the weights from the request.
        Supports epochs (epochs) and learning rate (lr)."""
        try:
            data = await request.json()

            if "lr" in data:
                try:
                    learn_rate = float(data["lr"])
                    client.learn_rate = learn_rate
                except Exception:
                    return {"Status": "Error: lr not of type float."}
            if "epochs" in data:
                try:
                    epochs = int(data["epochs"])
                    client.iterative_epochs = epochs
                except Exception:
                    return {"Status": "Error: epochs not of type int"}

        except KeyError:
            raise HTTPException(status_code=400, detail="No data provided")

        return {"Status": "Success"}

    @app.post("/update_weights")
    async def update_weights(request: Request):
        """Updates the models' weights with the weights from the request."""
        try:
            data = await request.json()
            state_dict = TorchSerializer.deserialize(data, client.device)
        except KeyError:
            raise HTTPException(status_code=400, detail="No data provided")
        # this is where the new weights are assigned to the model
        with weights_lock:
            client.model = client.make_model(state_dict, optimizer=client.model.optimizer)

        # model is stored locally for resistance to disturbances
        client.model.store(client.models_dir + "model_weights.pth")
        return {"status": "Success"}

    return app


class ClientServer:
    """Provides API and Kafka-Consumer/Model Training."""
    def __init__(self, cli_id):  # id, as we have several clients
        self.label_map = {-1: 0, 1: 1}  # map usual sentiment classes to 0/1 - we do not deal in neutral sentiment
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # maybe the machine has cuda
        self.embedding_size = 15000
        self.data_dir = "src/data/"
        self.models_dir = "src/models/"
        self.client_id = cli_id

        self.learn_rate = 0.01
        self.iterative_epochs = 5

        self.weights_lock = Lock()  # shared resources need a lock
        self.model = self.initialize_model()  # loading the model file
        self.most_recent_ts = 0.0  # for time reference
        self.app = define_app(self.weights_lock, self)  # for (fast-)api endpoints

    def initialize_model(self):
        """Loads sentiment analysis model from file."""
        # input length refers to "number of embeddings", which ended up being once, as we use tfidf vectorizer
        mdl = SentimentAnalysisClassifier(input_length=1,
                                          model_path=self.models_dir + "model_weights.pth")

        try:
            mdl.load_model(self.device)
        except Exception as e:
            logging.exception(f"Client:::Exception {e}")

        return mdl

    def make_model(self, state_dict, optimizer):
        """Makes sentiment analysis model from state-dict. """
        mdl = SentimentAnalysisClassifier(input_length=1,
                                          model_path=self.models_dir + "model_weights.pth",
                                          optimizer=optimizer)
        if state_dict:
            mdl.load_state_dict(state_dict)
        mdl.to(self.device)
        return mdl

    def kafka_consumer_thread(self):
        """THREAD for kafka consuming and model-training."""
        while True:
            # restart, if ended.
            try:
                logging.info("Consumer thread started.")

                # model defined in environment
                topic = os.getenv("TOPIC", ["sg1", "sg2", "sg3", "sg4"])
                if not isinstance(topic, list):
                    topic = [topic]

                # need to get kafka address
                bootstrap_server = os.getenv("BOOTSTRAP_SERVER", 'localhost:29093')
                consumer = KafkaConsumer(
                    *topic,
                    bootstrap_servers=[bootstrap_server],
                    group_id=str(self.client_id),
                    auto_offset_reset='earliest',
                    enable_auto_commit=False
                )

                try:
                    batch = []
                    labels = []
                    # consuming kafka-2
                    for message in consumer:

                        try:
                            # deserializing
                            decoded_message_value = message.value.decode('utf-8')
                            data = json.loads(decoded_message_value)

                            compressed_embedding = json.loads(data["embedding"])

                            embedding = [0.0 if str(index) not in compressed_embedding else compressed_embedding[str(index)]
                                         for index in range(self.embedding_size)]
                            # batching
                            batch += [np.array(embedding)]
                            labels += [self.label_map[data['label']]]
                            #
                            # # extracting data from message
                            # embedding = np.array([np.array(embedding)])
                            # label = [self.label_map[data['label']]]
                            created_utc = data['created_utc']
                            self.most_recent_ts = created_utc
                            if len(batch) % 10 == 0:
                                logging.info(f"accumulated {len(batch)} samples in batch.")
                            if len(batch) == 128:
                                logging.info(f"training...")
                                with self.weights_lock:
                                    self.model.fit(num_epochs=self.iterative_epochs,
                                                   data=np.array(batch),
                                                   labels=labels,
                                                   device=self.device,
                                                   batch_size=128,
                                                   verbose=0,
                                                   lr=self.learn_rate)
                                batch = []
                                labels = []
                                logging.info(
                                    f"Processing batch of messages: created_utc={created_utc}")
                                # give some time off to let aggregator query etc
                                time.sleep(2)

                            # the actual federated training. we increase the training rate here for stronger effect

                        except json.decoder.JSONDecodeError:
                            # it is expected that some malformed datapoints arrive.
                            # this should not lead to a flooding of logs.
                            pass
                        except Exception as e:
                            logging.exception(f"Error processing message: {e}")

                except KeyboardInterrupt:  # just some graceful shutdown logic
                    logging.info("Consumer interrupted, closing...")
                finally:
                    consumer.close()
                    logging.info("Consumer closed.")
            except Exception as e:
                if isinstance(e, KeyboardInterrupt):
                    raise e
                logging.exception(f"Problem: {e}")

    def run(self):
        # starting kafka consumer thread
        consumer_thread = Thread(target=self.kafka_consumer_thread)
        consumer_thread.start()

        port = int(os.getenv("PORT", 1337))  # from env, again
        # and starting uvicorn app
        uvicorn.run(self.app, host="0.0.0.0", port=port)




# this allows you to test the update weights api point. ou need the model_weights.pth and the client has to be running
# from federated_sentiment_analysis.TorchSerializer import TorchSerializer
# import requests
# import torch
#
# state_dict = torch.load("./federated_sentiment_analysis/model_weights.pth")
# weights_str = TorchSerializer.serialize(state_dict)
# response = requests.post("http://localhost:8000" + "/update_weights", json=weights_str)
# if response.status_code != 200:
#     # something go wrong, give some feedback and skip to next client
#     logging.error(f"Error in query client {"http://localhost:8000"}: {response.status_code}, {response.text}")
