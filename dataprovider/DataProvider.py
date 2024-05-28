import logging
import os

import json
import random
import time

from collections import defaultdict
import datetime

import pandas as pd
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

# localhost:29092 if the dataprovider is external, i.e. not part of the network. otherwise its kafka-container-name:9092
BOOTSTRAP_SERVER = os.getenv("BOOTSTRAP_SERVER", "localhost:29092")

DATA_DIR = os.getenv("DATA_DIR", '/home/gabst/projects/big_data/data/')
# delay factor = up to second * delay_factor
SLOW_MODE = os.getenv("SLOW_MODE", True)

DELAY_FACTOR = int(os.getenv("DELAY_FACTOR", 1))
VERBOSE = os.getenv("VERBOSE", 1)

RETRIES = int(os.getenv("RETRIES", 3))


def serialize(obj):
    if isinstance(obj, datetime.datetime):
        return obj.isoformat(timespec="milliseconds")
    if isinstance(obj, datetime.date):
        return str(obj)
    return obj


def read_subreddit_distribution_and_define_topic_split():
    logging.info("Reading json-File to determine subreddit distribution and define topic split")
    df = pd.read_json("subreddits.json", typ='series')
    sorted_series = df.sort_values(ascending=False)

    # Initialize four empty groups and their sums
    groups = {i: [] for i in range(4)}
    group_sums = {i: 0 for i in range(4)}

    for subreddit, count in sorted_series.items():
        # Find the group with the smallest sum
        smallest_group = min(group_sums, key=group_sums.get)
        # Add the subreddit to this group
        groups[smallest_group].append((subreddit, count))
        # Update the sum of this group
        group_sums[smallest_group] += count
    # Convert groups to DataFrame for better visualization
    grouped_df = {i: pd.DataFrame(group, columns=["Subreddit", "Count"]) for i, group in groups.items()}

    sets = []

    # sanity check
    if VERBOSE:
        logging.info(f"Sanity Check: the following groups should be roughly equal in size.")
        for key, dff in grouped_df.items():
            logging.info(f"{key}, {sum(dff['Count'])}")
            sets += [set(dff["Subreddit"])]

    # this is to be used for separation for now
    topic_split_map = {idx: f"sg{idx + 1}" for idx in range(len(sets))}
    return topic_split_map, sets


class DataProvider:
    def __init__(self, path):
        logging.info(f"Initializing DataProvider. Verbose mode is {'on' if VERBOSE else 'off'}.")
        self.path = path
        retries = 0
        self.producer = None
        while retries < RETRIES:
            try:
                self.producer = KafkaProducer(bootstrap_servers=BOOTSTRAP_SERVER,
                                              key_serializer=lambda v: json.dumps(v, default=serialize).encode("utf-8"),
                                              value_serializer=lambda v: json.dumps(v, default=serialize).encode(
                                                  "utf-8"),
                                              api_version=(2, 5, 0)
                                              )
                break
            except NoBrokersAvailable:
                if VERBOSE:
                    logging.info(f"NoBrokersAvailable-Exception occurred."
                                 f" Will retry in 5 seconds {RETRIES - retries} more time(s).")
                retries += 1
            time.sleep(5)
        logging.info(f"KafkaProducer initialized successfully.")

        if not self.producer:
            message = f"Kafka producer initialization failed. Is kafka under {BOOTSTRAP_SERVER} running?"
            raise NoBrokersAvailable(message)

        # this is the
        self.topic = "rce2"
        # id: identifier, body: the comment, created_utc: timestamp of commenting, subreddit: where it was posted
        # these fields are the ones we filter for, not the entire data point sent by kafka
        self.relevant_fields = ["id", "body", "created_utc", "subreddit"]
        self.topic_split_map, self.sets = read_subreddit_distribution_and_define_topic_split()
        logging.info("Startup complete.")

    def send_data(self):
        logging.info("Sending data.")
        for comment in self.load_json_chunks():
            self.producer.send(self.topic, comment)
            if SLOW_MODE:
                # sleep to keep the waste of space contained
                time.sleep(random.random() * DELAY_FACTOR)

    def load_json_chunks(self):
        subreddits = defaultdict(int)

        with open(self.path, 'rb') as file:
            data_buffer = b''
            for line in file:

                data_buffer += line
                if b'"total_awards_received":' in line:
                    try:
                        data_buffer = json.loads(data_buffer.decode(encoding="utf-8", errors="ignore"))

                        # drop 'unnecessary' data
                        data_buffer = {key: data_buffer[key] for key in self.relevant_fields
                                       if data_buffer["body"] != "[deleted]" and data_buffer["body"]}
                        try:
                            subreddits[data_buffer["subreddit"]] += 1
                        except KeyError:
                            pass

                        if data_buffer:
                            self.add_split_group(data_buffer)
                            if VERBOSE:
                                logging.info(data_buffer)

                            yield data_buffer

                    except json.JSONDecodeError:
                        pass
                    data_buffer = b''
        with open(os.getcwd() + "/subreddits.json", "w") as file:
            json.dump(subreddits, file)

    def add_split_group(self, data_buffer):
        for idx, s in enumerate(self.sets):
            if data_buffer["subreddit"] in s:
                data_buffer["split_group"] = self.topic_split_map[idx]
                return


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    DataProvider(os.path.join(DATA_DIR, "RC_2019-04")).send_data()
