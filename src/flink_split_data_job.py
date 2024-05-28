import os
import logging

from pyflink.common import WatermarkStrategy
from pyflink.datastream import StreamExecutionEnvironment, RuntimeExecutionMode
from pyflink.datastream.connectors.kafka import KafkaSource, KafkaOffsetsInitializer, KafkaSink, \
    KafkaRecordSerializationSchema
from pyflink.datastream.formats.json import JsonRowSerializationSchema, JsonRowDeserializationSchema
from RedditComment import SubredditKafkaTopicSelector, RedditComment

RUNTIME_ENV = os.getenv("RUNTIME_ENV", "local")
SOURCE_BOOTSTRAP_SERVERS = os.getenv("KAFKA_1_BOOTSTRAP_SERVERS")
SINK_BOOTSTRAP_SERVERS = os.getenv("KAFKA_2_BOOTSTRAP_SERVERS")


if __name__ == "__main__":
    """
    ## cluster execution
    docker exec jobmanager /opt/flink/bin/flink run \
        --python /tmp/src/flink_split_data_job.py
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.info(f"RUNTIME_ENV - {RUNTIME_ENV}, SOURCE_BOOTSTRAP_SERVERS - {SOURCE_BOOTSTRAP_SERVERS}")

    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_runtime_mode(RuntimeExecutionMode.STREAMING)
    # env.set_parallelism(5)
    if RUNTIME_ENV != "docker":
        CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
        jar_files = ["flink-sql-connector-kafka-1.17.1.jar"]
        jar_paths = tuple(
            [f"file://{os.path.join(CURRENT_DIR, 'jars', name)}" for name in jar_files]
        )
        logging.info(f"adding local jars - {', '.join(jar_files)}")
        env.add_jars(*jar_paths)

    deserializer = JsonRowDeserializationSchema.builder().type_info(
        RedditComment.get_value_type_info()).build()

    reddit_comment_source = (
        KafkaSource.builder()
        .set_bootstrap_servers(SOURCE_BOOTSTRAP_SERVERS)
        .set_topics("rce2")
        .set_starting_offsets(KafkaOffsetsInitializer.latest())
        .set_value_only_deserializer(
            deserializer
        )
        .build()
    )

    reddit_comment_stream = env.from_source(
        reddit_comment_source, WatermarkStrategy.no_watermarks(), "reddit_comment_source"
    )

    reddit_comment_stream.print()
    # sr1 = {"teenagers"}
    # sr2 = {"funny"}
    # sr3 = {"soccer"}
    # sr4 = {"memes"}
    #
    # subreddits_1 = reddit_comment_stream.filter(lambda r: r.subreddit in sr1)
    # subreddits_2 = reddit_comment_stream.filter(lambda r: r.subreddit in sr2)
    # subreddits_3 = reddit_comment_stream.filter(lambda r: r.subreddit in sr3)
    # subreddits_4 = reddit_comment_stream.filter(lambda r: r.subreddit in sr4)
    reddit_comment_sink = (KafkaSink.builder()
    .set_bootstrap_servers(SINK_BOOTSTRAP_SERVERS)
    .set_record_serializer(
        KafkaRecordSerializationSchema.builder()
        .set_topic_selector(SubredditKafkaTopicSelector())
        .set_key_serialization_schema(
            JsonRowSerializationSchema.builder().with_type_info(
                RedditComment.get_key_type_info()).build()
        )
        .set_value_serialization_schema(
            JsonRowSerializationSchema.builder().with_type_info(
                RedditComment.get_value_type_info()).build()
        )
        .build()
    ).build())

    (reddit_comment_stream
     .sink_to(reddit_comment_sink).name("comment_splitter_out"))

    env.execute("comment-splitter")
