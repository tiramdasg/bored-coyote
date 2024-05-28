import dataclasses
from pyflink.common import Row
from pyflink.datastream.connectors.kafka import KafkaTopicSelector
from pyflink.common.typeinfo import Types


@dataclasses.dataclass
class RedditComment:
    id: str
    subreddit: str
    body: str
    created_utc: int
    split_group: str

    def get_subreddit(self):
        return self.subreddit

    def get_key(self):
        return self.id

    def get_value(self):
        return {
            'content': self.body,
            'timestamp': self.created_utc,
            'subreddit': self.subreddit,
            'split_group': self.split_group,
        }

    @staticmethod
    def get_key_type_info():
        return Types.ROW([Types.STRING()])

    @staticmethod
    def get_value_type_info():
        return Types.ROW_NAMED(
            field_names=[
                            "id",
                            "body",
                            "created_utc",
                            "subreddit",
                            "split_group",
            ],
            field_types=[
                        Types.STRING(),
                        Types.STRING(),
                        Types.INT(),
                        Types.STRING(),
                        Types.STRING(),
            ],
        )

    @classmethod
    def get_topic(cls, row):
        """Assuming split_group was defined in the data provider.
         This might not be what we end up using. """
        return row.as_dict()['split_group']

    @classmethod
    def from_row(cls, row: Row):
        return cls(row.comment_id,
                   row.subreddit,
                   row.body,
                   row.timestamp,
                   row.split_group)

    def to_row(self):
        return Row(
            id=self.id,
            subreddit=self.subreddit,
            body=self.body,
            created_utc=self.created_utc,
            split_group=self.split_group
        )


class SubredditKafkaTopicSelector(KafkaTopicSelector):
    """This allows to dynamically select the subreddit in kafka sink (flink).
    """
    def apply(self, value):
        return RedditComment.get_topic(value)
