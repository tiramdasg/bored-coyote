import json
import torch
# from SentimentAnalysisClassifier import SentimentAnalysisClassifier


class TorchSerializer:
    @staticmethod
    def serialize(weights):
        config = {}
        for k, v in weights.items():

            if isinstance(v, torch.Tensor):
                config[k] = v.cpu().data.numpy().tolist()
            else:
                config[k] = v

        return json.dumps(config)

    @staticmethod
    def deserialize(weights_str, device):
        weights = json.loads(weights_str)
        config = {}
        for k, v in weights.items():
            try:
                config[k] = torch.tensor(v, device=device)  # Attempt to convert string back to tensor
            except ValueError:
                config[k] = v

        return config


# def main():
#     """This is a test for torchserializer"""
#     # model a
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = SentimentAnalysisClassifier(embedding_dim=300, hidden_dim=256, num_classes=3).to(device)
#
#     original_state_dict = model.state_dict()
#
#     serialized_state_dict = TorchSerializer.serialize(original_state_dict)
#
#     deserialized_state_dict = TorchSerializer.deserialize(serialized_state_dict, device)
#     # model b for loading deserialized state dict into
#     new_model = SentimentAnalysisClassifier(embedding_dim=300, hidden_dim=256, num_classes=3).to(device)
#     new_model.load_state_dict(deserialized_state_dict)
#
#     # if this passes, (de-)serialization works perfectly
#     for k in original_state_dict:
#         assert torch.equal(original_state_dict[k], new_model.state_dict()[k]), f"Mismatch found at {k}"
#
#     print("Serialization and deserialization tests passed.")
#
#
# if __name__ == "__main__":
#     main()
