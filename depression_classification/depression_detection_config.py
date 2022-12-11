from transformers import PretrainedConfig


class DepressionDetectionConfig(PretrainedConfig):
    model_type = "DepressionDetection"

    def __init__(
            self,
            model_name_or_path: str = "depression-classification",
            num_layers: int = 1,
            hidden_size: int = 768,
            num_classes: int = 2,
            **kwargs,
    ):
        self.model_name_or_path = model_name_or_path
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        super().__init__(**kwargs)
