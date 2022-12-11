from transformers import (
    AutoTokenizer,
)
from transformers import pipeline

from depression_classification import (
    DepressionDetectionConfig,
    DepressionDetectionModel,
    DepressionPipeline
)

directory = "../depression-classification"
tokenizer = AutoTokenizer.from_pretrained(directory)

config = DepressionDetectionConfig.from_pretrained(
    directory,
)
model = DepressionDetectionModel.from_pretrained(
    pretrained_model_name_or_path=directory,
    config=config,
)

pipe = pipeline("text-classification",
                model=model,
                tokenizer=tokenizer,
                pipeline_class=DepressionPipeline,
                )

result = pipe("I don't feel hopeful.")
print(result)
