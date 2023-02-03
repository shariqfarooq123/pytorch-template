from registry import create_model, create_dataset, list_models, list_datasets
from utils.config import get_config

conf = get_config("toymodel")
model = create_model("toymodel", **conf)
print(list(model.parameters()))