from utils.model_io import load_state_from_resource
from inspect import isclass

def _create_registry():
    class Registry:
        """Mixin to register objects with the registry. Makes use of the __init_subclass__ hook to register the model with the registry.
        """
        _registry = {}

        def __init_subclass__(cls, name, version="v1", **kwargs):
            super().__init_subclass__(**kwargs)
            name = f"{name}_{version}"
            if name in cls._registry:
                raise ValueError(f"object name {name} already registered")
            
            if name == 'ignore':
                return
            
            cls._registry[name] = cls

        @classmethod
        def get_object(cls, object_name):
            """Get a object from the registry.
            Args:
                object_name (str): Name of the object to get.
            Returns:
                Baseobject: object class.
            """
            return cls._registry[object_name]

        @classmethod
        def get_object_names(cls):
            """Get a list of all object names in the registry.
            Returns:
                list: List of object names.
            """
            return list(cls._registry.keys())
        
        @classmethod
        def register(cls, object_name):
            """Decorator to register a object.
                object_name (str): Name of the object.
            """
            def register_object_(func):
                cls._registry[object_name] = func
                return func
            return register_object_
    
    return Registry

ModelRegistry = _create_registry()
DatasetRegistry = _create_registry()
LossRegistry = _create_registry()

def list_models():
    """List all models in the registry.
    Returns:
        list: List of model names.
    """
    return ModelRegistry.get_object_names()

def create_model(model_name, model_version="v1", pretrained="", **kwargs):
    model_name = f"{model_name}_{model_version}"
    model_cls = ModelRegistry.get_object(model_name)
    model = model_cls(**kwargs)
    if pretrained:
        model = load_state_from_resource(model, pretrained)
    return model



def list_datasets():
    """List all datasets in the registry.
    Returns:
        list: List of dataset names.
    """
    return DatasetRegistry.get_object_names()

def create_dataset(dataset_name, **kwargs):
    config = kwargs.pop('config') if 'config' in kwargs else {}
    dataset_cls = DatasetRegistry.get_object(dataset_name)
    dataset = dataset_cls.build_loader(config, **kwargs)
    return dataset


def create_loss(loss_name, **kwargs):
    """Builds a loss from a config dict.
    Args:
        loss_name (str): Loss dict.
    Returns:
        Loss object
    """
    loss_cls = LossRegistry.get_object(loss_name)
    if isclass(loss_cls):
        loss = loss_cls(**kwargs)
    else:
        loss = loss_cls
    return loss

def list_losses():
    """List all losss in the registry.
    Returns:
        list: List of loss names.
    """
    return LossRegistry.get_object_names()

def get_trainer(trainer):
    """Builds and returns a trainer based on the config.

    Args:
        trainer (str): the name of the trainer to use. The module named "{config.trainer}_trainer" must exist in trainers root module

    Raises:
        ValueError: If the specified trainer does not exist under trainers/ folder

    Returns:
        Trainer (inherited from trainers.BaseTrainer): The Trainer object
    """
    try:
        Trainer = getattr(import_module(
            f"trainers.{trainer}"), 'Trainer')
    except ModuleNotFoundError as e:
        raise ValueError(f"Trainer {trainer} not found.") from e
    return Trainer

from losses import *
from datasets import *
from models import *
