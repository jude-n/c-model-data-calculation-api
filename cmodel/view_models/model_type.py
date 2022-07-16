from enum import Enum


class ModelType(str, Enum):
    CMODEL = "C-model"
    LINEAR = "Linear"
    EXPONENTIAL = "Exponential"
