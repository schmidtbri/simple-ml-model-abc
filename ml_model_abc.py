from abc import ABC, abstractmethod


class MLModel(ABC):
    """ An abstract base class for ML model prediction code  """
    @property
    @abstractmethod
    def input_schema(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def output_schema(self):
        raise NotImplementedError()

    @abstractmethod
    def __init__(self):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, data):
        self.input_schema.validate(data)
