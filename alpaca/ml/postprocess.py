from abc import ABCMeta, abstractmethod


class BaseAverage(metaclass=ABCMeta):

    def __init__(self):
        pass

    @abstractmethod
    def get_result(self, df):
        raise NotImplementedError()


class MeanAverage(metaclass=ABCMeta):

    def __init__(self):
        pass

    def get_result(self, df):
        return 0, 0
