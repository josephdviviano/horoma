from abc import ABC, abstractmethod

import torch.optim

import models
import trainer


class Factory(ABC):
    """
    Base Factory class
    """

    @staticmethod
    @abstractmethod
    def get(config, *args):
        """
        Return the class instantiated with the config and args.

        :param config: config to use to create the class
        :param args: args to use to create the class
        :return: class instance
        """
        raise NotImplementedError


class ModelFactory(Factory):
    """
    Factory for creating models
    """

    @staticmethod
    def getclass(model_class):
        """
        Return the class of a model.

        :param model_class: model class to get
        :return: model class
        """
        return getattr(models, model_class)

    @staticmethod
    def get(config, *args):
        """
        Return the model instantiated with the config and args.

        :param config: config to use to create the model
        :param args: args to use to create the model
        :return: model instance
        """
        return getattr(models, config["model"]['type'])(
            *args,
            **config["model"]['args']
        )


class OptimizerFactory(Factory):
    """
    Factory for creating optimizers
    """

    @staticmethod
    def getclass(optim_class):
        """
        Return the class of an optimizer.

        :param model_class: optimizer class to get
        :return: optimizer class
        """
        return getattr(torch.optim, optim_class)

    @staticmethod
    def get(config, *args):
        """
        Return the optimizer instantiated with the config and args.

        :param config: config to use to create the optimizer
        :param args: args to use to create the optimizer
        :return: optimizer instance
        """
        return getattr(torch.optim, config["optimizer"]['type'])(
            *args,
            **config["optimizer"]['args']
        )


class TrainerFactory(Factory):
    """
    Factory for creating trainers
    """

    @staticmethod
    def get(config, *args):
        """
        Return the trainer instantiated with the config and args.

        :param config: config to use to create the trainer
        :param args: args to use to create the trainer
        :return: trainer instance
        """
        return getattr(trainer, config['trainer']['type'])
