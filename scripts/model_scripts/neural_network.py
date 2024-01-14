#! python
# -*- coding: UTF-8 -*-
"""
Обучение с помощью нейронной сети
"""

from dataclasses import dataclass
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from imblearn.under_sampling import NearMiss

from .train import *


@dataclass
class MLPParams:
    """
    Параметры алгоритма обучения модели
    """
    max_depth: int = 0
    learning_rate: float = 0.0
    verbose: bool = True
    hidden_layer_x: int = 0
    hidden_layer_y: int = 0


def train_model(dataset: pd.DataFrame, mlp_params: MLPParams):
    """
    Обучение модели с помощью XGBClassifier

    :param dataset: Исходный датасет
    :param mlp_params: Параметры обучения модели
    :return: Обученная модель
    """
    # Подготовка датасета
    x_train = dataset.drop('Y', axis=1)
    y_train = dataset['Y']

    # Так как данные не сбалансированы, применяем метод балансировки
    nm = NearMiss()
    x_train_miss, y_train_miss = nm.fit_resample(x_train, y_train)

    # Create and train model
    model = MLPClassifier(
        hidden_layer_sizes=(mlp_params.hidden_layer_x, mlp_params.hidden_layer_y),
        verbose=mlp_params.verbose,
        max_iter=mlp_params.max_depth,
        learning_rate_init=mlp_params.learning_rate)

    # Fit the model
    model.fit(x_train_miss, y_train_miss)

    return model


def get_train_params(params_yaml):
    """
    Получение параметров алгоритма обучения модели
    :param params_yaml: Объект yaml файла параметров
    :return: Объект TreeParams с параметрами модели
    """
    return MLPParams(
        max_depth=params_yaml["neural"]["max_depth"],
        learning_rate=params_yaml["neural"]["learning_rate_init"],
        verbose=params_yaml["neural"]["verbose"],
        hidden_layer_x=params_yaml["neural"]["hidden_layer_sizes_x"],
        hidden_layer_y=params_yaml["neural"]["hidden_layer_sizes_y"]
    )


if __name__ == "__main__":
    stage_name = Path(sys.argv[0]).stem
    train_stage(stage_name, train_model, get_train_params)
