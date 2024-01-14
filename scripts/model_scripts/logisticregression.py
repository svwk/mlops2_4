#! python
# -*- coding: UTF-8 -*-
from sklearn.linear_model import LogisticRegression
from pathlib import Path
from dataclasses import dataclass

from .train import *


@dataclass
class LogRegParams:
    """
    Параметры алгоритма обучения модели
    """
    max_iter: int = 0


def train_model(dataset: pd.DataFrame, train_params: LogRegParams):
    """
    Обучение модели с помощью алгоритма логистической регрессии

    :param dataset: Исходный датасет
    :param train_params: Параметры обучения модели
    :return: Обученная модель
    """

    # Подготовка датасета
    x_train = dataset.drop('Y', axis=1)
    y_train = dataset['Y']

    model = LogisticRegression(
        max_iter=train_params.max_iter
    )
    model.fit(x_train, y_train)

    return model


def get_train_params(params_yaml):
    """
    Получение параметров алгоритма обучения модели
    :param params_yaml: Объект yaml файла параметров
    :return: Объект LogRegParams с параметрами модели
    """
    return LogRegParams(
        max_iter=params_yaml["log_reg"]["max_iter"]
    )


if __name__ == "__main__":
    stage_name = Path(sys.argv[0]).stem
    train_stage(stage_name, train_model, get_train_params)
