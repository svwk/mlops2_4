#! python
# -*- coding: UTF-8 -*-
"""
Обучение с помощью XGBClassifier
"""

from dataclasses import dataclass
from pathlib import Path
from xgboost import XGBClassifier

from .train import *


@dataclass
class TreeParams:
    """
    Параметры алгоритма обучения модели
    """
    max_depth: int = 0
    n_estimators: int = 0
    eta: float = 0.0
    reg_lambda: int = 0
    reg_alpha: int = 0
    scale_pos_weight: int = 0
    device: str = 'gpu'
    sampling_method: str = 'gradient_based'


def train_model(dataset: pd.DataFrame, tree_params: TreeParams):
    """
    Обучение модели с помощью XGBClassifier

    :param dataset: Исходный датасет
    :param tree_params: Параметры обучения модели
    :return: Обученная модель
    """
    # Подготовка датасета
    x_train = dataset.drop('Y', axis=1)
    y_train = dataset['Y']

    # Если балансируем классы, то на выходе ~0,34 accuracy параметр для балансировки class_weight="balanced"
    # clf = DecisionTreeClassifier(max_depth=max_depth, max_features="auto", criterion="log_loss", max_leaf_nodes=0.5)
    # clf.fit(x_train, y_train)

    # Create and train the XGBoost model
    model = XGBClassifier(
        n_estimators=tree_params.n_estimators,
        max_depth=tree_params.max_depth,
        device=tree_params.device,
        eta=tree_params.eta,
        sampling_method=tree_params.sampling_method,
        reg_lambda=tree_params.reg_lambda,
        reg_alpha=tree_params.reg_alpha,
        scale_pos_weight=tree_params.scale_pos_weight)

    # Fit the model
    model.fit(x_train, y_train)

    return model


def get_train_params(params_yaml):
    """
    Получение параметров алгоритма обучения модели
    :param params_yaml: Объект yaml файла параметров
    :return: Объект TreeParams с параметрами модели
    """
    return TreeParams(
        max_depth=params_yaml["tree"]["max_depth"],
        n_estimators=params_yaml["tree"]["n_estimators"],
        eta=params_yaml["tree"]["eta"],
        reg_lambda=params_yaml["tree"]["reg_lambda"],
        reg_alpha=params_yaml["tree"]["reg_alpha"],
        scale_pos_weight=params_yaml["tree"]["scale_pos_weight"]
    )


if __name__ == "__main__":
    stage_name = Path(sys.argv[0]).stem
    train_stage(stage_name, train_model, get_train_params)
