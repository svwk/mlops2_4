"""
Методы работы с моделью
"""
import sys
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# from scripts import message_constants as mc

project_path = os.getcwd()


# %% Загрузим, обучим и сохраним модель
def train_model(train_dataset):
    """
    Обучение модели
    :param train_dataset:  Обучающий набор данных
    """

    target_column_name = "z"
    y_train = train_dataset[target_column_name].values
    x_train = train_dataset.drop(target_column_name, axis=1).values

    model = LogisticRegression(max_iter=100_000).fit(x_train, y_train)

    return model


def clear_train_test_data_frame(df, bank):
    out = df.drop(f'решение банка {bank}', axis=1)
    out = out.drop('образование', axis=1)
    out = out.drop('тип занятости', axis=1)
    out = out.drop('стаж работы', axis=1)
    out = out.drop('семейное положение', axis=1)
    out = out.drop('категория товара', axis=1)
    out = out.drop('стаж работы на последнем месте', axis=1)
    out = out.drop('срок кредита код', axis=1)
    out = out.drop('категория товара код', axis=1)
    out = out.drop('код магазина код', axis=1)
    out = out.drop('семейное положение код', axis=1)
    out = out.drop('образование код', axis=1)
    out = out.drop('тип занятости код', axis=1)
    out = out.drop('стаж работы код', axis=1)
    out = out.drop('последний стаж код', axis=1)

    return out
