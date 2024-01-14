#! python
# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np

from .data_methods import create_stage


def prepare_dataset(source_dataset):
    """
    Начальная подготовка датасета
    :param source_dataset:  Исходный датасет
    """

    df = source_dataset.copy()

    df['BirthDate'] = pd.to_datetime(df['BirthDate'])
    df['JobStartDate'] = pd.to_datetime(df['JobStartDate'])
    df['Gender'] = np.where(df['Gender'] > 0, 1, 0)
    df['ChildCount'] = df['ChildCount'].astype('int')
    df['SNILS'] = df['SNILS'].astype('int')
    df['Loan_amount'] = df['Loan_amount'].astype('int')
    df['Loan_term'] = df['Loan_term'].astype('int')
    df['MonthProfit'] = df['MonthProfit'].astype('int')
    df['MonthExpense'] = df['MonthExpense'].astype('int')
    df['Merch_code'] = df['Merch_code'].astype('int')

    return df


if __name__ == "__main__":
    create_stage("data_prepare", prepare_dataset)
