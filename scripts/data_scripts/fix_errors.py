#! python
# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
from datetime import datetime, date
from dateutil.relativedelta import relativedelta

from .data_methods import create_stage
from .utils.seniority_cats import months_seniority_to_cat, seniority_cat_to_month_count
from .utils.seniority_cats import set_last_seniority

# Прожиточный минимум
ADULT_LIVING_WAGE = 15669
CHILD_LIVING_WAGE = 13944


def fix_errors_in_dataset(source_dataset):
    """
    Исправление ошибок и аномалий в данных
    :param source_dataset:  Исходный датасет
    """

    df = source_dataset.copy()

    # Удаление выбросов
    features_to_clear = ["MonthProfit", "MonthExpense"]
    for feature in features_to_clear:
        lower_bound, upper_bound = get_bounds((df[feature]))
        df = df.drop(df[(df[feature] < lower_bound) | (df[feature] > upper_bound)].index)
    df = df.reset_index(drop=True)

    # Исправление аномалий
    df = df.apply(fix_seniority, axis=1)
    df = df.apply(fix_expense, axis=1)

    return df


def fix_expense(application_data):
    """
    Исправление аномальных значений расхода семьи
    :param application_data:  Данные заявки пользователя
    """

    # Реальный расход пользователя, если он один в семье
    real_expense = ADULT_LIVING_WAGE

    # Расходы на содержание супруга/супруги (при наличии)
    if check_family_status(application_data['Family status']):
        real_expense += ADULT_LIVING_WAGE

    # Расходы на содержание детей
    real_expense += CHILD_LIVING_WAGE * application_data['ChildCount']

    # Если рассчитанные расходы превышают указанные в заявлении,
    # исправляем на большее значение
    if real_expense > application_data['MonthExpense']:
        application_data['MonthExpense'] = real_expense

    return application_data


def check_family_status(status):
    """
        Проверка семейного положения, при наличии супруга/супруги
         возвращает True
        :param status:  Данные из заявки
    """
    status = str(status)

    if (status == 'Женат / замужем' or
            status == 'Гражданский брак / совместное проживание'):
        return True
    else:
        return False


def fix_seniority(application_data):
    """
    Исправление аномальных значений стажа работы
    :param application_data:  Данные заявки
    """

    # Общий стаж в месяцах
    total_seniority_in_months = int(seniority_cat_to_month_count(application_data['Value']))

    # Возраст
    age = relativedelta(datetime.today(), application_data['BirthDate'])

    # Проверка на соответствие трудовому законодательству
    if age.years < 16:
        application_data['Value'] = 'Нет стажа'
        application_data['JobStartDate'] = np.NAN
        return application_data

    # Максимально возможный трудовой стаж в месяцах
    # (возраст - 16) (ТК)
    max_seniority_in_months = age.months + (age.years - 16) * 12

    # Общий стаж не может быть больше,
    # чем максимально возможный трудовой стаж
    new_total_seniority = min(total_seniority_in_months, max_seniority_in_months)

    # Стаж работы на последнем месте
    if (not pd.isna(application_data['JobStartDate']) and
            application_data['employment status'] != "Не работаю"):
        # Стаж работы на последнем месте в месяцах
        last_seniority_in_months = set_last_seniority(application_data)

        # Стаж на последнем рабочем месте не может быть больше,
        # чем максимально возможный трудовой стаж
        # Общий стаж не может быть меньше, чем стаж на последнем рабочем  месте
        new_last_seniority = min(last_seniority_in_months, max_seniority_in_months, new_total_seniority)

        # Корректировка стажа на последнем рабочем  месте
        if new_last_seniority != last_seniority_in_months:
            application_data['JobStartDate'] = datetime.strftime((date.today() -
                                                                  relativedelta(months=new_last_seniority)),
                                                                 "%Y-%m-%d %H:%M:%S")

    # Корректировка общего стажа
    if new_total_seniority != total_seniority_in_months:
        application_data['Value'] = months_seniority_to_cat(new_total_seniority)

    return application_data


def search_outliers(feature):
    """
    Поиск выбросов в значениях признака
    :param feature: Значения признака
    :return: Количество выбросов в столбце
    """

    lower_bound, upper_bound = get_bounds(feature)
    return np.where((feature < lower_bound) | (feature > upper_bound))[0]


def get_bounds(feature):
    """
    Вычисляет пределы значений для нахождения выбросов
    :param feature: Значения признака
    :return: Нижняя и верхняя граница пределов
    """
    q1, q3 = np.percentile(feature, [25, 85])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return lower_bound, upper_bound


if __name__ == "__main__":
    create_stage("fix_errors", fix_errors_in_dataset)
