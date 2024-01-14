import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

DEFAULT_CAT = 'Нет стажа'

# Исходные категории стажа
SENIORITY_VALUES = {
    'Нет стажа': range(0, 1),
    'менее 4 месяцев': range(1, 4),
    '4 - 6 месяцев': range(4, 6),
    '6 месяцев - 1 год': range(6, 12),
    '1 - 2 года': range(12, 24),
    '2 - 3 года': range(24, 36),
    '3 - 4 года': range(36, 48),
    '4 - 5 лет': range(48, 60),
    '5 - 6 лет': range(60, 72),
    '6 - 7 лет': range(72, 84),
    '7 - 8 лет': range(84, 96),
    '8 - 9 лет': range(96, 108),
    '9 - 10 лет': range(108, 120),
    '10 и более лет': range(120, 1000),
}

# Новые категории стажа
NEW_SENIORITY_VALUES = {
    'Нет стажа': range(0, 1),
    'Менее 6 месяцев': range(1, 6),
    'Менее 2 лет': range(6, 24),
    'Менее 5 лет': range(24, 60),
    'Менее 10 лет': range(60, 120),
    '10 и более лет': range(120, 1000),
}


def months_seniority_to_cat(numeric_value):
    """
    Конвертация числового значения стажа в строковое
    представление исходной категории стажа
    :param numeric_value: стаж (количество месяцев)
    :return строковое представление исходной категории стажа
    """
    if numeric_value is None:
        return None

    numeric_value = int(numeric_value)

    for key, value_range in SENIORITY_VALUES.items():
        if numeric_value in value_range:
            return key

    return DEFAULT_CAT


def months_seniority_to_new_cat(numeric_value):
    """
    Конвертация числового значения стажа в строковое
    представление новой категории стажа
    :param numeric_value: стаж (количество месяцев)
    :return строковое представление новой категории стажа
    """
    if numeric_value is None:
        return None

    numeric_value = int(numeric_value)

    for key, value_range in NEW_SENIORITY_VALUES.items():
        if numeric_value in value_range:
            return key

    return DEFAULT_CAT


def seniority_cat_to_month_count(str_value):
    """
    Конвертация строкового представления исходной категории стажа
    в количество месяцев
    :param str_value:  строковое представление исходной категории стажа
    :return стаж (количество месяцев)
    """

    str_value = str(str_value)
    range_value = SENIORITY_VALUES.get(str_value, SENIORITY_VALUES[DEFAULT_CAT])

    return max(range_value)


def set_last_seniority(application_data):
    """
    Вычисление стажа работы на последнем месте в месяцах
    :param application_data:  Данные заявки
    :return: Стаж работы на последнем месте в месяцах
    """
    if not pd.isna(application_data['JobStartDate']):
        last_seniority = relativedelta(datetime.today(), pd.to_datetime(application_data['JobStartDate']))
        # Стаж работы на последнем месте в месяцах
        return last_seniority.months + last_seniority.years * 12

    return None


def set_last_seniority_cat(application_data):
    """
    Вычисление категории стажа работы на последнем месте
    :param application_data:  Данные заявки
    :return: Исходная категория стажа работы на последнем месте
    """
    last_seniority = set_last_seniority(application_data)

    return months_seniority_to_cat(last_seniority)


def set_last_seniority_new_cat(application_data):
    """
    Вычисление новой категории стажа работы на последнем месте
    :param application_data:  Данные заявки
    :return: Новая категория стажа работы на последнем месте
    """
    last_seniority = set_last_seniority(application_data)

    return months_seniority_to_new_cat(last_seniority)
