#! python
# -*- coding: UTF-8 -*-
import numpy as np

from .data_methods import create_stage
from .utils.seniority_cats import *


CATEGORIES_MERCH = list(range(1, 90))

DECISION_CATEGORIES = ['denied', 'success', 'error']

CHILDCOUNT_CATEGORIES = ['Без детей', '1 ребенок', '2 и более детей']

EDUCATION_CATEGORIES = ['Среднее', 'Среднее профессиональное', 'Высшее']

GOODS_CATEGORIES = ["Furniture", "Mobile_devices", "Travel", "Medical_services", "Education", "Fitness", "Other"]

EMPLOYMENT_CATEGORIES = ["Работаю по найму", "Собственное дело", "Иные виды"]

LOAN_TERM_CATEGORIES = [6, 12, 18, 24]

FAMILY_STATUS_CATEGORIES = ["Никогда в браке не состоял(а)", "Женат / замужем", "Разведён / Разведена"]

# Новые категории для признаков
SENIORITY_CATEGORIES = ['Нет стажа', 'Менее  6 месяцев', 'Менее 2 лет',
                        'Менее 5 лет', 'Менее 10 лет', '10 и более лет']


def create_features_in_dataset(source_dataset):
    """
    Создание новых признаков и удаление ненужных
    :param source_dataset:  Исходный датасет
    """

    df = source_dataset.copy()

    df = replace_features(df)
    df = replace_targets(df)

    return df


def replace_targets(dataset):
    """
    Преобразует и переименовывает целевые параметры
    :param dataset:  Исходный датасет
    :return: Преобразованный датасет
    """

    dataset['Решение_банка_A'] = pd.Categorical(dataset['BankA_decision'],
                                                categories=DECISION_CATEGORIES,
                                                ordered=True)
    dataset['Решение_банка_B'] = pd.Categorical(dataset['BankB_decision'],
                                                categories=DECISION_CATEGORIES,
                                                ordered=True)
    dataset['Решение_банка_C'] = pd.Categorical(dataset['BankC_decision'],
                                                categories=DECISION_CATEGORIES,
                                                ordered=True)
    dataset['Решение_банка_D'] = pd.Categorical(dataset['BankD_decision'],
                                                categories=DECISION_CATEGORIES,
                                                ordered=True)
    dataset['Решение_банка_E'] = pd.Categorical(dataset['BankE_decision'],
                                                categories=DECISION_CATEGORIES,
                                                ordered=True)
    dataset['Решение_банка_A'] = dataset['Решение_банка_A'].cat.codes
    dataset['Решение_банка_B'] = dataset['Решение_банка_B'].cat.codes
    dataset['Решение_банка_C'] = dataset['Решение_банка_C'].cat.codes
    dataset['Решение_банка_D'] = dataset['Решение_банка_D'].cat.codes
    dataset['Решение_банка_E'] = dataset['Решение_банка_E'].cat.codes

    # Удаление более ненужных признаков
    dataset = dataset.drop(columns=['BankA_decision', 'BankB_decision', 'BankC_decision',
                                    'BankD_decision', 'BankE_decision'])

    return dataset


def replace_features(dataset):
    """
    Преобразует и переименовывает признаки
    :param dataset:  Исходный датасет
    :return: Преобразованный датасет

    """

    # Создание новых числовых и бинарных признаков
    dataset['Имеет_доход'] = (
            (pd.notna(dataset['JobStartDate'])) & (dataset['employment status'] != "Не работаю")).astype('int')

    dataset['Кредитная_нагрузка'] = (dataset['MonthProfit'] - dataset['MonthExpense']) / (
            dataset['Loan_amount'] / dataset['Loan_term'])
    # df['Кредитная_нагрузка'] = np.where(df['Кредитная_нагрузка'] < 0, 0, df['Кредитная_нагрузка'])

    dataset['Кредит_возможен'] = np.where(dataset['Кредитная_нагрузка'] > 1.25, 1, 0)

    dataset['Возраст'] = dataset['BirthDate'].apply(lambda r: relativedelta(datetime.today(), r).years)

    # Создание новых категориальных признаков

    # Стаж работы на последнем месте в месяцах
    last_seniority = dataset.apply(set_last_seniority_new_cat, axis=1)
    dataset['Последний_стаж_работы'] = pd.Categorical(last_seniority, ordered=True,
                                                      categories=SENIORITY_CATEGORIES)
    last_seniority = pd.get_dummies(dataset['Последний_стаж_работы'], prefix="Посл_стаж", dtype=int)
    dataset['Код_Последний_стаж_работы'] = dataset['Последний_стаж_работы'].cat.codes

    dataset['Категория_товара'] = pd.Categorical(dataset['Goods_category'], ordered=True, categories=GOODS_CATEGORIES)
    dataset['Код_Категория_товара'] = dataset['Категория_товара'].cat.codes
    goods_category = pd.get_dummies(dataset['Категория_товара'], prefix="Кат_товара", dtype=int)

    dataset['Код_магазина'] = pd.Categorical(dataset['Merch_code'], ordered=True, categories=CATEGORIES_MERCH)
    merch_codes = pd.get_dummies(dataset['Код_магазина'], prefix="код_магазина", dtype=int)

    dataset['Family status'] = dataset['Family status'].apply(replace_family_status)
    dataset['Семейное_положение'] = pd.Categorical(dataset['Family status'], ordered=True,
                                                   categories=FAMILY_STATUS_CATEGORIES)
    dataset['Код_Семейное_положение'] = dataset['Семейное_положение'].cat.codes
    family_status = pd.get_dummies(dataset['Семейное_положение'], prefix="Сем_положение", dtype=int)

    dataset['education'] = dataset['education'].apply(replace_education)
    dataset['Образование'] = pd.Categorical(dataset['education'], ordered=True,
                                            categories=EDUCATION_CATEGORIES)
    dataset['Код_Образование'] = dataset['Образование'].cat.codes
    education = pd.get_dummies(dataset['Образование'], prefix="Образование", dtype=int)

    dataset['employment status'] = dataset['employment status'].apply(replace_employment_status)
    dataset['Тип_занятости'] = pd.Categorical(dataset['employment status'], ordered=True,
                                              categories=EMPLOYMENT_CATEGORIES)
    dataset['Код_Тип_занятости'] = dataset['Тип_занятости'].cat.codes
    employment_status = pd.get_dummies(dataset['Тип_занятости'], prefix="Занятость", dtype=int)

    dataset['Value'] = dataset['Value'].apply(replace_seniority)
    dataset['Стаж_работы'] = pd.Categorical(dataset['Value'], ordered=True, categories=SENIORITY_CATEGORIES)
    dataset['Код_Стаж_работы'] = dataset['Стаж_работы'].cat.codes
    value = pd.get_dummies(dataset['Стаж_работы'], prefix="Общий_стаж", dtype=int)

    dataset['Срок_кредита'] = pd.Categorical(dataset['Loan_term'], ordered=True, categories=LOAN_TERM_CATEGORIES)
    loan_term = pd.get_dummies(dataset['Срок_кредита'], prefix="Срок_кредита", dtype=int)

    dataset['Колво_детей'] = dataset['ChildCount'].apply(replace_childcount)
    dataset['Колво_детей'] = pd.Categorical(dataset['Колво_детей'], ordered=True,
                                            categories=CHILDCOUNT_CATEGORIES)
    dataset['Код_Колво_детей'] = dataset['Колво_детей'].cat.codes
    child_count = pd.get_dummies(dataset['Колво_детей'], prefix="Колво_детей", dtype=int)

    # Добавление кодированных признаков
    dataset = pd.concat(
        [dataset, value, education, employment_status, family_status, loan_term,
         goods_category, merch_codes, last_seniority, child_count],
        axis=1
    )

    # Удаление более ненужных признаков
    dataset = dataset.drop(columns=['BirthDate', 'JobStartDate', 'Goods_category',
                                    'Family status', 'education', 'employment status', 'Value', 'Loan_term',
                                    'ChildCount'])

    # Переименование признаков в более человекопонятные
    dataset = dataset.rename(columns={
        'MonthProfit': 'Ежемесячный_доход',
        'MonthExpense': 'Ежемесячный_расход',
        'Loan_amount': 'Сумма_заказа',
        'Gender': 'Пол',
        'SNILS': 'СНИЛС'
    })

    return dataset


def replace_family_status(old_value):
    """
    Заменяет старое значение категории признака 'Семейное_положение' на новое
    :param old_value: Старое значение категории признака 'Семейное_положение'
    :return: Новое значение категории признака 'Семейное_положение'
    """

    if old_value == 'Гражданский брак / совместное проживание':
        return 'Женат / замужем'
    if old_value == 'Вдовец / вдова':
        return 'Разведён / Разведена'
    return old_value


def replace_education(old_value):
    """
    Заменяет старое значение категории признака 'Образование' на новое
    :param old_value: Старое значение категории признака 'Образование'
    :return: Новое значение категории признака 'Образование'
    """
    if old_value in ['Высшее - специалист', 'Бакалавр', 'Магистр', 'Несколько высших']:
        return 'Высшее'
    if old_value in ['Неоконченное среднее', 'Среднее', 'Неоконченное высшее']:
        return 'Среднее'
    return old_value


def replace_employment_status(old_value):
    """
    Заменяет старое значение категории признака 'Тип_занятости' на новое
    :param old_value: Старое значение категории признака 'Тип_занятости'
    :return: Новое значение категории признака 'Тип_занятости'
    """
    if old_value in ['Работаю по найму полный рабочий день/служу', 'Работаю по найму неполный рабочий день']:
        return 'Работаю по найму'
    if old_value in ['Пенсионер', 'Студент', 'Декретный отпуск', 'Не работаю']:
        return 'Иные виды'
    return old_value


def replace_childcount(old_value):
    """
    Заменяет старое значение категории признака 'ChildCount' на новое
    :param old_value: Старое значение категории признака 'ChildCount'
    :return: Новое значение категории признака 'ChildCount'
    """
    if old_value == 0:
        return 'Без детей'
    if old_value == 1:
        return '1 ребенок'
    return '2 и более детей'


def replace_seniority(old_value):
    """
    Заменяет старое значение категории признака 'стаж работы' на новое
    :param old_value: Старое значение признака 'стаж работы'
    :return: Новое значение категории признака 'стаж работы'
    """

    # Общий стаж в месяцах
    total_seniority_in_months = int(seniority_cat_to_month_count(old_value))
    return months_seniority_to_new_cat(total_seniority_in_months)


if __name__ == "__main__":
    create_stage("create_features", create_features_in_dataset)
