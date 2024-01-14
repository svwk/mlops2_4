#! python
# -*- coding: UTF-8 -*-
"""
Подготговка датасета для определенного банка
"""
import sys
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# Признаки, не оказывающие влияния на решение банка, подлежат удалению
# из признакового пространства
all_drop_columns = {
    'A': ['Кат_товара_Education', 'Кат_товара_Education', 'Кат_товара_Other',
          'Занятость_Иные виды', 'Срок_кредита_18', 'Срок_кредита_24'],
    'B': ['Кат_товара_Education', 'Кат_товара_Travel', 'Кат_товара_Furniture',
          'Кат_товара_Education', 'Занятость_Иные виды', 'Срок_кредита_12',
          'Срок_кредита_6'],
    'C': ['Кат_товара_Travel', 'Кат_товара_Furniture', 'Кат_товара_Other',
          'Занятость_Собственное дело', 'Занятость_Иные виды', 'Занятость_Работаю по найму',
          'Срок_кредита_12', 'Срок_кредита_24'],
    'D': ['Кат_товара_Education', 'Кат_товара_Furniture', 'Кат_товара_Education',
          'Занятость_Собственное дело', 'Занятость_Работаю по найму',
          'Срок_кредита_12', 'Срок_кредита_24', 'Срок_кредита_6'],
    'E': ['Кат_товара_Other', 'Срок_кредита_18', 'Срок_кредита_24'],
}


def feature_prepare_for_bank(dataset, data_id, scaler, transform_columns):
    """
    Подготавливает датасет для определенного банка
    :param dataset: Исходный датасет
    :param data_id: Идентификатор банка
    :param scaler: Обученный объект для стандартизации числовых признаков
    :param transform_columns: Признаки, подлежащие стандартизации
    :return: Датасет для обучения модели определенного банка
    """
    # Степень полинома для создания новых признаков
    # для создания полиномиальной модели
    polynom_order = 3

    # Удаление признаков, которые не нужны для обучения модели любого банка
    dataset = dataset.drop(columns=['Последний_стаж_работы', 'Код_Последний_стаж_работы',
                                    'Категория_товара', 'Код_Категория_товара',
                                    'Код_магазина', 'Семейное_положение', 'Код_Семейное_положение',
                                    'Образование', 'Код_Образование', 'Тип_занятости', 'Код_Тип_занятости',
                                    'Стаж_работы', 'Код_Стаж_работы', 'Срок_кредита', 'Колво_детей',
                                    'Код_Колво_детей'])
    # Удаление признаков, которые не нужны для обучения модели конкретного банка
    bank_drop_columns = all_drop_columns.get(data_id, [])
    dataset = dataset.drop(columns=bank_drop_columns)

    # Масштабирование признаков
    dataset['Возраст'] = dataset['Возраст'] / 100

    scaled = scaler.transform(dataset[transform_columns])
    scaled = to_polynom(scaled, order=polynom_order)

    ext_transform_columns = transform_columns.copy()
    for o in range(2, polynom_order + 1):
        for el in transform_columns:
            ext_transform_columns = np.append(ext_transform_columns, f'{el}_{o}')
    df_standard = pd.DataFrame(scaled, columns=ext_transform_columns)

    # Удаление исходных признаков, которые уже отмасштабированы
    dataset = dataset.drop(columns=transform_columns)
    # Добавление новых отмасштабированных признаков
    dataset = pd.concat([dataset, df_standard], axis=1)

    return dataset


def to_polynom(x, order=2):
    """
    Преобразование признаков к полиному
    :param x: исходные данные
    :param order: степень полинома
    """

    order_range = range(2, order + 1, 1)
    out = np.copy(x)
    for i in order_range:
        out = np.hstack([out, np.power(x, i)])
    return out


if __name__ == "__main__":
    stage_name = "feature_prepare"

    if len(sys.argv) !=3:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write(f"\tpython3 bank_id  {stage_name}.py data-file\n")
        sys.exit(1)

    # Название файла загружаемого датасета
    f_input = sys.argv[2]
    bank_id = sys.argv[1]

    # %% Задание каталогов
    project_path = os.getcwd()
    stage_dir = os.path.join(project_path, "data", f"stage_{stage_name}")
    model_dir = os.path.join(project_path, "models")

    # %% Создание каталогов
    os.makedirs(stage_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # %% Задание путей для файлов
    filename_input = os.path.join(project_path, f_input)
    filename_output = os.path.join(stage_dir, f"dataset_{bank_id}.csv")
    scaler_filename = f'scaler_{bank_id}.pkl'
    scaler_full_filename = os.path.join(model_dir, scaler_filename)

    # %% Чтение файла данных
    df = pd.read_csv(filename_input, sep=';')

    # Подготовка датасета
    target = f'Решение_банка_{bank_id}'
    num_columns = ['Ежемесячный_доход', 'Ежемесячный_расход', 'Сумма_заказа', 'Кредитная_нагрузка']
    standard_scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    standard_scaler.fit(df[num_columns])

    df = df.drop(df[(df[target] == 2)].index)
    df = df.reset_index(drop=True)
    df = df.rename(columns={target: 'Y'})
    df = df.drop(columns=[column for column in df.columns
                          if column.startswith("Решение_банка")])
    df = feature_prepare_for_bank(df, bank_id, standard_scaler, num_columns)

    # Сохранение результатов в файлы
    joblib.dump(standard_scaler, scaler_full_filename)
    df.to_csv(filename_output, index=False, sep=';')
