#! python
# -*- coding: UTF-8 -*-

import sys
import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split


def separate_bank_dataset(source_dataset, target_name, p_split_ratio, random_state=42):
    """
     Разделение на обучающую и тестовую выборки
     :param source_dataset: Исходный датасет
    :param target_name: Название целевого параметра
    :param p_split_ratio: Отношение для разделения
    :param random_state: фиксированный сид случайных чисел (для повторяемости)
    :return: Два дата-фрейма с обучающими и тестовыми данными
    """

    x = source_dataset.drop(target_name, axis=1)
    y = source_dataset[target_name]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=p_split_ratio,
                                                        random_state=random_state,
                                                        stratify=y)

    df_train = pd.concat([x_train, y_train], axis=1)
    df_test = pd.concat([x_test, y_test], axis=1)

    return df_train, df_test


if __name__ == "__main__":
    stage_name = "train_test_split"

    if len(sys.argv) != 3:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write(f"\tpython3 bank_id {stage_name}.py data-file\n")
        sys.exit(1)

    # Название файла загружаемого датасета
    f_input = sys.argv[2]
    bank_id = sys.argv[1]

    # %% Задание путей для файлов
    project_path = os.getcwd()
    stage_dir = os.path.join(project_path, "data", f"stage_{stage_name}")
    filename_input = os.path.join(project_path, f_input)

    # %% Создание каталогов
    os.makedirs(stage_dir, exist_ok=True)

    # %% Загрузка параметров расчета
    params = yaml.safe_load(open(os.path.join(project_path, "params.yaml")))
    split_ratio = params["split"]["split_ratio"]
    random_state = params["split"]["random_state"]

    # %% Чтение файла данных
    df = pd.read_csv(filename_input, sep=';')
    print(f'Строк - {df.shape[0]}')

    # Подготовка датасетов
    target_column = 'Y'
    df_train, df_test = separate_bank_dataset(df, target_column, split_ratio, random_state)

    # Сохранение результатов в файлы
    train_filename_output = os.path.join(stage_dir, f"train_{bank_id}.csv")
    test_filename_output = os.path.join(stage_dir, f"test_{bank_id}.csv")
    df_train.to_csv(train_filename_output, index=False, sep=';')
    df_test.to_csv(test_filename_output, index=False, sep=';')
