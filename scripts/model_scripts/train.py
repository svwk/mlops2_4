#! python
# -*- coding: UTF-8 -*-
import sys
import os
import yaml
import pickle
import pandas as pd


def train_stage(stage_name, train_function, params_function):
    if len(sys.argv) != 3:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write(f"\tpython3 bank_id {stage_name}.py data-file\n")
        sys.exit(1)

    # Название файла загружаемого датасета
    f_input = sys.argv[2]
    bank_id = sys.argv[1]

    # %% Задание каталогов
    # Выбрать вариант в зависимости от операционной системы и способа запуска
    # project_path = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir, os.path.pardir))
    project_path = os.getcwd()
    model_dir = os.path.join(project_path, "models")

    # Загрузка параметров расчета
    params = yaml.safe_load(open(os.path.join(project_path, "params.yaml")))

    # %% Задание путей для файлов
    filename_input = os.path.join(project_path, f_input)
    filename_output = os.path.join(model_dir, f"model_{stage_name}_{bank_id}.pkl")

    # %% Чтение файла данных
    train_data = pd.read_csv(filename_input, sep=';')

    # Обучение модели
    model_params = params_function(params)
    model = train_function(train_data, model_params)

    # %% Сохранение результатов в файлы
    os.makedirs(model_dir, exist_ok=True)
    with open(filename_output, "wb") as fd:
        pickle.dump(model, fd)
