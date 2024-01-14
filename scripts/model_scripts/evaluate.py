#! python
# -*- coding: UTF-8 -*-
"""
Вычисление метрик
"""
import os
import sys
import pickle
import json
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from pathlib import Path


if __name__ == "__main__":
    stage_name = Path(sys.argv[0]).stem

    if len(sys.argv) != 5:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write(f"\tpython3 bank_id {stage_name}.py data-file  model json-file\n")
        sys.exit(1)

    # Название файла загружаемого датасета
    f_input = sys.argv[2]
    f_model = sys.argv[3]
    f_evaluate = sys.argv[4]
    bank_id = sys.argv[1]

    # %% Задание каталогов
    # Выбрать вариант в зависимости от операционной системы и способа запуска
    # project_path = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir, os.path.pardir))
    project_path = os.getcwd()
    model_dir = os.path.join(project_path, "models")
    evaluate_dir = os.path.join(project_path, "evaluate")

    # %% Создание каталогов
    os.makedirs(evaluate_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # %% Задание путей для файлов
    filename_input = os.path.join(project_path, f_input)
    filename_model = os.path.join(model_dir, f_model)
    filename_evaluate = os.path.join(evaluate_dir, f_evaluate)

    # %% Чтение файла данных
    test_data = pd.read_csv(filename_input, sep=';')
    with open(filename_model, "rb") as fd:
        clf = pickle.load(fd)

    # Подготовка датасета
    x_test = test_data.drop('Y', axis=1)
    y_test = test_data['Y']

    clf.set_params(device="cpu")
    score = clf.score(x_test, y_test)
    preds = clf.predict(x_test)
    f1 = classification_report(y_test, preds, target_names=['negative', 'positive'], zero_division=True)
    print(f1)
    f1_micro = f1_score(y_test, preds, average="micro")
    print("f1_micro=", f1_micro)

    with open(filename_evaluate, "w") as fd:
        json.dump({"micro_f1": f1_micro}, fd)
