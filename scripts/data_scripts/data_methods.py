"""
Общие методы работы с данными
"""
import sys
import os
import pandas as pd

# Выбрать вариант в зависимости от операционной системы и способа запуска
# project_path = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir, os.path.pardir))
project_path = os.getcwd()


def create_stage(stage_name, function):
    """
    Выполнение одного из этапов конвейера обработки данных
    :param stage_name: название этапа конвейера
    :param function: функция, выполняющая основную задачу этапа
    :return: путь для каталога с результатами работы этапа
    """

    if len(sys.argv) != 2:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write(f"\tpython3 {stage_name}.py data-file\n")
        sys.exit(1)

    # Название файла загружаемого датасета
    f_input = sys.argv[1]

    # %%  Задание путей для файлов
    stage_dir = os.path.join(project_path, "data", f"stage_{stage_name}")
    os.makedirs(stage_dir, exist_ok=True)
    filename_input = os.path.join(project_path, f_input)
    filename_output = os.path.join(stage_dir, "dataset.csv")

    # %% Чтение файла данных
    df = pd.read_csv(filename_input, sep=';', parse_dates=['JobStartDate', 'BirthDate'])

    # Подготовка датасета
    df = function(df)

    # Сохранение результатов в файлы
    df.to_csv(filename_output, index=False, sep=';')

    return stage_dir

