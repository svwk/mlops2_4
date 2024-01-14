#! python
# -*- coding: UTF-8 -*-

"""
Исследовательский анализ
"""
import pandas as pd
import os

# %% Задание пути для сохранения файлов
project_path = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir, os.path.pardir, os.path.pardir))
f_input = os.path.join("data", "raw", "SF_Mandarin_dataset_ver3_csv.csv")

# %% Чтение файла данных
filename = os.path.join(project_path, f_input)
df = pd.read_csv(filename, sep=';')



