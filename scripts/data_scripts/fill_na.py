#! python
# -*- coding: UTF-8 -*-

from .data_methods import create_stage


def fill_na_in_dataset(source_dataset):
    """
    Заполнение пропусков и удаление ненужных столбцов
    :param source_dataset:  Исходный датасет
    """

    df = source_dataset.copy()

    # Удаление пустых строк
    df = df.dropna(how='all')

    # Удаление дубликатов
    df = df.drop_duplicates()

    # Удаление ненужных столбцов
    df = df.drop(columns=['SkillFactory_Id', 'Position'''])
    df = df.reset_index(drop=True)

    # Заполнение пустых значений
    mode_gender = df['Gender'].value_counts().idxmax()
    mode_family_status = df['Family status'].value_counts().idxmax()
    mode_loan_term = df['Loan_term'].value_counts().idxmax()

    df['Value'] = df['Value'].fillna('Нет стажа')
    df['Gender'] = df['Gender'].fillna(mode_gender)
    df['Family status'] = df['Family status'].fillna(mode_family_status)
    df['ChildCount'] = df['ChildCount'].fillna(0)
    df['SNILS'] = df['SNILS'].fillna(0)
    df['Loan_amount'] = df['Loan_amount'].fillna(df['Loan_amount'].median())
    df['Loan_term'] = df['Loan_term'].fillna(mode_loan_term)

    return df


if __name__ == "__main__":
    create_stage("fill_na", fill_na_in_dataset)
