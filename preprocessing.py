from typing import Tuple, List

import pandas as pd
import sqlite3
import math

conn = sqlite3.connect('./cars.sqlite')


def format_condition(con: str) -> tuple[str, list[str]]:
    if ' = ' in con:
        elements = con.split(' = ')
        return elements[0], [elements[1]]
    else:
        elements = con.split(' IN ')
        attribute = elements[0]
        values = elements[1][1:-1].split(',')
        return attribute, values


def parse_workload_data():
    with open('./workload.txt') as f:
        lines = f.readlines()

    workload_elements = []
    for line in lines[2::]:
        cons = line.split('WHERE ')[1].replace('\n', '').replace("'", '').split(' AND ')
        print(cons)

        for con in cons:
            key, values = format_condition(con)
            workload_elements.append([key, values])

    for ele in workload_elements:
        print(ele)


def read_categorical_data() -> pd.DataFrame:
    query = """
        SELECT
            brand,
            model,
            type
        FROM autompg
        """

    return pd.read_sql(query, conn)


def calculate_idf_categorical(df: pd.DataFrame) -> pd.DataFrame:
    pass


def idf_categorical() -> pd.DataFrame:
    df = read_categorical_data()

    # setup
    n = len(df)
    result_df = pd.DataFrame()

    for column in df.columns:
        # Count values
        temp_df = df[column]
        temp_df = temp_df.value_counts().reset_index()

        # Rename columns
        temp_df['attribute'] = column
        temp_df = temp_df.rename(columns={column: 'value', 'count': 'frequency'})

        # Calculate idf score
        temp_df['idf'] = temp_df['frequency'].apply(lambda freq: math.log(n / freq))

        # Append to result
        result_df = pd.concat([result_df, temp_df])

    return result_df[['attribute', 'value', 'frequency', 'idf']]


def main():
    idf_cat_df = idf_categorical()
    parse_workload_data()


if __name__ == '__main__':
    main()