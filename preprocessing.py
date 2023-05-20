import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import math

CONN = sqlite3.connect('./cars.sqlite')
CURSOR = CONN.cursor()
CATS = ['brand', 'model', 'type', 'cylinders', 'origin']
NUMS = ['mpg', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year']


# <editor-fold desc="Read data">
def format_condition(con: str) -> tuple[str, list[str]]:
    if ' = ' in con:
        elements = con.split(' = ')
        return elements[0], [elements[1]]
    else:
        elements = con.split(' IN ')
        attribute = elements[0]
        values = elements[1][1:-1].split(',')
        return attribute, values


def read_workload_data() -> pd.DataFrame:
    # Read workload file
    with open('./workload.txt') as f:
        lines = f.readlines()

    # Parse and format entries
    workload_elements = []
    for line in lines[2::]:
        # Dissect line into weight, attribute and values
        elements = line.split('WHERE ')
        weight = int(elements[0].split(' times: ')[0])
        cons = elements[1].replace('\n', '').replace("'", '').split(' AND ')

        # Add each condition to the df (weighted)
        for con in cons:
            key, values = format_condition(con)
            workload_elements += [(key, values) for _ in range(weight)]

    # Create DataFrame
    df = pd.DataFrame(workload_elements, columns=['attribute', 'value'])
    df['clauses'] = df['value'].apply(lambda l: len(l))
    return df


def read_database_data() -> pd.DataFrame:
    query = """
        SELECT
            *
        FROM autompg
        """

    return pd.read_sql(query, CONN)
# </editor-fold>


# <editor-fold desc="Score formulas">
def calculate_qf_frequency_categorical(rqf: int, rqf_max: int) -> float:
    return rqf / rqf_max


def calculate_idf_categorical(n: int, frequency: int) -> float:
    return math.log(n / frequency)


def calculate_idf_numerical(h: float, n: int, t: float | int, ts: pd.Series):
    rs = ts.apply(lambda ti: math.e**((-1/2) * (((ti - t) / h)**2)))
    return math.log(n / rs.sum())
# </editor-fold>


# <editor-fold desc="Data aggregation">
def get_qf_frequency_categorical(workload_df: pd.DataFrame) -> pd.DataFrame:
    df = workload_df.copy()

    # Get categorical attributes
    df = df[df['attribute'].isin(['brand', 'model', 'type', 'cylinders', 'origin'])]

    # Explode list elements and count
    df = df[['attribute', 'value']].explode('value').value_counts().reset_index()

    # Format
    df = df.rename(columns={'count': 'frequency'})

    # Calculate QF score
    rqf_max = df['frequency'].max()
    df['qf'] = df['frequency'].apply(lambda rqf: calculate_qf_frequency_categorical(rqf, rqf_max))
    return df


def get_qf_jaccard_categorical(workload_df: pd.DataFrame) -> pd.DataFrame:
    df = workload_df.copy()

    # Get IN clauses
    df = df[df['clauses'] > 1]

    # Explode list elements
    df['query_id'] = df.index
    df = df.explode('value').reset_index()

    # Result
    return df[['attribute', 'value', 'query_id']]


def get_idf_categorical(database_df: pd.DataFrame) -> pd.DataFrame:
    df = database_df.copy()

    # Get categorical attributes
    df = df[CATS]

    # Setup
    n = len(df)
    result_df = pd.DataFrame()

    for column in df.columns:
        # Count values
        temp_df = df[column]
        temp_df = temp_df.value_counts().reset_index()

        # Format
        temp_df['attribute'] = column
        temp_df = temp_df.rename(columns={column: 'value', 'count': 'frequency'})

        # Calculate IDF score
        temp_df['idf'] = temp_df['frequency'].apply(lambda freq: calculate_idf_categorical(n, freq))

        # Append to result
        result_df = pd.concat([result_df, temp_df])

    return result_df[['attribute', 'value', 'frequency', 'idf']]


def get_idf_numerical(database_df: pd.DataFrame) -> pd.DataFrame:
    df = database_df.copy()

    # Get categorical attributes
    df = df[NUMS]

    # Setup
    n = len(df)
    result_df = pd.DataFrame()

    for column in df.columns:
        temp_df = df[[column]].copy()

        # Format
        temp_df['attribute'] = column
        temp_df = temp_df.rename(columns={column: 'value'})

        # Determine bandwidth
        std = temp_df['value'].std()
        h = 1.06 * std * (n ** (-1/5))

        # Calculate IDF score
        temp_df['idf'] = temp_df['value'].apply(lambda t: calculate_idf_numerical(h, n, t, temp_df['value']))

        # Append result
        temp_df = temp_df.groupby(['attribute', 'value']).agg(idf=('idf', 'mean')).reset_index()
        result_df = pd.concat([result_df, temp_df])

    return result_df[['attribute', 'value', 'idf']]
# </editor-fold>


def export(name: str, df: pd.DataFrame):
    CURSOR.execute(f'DELETE FROM {name}')
    df.to_sql(name, CONN, if_exists='append', index=False, method='multi')
    print(f'Exported data to [{name}]')


def main():
    # Fetch data
    workload_df = read_workload_data()
    database_df = read_database_data()

    # Calculate scores
    idf_cat_df = get_idf_categorical(database_df)
    idf_num_df = get_idf_numerical(database_df)
    qf_rqf_cat_df = get_qf_frequency_categorical(workload_df)
    qf_jac_cat_df = get_qf_jaccard_categorical(workload_df)

    # Export data
    export('idf_cat', idf_cat_df)
    export('idf_num', idf_num_df)
    export('qf_rqf_cat', qf_rqf_cat_df)
    export('qf_jac_cat', qf_jac_cat_df)

    # Debug
    print('\nIDF numerical')
    print(idf_num_df)

    print('\nIDF categorical')
    print(idf_cat_df)

    print('\nQF rqf categorical')
    print(qf_rqf_cat_df)

    print('\nQF jaccard categorical')
    print(qf_jac_cat_df)


if __name__ == '__main__':
    main()
