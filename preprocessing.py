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
            workload_elements += [(key, values) for _ in range(1)]

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


def calculate_qf_similarity_categorical(p: list, q: list) -> float:
    p_set = set(p)
    q_set = set(q)

    intersect = len(p_set.intersection(q_set))
    return intersect / (len(p_set) + len(q_set) - intersect)


def calculate_idf_categorical(n: int, frequency: int) -> float:
    return math.log(n / frequency)


def calculate_idf_numerical(h: float, n: int, t: float | int, ts: pd.Series):
    rs = ts.apply(lambda ti: math.e**((-1/2) * (((ti - t) / h)**2)))
    return math.log(n / rs.sum())


def calculate_qf_numerical(h: float, n: int, t: float | int, ts: pd.Series):
    rs = ts.apply(lambda ti: math.e**((-1/2) * (((ti - t) / h)**2)))
    return math.log(rs.sum() / n)
# </editor-fold>


# <editor-fold desc="Data aggregation">
def get_qf_frequency_categorical(workload_df: pd.DataFrame) -> pd.DataFrame:
    df = workload_df.copy()

    # Get categorical attributes
    df = df[df['attribute'].isin(CATS)]

    # Explode list elements and count
    df = df[['attribute', 'value']].explode('value').value_counts().reset_index()

    # Format
    df = df.rename(columns={'count': 'frequency'})

    # Calculate QF score
    rqf_max = df['frequency'].max()
    df['qf'] = df['frequency'].apply(lambda rqf: calculate_qf_frequency_categorical(rqf, rqf_max))
    return df


def add_qf_frequency_numerical_attribute(df: pd.DataFrame, attribute: str, result_df: pd.DataFrame, n: int) -> pd.DataFrame:
    temp_df = df.copy()
    temp_df = temp_df[temp_df['attribute'] == attribute]

    # Determine bandwidth
    std = temp_df['value'].std()
    h = 1.06 * std * (n ** (-1 / 5))

    # Calculate IDF score
    temp_df['qf'] = temp_df['value'].apply(lambda t: calculate_qf_numerical(h, n, t, temp_df['value']))

    # Append result
    temp_df = temp_df.groupby(['attribute', 'value']).agg(qf=('qf', 'mean')).reset_index()
    return pd.concat([result_df, temp_df])


def get_qf_frequency_numerical(workload_df: pd.DataFrame) -> pd.DataFrame:
    df = workload_df[['attribute', 'value']].copy()

    # Get numerical attributes
    df = df[df['attribute'].isin(NUMS)]
    df = df.explode('value')
    df['value'] = df['value'].astype(float)

    # Setup
    n = len(df)
    result_df = pd.DataFrame()

    for attribute in NUMS:
        result_df = add_qf_frequency_numerical_attribute(df, attribute, result_df, n)
    return result_df


def get_qf_jaccard_categorical(workload_df: pd.DataFrame) -> pd.DataFrame:
    df = workload_df.copy()

    # Get IN clauses
    df = df[df['clauses'] > 1]

    # Explode list elements
    df['query_id'] = df.index
    df = df.explode('value').reset_index()
    df = df.groupby(['attribute', 'value']).agg(({'query_id': lambda x: list(x)})).reset_index()

    # Cross join
    df['key'] = 1
    df1 = df.copy().drop(['attribute'], axis=1)
    score_df = pd.merge(df, df1, on='key').drop('key', axis=1)

    # Calculate QF score
    score_df['qf'] = score_df.apply(lambda r: calculate_qf_similarity_categorical(r['query_id_x'], r['query_id_y']), axis=1)
    return score_df[['attribute', 'value_x', 'value_y', 'qf']]


def add_idf_categorical_attribute(df: pd.DataFrame, column: str, result_df: pd.DataFrame, n: int) -> pd.DataFrame:
    # Count values
    temp_df = df[column]
    temp_df = temp_df.value_counts().reset_index()

    # Format
    temp_df['attribute'] = column
    temp_df = temp_df.rename(columns={column: 'value', 'count': 'frequency'})

    # Calculate IDF score
    temp_df['idf'] = temp_df['frequency'].apply(lambda freq: calculate_idf_categorical(n, freq))

    # Append to result
    return pd.concat([result_df, temp_df])


def get_idf_categorical(database_df: pd.DataFrame) -> pd.DataFrame:
    df = database_df.copy()

    # Get categorical attributes
    df = df[CATS]

    # Setup
    n = len(df)
    result_df = pd.DataFrame()

    for column in df.columns:
        result_df = add_idf_categorical_attribute(df, column, result_df, n)

    return result_df[['attribute', 'value', 'frequency', 'idf']]


def add_idf_numerical_attribute(df: pd.DataFrame, column: str, result_df: pd.DataFrame, n: int) -> pd.DataFrame:
    temp_df = df[[column]].copy()

    # Format
    temp_df['attribute'] = column
    temp_df = temp_df.rename(columns={column: 'value'})

    # Determine bandwidth
    std = temp_df['value'].std()
    h = 1.06 * std * (n ** (-1 / 5))

    # Calculate IDF score
    temp_df['idf'] = temp_df['value'].apply(lambda t: calculate_idf_numerical(h, n, t, temp_df['value']))

    # Append result
    temp_df = temp_df.groupby(['attribute', 'value']).agg(idf=('idf', 'mean')).reset_index()
    return pd.concat([result_df, temp_df])


def get_idf_numerical(database_df: pd.DataFrame) -> pd.DataFrame:
    df = database_df.copy()

    # Get categorical attributes
    df = df[NUMS]

    # Setup
    n = len(df)
    result_df = pd.DataFrame()

    for column in df.columns:
        result_df = add_idf_numerical_attribute(df, column, result_df, n)

    return result_df[['attribute', 'value', 'idf']]
# </editor-fold>


# <editor-fold desc="Export">
def export(name: str, df: pd.DataFrame):
    CURSOR.execute(f'DELETE FROM {name}')
    df.to_sql(name, CONN, if_exists='append', index=False, method='multi')
    print(f'Exported data to [{name}]')
# </editor-fold>


def main():
    # Fetch data
    workload_df = read_workload_data()
    database_df = read_database_data()

    # Calculate scores
    idf_cat_df = get_idf_categorical(database_df)
    idf_num_df = get_idf_numerical(database_df)
    qf_rqf_cat_df = get_qf_frequency_categorical(workload_df)
    qf_jac_cat_df = get_qf_jaccard_categorical(workload_df)
    qf_rqf_num_df = get_qf_frequency_numerical(workload_df)

    # Export data
    # export('idf_cat', idf_cat_df)
    # export('idf_num', idf_num_df)
    # export('qf_rqf_cat', qf_rqf_cat_df)
    # export('qf_jac_cat', qf_jac_cat_df)

    # Debug
    # print('\nIDF numerical')
    # print(idf_num_df)
    #
    # print('\nIDF categorical')
    # print(idf_cat_df)
    #
    # print('\nQF rqf categorical')
    # print(qf_rqf_cat_df)
    #
    # print('\nQF jaccard categorical')
    # print(qf_jac_cat_df)
    #
    print('\nQF rqf numerical')
    print(qf_rqf_num_df)

    rf = qf_rqf_num_df[qf_rqf_num_df['attribute'] == 'acceleration']

    print(rf)
    rf['value'].plot(kind='hist')
    rf.plot.scatter(x='value', y='qf')
    plt.show()


if __name__ == '__main__':
    main()
