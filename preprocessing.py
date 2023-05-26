import sys

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
    """
    Retrieve tuple of attribute A and term t.
    In case of an IN condition, return a list of t's.
    :param con: condition string
    :return: (A, t[])
    """
    if ' = ' in con:
        elements = con.split(' = ')
        return elements[0], [elements[1]]
    else:
        elements = con.split(' IN ')
        attribute = elements[0]
        values = elements[1][1:-1].split(',')
        return attribute, values


def read_workload_data() -> pd.DataFrame:
    """
    Parse workload file and convert data to a DataFrame.
    :return:
    """
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
    """
    Read table K from database.
    :return:
    """
    query = """
        SELECT
            *
        FROM autompg
        """

    return pd.read_sql(query, CONN)
# </editor-fold>


def shift_merge(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pair each tuple with the next tuple in the DataFrame, allowing for easier interpolation lookup.
    :param df: numeric data
    :return: DataFrame with ranges min and max
    """
    df = df.sort_values(by='value')
    df['key'] = df.index
    join_df = df.copy()
    join_df['key'] = join_df['key'].shift(1)
    df = pd.merge(df, join_df, on='key')
    df = df.drop([])
    df = df.rename(columns={'attribute_x': 'attribute'})
    df = df.drop(['attribute_y', 'key'], axis=1)
    return df


def add_boundaries(df: pd.DataFrame, attribute: str) -> pd.DataFrame:
    idx = len(df)
    df.loc[idx] = [0, attribute, 0, 0]
    df.loc[idx + 1] = [9999, attribute, 0, 0]
    return df


# <editor-fold desc="Score formulas">
def calculate_rqf(rqf: int, rqf_max: int) -> float:
    """
    Calculate the RQF score of a TF.
    :param rqf: term frequency
    :param rqf_max: max frequency
    :return: RQF
    """
    return (rqf + 1) / (rqf_max + 1)


def calculate_numerical_tf(h: float, t: float | int, ts: pd.Series):
    """
    Smooth curve function to approximate frequency distribution.
    :param h: bandwidth constant
    :param t: term t
    :param ts: row in which term t is present
    :return: approximated TF
    """
    rs = ts.apply(lambda ti: math.e**((-1/2) * (((ti - t) / h)**2)))
    return rs.sum()


def calculate_jaccard(p: list, q: list) -> float:
    """
    Calculate similarity between two terms, based on workload data.
    :param p: query set of term p
    :param q: query set of term q
    :return:
    """
    p_set = set(p)
    q_set = set(q)

    intersect = len(p_set.intersection(q_set))
    return intersect / (len(p_set) + len(q_set) - intersect)


def calculate_idf(n: int, tf: int) -> float:
    """
    Calculate the IDF score of a tf.
    :param n: total amount of tuples
    :param tf: term frequency
    :return:
    """
    return math.log(n / tf)
# </editor-fold>


# <editor-fold desc="Data aggregation">
# <editor-fold desc="QF categorical">
def get_categorical_qf(workload_df: pd.DataFrame) -> pd.DataFrame:
    """
    Categorical QF pipeline.
    :param workload_df:
    :return:
    """
    df = workload_df.copy()

    # Get categorical attributes
    df = df[df['attribute'].isin(CATS)]

    # Explode list elements and count
    df = df[['attribute', 'value']].explode('value').value_counts().reset_index()

    # Format
    df = df.rename(columns={'count': 'tf'})

    # Calculate QF score
    rqf_max = df['tf'].max()
    df['qf'] = df['tf'].apply(lambda rqf: calculate_rqf(rqf, rqf_max))
    return df
# </editor-fold>


# <editor-fold desc="QF numerical">
def add_numerical_qf_attribute(df: pd.DataFrame, attribute: str, result_df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Numerical QF pipeline for a single attribute / column.
    :param df:
    :param attribute:
    :param result_df:
    :param n:
    :return:
    """
    temp_df = df.copy()
    temp_df = temp_df[temp_df['attribute'] == attribute]

    # Determine bandwidth
    std = temp_df['value'].std()
    h = 1.06 * std * (n ** (-1 / 5))

    # Calculate QF score
    temp_df['tf'] = temp_df['value'].apply(lambda t: calculate_numerical_tf(h, t, temp_df['value']))
    rqf_max = temp_df['tf'].max()
    temp_df['qf'] = temp_df['tf'].apply(lambda rqf: calculate_rqf(rqf, rqf_max))

    # Add boundaries
    temp_df = add_boundaries(temp_df, attribute)

    # Append result
    temp_df = temp_df.groupby(['attribute', 'value']).agg(tf=('tf', 'mean'), qf=('qf', 'mean')).reset_index()

    # Shift merge
    temp_df = shift_merge(temp_df)
    return pd.concat([result_df, temp_df])


def get_numerical_qf(workload_df: pd.DataFrame) -> pd.DataFrame:
    """
    Numerical QF pipeline.
    :param workload_df:
    :return:
    """
    df = workload_df[['attribute', 'value']].copy()

    # Get numerical attributes
    df = df[df['attribute'].isin(NUMS)]
    df = df.explode('value')
    df['value'] = df['value'].astype(float)

    # Setup
    n = len(df)
    result_df = pd.DataFrame()

    for attribute in NUMS:
        result_df = add_numerical_qf_attribute(df, attribute, result_df, n)

    return result_df[['attribute', 'value_x', 'value_y', 'tf_x', 'tf_y', 'qf_x', 'qf_y']]
# </editor-fold>


# <editor-fold desc="Jaccard">
def get_jaccard(workload_df: pd.DataFrame) -> pd.DataFrame:
    """
    Categorical Jaccard pipeline
    :param workload_df:
    :return:
    """
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
    score_df['qf'] = score_df.apply(lambda r: calculate_jaccard(r['query_id_x'], r['query_id_y']), axis=1)
    return score_df[['attribute', 'value_x', 'value_y', 'qf']]
# </editor-fold>


# <editor-fold desc="IDF categorical">
def add_categorical_idf_attribute(df: pd.DataFrame, column: str, result_df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Categorical IDF pipeline for a single attribute / column.
    :param df:
    :param column:
    :param result_df:
    :param n:
    :return:
    """
    # Count values
    temp_df = df[column]
    temp_df = temp_df.value_counts().reset_index()

    # Format
    temp_df['attribute'] = column
    temp_df = temp_df.rename(columns={column: 'value', 'count': 'tf'})

    # Calculate IDF score
    temp_df['idf'] = temp_df['tf'].apply(lambda freq: calculate_idf(n, freq))

    # Append to result
    return pd.concat([result_df, temp_df])


def get_categorical_idf(database_df: pd.DataFrame) -> pd.DataFrame:
    """
    Categorical IDF pipeline.
    :param database_df:
    :return:
    """
    df = database_df.copy()

    # Get categorical attributes
    df = df[CATS]

    # Setup
    n = len(df)
    result_df = pd.DataFrame()

    for column in df.columns:
        result_df = add_categorical_idf_attribute(df, column, result_df, n)

    return result_df[['attribute', 'value', 'tf', 'idf']]
# </editor-fold>


# <editor-fold desc="IDF numerical">
def add_numerical_idf_attribute(df: pd.DataFrame, column: str, result_df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Numerical IDF pipeline for a single attribute / column.
    :param df:
    :param column:
    :param result_df:
    :param n:
    :return:
    """
    temp_df = df[[column]].copy()

    # Format
    temp_df['attribute'] = column
    temp_df = temp_df.rename(columns={column: 'value'})

    # Determine bandwidth
    std = temp_df['value'].std()
    h = 1.06 * std * (n ** (-1 / 5))

    # Calculate IDF score
    temp_df['tf'] = temp_df['value'].apply(lambda t: calculate_numerical_tf(h, t, temp_df['value']))
    temp_df['idf'] = temp_df['tf'].apply(lambda f: calculate_idf(n, f))

    # Add boundaries
    temp_df = add_boundaries(temp_df, column)

    # Append result
    temp_df = temp_df.groupby(['attribute', 'value']).agg(tf=('tf', 'mean'), idf=('idf', 'mean')).reset_index()

    # Merge with shift
    temp_df = shift_merge(temp_df)
    return pd.concat([result_df, temp_df])


def get_numerical_idf(database_df: pd.DataFrame) -> pd.DataFrame:
    """
    Numerical IDF pipeline.
    :param database_df:
    :return:
    """
    df = database_df.copy()

    # Get categorical attributes
    df = df[NUMS]

    # Setup
    n = len(df)
    result_df = pd.DataFrame()

    for column in df.columns:
        result_df = add_numerical_idf_attribute(df, column, result_df, n)

    return result_df[['attribute', 'value_x', 'value_y', 'tf_x', 'tf_y', 'idf_x', 'idf_y']]
# </editor-fold>
# </editor-fold>


# <editor-fold desc="Export">
def export(name: str, df: pd.DataFrame):
    """
    Export data to database table.
    :param name:
    :param df:
    :return:
    """
    CURSOR.execute(f'DELETE FROM {name}')
    df.to_sql(name, CONN, if_exists='append', index=False, method='multi')
    print(f'Exported data to [{name}]')
# </editor-fold>


def main():
    """
    Run method of pre-processing unit.
    :return:
    """
    # Fetch data
    workload_df = read_workload_data()
    database_df = read_database_data()

    # Calculate scores
    idf_cat_df = get_categorical_idf(database_df)
    idf_num_df = get_numerical_idf(database_df)
    qf_rqf_cat_df = get_categorical_qf(workload_df)
    qf_jac_cat_df = get_jaccard(workload_df)
    qf_rqf_num_df = get_numerical_qf(workload_df)

    # Export data
    export('idf_cat', idf_cat_df)
    export('idf_num', idf_num_df)
    export('qf_rqf_cat', qf_rqf_cat_df)
    export('qf_jac_cat', qf_jac_cat_df)
    export('qf_rqf_num', qf_rqf_num_df)

    # Debug
    print('\nIDF numerical')
    print(idf_num_df)

    print('\nIDF categorical')
    print(idf_cat_df)

    print('\nRQF categorical')
    print(qf_rqf_cat_df)

    print('\nRQF numerical')
    print(qf_rqf_num_df)

    print('\nJaccard categorical')
    print(qf_jac_cat_df)

    # Plot
    # test = qf_rqf_num_df[qf_rqf_num_df['attribute'] == 'acceleration']
    # test['value'].plot(kind='hist')
    # test.plot.scatter(x='value', y='qf')
    # plt.show()


if __name__ == '__main__':
    main()
