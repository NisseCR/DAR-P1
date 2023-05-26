import pandas as pd
import sqlite3

CONN = sqlite3.connect('./cars.sqlite')


def read_data(name: str) -> pd.DataFrame:
    """
    Read table K from database.
    :return:
    """
    query = f"""
        SELECT
            *
        FROM {name}
        """

    return pd.read_sql(query, CONN)


def value_to_entry(v: str | float | int) -> str:
    if type(v) is str:
        return f"'{v}'"

    return f'{v}'


def row_to_entry(r: pd.Series) -> str:
    entry = '('
    for _, v in r.items():
        entry += value_to_entry(v) + ', '
    return entry[:-2] + '), \n'


def build_query(df: pd.DataFrame, name: str) -> str:
    columns = ','.join(df.columns)
    query = f'insert into {name} ({columns})\n values \n'

    for idx, r in df.iterrows():
        query += row_to_entry(r)

    query = query[:-3] + ';'
    return query


def write_query(query: str):
    with open('metaload.txt', 'a') as f:
        f.write(query)


def snapshot(name: str):
    df = read_data(name)
    query = build_query(df, name)
    write_query(query)


def main():
    snapshot('idf_cat')
    snapshot('idf_num')
    snapshot('qf_jac_cat')
    snapshot('qf_rqf_cat')
    snapshot('qf_rqf_num')


if __name__ == '__main__':
    main()
