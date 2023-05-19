import pandas as pd
import sqlite3
import math

conn = sqlite3.connect('./cars.sqlite')


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
    with open('./workload.txt') as f:
        lines = f.readlines()

    workload_elements = []
    for line in lines[2::]:
        cons = line.split('WHERE ')[1].replace('\n', '').replace("'", '').split(' AND ')

        for con in cons:
            key, values = format_condition(con)
            workload_elements.append((key, values))

    return pd.DataFrame(workload_elements, columns=['attribute', 'values'])


def read_categorical_data() -> pd.DataFrame:
    query = """
        SELECT
            brand,
            model,
            type
        FROM autompg
        """

    return pd.read_sql(query, conn)
# </editor-fold>


# <editor-fold desc="Score formulas">
def calculate_qf_frequency_categorical(rqf: int, rqf_max: int) -> float:
    return rqf / rqf_max


def calculate_idf_categorical(n: int, frequency: int) -> float:
    return math.log(n / frequency)
# </editor-fold>


# <editor-fold desc="Data aggregation">
def get_qf_frequency_categorical(workload: pd.DataFrame) -> pd.DataFrame:
    df = workload

    # Get categorical attributes
    df = df[df['attribute'].isin(['brand', 'model', 'type'])]

    # Explode list elements and count
    df = df.explode('values').value_counts().reset_index()

    # Format
    df = df.rename(columns={'count': 'frequency'})

    # Calculate QF score
    rqf_max = df['frequency'].max()
    df['qf'] = df['frequency'].apply(lambda rqf: calculate_qf_frequency_categorical(rqf, rqf_max))
    return df


def get_idf_categorical() -> pd.DataFrame:
    df = read_categorical_data()

    # setup
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
# </editor-fold>


def main():
    # Parse workload
    workload_df = read_workload_data()
    print(workload_df)

    # Calculate scores
    idf_cat_df = get_idf_categorical()
    qf_freq_cat_df = get_qf_frequency_categorical(workload_df)

    # Debug
    print(qf_freq_cat_df)


if __name__ == '__main__':
    main()
