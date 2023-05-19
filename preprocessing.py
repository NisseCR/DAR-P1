import pandas as pd
import sqlite3
import math

conn = sqlite3.connect('./cars.sqlite')


def get_categorical_data() -> pd.DataFrame:
    query = """
        SELECT
            brand,
            model,
            type
        FROM autompg
        """

    df = pd.read_sql(query, conn)
    return df


def calculate_idf_categorical(df: pd.DataFrame) -> pd.DataFrame:
    pass


def idf_categorical() -> pd.DataFrame:
    df = get_categorical_data()
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
    print(idf_cat_df)


if __name__ == '__main__':
    main()