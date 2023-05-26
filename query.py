import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from math import e
from preprocessing import read_database_data, CATS, NUMS, CURSOR, CONN

pd.options.mode.chained_assignment = None

def main():
    dict = parse()
    result = retrieve_tuples(dict)
    pd.set_option('display.max_columns', 15)
    print(result)


def retrieve_tuples(dict):
    k = dict.pop('k')
    tuples = calculate_scores(dict)
    sorted = tuples.sort_values('score', ascending=False)
    max_score = sorted.iloc[0]['score']
    tuples_with_max_score = sorted[sorted['score'] == max_score]
    #if the top k tuples all have the same score a sort of "tiebreaker" is needed
    if (len(tuples_with_max_score.index) > k):
        attributes_not_in_query = [attribute for attribute in CATS + NUMS if attribute not in dict.keys()]
        res = calculate_many_answers(attributes_not_in_query, tuples_with_max_score)
        sorted2 = res.sort_values('score2', ascending=False)
        return sorted2.head(k)
    else:
        return sorted.head(k)


def calculate_scores(dict):
    #fetch entire autompg database
    tuples = read_database_data()
    num_data = {cat: (idf_num(cat, val), qf_num(cat, val), calc_h(cat)) for cat, val in dict.items() if cat in NUMS}
    tuples['score'] = tuples.apply(lambda tup: sim(tup, dict, num_data), axis=1)
    return tuples

# to break ties the sum of the qfs of none query attributes is used as the score
def calculate_many_answers(attributes, tuples):
    tuples['score2'] = tuples.apply(lambda tup: row_qf_score(tup, attributes), axis=1)
    return tuples


def row_qf_score(tuple, attributes):
    score = 0
    for attribute in attributes:
        if attribute in CATS:
            score += get_qf(attribute, tuple[attribute])
        elif attribute in NUMS:
            score += qf_num(attribute, tuple[attribute])
    return score


def calc_h(cat):
    query = f"SELECT {cat} FROM autompg"
    cur = CONN.cursor()
    cur.execute(query)
    res = [row[0] for row in cur.fetchall()]
    stdev = np.std(res)
    return 1.06*stdev*len(res)**(-1/5)


def sim(tup: pd.Series, query: dict, num_data: dict) -> float:
    score = 0
    for queryCat, queryVal in query.items():
        if queryCat in CATS:
            score += s_cat(queryCat, tup[queryCat], queryVal)
        elif queryCat in NUMS:
            score += s_num(queryCat, tup[queryCat], queryVal, num_data)
        else:
            raise ValueError("Invalid category in input")
    return score


def s_cat(cat, t_val, q_val):
    idf = idf_cat(cat, t_val, q_val)
    qf = qf_cat(cat,t_val,q_val)
    res = idf*qf
    return res

def idf_cat(cat, t_val, q_val) -> float:
    if (t_val==q_val):
        return get_idf_cat(cat, t_val)
    else:
        return 0


def qf_cat(cat, t_val, q_val):
    qf = get_qf(cat, t_val)
    return qf*jacar(cat, t_val, q_val)


def jacar(cat, t_val, q_val):
    if(t_val==q_val):
        return 0
    query = f"SELECT qf FROM qf_jac_cat WHERE attribute='{cat}' AND value_x='{t_val}' AND value_y='{q_val}'"
    cur = CONN.cursor()
    cur.execute(query)
    res = cur.fetchone()
    if res:
        return res[0]
    else:
        return 0

def get_qf(cat, val):
    query = f"SELECT qf FROM qf_rqf_Cat WHERE attribute='{cat}' AND value='{val}'"
    cur = CONN.cursor()
    cur.execute(query)
    res = cur.fetchone()
    if res:
        return res[0]
    else:
        RQFMaxQuery = "SELECT MAX(tf) FROM qf_rqf_cat"
        cur.execute(RQFMaxQuery)
        RQFMax = cur.fetchone()[0]
        return 1/(RQFMax+1)


def get_idf_cat(cat, val):
    query = f"SELECT idf FROM idf_Cat WHERE attribute='{cat}' AND value='{val}'"
    cur = CONN.cursor()
    cur.execute(query)
    res = cur.fetchone()
    if res:
        return res[0]
    else:
        return 0


def s_num(cat, t_val, q_val, num_data):
    idf, qf, h = num_data[cat]
    factor = e**(-0.5*((t_val-q_val)/h)**2)
    if(t_val==q_val):
        return idf*qf
    res = idf*factor*qf
    return res


def get_between_and_interpolate(cat, table, res_attribute, val):
    query = f"SELECT {res_attribute}_x,{res_attribute}_y,value_x,value_y FROM {table} WHERE attribute='{cat}' AND {val} between value_x and value_y"
    cur = CONN.cursor()
    cur.execute(query)
    res = cur.fetchone()
    if res:
        xres, yres, xval, yval = res
        dif = yval - xval
        factor = (val-xval)/dif
        difres = yres - xres
        return yres + difres*factor
    else:
        return 0


def idf_num(cat, t_val):
    return get_between_and_interpolate(cat, "idf_num", "idf", t_val)


def qf_num(cat, t_val):
    return get_between_and_interpolate(cat, "qf_rqf_num", "qf", t_val)


def parse()->dict:
    prompt = input()
    terminatedRE = re.compile(r"[^;]+")
    terminated = terminatedRE.match(prompt).group()
    predicatesRE = re.compile(r"[^,]+")
    predicates = predicatesRE.findall(terminated)
    dict = {}
    for predicate in predicates:
        splitre = re.compile(r"\s?=\s?")
        attribute, value = splitre.split(predicate)
        attribute = attribute.strip()
        value = eval(value.strip())
        dict[attribute] = value
    if 'k' not in dict:
        dict['k'] = 10
    return dict


if __name__== '__main__':
    main()
