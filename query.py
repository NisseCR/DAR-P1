import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from math import e
from preprocessing import read_database_data, CATS, NUMS, CURSOR, CONN


def main():
    dict = parse()
    k = dict.pop('k')
    tuples = read_database_data()
    num_data = {cat: (idf_num(cat, val), qf_num(cat,val), calc_h(cat)) for cat, val in dict.items() if cat in NUMS}
    tuples['score'] = tuples.apply(lambda tup: sim(tup, dict, num_data), axis=1)
    sorted = tuples.sort_values('score', ascending=False)
    print(sorted.head(k))


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
    return idf*qf

def idf_cat(cat, t_val, q_val) -> float:
    if (t_val==q_val):
        return get_idf_cat(cat, t_val)
    else:
        return 0


def qf_cat(cat, t_val, q_val):
    qf = get_qf(cat, t_val)
    return qf*jacar(cat, t_val, q_val)


def jacar(cat, t_val, q_val):
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
    return e**(-0.5*((t_val-q_val)/2)**2)*idf*qf


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
    print(terminated)
    predicatesRE = re.compile(r"[^,]+")
    predicates = predicatesRE.findall(terminated)
    print(predicates)
    dict = {}
    for predicate in predicates:
        attribute, value = predicate.split(" = ")
        attribute = attribute.strip()
        value = eval(value.strip())
        dict[attribute] = value
    if 'k' not in dict:
        dict['k'] = 10
    return dict


if __name__== '__main__':
    main()
