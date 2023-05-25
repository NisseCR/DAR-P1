import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import math
from preprocessing import read_database_data, CATS, NUMS, CURSOR, CONN


def main():
    dict = parse()
    k = dict.pop('k')
    tuples = read_database_data()
    tuples['score'] = tuples.apply(lambda tup: sim(tup, dict), axis=1)
    #tuples.sort_values('score')
    print(tuples)


def sim(tup: pd.Series, query: dict) -> float:
    score = 0
    for queryCat, queryVal in query.items():
        if queryCat in CATS:
            score += s_cat(queryCat, tup[queryCat], queryVal)
        elif queryCat in NUMS:
            score += s_num(queryCat, tup[queryCat], queryVal)
        else:
            print("cat not found")
    return score


def s_cat(cat, t_val, q_val):
    idf = idf_cat(cat, t_val, q_val)
    qf = qf_cat(cat,t_val,q_val)
    return idf*qf

def idf_cat(cat, t_val, q_val) -> float:
    if (t_val==q_val):
        print(f"IDF RES:::::{t_val}, {get_idf_cat(cat,t_val)}")
        return get_idf_cat(cat, t_val)
    else:
        return 0


def qf_cat(cat, t_val, q_val):
    return 1

def get_idf_cat(cat, val):
    query = f"SELECT idf FROM idf_Cat WHERE attribute='{cat}' AND value='{val}'"
    cur = CONN.cursor()
    cur.execute(query)
    return cur.fetchone()[0]


def s_num(cat, t_val, q_val):
    print(f"{cat},{t_val},{q_val}")
    return 1


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