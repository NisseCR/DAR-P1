import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import math
from preprocessing import read_database_data, CATS, NUMS


def main():
    dict = parse()
    k = dict.pop('k')
    tuples = read_database_data()
    tuples['score'] = tuples.apply(lambda tup: sim(tup, dict), axis=1)
    print(tuples)


def sim(tup: pd.Series, query: dict) -> int:
    score = 0
    for queryCat,queryVal in query.items():
        if queryCat in CATS:
            score += s_cat(queryCat, queryVal, tup[queryCat])
        elif queryCat in NUMS:
            score += s_num(queryCat, queryVal, tup[queryCat])
        else:
            print("cat not found")
    return 1


def s_cat(cat, qVal, tVal):
    print(f"{cat},{qVal},{tVal}")
    return 1


def s_num(cat, qVal, tVal):
    print(f"{cat},{qVal},{tVal}")
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
