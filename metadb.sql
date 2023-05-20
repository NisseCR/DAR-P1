/* Reset data */
DROP TABLE IF EXISTS idf_num;
DROP TABLE IF EXISTS idf_cat;
DROP TABLE IF EXISTS qf_rqf_cat;
DROP TABLE IF EXISTS qf_jac_cat;

/* IDF numerical */
CREATE TABLE idf_num (
    attribute VARCHAR(255),
    value REAL,
    idf REAL,
    PRIMARY KEY (attribute, value)
);

/* IDF categorical */
CREATE TABLE idf_cat (
    attribute VARCHAR(255),
    value REAL,
    frequency INTEGER,
    idf REAL,
    PRIMARY KEY (attribute, value)
);

/* QF rqf categorical */
CREATE TABLE qf_rqf_cat (
    attribute VARCHAR(255),
    value REAL,
    frequency INTEGER,
    qf REAL,
    PRIMARY KEY (attribute, value)
);

/* QF jaccard categorical */
CREATE TABLE qf_jac_cat (
    attribute VARCHAR(255),
    value REAL,
    query_id INTEGER,
    PRIMARY KEY (attribute, value, query_id)
);