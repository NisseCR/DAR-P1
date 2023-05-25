/* Reset data */
DROP TABLE IF EXISTS idf_num;
DROP TABLE IF EXISTS idf_cat;
DROP TABLE IF EXISTS qf_rqf_cat;
DROP TABLE IF EXISTS qf_jac_cat;
DROP TABLE IF EXISTS qf_rqf_num;

/* IDF numerical */
CREATE TABLE idf_num (
    attribute VARCHAR(255),
    value REAL,
    tf REAL,
    idf REAL,
    PRIMARY KEY (attribute, value)
);

/* IDF categorical */
CREATE TABLE idf_cat (
    attribute VARCHAR(255),
    value VARCHAR(255),
    tf INTEGER,
    idf REAL,
    PRIMARY KEY (attribute, value)
);

/* QF rqf categorical */
CREATE TABLE qf_rqf_cat (
    attribute VARCHAR(255),
    value VARCHAR(255),
    tf INTEGER,
    qf REAL,
    PRIMARY KEY (attribute, value)
);

/* IDF numerical */
CREATE TABLE qf_rqf_num (
    attribute VARCHAR(255),
    value REAL,
    tf REAL,
    qf REAL,
    PRIMARY KEY (attribute, value)
);

/* QF jaccard categorical */
CREATE TABLE qf_jac_cat (
    attribute VARCHAR(255),
    value_x VARCHAR(255),
    value_y VARCHAR(255),
    qf INTEGER,
    PRIMARY KEY (attribute, value_x, value_y)
);