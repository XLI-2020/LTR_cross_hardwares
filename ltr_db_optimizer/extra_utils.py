import random
from ltr_db_optimizer.parser import SQLParser
from ltr_db_optimizer.enumeration_algorithm.table_info import TPCHTableInformation
from ltr_db_optimizer.enumeration_algorithm.table_info_imdb import IMDBTableInformation

from ltr_db_optimizer.enumeration_algorithm.table_info_stats import STATSTableInformation


import pandas as pd
from ltr_db_optimizer.model.metrics import ndcg
import copy
import pyodbc
import os
import pandas as pd
from datetime import datetime


def get_the_split_of_training_data(plan_keys, valid_ratio, test_ratio, workload):
    '''
    group training data by the distribution of runtime or the number of joins
    '''
    seed = 139
    random.seed(seed)

    key_df = pd.DataFrame(plan_keys, columns=['key'])

    print('key df head: ', key_df.head(5))

    key_df["Job_nr"] = list(map(lambda x: x.split("_")[0], key_df['key'].values.tolist()))  # Job443v1_be_sp0
    key_df["Subquery"] = list(map(lambda x: '_'.join(x.split("_")[:-1]), key_df['key'].values.tolist()))

    print('key_df, before drop duplicates: ', len(key_df), key_df.head(5))
    key_df = key_df.drop_duplicates(keep='first', subset=['Subquery'])
    print('the nr of subquery after drop duplicates: ', len(key_df))
    print('key_df, after drop duplicates: ', key_df.head(5))

    subquery_list = key_df['key'].values.tolist()

    subquery_joins = {}
    nr_joins_list = []

    for subquery in subquery_list:
        job_nr, subquery_nr, plan_nr = subquery.split("_")
        if "tpch" in workload:
            with open(f"/home/xliq/Documents/LTR_DP/Data/subplans_{workload}/{job_nr}/{subquery_nr}/{plan_nr}.txt", "r") as f:
                plan_xml = f.read()
                sql_full = plan_xml.split("OPTION")[0]

            sql_full = sql_full.replace("tcph", "tpch")
            sql_dict, alias_dict = SQLParser.from_sql(sql_full, temp_table_info=TPCHTableInformation())

        elif ("job" in workload) or ('imdb' in workload):
            with open(f"/home/xliq/Documents/LTR_DP/Data/subplans_{workload}/{job_nr}/{subquery_nr}/{plan_nr}.txt", "r") as f:
                plan_xml = f.read()
                sql_full = plan_xml.split("OPTION")[0]
            # print('sql full:', sql_full)
            ### parse query
            sql_dict, alias_dict = SQLParser.from_sql(sql_full, temp_table_info=IMDBTableInformation())
            # print('how many joins: ', len(sql_dict["Joins"]))
        elif "stats" in workload:
            with open(f"/home/xliq/Documents/LTR_DP/Data/subplans_{workload}/{job_nr}/{subquery_nr}/{plan_nr}.txt",
                      "r") as f:
                plan_xml = f.read()
                sql_full = plan_xml.split("OPTION")[0]
            # print('sql full:', sql_full)
            ### parse query
            sql_dict, alias_dict = SQLParser.from_sql(sql_full, temp_table_info=STATSTableInformation())

        sq_name = "_".join([job_nr, subquery_nr])
        subquery_joins[sq_name] = [len(sql_dict["Joins"])]

        nr_joins_list.append(len(sql_dict["Joins"]))

    job_joins_df = pd.DataFrame(subquery_joins).T
    job_joins_df.columns = ['nr_joins']
    unique_nr_joins_list = list(set(nr_joins_list))

    split_valid = []
    split_test = []
    print('unique_nr_joins_list:', unique_nr_joins_list)
    for nr_jo in unique_nr_joins_list:
        temp_df = job_joins_df[job_joins_df['nr_joins'] == nr_jo]
        print('number of joins and number of jobs having that number of joins: ', nr_jo, len(temp_df))
        valid_len = int(len(temp_df)*valid_ratio)
        test_len = int(len(temp_df)*test_ratio)
        valid_test_portion = random.sample(list(temp_df.index), valid_len + test_len)
        valid_portion = random.sample(valid_test_portion, valid_len)
        test_portion = [job for job in valid_test_portion if job not in valid_portion]

        split_valid.extend(valid_portion)
        split_test.extend(test_portion)

    random.shuffle(split_valid)
    random.shuffle(split_test)
    print('the number of jobs for validation: ', len(split_valid))
    print('the number of jobs for test: ', len(split_test))
    print('the number of jobs for training: ', len([job for job in key_df["Subquery"].values.tolist() if job not in (split_valid+split_test)]))
    return split_valid, split_test


def get_the_split_of_jobs_list(X_train_vecs, train_ratio, workload):
    '''
    group training data by the distribution of runtime or the number of joins
    '''
    # seed = 139
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)

    subquery_list = X_train_vecs.keys()

    print('the total number of jobs before split: ', len(subquery_list))
    subquery_joins = {}
    nr_joins_list = []
    print('workload in the training data split:', workload)
    for subquery in subquery_list:
        job_nr, subquery_nr = subquery.split("_")
        plan_nr = list(X_train_vecs[subquery].keys())[0].split("_")[2]

        if "tpch" in workload:
            with open(f"/home/xliq/Documents/LTR_DP/Data/subplans_{workload}/{job_nr}/{subquery_nr}/{plan_nr}.txt", "r") as f:
                plan_xml = f.read()
            sql_full = plan_xml.split("OPTION")[0]
            sql_full = sql_full.replace("tcph", "tpch")
            ### parse query
            sql_dict, alias_dict = SQLParser.from_sql(sql_full, temp_table_info=TPCHTableInformation())
            # print('how many joins: ', len(sql_dict["Joins"]))
            subquery_joins[subquery] = [len(sql_dict["Joins"])]
            nr_joins_list.append(len(sql_dict["Joins"]))

        elif "job" in workload or 'imdb' in workload:
            with open(f"/home/xliq/Documents/LTR_DP/Data/subplans_{workload}/{job_nr}/{subquery_nr}/{plan_nr}.txt", "r") as f:
                plan_xml = f.read()
            sql_full = plan_xml.split("OPTION")[0]
            ### parse query
            sql_dict, alias_dict = SQLParser.from_sql(sql_full, temp_table_info=IMDBTableInformation())
            # print('how many joins: ', len(sql_dict["Joins"]))
            subquery_joins[subquery] = [len(sql_dict["Joins"])]
            nr_joins_list.append(len(sql_dict["Joins"]))

        elif "stats" in workload:
            with open(f"/home/xliq/Documents/LTR_DP/Data/subplans_{workload}/{job_nr}/{subquery_nr}/{plan_nr}.txt", "r") as f:
                plan_xml = f.read()
            sql_full = plan_xml.split("OPTION")[0]
            ### parse query
            sql_dict, alias_dict = SQLParser.from_sql(sql_full, temp_table_info=STATSTableInformation())
            # print('how many joins: ', len(sql_dict["Joins"]))
            subquery_joins[subquery] = [len(sql_dict["Joins"])]
            nr_joins_list.append(len(sql_dict["Joins"]))

    job_joins_df = pd.DataFrame(subquery_joins).T
    job_joins_df.columns = ['nr_joins']
    validation_ratio = train_ratio
    split = []

    unique_nr_joins_list = list(set(nr_joins_list))
    print('unique nr joins list:', list(unique_nr_joins_list))


    for nr_jo in unique_nr_joins_list:
        temp_df = job_joins_df[job_joins_df['nr_joins'] == nr_jo]
        print('number of joins and number of jobs having that number of joins: ', nr_jo, len(temp_df))
        portion = random.sample(list(temp_df.index), int(len(temp_df)*validation_ratio))
        split.extend(portion)
    print('the number of jobs for split: ', len(split))

    random.shuffle(split)

    return split

def ndcg_wrap(y_pred, y_true, ats=None):
    return ndcg(y_pred, y_true, ats=ats).numpy()

def add_force_scan_to_sql(sql):
    # input_sql = sql.upper()
    print('initial sql: ', sql)
    input_sql = copy.copy(sql)
    before_from, after_from = input_sql.split('FROM')
    if "WHERE" in after_from:
        before_where, after_where = after_from.split('WHERE')
        tables = before_where.split(',')

        tables_added_hint = ','.join(list(map(lambda x: x + " WITH(FORCESCAN)", tables)))

        # print(tables_added_hint)

        sql_with_hint = before_from + " FROM" + tables_added_hint + " WHERE" + after_where
        # print('input sql', input_sql)
    else:
        if "GROUP BY" in after_from:
            before_group_by, after_group_by = after_from.split('GROUP BY')
            tables = before_group_by.split(',')
            tables_added_hint = ','.join(list(map(lambda x: x + " WITH(FORCESCAN)", tables)))
            sql_with_hint = before_from + " FROM" + tables_added_hint + " GROUP BY" + after_group_by
        elif "ORDER BY" in after_from:
            before_order_by, after_order_by = after_from.split('ORDER BY')
            tables = before_order_by.split(',')
            tables_added_hint = ','.join(list(map(lambda x: x + " WITH(FORCESCAN)", tables)))
            sql_with_hint = before_from + " FROM" + tables_added_hint + " ORDER BY" + after_order_by
        else:
            tables = after_from.split(',')
            tables_added_hint = ','.join(list(map(lambda x: x + " WITH(FORCESCAN)", tables)))
            sql_with_hint = before_from + " FROM" + tables_added_hint
        # print('input sql: ', input_sql)
        # print('sql_with_hint: ', sql_with_hint)

    # sql_with_hint = sql_with_hint + " OPTION (MAXDOP 1, USE HINT ('QUERY_OPTIMIZER_COMPATIBILITY_LEVEL_130', 'FORCE_LEGACY_CARDINALITY_ESTIMATION')) "

    sql_with_hint = sql_with_hint + " OPTION (MAXDOP 1, USE HINT ('QUERY_OPTIMIZER_COMPATIBILITY_LEVEL_140', 'FORCE_DEFAULT_CARDINALITY_ESTIMATION')) "

    print("####")
    print('sql_with_hint: ', sql_with_hint)

    return sql_with_hint

def set_DBMS(db):
    SERVER = 'localhost,1433'
    DATABASE = db
    USERNAME = 'sa'
    PASSWORD = 'LX##1992'

    conn = pyodbc.connect(driver='{ODBC Driver 18 for SQL Server}',
                          server=SERVER,
                          database=DATABASE,
                          UID=USERNAME,
                          PWD=PASSWORD, TrustServerCertificate='Yes')


    conn.timeout = 21600  # 180
    cursor = conn.cursor()
    return cursor

def enumerate_by_SQL_Server(job, sql_full, folder_name, cursor, iter):

    cursor.execute("SET SHOWPLAN_XML ON")

    start_time = datetime.now()

    sql_full = add_force_scan_to_sql(sql_full)

    rows = cursor.execute(sql_full).fetchall()

    enum_end_time = datetime.now()
    enum_delta_time = (enum_end_time - start_time).total_seconds()
    # enum_plan_time.append(enum_delta_time)

    print('planning time:', enum_delta_time)

    # print('rows[0]', rows[0])
    # print('rows[0][0]', rows[0][0])
    path_out = f"/home/xliq/Documents/LTR_DP/results/enumerated_plans_DB/{folder_name}/iter{iter}/{job}/"

    os.system(f"mkdir -p {path_out}")


    # sql_full_without_addtional_string = sql_full.replace("OPTION (MAXDOP 1, USE HINT ('QUERY_OPTIMIZER_COMPATIBILITY_LEVEL_130', 'FORCE_LEGACY_CARDINALITY_ESTIMATION'))", "")

    sql_full_without_addtional_string = sql_full.replace("OPTION (MAXDOP 1, USE HINT ('QUERY_OPTIMIZER_COMPATIBILITY_LEVEL_140', 'FORCE_DEFAULT_CARDINALITY_ESTIMATION'))", "")


    plan_sql = sql_full_without_addtional_string + " OPTION (RECOMPILE, USE PLAN N'" + rows[0][0] + "')"

    with open(path_out + str(0) + ".txt", "w") as f:
        f.write(plan_sql)

    cursor.execute("SET SHOWPLAN_XML OFF")


def preprocess_job_queries(sql_full):
    print('original sql: ', sql_full)

    if sql_full.strip().endswith(";"):
        sql_full = sql_full.strip()
        sql_full = sql_full[:-1]

    print('sql to enum: ', sql_full)

    # with open(f"{query_path}/{job}.sql", "w+") as f:
    #     f.write(sql_full)

