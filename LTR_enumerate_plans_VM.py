
from ltr_db_optimizer.enumeration_algorithm.table_info import TPCHTableInformation
from ltr_db_optimizer.enumeration_algorithm.table_info_5 import TPCHTableInformation5

from ltr_db_optimizer.enumeration_algorithm.table_info_imdb import IMDBTableInformation

from ltr_db_optimizer.enumeration_algorithm.table_info_stats import STATSTableInformation

from pyodbc import ProgrammingError, OperationalError
import os
import pandas as pd
from datetime import datetime
import time
from ltr_db_optimizer.parser import SQLParser
from argparse import ArgumentParser

from ltr_db_optimizer.extra_utils import set_DBMS




if __name__ == "__main__":
    parser = ArgumentParser()

    # random.seed(248)
    # np.random.seed(248)
    # torch.manual_seed(248)

    parser.add_argument("--emd", type=str, default='HM')

    parser.add_argument("--db", type=str, default="tpch")

    parser.add_argument('--tq', type=str, default="tpch-o")

    parser.add_argument('--topk', type=int, default=20)

    parser.add_argument('--do_enum', type=bool, default=True)

    parser.add_argument('--do_runtime', type=bool, default=False)

    parser.add_argument('--iter', type=str, default='None')

    parser.add_argument("--mn", type=str, default='special_50_97_MODEL_LTRankModel1_lambdaloss_sigma_0.5_k_5_mu_1.0_NDCG_Loss2++_presort_False')


    print('start time: ', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    Start_time = datetime.now()

    args = parser.parse_args()

    targeted_enum_method = args.emd

    targeted_database = args.db

    targeted_test_query = args.tq

    targeted_model_name = args.mn

    iter = args.iter


    print('Argumments: ', args)

    cursor = set_DBMS(db=targeted_database)


    ### dataset info.
    if targeted_database == "tpch":
        TableInformation = TPCHTableInformation()
    elif targeted_database == "imdb":
        TableInformation = IMDBTableInformation()
    elif targeted_database == "tpch5":
        TableInformation = TPCHTableInformation5()
    elif targeted_database == "stats":
        TableInformation = STATSTableInformation()


    ### enum method info.


    if targeted_enum_method == "HM":
        from ltr_db_optimizer.enumeration_algorithm.enumeration_algorithm import EnumerationAlgorithm
        print("from ltr_db_optimizer.enumeration_algorithm.enumeration_algorithm import EnumerationAlgorithm")



    ### trained ranking model to rank plans in enumeration process
    # model = f"/home/xliq/Documents/LTR_DP/ltr_db_optimizer/model/saved_models/{targeted_model_name}/best_avg_ndcg.pth"

    # model = f"/home/xliq/Documents/LTR_DP/ltr_db_optimizer/model/saved_models/{targeted_model_name}/min_avg_valid_loss.pth"

    model = None



    ### test queries info.
    if targeted_test_query == "tpch-o":
        query_folder_name = "tpch_queries"
        query_path = f"/home/xliq/Documents/LTR_DP/Data/testing_data/{query_folder_name}/"
        testquery = ['1', '3', '5', '6', '10', '12', '14'] #'7', '8', '9' has subquery


    elif targeted_test_query in ["tpch-d", "tpch-l"]:
        df = pd.read_csv('/home/xliq/Documents/LTR_DP/Data/testing_data/HM_TPCH_test_results.csv', header=0)  ### 136 TPCH queries
        df = df[df['Time'] != 0]  # filter those bad queries
        testquery = df['Job'].values.tolist()
        query_folder_name = "HM_TPCH_test_queries"  # tpch_query
        query_path = f'/home/xliq/Documents/LTR_DP/Data/testing_data/{query_folder_name}/'


    elif targeted_test_query == "tpch-s": #tpch short queries
        df = pd.read_csv('/home/xliq/Documents/LTR_DP/results/dataset_tpch_workload_tpch-s_enum_DB_iter_tpchdThre50sNoBitMapCurrentCompaLevelCardEstimateCL140_runtime.csv', header=0)  ### 136 TPCH queries
        df = df[(df['Time'] != 0)]  # filter those bad queries
        print('total number of tpch datafarm queries: ', len(df))
        df = df[(df['Time'] < 50000)]
        print('the number of tpch datafarm queries less than 50 seconds: ', len(df))
        testquery = df['Job'].values.tolist()
        query_folder_name = "HM_TPCH_test_queries"  # tpch_query
        query_path = f'/home/xliq/Documents/LTR_DP/Data/testing_data/{query_folder_name}/'


    elif targeted_test_query == "imdb-o":
        query_folder_name = "imdb_queries"
        query_path = f"/home/xliq/Documents/LTR_DP/Data/testing_data/{query_folder_name}/"
        query_file_names = os.listdir(query_path)
        testquery = []
        for imdb_query in query_file_names:
            if imdb_query.endswith(".txt"):
                testquery.append(imdb_query.split(".")[0])

    elif targeted_test_query == "job-o":
        query_folder_name = "job"
        query_path = f"/home/xliq/Documents/LTR_DP/Data/testing_data/{query_folder_name}/"
        query_file_names = os.listdir(query_path)
        testquery = []
        for imdb_query in query_file_names:
            if imdb_query.endswith(".sql"):
                testquery.append(imdb_query.split(".")[0])

        testquery = sorted(testquery, reverse=False)

    elif targeted_test_query == "stats-o":
        query_folder_name = "stats_queries"
        query_path = f"/home/xliq/Documents/LTR_DP/Data/testing_data/{query_folder_name}/"
        query_file_names = os.listdir(query_path)
        testquery = []
        for stats_query in query_file_names:
            if stats_query.endswith(".txt") and stats_query.startswith("qt"):
                testquery.append(stats_query.split(".")[0])

    elif targeted_test_query == "stats-r":
        query_folder_name = "stats_test2"
        query_path = f"/home/xliq/Documents/LTR_DP/Data/testing_data/{query_folder_name}/"
        query_file_names = os.listdir(query_path)
        testquery = []
        for stats_query in query_file_names:
            if stats_query.endswith(".txt") and (stats_query.startswith("qt") or stats_query.startswith("tq")):
                testquery.append(stats_query.split(".")[0])

    elif targeted_test_query == "imdb-r":
        query_folder_name = "imdb_test2"
        query_path = f"/home/xliq/Documents/LTR_DP/Data/testing_data/{query_folder_name}/"
        query_file_names = os.listdir(query_path)
        testquery = []
        for imdb_query in query_file_names:
            if imdb_query.endswith(".sql"):
                testquery.append(imdb_query.split(".")[0])

        testquery = sorted(testquery, reverse=False)

    elif targeted_test_query == "job-t3":
        query_folder_name = "imdb_test3"
        query_path = f"/home/xliq/Documents/LTR_DP/Data/testing_data/{query_folder_name}/"
        query_file_names = os.listdir(query_path)
        testquery = []
        for imdb_query in query_file_names:
            if imdb_query.endswith(".sql"):
                testquery.append(imdb_query.split(".")[0])

        testquery = sorted(testquery, reverse=False)

    k = args.topk

    enum_plan_time = []

    if args.do_enum:
        for job_index, job in enumerate(testquery):

            if targeted_test_query == "tpch-d" and job in ["Job17v3", "Job1131v3"]:
                continue

            if targeted_test_query == "tpch-l" and job not in ['Job49v1', 'Job44v3', 'Job25v1', 'Job253v4', 'Job253v0',
                                                               'Job185v3', 'Job185v2', 'Job16v0', 'Job161v0', 'Job1325v3',
                                                               'Job1300v2', 'Job128v4', 'Job128v3', 'Job128v0', 'Job1252v1',
                                                               'Job1103v3', 'Job1034v4', 'Job1034v0', 'Job1033v4', 'Job1028v2',
                                                               'Job1021v1', 'Job1017v3', 'Job1017v1', 'Job1011v2']:


                continue

            if targeted_test_query == "tpch-s" and job in ["Job1300v2"]:
                continue


            print('enumerate job: ', job_index, job)

            ###load query
            if os.path.exists(f"{query_path}/{job}.txt"):
                with open(f"{query_path}/{job}.txt", "r") as f:
                    sql_full = f.read()
            else:
                with open(f"{query_path}/{job}.sql", "r") as f:
                    sql_full = f.read()

            sql_full = sql_full.replace("tcph", "tpch")


            path_out = f"/home/xliq/Documents/LTR_DP/results/enumerated_plans_{targeted_enum_method}_{targeted_model_name}/{query_folder_name}/iter{iter}/{job}/"

            if targeted_test_query in ["job-o", "stats-o"] and os.path.exists(path_out):
                print(f'{job} in {targeted_test_query} alreay enumerated and thus ignored!')
                continue


            ### parse query
            sql_dict, alias_dict = SQLParser.from_sql(sql_full, temp_table_info=TableInformation)
            # print('sql_dict', sql_dict)
            # print('alias_dict', alias_dict)



            ###enumeration
            enum = EnumerationAlgorithm(sql_dict,
                                        TableInformation,
                                        model,
                                        sql_full,
                                        k,
                                        alias_dict=alias_dict, train_wk=[])

            start_time = datetime.now()

            best_plans = enum.find_best_plan()

            enum_end_time = datetime.now()
            enum_delta_time = (enum_end_time - start_time).total_seconds()



            print(f"{job}'s planning time:", enum_delta_time) #planning time: 0.11666666666666667

            print('the number of returned best plans', len(best_plans))

            # path_out = f"/home/xliq/Documents/LTR_DP/results/enumerated_plans_{targeted_enum_method}_{targeted_model_name}/{query_folder_name}/iter{iter}/{job}/"

            # os.mkdir(path_out)
            os.system(f"mkdir -p {path_out}")
            # pathlib.Path(path_out).mkdir(parents=True, exist_ok=True)

            for idx, plan in enumerate(best_plans):
                xml = enum.to_xml(plan)

                with open(path_out + str(idx) + ".txt", "w") as f:
                    f.write(xml)
                # with open(path_out + str(idx) + ".pickle", "wb") as f:
                #     pickle.dump(plan, f)


    if args.do_runtime:
        ### get runtime of plans
        print('Enumeration done! Continue to obtain runtime of enumerated plans !!!')
        print('Current time: ', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        all_runtime_list = []
        for job_index, job in enumerate(testquery):


            if targeted_test_query == "tpch-d" and job in ["Job17v3", "Job1131v3", "Job128v3"]:
                continue

            if targeted_test_query == "tpch-l" and job not in ['Job49v1', 'Job44v3', 'Job25v1', 'Job253v4', 'Job253v0',
                                                               'Job185v3', 'Job185v2', 'Job16v0', 'Job161v0',
                                                               'Job1325v3',
                                                               'Job1300v2', 'Job128v4', 'Job128v3', 'Job128v0',
                                                               'Job1252v1',
                                                               'Job1103v3', 'Job1034v4', 'Job1034v0', 'Job1033v4',
                                                               'Job1028v2',
                                                               'Job1021v1', 'Job1017v3', 'Job1017v1', 'Job1011v2']:
                continue


            print('run job: ', job_index, job)
            print('Job Start time: ', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            methods_runtime_list = []

            if targeted_enum_method == "DB":
                plan_path = f"/home/xliq/Documents/LTR_DP/results/enumerated_plans_{targeted_enum_method}/{query_folder_name}/iter{iter}/{job}/"
            else:
                plan_folder_postfix = "_".join([targeted_enum_method, targeted_model_name])
                plan_path = f"/home/xliq/Documents/LTR_DP/results/enumerated_plans_{plan_folder_postfix}/{query_folder_name}/iter{iter}/{job}/"


            with open(plan_path + str(0) + ".txt", "r") as f:
                plan_sql = f.read()
            try:
                cursor.execute("SET STATISTICS TIME ON")
                cursor.execute(plan_sql)
                while (cursor.nextset()):
                    mess = cursor.messages
                    # print('results:', mess)
                    if len(mess[0][1].split(",")) == 3:
                        cpu = int(mess[0][1].split(",")[1].split("=")[1][:-3])
                        cpu_unit = mess[0][1].split(",")[1].split("=")[1][-3:]
                        time = int(mess[0][1].split(",")[2].split("=")[1][:-3])
                        time_unit = mess[0][1].split(",")[2].split("=")[1][-3:]
                    elif len(mess[0][1].split(",")) == 2:
                        cpu = int(mess[0][1].split(",")[0].split("=")[1][:-3])
                        cpu_unit = mess[0][1].split(",")[0].split("=")[1][-3:]
                        time = int(mess[0][1].split(",")[1].split("=")[1][:-3])
                        time_unit = mess[0][1].split(",")[1].split("=")[1][-3:]
                    else:
                        print('mess', mess)
                        continue
                result = [job, cpu, time, cpu + time, cpu_unit, time_unit]

            except OperationalError as err:
                print("Timeout")
                print(job, targeted_enum_method, targeted_model_name, err)
                result = [job, -1, -1, -1, 'ms', 'ms']
                print("----")

            except ProgrammingError as err:
                print('WeridError')
                print(job, targeted_enum_method, targeted_model_name, err)
                if targeted_enum_method != "DB":
                    result = [job, -2, -2, -2, 'ms', 'ms']
                    print("----")
                else:
                    pure_sql = plan_sql.split("OPTION")[0] + " OPTION (MAXDOP 1) "
                    ### rerun the pure sql instead of plan
                    cursor.execute("SET STATISTICS TIME ON")
                    cursor.execute(pure_sql)
                    while (cursor.nextset()):
                        mess = cursor.messages
                        # print('results:', mess)
                        if len(mess[0][1].split(",")) == 3:
                            cpu = int(mess[0][1].split(",")[1].split("=")[1][:-3])
                            cpu_unit = mess[0][1].split(",")[1].split("=")[1][-3:]
                            time = int(mess[0][1].split(",")[2].split("=")[1][:-3])
                            time_unit = mess[0][1].split(",")[2].split("=")[1][-3:]
                        elif len(mess[0][1].split(",")) == 2:
                            cpu = int(mess[0][1].split(",")[0].split("=")[1][:-3])
                            cpu_unit = mess[0][1].split(",")[0].split("=")[1][-3:]
                            time = int(mess[0][1].split(",")[1].split("=")[1][:-3])
                            time_unit = mess[0][1].split(",")[1].split("=")[1][-3:]
                        else:
                            print('mess', mess)
                            continue
                    result = [job, cpu, time, cpu + time, cpu_unit, time_unit]
            except Exception as exce:
                print('Exception:', Exception)
                continue

            print('statistic:', result)
            print('Job End time: ', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            methods_runtime_list.append(result)

            df = pd.DataFrame(methods_runtime_list, index=[targeted_enum_method])
            print('df', df)
            df.columns = "Job,CPU time,Time,Sum,CPU unit,Time unit".split(',')
            all_runtime_list.append(df)

            all_runtime_df = pd.concat(all_runtime_list, axis=0)
            if targeted_enum_method == "DB":
                all_df_path = f"/home/xliq/Documents/LTR_DP/results/dataset_{targeted_database}_workload_{targeted_test_query}_enum_{targeted_enum_method}_iter_{iter}_runtime.csv"
            else:
                all_df_path = f"/home/xliq/Documents/LTR_DP/results/dataset_{targeted_database}_workload_{targeted_test_query}_enum_{targeted_enum_method}_rank_{targeted_model_name}_iter_{iter}_runtime.csv"
                # all_df_path = f"/home/xliq/Documents/LTR_DP/results/dataset_{targeted_database}_workload_{targeted_test_query}_enum_{targeted_enum_method}_rank_{targeted_model_name}_runtime.csv"

            all_runtime_df.to_csv(all_df_path, index=True, header=True)

        ### describe the runtime distributions of test queries
        runtime_desc_list = []
        for method in [targeted_enum_method]:
            method_runtime_df = all_runtime_df[all_runtime_df.index == method]
            method_runtime_desc_df = method_runtime_df['Time'].describe()
            runtime_desc_list.append(method_runtime_desc_df)

        runtime_desc_df = pd.concat(runtime_desc_list, axis=1)
        runtime_desc_df.columns = [targeted_enum_method]
        runtime_desc_df = runtime_desc_df.round(1)

        if targeted_enum_method == "DB":
            runtime_desc_file_path = f"/home/xliq/Documents/LTR_DP/results/dataset_{targeted_database}_workload_{targeted_test_query}_enum_{targeted_enum_method}_iter_{iter}_runtime_descr.csv"
        else:
            runtime_desc_file_path = f"/home/xliq/Documents/LTR_DP/results/dataset_{targeted_database}_workload_{targeted_test_query}_enum_{targeted_enum_method}_rank_{targeted_model_name}_iter_{iter}_runtime_descr.csv"
            # runtime_desc_file_path = f"/home/xliq/Documents/LTR_DP/results/dataset_{targeted_database}_workload_{targeted_test_query}_enum_{targeted_enum_method}_rank_{targeted_model_name}_runtime_descr.csv"

        runtime_desc_df.to_csv(runtime_desc_file_path, index=True, header=True)


    print('end time: ', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    End_time = datetime.now()

    Elapsed_time = round((End_time - Start_time).total_seconds()/60, 2)
    print('Total Elapsed Time: ', Elapsed_time)











