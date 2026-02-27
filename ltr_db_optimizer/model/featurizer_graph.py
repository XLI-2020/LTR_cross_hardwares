import copy
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from ltr_db_optimizer.enumeration_algorithm.table_info_imdb import IMDBTableInformation
from ltr_db_optimizer.enumeration_algorithm.table_info_stats import STATSTableInformation
from ltr_db_optimizer.parser import SQLParser
import ltr_db_optimizer.enumeration_algorithm.enumeration_node as nodes

from ltr_db_optimizer.enumeration_algorithm.table_info import TPCHTableInformation

class FeatureExtractorGraph:
    # We used the following SQL parts:
    # Sort
    # Join: Merge Join, Nested Loop Join, Hash Join
    # Aggregate: Stream Aggregate, Hash Aggregate
    # Scan: Index Scan, Table Scan

    type_vector = ["sort", "stream_aggregate", "hash_aggregate", "merge_join", "nested_loop_join", "hash_join", "index_scan", "table_scan", "null"]

    # type_vector = ["sort", "stream_aggregate", "hash_aggregate", "merge_join", "nested_loop_join", "hash_join", "table_scan", "null"]

    
    def __init__(self, with_cost = True, with_issort = False, workload = 'tpch', tree_high=None, tree_low=None):
        self.vector_length = len(self.type_vector)
        self.with_cost = with_cost
        if with_cost:
            self.vector_length += 1#2 # for estimated subtree cost and estimated rows
        self.with_issort = with_issort
        if with_issort:
            self.vector_length += 1
        self.rows = {}
        self.workload = workload
        if 'tpch' in workload or "Db" in workload:
            print('use tpch workload')
            # self.min_vec = np.array([0.0, 0.0, 0.0, 1.0, 5.0, 5.0])
            # self.max_vec = np.array([1.0, 1.0, 7.0, 53435600, 6.001215e+06, 6.001215e+06])
            self.tree_high =  tree_high #53435600 #28796000000000.0 #53435600.0 #4.801740e+08
            self.tree_low = tree_low
            # self.tree_mean = 243752575
            # self.tree_std = 7325942834

        elif ('job' in workload) or ('imdb' in workload):
            print('use job workload')
            # self.min_vec = np.array([0.0, 0.0, 0.0, 1.0, 4.0, 4.0])
            # self.max_vec = np.array([1.0, 1.0, 23.0, 53435600, 36244343, 36244343])
            self.tree_high = tree_high #53435600 # 53435600.0 # 8.6e+09
            self.tree_low = tree_low

        elif 'stats' in workload:
            print('use stats workload')
            # self.min_vec = np.array([0.0, 0.0, 1.0, 1.0, 1032, 1032])
            # self.max_vec = np.array([1.0, 1.0, 5.0, 18349800.0 , 328064, 328064])
            self.tree_high = tree_high # 691628.0 98P # 328064.0 95P # 52426.7
            self.tree_low = tree_low #1

        self.parts_cost_list = []
        self.max_tree_high = 0
        self.min_tree_high = np.inf

        self.tree_len_list = []

    def featurize_node(self, node):
        none_child = np.zeros(self.vector_length)
        none_child[8] = 1
        # print('curr plan node name & id: ', node.name, node.id, node.estimated_rows)
        if node.has_featurized_plan():
            return node.get_featurized_plan(), node.estimated_rows
        
        if node.name in self.type_vector:
            this_vector = np.zeros(self.vector_length)
            this_vector[self.type_vector.index(node.name)] = 1
        else:
            # If it is not in type vector, there should also be only one child
            # print('Not considered operator types, ignored!')
            return self.featurize_node(node.get_left_child())

        if self.with_cost:
            # insert cost here.
            this_vector[-1] = node.estimated_rows # changed -2
            rows = node.estimated_rows
            # print('curr plan node vec:, this_vector', this_vector)

        # if self.with_issort:
        #     this_vector[-2] = node.is_sorted

        if self.with_issort and node.has_right_child():
            left_child = node.get_left_child()
            right_child = node.get_right_child()
            if self.with_issort:
                left_column, right_column = node.left_column, node.right_column
                # print('to be joined: left, right columns:', left_column, right_column)

                if isinstance(left_child, nodes.ScanNode) and left_child.name == "index_scan":
                    # print('left_child.sorted_columns: ', left_child.sorted_columns)
                    if left_child.sorted_columns.split('.')[-1] == left_column:
                        this_vector[-2] = 1

                if isinstance(right_child, nodes.ScanNode) and right_child.name == "index_scan":
                    # print('right_child.sorted_columns: ', right_child.sorted_columns)
                    if right_child.sorted_columns.split('.')[-1] == right_column:
                        this_vector[-2] = 1

        #     # print('curr plan node vec after sort encoding:, this_vector', this_vector)


        if node.has_right_child():
            # print('## two childs!')
            # print('left child name, id, rows', node.get_left_child().name, node.get_left_child().id, node.get_left_child().estimated_rows)
            left_child, _ = self.featurize_node(node.get_left_child())
            # print('right child name, id rows', node.get_right_child().name, node.get_right_child().id, node.get_right_child().estimated_rows)
            right_child, _ = self.featurize_node(node.get_right_child())

        elif node.has_left_child():
            # print('## left child only!')
            # print('left child name, id', node.get_left_child().name, node.get_left_child().id, node.get_left_child().estimated_rows)
            left_child,_ = self.featurize_node(node.get_left_child())
            right_child = (none_child)

            # if self.with_issort:
            #     this_vector[-2] = float(left_child.is_sorted)

        else:
            # print('## no childs!')
            # print('leaf node: ', this_vector)
            node.set_featurized_plan((this_vector))
            return (this_vector), node.estimated_rows

        node.set_featurized_plan((this_vector, left_child, right_child))
        return (this_vector, left_child, right_child), node.estimated_rows
    
    def match_cost_plan(self, execution_plan, cost_plan):
        cost_parts = cost_plan.split("<")
        parts_cost = []
        # print('cost_parts: ', cost_parts[:10])
        
        for part_num, part in enumerate(cost_parts):
            # print('part: ', part)
            if part.startswith("RelOp"):
                sub_parts = part.split('"')
                sub_parts_cost = []
                for idx, sub in enumerate(sub_parts):
                    if "PhysicalOp" in sub:
                        # print('###PhysicalOp: ', sub_parts[idx+1])
                        sub_parts_cost.append(sub_parts[idx+1])
                    if "EstimateRows" in sub:
                        # print('###EstimateRows: ', sub_parts[idx+1])
                        sub_parts_cost.append(float(sub_parts[idx+1]))
                parts_cost.append(tuple(sub_parts_cost))

        # print('parts_cost',parts_cost)
        # self.parts_cost_list.extend(parts_cost)

        extended_plan, _ =  self.append_features(execution_plan, parts_cost)

        return extended_plan


    def append_features(self, execution_plan, label_parts):
            # print('self.rows dict:', self.rows)
        # print('current execution_plan name & id: ', execution_plan.name, execution_plan.id)
        if not(execution_plan.name == "top" and execution_plan.get_left_child().name == "sort"):
            if execution_plan.name == "compute_scalar" and not label_parts[0][0] == "Compute Scalar":
                # print('compute_scalar: ', label_parts[0])
                curr_part = label_parts[0]
            else:
                curr_part = label_parts.pop(0)
                # print('curr part cardinality: ', curr_part)
                if curr_part[0] == 'Filter':
                    curr_part = label_parts.pop(0)
                    # print('curr part actually used 1: ', curr_part)
                if curr_part[0] == "Compute Scalar" and execution_plan.name != "compute_scalar":
                    curr_part = label_parts.pop(0)
                    # print('curr part actually used 2: ', curr_part)

            if not execution_plan.has_rows():
                # print('enter setting rows!!')
                assert len(curr_part) == 2 or len(curr_part) == 3, "length of curr_part abnormal!!!"
                if len(curr_part) == 2:
                    estimated_rows = curr_part[1]
                elif len(curr_part) == 3:
                    estimated_rows = curr_part[2]

                print('actual estimated_rows: ', estimated_rows)
                rows_calc = (estimated_rows-self.tree_low)/(self.tree_high-self.tree_low)

                # print('execution_plan.name: ', execution_plan.name)
                # print('execution_plan.id: ', execution_plan.id)
                # print('estimated number of rows: ', estimated_rows)
                # print('rows_calc: ', rows_calc)

                if estimated_rows > self.max_tree_high:
                    self.max_tree_high = estimated_rows
                    print('max_tree_high: ', self.max_tree_high)
                if estimated_rows < self.min_tree_high:
                    self.min_tree_high = estimated_rows
                    print('min_tree_high: ', self.min_tree_high)

                self.tree_len_list.append(estimated_rows)

                execution_plan.set_estimated_rows(rows_calc, workload=self.workload)
                self.rows[execution_plan.id] = rows_calc
        if execution_plan.name not in ["index_scan", "table_scan"]:
            # print("current execution_plan's left_child name & id", execution_plan.left_child.name, execution_plan.left_child.id)
            # print("current execution_plan's left_child contained_tables", execution_plan.left_child.contained_tables)

            # if execution_plan.has_right_child():
            #     print("current execution_plan's right_child name & id", execution_plan.right_child.name, execution_plan.right_child.id)
            #     print("current execution_plan's right_child contained_tables", execution_plan.right_child.contained_tables)
            # else:
            #     print('no right child')

            for child in execution_plan.get_children()[::-1]:
                self.append_features(child, label_parts)
        return execution_plan, label_parts



    def append_cost(self, plan):
        if plan.has_rows():
            # print('plan.has_rows()!')
            return False
        if plan.id in self.rows and plan.name != "sort":
            # print('plan.id in self.rows!')
            plan.set_estimated_rows(self.rows[plan.id], workload=self.workload)
            toggle = False
            if plan.has_right_child():
                toggle = toggle or self.append_cost(plan.get_right_child())
            if plan.has_left_child():
                toggle = toggle or self.append_cost(plan.get_left_child())
            return toggle
        elif plan.name == "sort":
            # print('plan.name == "sort"!')
            if plan.left_child.has_rows():
                plan.set_estimated_rows(plan.left_child.estimated_rows, workload=self.workload)
                return False
            else:
                return_val = self.append_cost(plan.get_left_child())
                if return_val:
                    return True
                plan.set_estimated_rows(plan.left_child.estimated_rows, workload=self.workload)
                return False
        return True


def get_sql_parse_info(job):
    with open(f"/home/xliq/Documents/LTR_DP/Data/output_jobs/{job}.txt", "r") as f:
        sql_full = f.read()
    sql_full = sql_full.replace("tcph", "tpch")
    ### parse query
    sql_dict, alias_dict = SQLParser.from_sql(sql_full)
    # print('how many joins: ', len(sql_dict["Joins"]))
    return sql_dict

def recalculate_query_encoding(node):
    # print('start query-encode from current plan node name & id: ', node.name, node.id)
    if node.has_right_child():
        # print('enter query-encode left&right childs name, id', node.get_left_child().name, node.get_left_child().id,
        #       node.get_right_child().name, node.get_right_child().id)
        left_child_qv = recalculate_query_encoding(node.get_left_child())
        right_child_qv = recalculate_query_encoding(node.get_right_child())
    elif node.has_left_child():
        # print('no right child!!')
        # print('enter query-encode  left child name, id', node.get_left_child().name, node.get_left_child().id)
        left_child_qv = recalculate_query_encoding(node.get_left_child())
    else:
        # print('no childs!')
        node.calculate_query_encoding()
        return node

    node.calculate_query_encoding()
    # print('end query-encode from current plan node name & id: ', node.get_query_encoding())
    return node

def obtain_new_query_enc(node, sql_dict):
    execution_plan_cp1 = copy.deepcopy(node)

    new_node = recalculate_query_encoding(execution_plan_cp1)
    # print('new node query enc before normalize: ', np.array(new_node.get_query_encoding()))
    new_node.normalize_query()
    if len(sql_dict["Sort"]):
        new_node.query_encoding[0] = 1
    # print('new node query enc after normalize: ', np.array(new_node.get_query_encoding()))
    # print('####')
    return new_node.get_query_encoding()


def long_query_encode(sql_full, table_info=TPCHTableInformation()):

    if table_info == TPCHTableInformation():
        sql_full = sql_full.replace("tcph", "tpch")

    ### parse query
    sql_dict, alias_dict = SQLParser.from_sql(sql_full, temp_table_info=table_info)
    nr_joins = len(sql_dict["Joins"])

    has_sort = 0
    if len(sql_dict["Sort"]):
        has_sort = 1
    has_groupby = 0
    if sql_dict["Aggregation"] != {}:
        aggregation = sql_dict["Aggregation"]
        if len(aggregation["Group By"]):
            has_groupby = 1

    tables = sql_dict["Tables"]
    max_relation = 0
    min_relation = 1e14

    # for table in tables:
    #     # print('table name: ', table)
    #
    #     table_name = table[0].lower()
    #     alias = None
    #     if table_name in alias_dict["Tables"].keys():
    #         alias = alias_dict["Tables"][table_name]
    #
    #     if alias is not None:
    #         relation = table_info.get_table(alias).row_count
    #     else:
    #         relation = table_info.get_table(table_name).row_count
    #
    #     if relation > max_relation:
    #         max_relation = relation
    #     if relation < min_relation:
    #         min_relation = relation

    # result = [has_sort, has_groupby, nr_joins, max_relation, min_relation]
    result = [has_sort, has_groupby]

    # print('query encode result:', result)

    # min_vec = np.array([0.0, 0.0, 0.0, 5.0, 5.0])
    # max_vec = np.array([1.0, 1.0, 7.0, 6.001215e+06, 6.001215e+06])
    # query_encoding = list((np.array(result) - min_vec) / (max_vec - min_vec))
    return result


def get_features_with_cost_from_folder_Random(plans_folder, cost_folder, workload='tpch', plans_list=None, return_featurized=True, postfix=None):
    tree_high, tree_low = postfix.split('Tree')[1].split('%')
    tree_high = int(tree_high)
    tree_low = int(tree_low)
    print('tree info: ', tree_high, tree_low)

    feature_ext = FeatureExtractorGraph(workload=workload, tree_high=tree_high, tree_low=tree_low)
    featurized_trees = {}
    featurized_vecs = {}
    print('len of cost_folder: ', len(os.listdir(cost_folder)))
    print('len of plans_folder: ', len(os.listdir(plans_folder)))

    # if 'tpch' in workload or "Db" in workload:
    #     table_info = TPCHTableInformation()
    # elif "job" in workload or "imdb" in workload:
    #     table_info = IMDBTableInformation()
    # elif "stats" in workload:
    #     table_info = STATSTableInformation()

    # jobs_list = list(set(list(map(lambda x:x.split("_")[0], plans_list))))
    # query_enc_dict = {}
    # for job in jobs_list:
    #     # get the long query encoding
    #     for i in range(50):
    #         if os.path.exists(f"{cost_folder}/{job}/last/sp{i}.txt"):
    #             break
    #     with open(f"{cost_folder}/{job}/last/sp{i}.txt", "r") as f:
    #         plan_xml = f.read()
    #     sql_full = plan_xml.split("OPTION")[0]
    #
    #     # print('to encode sql: ', sql_full)
    #     long_query_enc = long_query_encode(sql_full, table_info=table_info)
    #     # print('long_query_enc', long_query_enc)
    #     query_enc_dict[job] = long_query_enc

    for plan_nr in plans_list:
        job_nr = plan_nr.split("_")[0]
        sub_query_nr = plan_nr.split("_")[1]
        plan_file_nr = plan_nr.split("_")[2]
        version_nr = plan_file_nr.replace("sp", "")
        cost_file_name =  f"cost{version_nr}.txt"
        plan_file_name = f"{plan_file_nr}.pickle"

        with open(f"{cost_folder}/{job_nr}/{sub_query_nr}/{cost_file_name}", "r") as f:
            cost_plan = f.read()
        with open(f"{plans_folder}/{job_nr}/{sub_query_nr}/{plan_file_name}", "rb") as d:
            execution_plan = pickle.load(d)

        full_execution_plan = feature_ext.match_cost_plan(execution_plan, cost_plan)

        if return_featurized:
            # print('####')
            # print('plan file: ', file)
            # sql = cost_plan.split('StatementText="')[1].split('OPTION')[0].strip()
            #
            # print('sql: ', sql)

            # print('execution_plan node: ', execution_plan, execution_plan.name, execution_plan.id)

            featurized_plan, rows = feature_ext.featurize_node(full_execution_plan)

            # print('done with curr plan enc!\n', featurized_plan, rows)
            # print('####')
            # query_vec = obtain_new_query_enc(full_execution_plan, sql_dict)

            query_vec = full_execution_plan.get_query_encoding()

            # total_query_enc = query_enc_dict[job_nr] + query_vec[2:]

            # total_query_enc = query_vec

            # total_query_enc = [query_vec[0], query_vec[1], query_vec[2], query_vec[4], query_vec[5]]

            # total_query_enc = [query_vec[2], query_vec[3], query_vec[4], query_vec[5]]


            featurized_trees[plan_nr] = featurized_plan
            featurized_vecs[plan_nr] = query_vec
        else:
            featurized_trees[plan_nr] = full_execution_plan

    tree_len = feature_ext.tree_len_list

    min_value = np.min(tree_len)
    percent_5 = np.quantile(tree_len, 0.05)
    percent_15 = np.quantile(tree_len, 0.15)
    percent_25 = np.quantile(tree_len, 0.25)
    median = np.quantile(tree_len, q=0.5)
    percen_75 = np.quantile(tree_len, q=0.75)
    percen_90 = np.quantile(tree_len, q=0.90)
    percen_95 = np.quantile(tree_len, q=0.95)
    percen_98 = np.quantile(tree_len, q=0.98)
    max_value = np.max(tree_len)
    print('min_value: ', min_value)
    print('percent_5: ', percent_5)
    print('percent_15: ', percent_15)
    print('percent_25: ', percent_25)
    print('median: ', median)
    print('percen_75: ', percen_75)
    print('percen_90: ', percen_90)
    print('percen_95: ', percen_95)
    print('percen_98: ', percen_98)
    print('max_value: ', max_value)

    return featurized_vecs, featurized_trees


def featurize_with_labels_Random(plans_folder, cost_folder, label_csv, workload="tpch", nr_jobs=2000, max_score = 50, score_function = "special",  extra_for_min = True, special_border = 0.95, postfix=None):

    print('postfix: ', postfix)
    label_dict = {}
    df = pd.read_csv(label_csv, index_col = 0)
    df = df.set_index(['sp_name'], drop=True)
    print('df after set_index: ', df.head(3))

    df["Job_nr"] = list(map(lambda x: x.split("_")[0], df.index)) #Job443v1_be_sp0
    df["Subquery"] = list(map(lambda x: '_'.join(x.split("_")[:-1]), df.index))

    ### remove queries without "last" subquery (which are bad queries)
    total_number_jobs_before = len(list(df["Job_nr"].unique()))
    total_number_subquery_before = len(list(df["Subquery"].unique()))

    print('total number of jobs before filtering: ', total_number_jobs_before, total_number_subquery_before)

    plans_with_last = list(filter(lambda x:x.split("_")[1] == "last", df.index))
    normal_jobs = list(set(list(map(lambda x: x.split("_")[0], plans_with_last))))
    print('the number of jobs after filtering: ', len(normal_jobs))

    df = df[df["Job_nr"].isin(normal_jobs)]
    print('the number of jobs/subqueries after filtering: ', len(list(df["Job_nr"].unique())), len(list(df["Subquery"].unique())))
    ### remove queries without "last" subquery (which are bad queries)

    total_unique_job_list = pd.unique(df["Job_nr"]).tolist()
    involved_job_list = total_unique_job_list[:nr_jobs]
    print('# total number of plans: ', len(df))
    df = df[df["Job_nr"].isin(involved_job_list)]
    print('# only involved plans: ', len(df))

    involved_plans_list = list(df.index)

    involved_subquery_list = pd.unique(df["Subquery"]).tolist()

    featurized_vecs, featurized_trees = get_features_with_cost_from_folder_Random(plans_folder, cost_folder, workload=workload, plans_list=involved_plans_list, postfix=postfix)

    print('number of featurized_vecs, featurized_trees', len(featurized_vecs.keys()), len(featurized_trees.keys()))

    times = []

    global_sum_max = df['cost'].max()

    for subquery in involved_subquery_list:

        temp_df = df[df["Subquery"]==subquery].copy()
        print('subquery: ', subquery)
        print('temp_df&length: ', temp_df, len(temp_df))

        a = np.array(temp_df["cost"])
        local_sum_max = a.max()

        if score_function == "special":
            temp = df[df["Subquery"] == subquery]
            labels = temp.index
            x = np.array(temp["cost"])
            if len(x) == 0:
                continue
            if np.min(x[x>=0]) != 0:
                x[x>=0] = x[x>=0]/np.min(x[x>=0])
            else:
                x = x + 1
            times.extend(list(zip(labels,x)))
        else:
            if score_function == "linear":
                a[a==-2] = max(a)*2 # Necessary, otherwise min(a) == -2
                temp_df["scores"] = calculate_linear_scores(a, n = max_score)
            elif score_function == "histogram":
                temp_df["scores"] = calculate_histogram_score(a, nr_bins = max_score, extra_bin_for_min = extra_for_min)
            elif score_function == "agglomerative":
                temp_df["scores"] = calculate_agglomerative_score(a, nr_clusters = max_score, extra_bin_for_min = extra_for_min)
            elif score_function == "linearG":
                temp_df["scores"] = calculate_linear_global_scores(a, global_sum_max)
            elif score_function == "linearL":
                temp_df["scores"] = calculate_linear_local_scores(a, local_sum_max)
            elif score_function == "linearR":
                temp_df["scores"] = calculate_linear_raw_scores(a)
            elif score_function == "linearLN":
                temp_df["scores"] = calculate_linear_local_norm_scores(a, local_sum_max)
            elif score_function == "linearNEG":
                temp_df["scores"] = calculate_negative_scores(a)

            for idx, row in temp_df.iterrows():
                assert not idx in label_dict.keys(), f"labels already exist {idx}!!!"
                label_dict[idx] = row["scores"]

    if score_function == "special":
        labels, scores = calculate_special_score(times, max_score, special_border)
        for idx, s in enumerate(scores):
            label_dict[labels[idx]] = s

    return featurized_vecs, featurized_trees, label_dict



def calculate_special_score(scores, n, border_value):
    s = np.array([s[1] for s in scores])
    border = np.quantile(s[s >= 0], border_value)
    print(border)
    times = []
    for s in scores:
        if s[1] > border or s[1] < 0:
            times.append(border)
        else:
            times.append(s[1])
    times = np.array(times)
    labels = AgglomerativeClustering(n_clusters=n+1).fit_predict(times.reshape(-1,1))
    #labels = clustering.predict(times.reshape(-1,1))
    maxima = [np.max(times[np.where(labels == i)]) for i in range(n)]
    sort = np.concatenate((np.sort(maxima)[::-1],np.array([0])))
    result = np.digitize(times,sort)
    return [s[0] for s in scores], result
    
def calculate_linear_scores(scores, n = 5):
    best = min(scores)
    ten_best = best*n
    if not ten_best:
        ten_best = n    
    # apply linear scores:
    m = -n/(ten_best - best)
    b = -1*m*(ten_best)
    
    scores = m*scores+b
    return scores


def calculate_linear_global_scores(scores, max_score):
    print('global max_score:', max_score)
    print('scores: ', scores)

    normalized_scores = max_score/(scores + 0.001)
    print('normalized scores: ', normalized_scores)
    return normalized_scores


def calculate_linear_local_scores(scores, max_score):
    print('linearL, local max_score:', max_score)
    print('scores: ', scores)
    normalized_scores = (max_score+0.001) / (scores+0.001)

    print('normalized scores: ', normalized_scores)
    return normalized_scores

def calculate_negative_scores(scores):
    neg_scores = (-1)*scores
    print('negative scores: ', neg_scores)

    return neg_scores

def calculate_linear_local_norm_scores(scores, max_score):
    print('linearLN, local min_score:', max_score)
    print('scores: ', scores)

    scores = max_score/scores
    print('min max scores: ', min(scores), max(scores))

    normalized_scores = (scores - min(scores) + 0.01) / (max(scores) - min(scores) + 0.001)
    print('normalized scores: ', normalized_scores)

    return normalized_scores

def calculate_linear_raw_scores(scores):
    print('runtimes: ', scores)
    scores_sorted = list(np.sort(scores)[::-1])

    # print('scores_sorted: ', scores_sorted)
    score_sorted_indices = []
    for idx, i in enumerate(scores):
        index = scores_sorted.index(i)
        score_sorted_indices.append(index)
        scores_sorted[index] = "abcde"

    print('scores: ', score_sorted_indices)

    return score_sorted_indices

def calculate_histogram_score(labels, nr_bins = 10, extra_bin_for_min = True):
    labels_copy = labels[labels != -2]
    hist, edges = np.histogram(labels_copy, nr_bins)
    edges_inv = edges[::-1]
    result = np.digitize(labels,edges_inv)
    result[labels == -2] = -1
    if extra_bin_for_min:
        result[labels == min(labels_copy)] = nr_bins+1
    return result

def calculate_agglomerative_score(labels, nr_clusters=10, extra_bin_for_min = True):    
    # Todo: This won't work yet because there is e.g. a max score of 3 for list of length 3 
    
    labels_copy = labels[labels != -2].reshape(-1, 1)
    if len(labels_copy) < nr_clusters:
        nr_clusters = len(labels_copy)
    
    clustering = AgglomerativeClustering(n_clusters = nr_clusters).fit_predict(labels_copy)
    maxima = [np.max(labels_copy[np.where(clustering == i)]) for i in range(nr_clusters)]
    sort = np.concatenate((np.sort(maxima)[::-1],np.array([0])))
    result = np.digitize(labels,sort)
    result[labels == -2] = -1
    if extra_bin_for_min:
        result[labels == min(labels_copy)] = nr_clusters+1
    return result


def get_left_child(node):
    if len(node) != 3:
        return None
    return node[1]

def get_right_child(node):
    if len(node) != 3:
        return None
    return node[2]
    
def get_features(node):
    if len(node) != 3:
        return node
    return node[0]


# scores = [34234,23, 342, 54, 1,25]
# calculate_linear_raw_scores(scores)

# runtimes = [4, 34, 5, 65, 78, 3, 6, 7, 9]
# scores = calculate_linear_raw_scores(runtimes)
