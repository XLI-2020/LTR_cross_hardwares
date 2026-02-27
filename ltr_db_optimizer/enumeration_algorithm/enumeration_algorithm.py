import copy

import networkx as nx
import pyodbc
import string

from ltr_db_optimizer.enumeration_algorithm.dpccp import DPccp

from ltr_db_optimizer.enumeration_algorithm.joiner import Joiner
from ltr_db_optimizer.enumeration_algorithm.aggregator import Aggregator
import ltr_db_optimizer.enumeration_algorithm.enumeration_node as nodes
from ltr_db_optimizer.model.featurizer_graph import FeatureExtractorGraph, long_query_encode
   
from ltr_db_optimizer.parser.xml_parser import XMLParser
from ltr_db_optimizer.parser.SQLParser import to_sql
import numpy as np

class EnumerationAlgorithm(DPccp):
    def __init__(self, sql_query, table_info, model, sql_text, top_k = 1, alias_dict = {}, train_wk=None, tree_high=None, tree_low=None, comparison=False):
        self.path = model
        super().__init__(model = model, joiner = Joiner(sql_query, alias_dict, table_info), comparison=comparison)
        self.top_k = top_k
        self.sql_query = sql_query
        self.sql_text = sql_text
        self.to_bfs_graph()
        self.table_info = table_info
        self.aggregator = Aggregator(sql_query, self.joiner, self.table_info.database == "imdb" or self.table_info.database == "stats")
        
        # self.connection = 'DSN=jettesDSN;Trusted_Connection=yes;'

        self.saved_rows = {}
        self.featurizer = FeatureExtractorGraph(workload=train_wk, tree_high=tree_high, tree_low=tree_low)
        
        self.regard_subquery = False
        self.alias_dict = alias_dict
        self.database = self.table_info.database

    def find_best_plan(self):
        if not self.regard_subquery:
            # print(111)
            if self.sql_query["Subquery"] is not None:
                print(111)
                # pretty ugly, can be changed hopefully
                sub_enum = EnumerationAlgorithm(self.sql_query["Subquery"], self.table_info,
                                                self.path, "", self.top_k, self.alias_dict)
                sub_enum.regard_subquery = True
                self.best_subquery = sub_enum.find_best_plan()
                self.joiner.set_subquery(self.best_subquery)
        if len(self.sql_query["Joins"]) > 0:
            print(222)
            best_plan = self.enumerate(name_in_data=True)
            print('type of each plan', type(best_plan[0]))

        else:
            print(333)
            assert len(self.sql_query["Tables"]) == 1
            best_plan = self.joiner.get_scan(self.sql_query["Tables"][0][0])

        best_plan = self.add_additional_nodes(best_plan)

        if self.regard_subquery:
            return best_plan

        if len(best_plan) > 1:
            best_plan = self.reduce(best_plan, True)

        # print('best plan type', type(best_plan), best_plan)
        # print(444, "IndexScan" in self.to_xml(best_plan))

        # return self.to_xml(best_plan)
        return best_plan

        
    def best_plan_to_xml(self, plan):
        plan = self.resolve_output_columns(plan)
        parser = XMLParser()
        return parser.generate_from_graph(plan)
        
    def to_bfs_graph(self):
        """
        Using the join information in sql_query, a networkx.Graph is constructed with the relations as nodes
        and the join-relations as edges. To simplify the enumeration, a breadth-first search is applied and the
        node names are changed to increasing integers.
        """        
        self.graph = nx.Graph()
        self.graph.add_nodes_from([table[0].lower() for table in self.sql_query["Tables"]])
        self.graph.add_edges_from([(edge[0], edge[2]) for edge in self.sql_query["Joins"]])
        self.graph = nx.bfs_tree(self.graph, source=list(self.graph.nodes)[0])
        print('graph nodes labels 1: ', list(self.graph.nodes))
        self.graph = nx.convert_node_labels_to_integers(self.graph, label_attribute="old_name")
        print('graph nodes labels 2: ', list(self.graph.nodes))
        ## relabeling nodes integer labels, e.g, "1, 2, 3, ..." to "a, b, c, d, ..."
        # mapping = {10: "a", 11: "b", 12: "c", "13":c, "14"}
        mapping = dict(zip(self.graph, string.ascii_lowercase))
        self.graph = nx.relabel_nodes(self.graph, mapping, copy=False)
        print('graph nodes labels 3: ', list(self.graph.nodes))

    def add_additional_nodes(self, best_plans):
        temp_result = []
        for plan in best_plans:
            if self.aggregator.has_aggregate():
                best_plan = self.aggregator.add_aggregate(plan)
                temp_result.extend(best_plan)                
            else:
                temp_result.append(plan)
        result = []
        for plan in temp_result:
            best_plan = plan
            if len(self.sql_query["Sort"]):
                plan.query_encoding[0] = 1
                if ((not all([o[1] in best_plan.sorted_columns for o in self.sql_query["Sort"]]) and
                    not all([self.joiner.is_restricted(s[1]) for s in self.sql_query["Sort"]])) or 
                    any([s[0] == "desc" for s in self.sql_query["Sort"]])):
                    columns = []
                    ascending = []
                    for order in self.sql_query["Sort"]:
                        columns.append(self.aggregator.get_translated_column(order[1]))
                        asc = "true" if order[0] == "asc" else "false"
                        ascending.append(asc)
                    best_plan = nodes.SortNode(columns, ascending, name = "sort", left_child = best_plan,
                                               is_sorted = True, contained_tables = best_plan.contained_tables,
                                               sorted_columns = columns)
                    print('best_plan.contained_tables', best_plan.contained_tables)
                    print('sorted_columns', columns)
            if self.sql_query["Top"] is not None:
                best_plan = nodes.TopNode(self.sql_query["Top"], name = "top", left_child = best_plan,
                                           is_sorted = best_plan.is_sorted, contained_tables = best_plan.contained_tables,
                                           sorted_columns = best_plan.sorted_columns)
            result.append(best_plan) 
        if self.table_info.database == "imdb" or self.table_info.database == "stats":
            temp_result = []
            for plan in result:
                temp_result.append(plan.down_propagate())
            return temp_result
        return result
    
    def resolve_output_columns(self, plan, output_columns=None, sort = False):
        if output_columns is None:
            output_columns = []
            for s in self.sql_query["Select"]:
                output_columns.append(self.aggregator.get_translated_column(s[1]))
        plan.set_output_columns(output_columns)
        # True for everything except Scans
        if plan.has_children():
            # True 
            if not plan.has_right_child():
                columns = []
                if type(plan) == nodes.ComputeScalarNode or type(plan) == nodes.AggregateNode:
                    for col in output_columns:
                        matched = self.aggregator.get_translated_column(col)
                        if type(matched) == list:
                            columns.extend(matched)
                        elif matched is None or matched.startswith("tempcount"):
                            continue
                        else:
                            columns.append(matched)
                    if plan.name == "stream_aggregate":
                        sort = True
                    plan.left_child = self.resolve_output_columns(plan.left_child, set(columns), sort)
                    
                else: # For Top and Sort (?)
                    if plan.name == "sort":
                        sort = False
                    plan.left_child = self.resolve_output_columns(plan.left_child, set(output_columns), sort)
            else:
                # Handle Joins
                right_columns = []
                left_columns = []  
                for col in output_columns:
                    if "." in col:
                        table = col.split(".")[0]
                    else:
                        pre = col.split("_")[0]+"_"
                        table = self.table_info.match_prefix(pre).table_name
                    if table in plan.left_child.contained_tables:
                        left_columns.append(col)
                    elif table in plan.right_child.contained_tables:
                        right_columns.append(col)
                    else:
                        raise Exception(f"Table {table} not found when trying to split the columns.")
                if plan.left_column not in left_columns:
                    left_columns.append(plan.left_column)
                if plan.right_column not in right_columns:
                    right_columns.append(plan.right_column)
                if plan.name == "merge_join":
                    sort = True
                plan.left_child = self.resolve_output_columns(plan.left_child, set(left_columns), sort)
                plan.right_child = self.resolve_output_columns(plan.right_child, set(right_columns), sort)
        else:
            plan.is_sorted = sort
        return plan


    def add_force_scan_to_sql(self, sql):
        # input_sql = sql.upper()
        # print('initial sql: ', sql)
        input_sql = copy.copy(sql)
        before_from, after_from = input_sql.split('FROM')
        if "WHERE" in after_from:
            before_where, after_where = after_from.split('WHERE')
            tables = before_where.split(',')

            tables_added_hint = ','.join(list(map(lambda x: x + " WITH(FORCESCAN)", tables)))

            sql_with_hint = before_from + " FROM" + tables_added_hint + " WHERE" + after_where

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

        print('input sql', input_sql)
        print('sql_with_hint', sql_with_hint)

        # sql_with_hint = sql_with_hint.replace(';', '')

        return sql_with_hint


    def prepare_plans_long_query_vec(self, plans, last = False):
        if last:
            sql = self.sql_text
        else:
            sql = to_sql(plans[0], self.table_info)
        # print('last', last)
        # print('length of plans', len(plans))
        # print('to sql: ', sql)

        sql = self.add_force_scan_to_sql(sql)

        long_query_enc = long_query_encode(self.sql_text, table_info = self.table_info)

        feat_plans, rows = self.plans_to_feature_vecs(sql, plans, last)
        # query_enc = plans[0].get_query_encoding()
        query_enc = [long_query_enc + plan.get_query_encoding()[2:] for plan in plans]

        return query_enc, feat_plans


    def prepare_plans(self, plans, last=False):
        if last:
            sql = self.sql_text
        else:
            sql = to_sql(plans[0], self.table_info)
        # print('last', last)
        # print('length of plans', len(plans))
        # print('to sql: ', sql)

        sql = self.add_force_scan_to_sql(sql)


        feat_plans, rows = self.plans_to_feature_vecs(sql, plans, last)
        # query_enc = plans[0].get_query_encoding()
        query_enc = [plan.get_query_encoding() for plan in plans]

        return query_enc, feat_plans

    def prepare_plans_adapt_queryvec_length(self, plans, last=False):
        if last:
            sql = self.sql_text
        else:
            sql = to_sql(plans[0], self.table_info)
        # print('last', last)
        # print('length of plans', len(plans))
        # print('to sql: ', sql)

        sql = self.add_force_scan_to_sql(sql)

        feat_plans, rows = self.plans_to_feature_vecs(sql, plans, last)
        # query_enc = plans[0].get_query_encoding()
        query_enc_list = []
        for plan in plans:
            query_vec = plan.get_query_encoding()
            # reduced_query_enc = [query_vec[0], query_vec[1], query_vec[2], query_vec[4], query_vec[5]]

            # reduced_query_enc = [query_vec[2], query_vec[3], query_vec[4], query_vec[5]]

            # query_enc_list.append(reduced_query_enc)
            query_enc_list.append(query_vec)

        return query_enc_list, feat_plans


    def plans_to_feature_vecs(self, sql, plans, last):
        parser = XMLParser(table_info=self.table_info, small_version=True)
        rows = 0
        result_featurized = []


        # SERVER = 'xiaoli,1433' # this is for Mac
        SERVER = 'localhost,1433'

        DATABASE = self.database
        USERNAME = 'sa'
        # PASSWORD = 'Lx##1992' # this is for Mac
        PASSWORD = 'LX##1992'
        conn_timeout = 10

        # SQL_ATTR_CONNECTION_TIMEOUT = 113

        conn = pyodbc.connect(driver='{ODBC Driver 18 for SQL Server}',
                              server=SERVER,
                              database=DATABASE,
                              UID=USERNAME,
                              PWD=PASSWORD, TrustServerCertificate='Yes', timeout=conn_timeout)



        # conn = pyodbc.connect(self.connection)
        cursor = conn.cursor()
        for it, plan in enumerate(plans):
            # print('index of plan: ', it)
            # print('plan.id: ', plan.id)
            # print('plan.name: ', plan.name)
            #
            # print('plan.left_child.name', plan.left_child.name)
            # print('plan.left_child.contained_tables', plan.left_child.contained_tables)

            # if plan.has_right_child():
            #     print('plan.right_child.name', plan.right_child.name)
            #     print('plan.right_child.contained_tables', plan.right_child.contained_tables)
            # else:
            #     print('plan has no right child')

            need_sql = self.featurizer.append_cost(plan)
            if need_sql:
                print('need_sql')
                xml = parser.generate_from_graph(plan)

                plan_sql = sql + " OPTION (RECOMPILE, USE PLAN N'"+xml+"')"
                cursor.execute("SET SHOWPLAN_XML ON")
                rows = cursor.execute(plan_sql).fetchall()
                cost_plan = rows[0][0]

                plan = self.featurizer.match_cost_plan(plan, cost_plan)
            temp, rows = self.featurizer.featurize_node(plan)
            # print('temp:', len(temp), temp)
            result_featurized.append(temp)

        conn.close()
        # unique_cost_parts = list(set(list(map(lambda x: x[0], self.featurizer.parts_cost_list))))
        # card_values_list = list(map(lambda x: x[1], self.featurizer.parts_cost_list))
        # print('length of cost parts: ', len(self.featurizer.parts_cost_list))
        # print('unique_cost_parts operators: ', unique_cost_parts)
        # print('max min card. values: ', max(card_values_list), min(card_values_list))

        tree_len = self.featurizer.tree_len_list

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
        print('test min_value: ', min_value)
        print('percent_5: ', percent_5)
        print('percent_15: ', percent_15)
        print('percent_25: ', percent_25)
        print('test median: ', median)
        print('percen_75: ', percen_75)
        print('percen_90: ', percen_90)
        print('percen_95: ', percen_95)
        print('percen_98: ', percen_98)
        print('max_value: ', max_value)

        return result_featurized, rows


    def to_xml(self, plan):
        print('to xml')
        parser = XMLParser(table_info=self.table_info, small_version=True)
        sql = self.sql_text
        xml = parser.generate_from_graph(plan)
        plan_sql = sql + " OPTION (RECOMPILE, USE PLAN N'"+xml+"')"
        # plan_sql = xml

        print('done xml!')
        return plan_sql
        
        
        
        