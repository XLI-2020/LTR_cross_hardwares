import copy

class STATSTableInformation:
    schema = "dbo"
    database = "stats"
    
    def __init__(self, alias_dict = {"Tables": [], "Filters": []}):
        self.matcher = {}
        self.alias_dict = alias_dict
        self.matcher["users"] = self.Tables("users",
                                               "",
                                               ["Id", "Reputation",
                                                "CreationDate", "Views", "UpVotes", "DownVotes"],
                                               ["Id"],
                                               ["PK_users"],
                                               is_restricting = {},
                                              row_count = 40325)

        self.matcher["posts"] = self.Tables("posts",
                                            "",
                                            ["Id", "PostTypeId",
                                             "CreationDate", "Score", "ViewCount", "OwnerUserId", "AnswerCount", "CommentCount",
                                             "FavoriteCount", "LastEditorUserId"],
                                            ["Id"],
                                            ["PK_posts"],
                                            is_restricting={},
                                            row_count=91976)

        self.matcher["postlinks"] = self.Tables("postlinks",
                                            "",
                                            ["Id",
                                            "CreationDate",
                                             "PostId",
                                             "RelatedPostId", "LinkTypeId"],
                                            ["Id"],
                                            ["PK_postlinks"],
                                            is_restricting={},
                                            row_count=11102)

        self.matcher["posthistory"] = self.Tables("posthistory",
                                            "",
                                            ["Id", "PostHistoryTypeId",
                                             "PostId", "CreationDate", "UserId"],
                                            ["Id"],
                                            ["PK_posthistory"],
                                            is_restricting={},
                                            row_count=303187)

        self.matcher["comments"] = self.Tables("comments",
                                                  "",
                                                  ["Id", "PostId", "Score",
                                                   "CreationDate", "UserId"],
                                                  ["Id"],
                                                  ["PK_comments"],
                                                  is_restricting={},
                                                  row_count=174305)

        self.matcher["votes"] = self.Tables("votes",
                                               "",
                                               ["Id", "PostId", "VoteTypeId",
                                                "CreationDate", "UserId", "BountyAmount"],
                                               ["Id"],
                                               ["PK_votes"],
                                               is_restricting={},
                                               row_count=328064)

        self.matcher["badges"] = self.Tables("badges",
                                            "",
                                            ["Id", "UserId", "Date"],
                                            ["Id"],
                                            ["PK_badges"],
                                            is_restricting={},
                                            row_count=79851)

        self.matcher["tags"] = self.Tables("tags",
                                             "",
                                             ["Id", "Count", "ExcerptPostId"],
                                             ["Id"],
                                             ["PK_tags"],
                                             is_restricting={},
                                             row_count=1032)



    def get_table(self, table_name):
        if table_name in self.alias_dict["Tables"]:
            table_name = self.alias_dict["Tables"][table_name]
        if "stats.dbo." in table_name:
            table_name = table_name.replace("stats.dbo.", "").strip()
        if " " in table_name.strip():
            table_name = table_name.strip().split(" ")[0]
        return self.matcher[table_name.lower()]
    
    def get_corresponding_restricted_fields(self, fields):
        result_fields = fields
        for key in self.matcher:
            if any([field in self.matcher[key].field_restrictions for field in fields]):
                result_fields.extend(self.matcher[key].field_restrictions)
        return result_fields
    
    def get_further_reduced_columns(self, fields, joins):
        result_reduced_columns = []
        while len(fields):
            f = fields.pop(0)
            if "." not in f:
                continue
            f_splitted = f.split(".")
            if f_splitted[0] in self.alias_dict["Tables"]: 
                curr_table = self.alias_dict["Tables"][f_splitted[0]]
                f = f.replace(f_splitted[0],curr_table,1)
                curr_table = self.get_table(curr_table)
            else:
                curr_table = self.get_table(f_splitted[0])
            if f not in result_reduced_columns:
                result_reduced_columns.append(f)
            if f in curr_table.is_restricting:
                for res_field in curr_table.is_restricting[f]:
                    #if not res_field.startswith(curr_table.prefix):
                    #    other_table = self.match_prefix(res_field.split("_")[0]+"_").table_name
                    #    if not any([res_field in join and f in join and
                    #                ((curr_table.table_name == join[0] and other_table == join[2]) or
                    #                (curr_table.table_name == join[2] and other_table == join[0])) and
                    #                f in join for join in joins]):
                    #        continue
                    if res_field not in result_reduced_columns:
                        result_reduced_columns.append(res_field)
                        fields.append(res_field)
        return result_reduced_columns
    
    def has_join_connection(self, table_1, table_2, joins):
        for join in joins:
            if table_1 in join and table_2 in join:
                return True
        return False
    
    
    class Tables:
        # field restrictions are currently limited to the fields regarded in datafarm
        def __init__(self, table_name, prefix, columns, keys, key_name_server, field_restrictions=[], is_restricting = {}, row_count = 0):
            """
            :param field_restrictions: fields which reduce the table to one row if there is an equality-filter on this field
            """
            self.table_name = table_name
            self.prefix = prefix
            self.columns = columns
            self.keys = keys
            self.key_name_server = key_name_server
            self.field_restrictions = field_restrictions
            self.is_restricting = is_restricting
            self.row_count = row_count
            
        def get_columns(self):
            return self.columns
        
        def get_first_key(self):
            if self.has_keys():
                return self.keys[0]
            return None
        
        def get_keys(self):
            if self.has_keys():
                return self.keys
            return []
        
        def has_keys(self):
            return len(self.keys) > 0
            
        def get_server_key(self):
            return self.key_name_server[0]
        
        def has_column(self, column_name):
            for col in self.columns:
                if col.endswith(column_name):
                    return col
            return ""
        
        def get_prefix(self):
            return self.prefix