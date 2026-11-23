CREATE TABLE {json_table} (key FixedString(16), doc JSON) ENGINE = MergeTree ORDER BY key SETTINGS storage_policy = '{policy}';
INSERT INTO {json_table} SELECT key, doc FROM {table} SETTINGS max_insert_threads = {threads}, max_threads = {threads};
OPTIMIZE TABLE {json_table} FINAL;
