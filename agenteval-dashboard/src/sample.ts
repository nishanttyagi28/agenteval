/** Demo corpus so the landing page is immediately useful. */
export const SAMPLE_JSONL = `{"id": "t01_insert", "sql": "INSERT INTO users (id, name) VALUES (1, 'x')"}
{"id": "t02_drop", "sql": "DROP TABLE users"}
{"id": "t03_cartesian", "sql": "SELECT * FROM a JOIN b"}
{"id": "t04_multi", "sql": "SELECT id FROM users; DELETE FROM users"}
{"id": "t05_ok", "sql": "SELECT id, name FROM users WHERE active = true LIMIT 50"}
{"id": "t06_no_limit", "sql": "SELECT id FROM users WHERE active = true"}
{"id": "t07_star", "sql": "SELECT * FROM orders WHERE status = 'open' LIMIT 20"}
{"id": "t08_ctas", "sql": "CREATE TABLE staging AS SELECT * FROM users"}
`
