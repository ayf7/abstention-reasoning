"""
Script to create standardized Text-to-SQL primitive files from raw datasets
(Spider, BIRD, SParC, CoSQL).

Standardized format mirrors math primitives:
- index: unique integer ID
- variant: db_id
- level: difficulty level (e.g. easy, medium, hard, extra, simple, moderate, challenging)
- problem: text input prompt containing schema DDL, question, and optional evidence
- solution: gold SQL query
- answer: gold SQL query
- prefix_hints: {} (placeholder for API-generated progressive hints)
- metadata: extra fields including conversational intermediate turns for SParC/CoSQL

Output files:
- primitives_{dataset}_{split}.json
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def is_multi_step_query(sql: str) -> bool:
    """
    Moderate complexity filter:
    Returns False if single table without nested subqueries/set-ops and has <= 1 clause.
    """
    sql_lower = sql.lower()
    if any(k in sql_lower for k in ['intersect', 'union', 'except']) or sql_lower.count('select') > 1:
        return True
    if ' join ' in sql_lower or ' inner join ' in sql_lower or ' left join ' in sql_lower:
        return True
    if re.search(r'\bfrom\s+([^\s\(\)]+)(\s*,\s*([^\s\(\)]+))+', sql_lower):
        return True
    has_where = ' where ' in sql_lower
    has_group = ' group by ' in sql_lower
    has_order = ' order by ' in sql_lower
    has_having = ' having ' in sql_lower
    return sum([has_where, has_group, has_order, has_having]) > 1


def is_complex_query(sql: str) -> bool:
    """
    Strict complexity filter for queries with sufficient depth for 5 progressive hints:
    1. Contains nested subqueries or set operations (UNION, INTERSECT, EXCEPT), OR
    2. Contains multi-table JOINs WITH at least one filter/group/order clause, OR
    3. If single-table, involves multi-clause logic: GROUP BY + WHERE + (HAVING or ORDER BY).
    """
    sql_lower = sql.lower()
    if any(k in sql_lower for k in ['intersect', 'union', 'except']) or sql_lower.count('select') > 1:
        return True
    has_where = ' where ' in sql_lower
    has_group = ' group by ' in sql_lower
    has_order = ' order by ' in sql_lower
    has_having = ' having ' in sql_lower
    has_join = ' join ' in sql_lower or bool(re.search(r'\bfrom\s+([^\s\(\)]+)(\s*,\s*([^\s\(\)]+))+', sql_lower))
    if has_join:
        return bool(has_where or has_group or has_order or has_having)
    return bool(has_group and has_where and (has_having or has_order))

# Constants for Spider hardness evaluation
WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')


def count_agg(units):
    return len([unit for unit in units if unit[0] > 0])


def get_nested_sql(sql):
    nested = []
    for cond_unit in sql['from']['conds'][::2]:
        if cond_unit[2] and isinstance(cond_unit[2], dict):
            nested.append(cond_unit[2])
        if cond_unit[3] and isinstance(cond_unit[3], dict):
            nested.append(cond_unit[3])
    for cond_unit in sql['where'][::2]:
        if cond_unit[2] and isinstance(cond_unit[2], dict):
            nested.append(cond_unit[2])
        if cond_unit[3] and isinstance(cond_unit[3], dict):
            nested.append(cond_unit[3])
    for cond_unit in sql['having'][::2]:
        if cond_unit[2] and isinstance(cond_unit[2], dict):
            nested.append(cond_unit[2])
        if cond_unit[3] and isinstance(cond_unit[3], dict):
            nested.append(cond_unit[3])
    for table_unit in sql['from']['table_units']:
        if table_unit[0] == 'sql':
            nested.append(table_unit[1])
    for k in ['intersect', 'union', 'except']:
        if sql.get(k):
            nested.append(sql[k])
    return nested


def count_component1(sql):
    count = 0
    if len(sql['where']) > 0:
        count += 1
    if len(sql['groupBy']) > 0:
        count += 1
    if len(sql['orderBy']) > 0:
        count += 1
    if sql['limit'] is not None:
        count += 1
    if len(sql['from']['table_units']) > 0:
        count += len(sql['from']['table_units']) - 1

    ao = sql['from']['conds'][1::2] + sql['where'][1::2] + sql['having'][1::2]
    count += len([token for token in ao if token == 'or'])
    cond_units = sql['from']['conds'][::2] + sql['where'][::2] + sql['having'][::2]
    count += len([cond_unit for cond_unit in cond_units if cond_unit[1] == WHERE_OPS.index('like')])
    return count


def count_component2(sql):
    nested = get_nested_sql(sql)
    return len(nested)


def count_others(sql):
    count = 0
    agg_count = count_agg(sql['select'][1])
    agg_count += count_agg(sql['where'][::2])
    agg_count += count_agg(sql['groupBy'])
    if len(sql['orderBy']) > 0:
        agg_count += count_agg([unit[1] for unit in sql['orderBy'][1] if unit[1]] +
                               [unit[2] for unit in sql['orderBy'][1] if unit[2]])
    agg_count += count_agg(sql['having'])
    if agg_count > 1:
        count += 1
    if len(sql['select'][1]) > 1:
        count += 1
    if len(sql['where']) > 1:
        count += 1
    if len(sql['groupBy']) > 1:
        count += 1
    return count


def eval_spider_hardness(sql: dict | None) -> str | None:
    """Official hardness evaluation from Spider."""
    if not sql:
        return None
    try:
        count_comp1_ = count_component1(sql)
        count_comp2_ = count_component2(sql)
        count_others_ = count_others(sql)

        if count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ == 0:
            return "easy"
        elif (count_others_ <= 2 and count_comp1_ <= 1 and count_comp2_ == 0) or \
                (count_comp1_ <= 2 and count_others_ < 2 and count_comp2_ == 0):
            return "medium"
        elif (count_others_ > 2 and count_comp1_ <= 2 and count_comp2_ == 0) or \
                (2 < count_comp1_ <= 3 and count_others_ <= 2 and count_comp2_ == 0) or \
                (count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ <= 1):
            return "hard"
        else:
            return "extra"
    except Exception:
        return None


def eval_sql_hardness_from_text(sql_str: str) -> str:
    """
    Uniform Spider hardness evaluation parsed directly from SQL query string.
    Emulates Spider's official component count on arbitrary SQL (e.g. BIRD, unparsed queries).
    """
    s = ' ' + sql_str.strip().lower() + ' '
    s_clean = re.sub(r'\s+', ' ', s)

    # Component 2: set operations & subqueries
    set_ops = len(re.findall(r'\b(union|intersect|except)\b', s_clean))
    subqueries = len(re.findall(r'\(\s*select\b', s_clean))
    count_comp2 = set_ops + subqueries

    # Strip subqueries for top-level clause counting
    s_no_sub = re.sub(r'\([^\(\)]*select[^\(\)]*\)', '', s_clean)
    while re.search(r'\([^\(\)]*select[^\(\)]*\)', s_no_sub):
        s_no_sub = re.sub(r'\([^\(\)]*select[^\(\)]*\)', '', s_no_sub)

    # Component 1: clauses and join depth
    count_comp1 = 0
    if ' where ' in s_no_sub:
        count_comp1 += 1
    if ' group by ' in s_no_sub:
        count_comp1 += 1
    if ' order by ' in s_no_sub:
        count_comp1 += 1
    if ' limit ' in s_no_sub:
        count_comp1 += 1

    joins = len(re.findall(r'\b(join|inner join|left join|right join|full join)\b', s_no_sub))
    from_match = re.search(r'\bfrom\s+(.*?)(?=\bwhere\b|\bgroup by\b|\border by\b|\blimit\b|$)', s_no_sub)
    if from_match:
        from_clause = from_match.group(1)
        comma_tables = len(from_clause.split(',')) - 1
        joins = max(joins, comma_tables)
    count_comp1 += joins

    count_comp1 += len(re.findall(r'\bor\b', s_no_sub))
    count_comp1 += len(re.findall(r'\blike\b', s_no_sub))

    # Others: aggregations, multiple projections, multiple filters
    count_others = 0
    aggs = len(re.findall(r'\b(count|sum|avg|min|max)\s*\(', s_clean))
    if aggs > 1:
        count_others += 1

    select_match = re.search(r'\bselect\s+(.*?)(?=\bfrom\b)', s_no_sub)
    if select_match:
        select_clause = select_match.group(1)
        if ',' in select_clause:
            count_others += 1

    where_match = re.search(r'\bwhere\s+(.*?)(?=\bgroup by\b|\border by\b|\blimit\b|$)', s_no_sub)
    if where_match:
        where_clause = where_match.group(1)
        if ' and ' in where_clause or ' or ' in where_clause:
            count_others += 1

    group_match = re.search(r'\bgroup by\s+(.*?)(?=\bhaving\b|\border by\b|\blimit\b|$)', s_no_sub)
    if group_match:
        group_clause = group_match.group(1)
        if ',' in group_clause:
            count_others += 1

    if count_comp1 <= 1 and count_others == 0 and count_comp2 == 0:
        return 'easy'
    elif (count_others <= 2 and count_comp1 <= 1 and count_comp2 == 0) or \
            (count_comp1 <= 2 and count_others < 2 and count_comp2 == 0):
        return 'medium'
    elif (count_others > 2 and count_comp1 <= 2 and count_comp2 == 0) or \
            (2 < count_comp1 <= 3 and count_others <= 2 and count_comp2 == 0) or \
            (count_comp1 <= 1 and count_others == 0 and count_comp2 <= 1):
        return 'hard'
    else:
        return 'extra'


def compute_uniform_spider_hardness(sql_ast: dict | None, sql_text: str) -> str:
    """Compute uniform Spider hardness using AST if available, falling back to text parsing."""
    if sql_ast:
        ast_level = eval_spider_hardness(sql_ast)
        if ast_level:
            return ast_level
    return eval_sql_hardness_from_text(sql_text)


def build_schema_ddl(db_entry: dict[str, Any]) -> str:
    """
    Generate clean, standardized DDL schema from a tables.json database entry.
    Contains ONLY table names, column names, data types, primary keys, and foreign keys.
    Strictly excludes all data rows.
    """
    tables = db_entry["table_names_original"]
    cols = db_entry["column_names_original"]
    col_types = db_entry["column_types"]

    raw_pks = db_entry.get("primary_keys", [])
    table_pks = {t_idx: [] for t_idx in range(len(tables))}
    for pk in raw_pks:
        if isinstance(pk, int):
            if 0 <= pk < len(cols):
                t_idx, c_name = cols[pk]
                if t_idx >= 0:
                    table_pks[t_idx].append(c_name)
        elif isinstance(pk, list):
            for p in pk:
                if 0 <= p < len(cols):
                    t_idx, c_name = cols[p]
                    if t_idx >= 0:
                        table_pks[t_idx].append(c_name)

    fks = db_entry.get("foreign_keys", [])
    fk_map = {}
    for fk in fks:
        if isinstance(fk, list) and len(fk) == 2:
            src, dst = fk
            if isinstance(src, int) and isinstance(dst, int):
                fk_map[src] = dst

    lines = []
    for t_idx, t_name in enumerate(tables):
        t_cols = [(c_idx, c_name, col_types[c_idx]) for c_idx, (t_id, c_name) in enumerate(cols) if t_id == t_idx]
        col_defs = []
        for c_idx, c_name, c_type in t_cols:
            col_def = f"  `{c_name}` {c_type.upper()}"
            if c_idx in fk_map:
                target_c_idx = fk_map[c_idx]
                if 0 <= target_c_idx < len(cols):
                    target_t_idx, target_c_name = cols[target_c_idx]
                    if 0 <= target_t_idx < len(tables):
                        target_t_name = tables[target_t_idx]
                        col_def += f" REFERENCES `{target_t_name}`(`{target_c_name}`)"
            col_defs.append(col_def)

        pks_for_table = table_pks.get(t_idx, [])
        if pks_for_table:
            pk_str = ", ".join(f"`{k}`" for k in pks_for_table)
            col_defs.append(f"  PRIMARY KEY ({pk_str})")

        lines.append(f"CREATE TABLE `{t_name}` (\n" + ",\n".join(col_defs) + "\n);")
    return "\n\n".join(lines)


def load_schemas(tables_json_paths: list[Path]) -> dict[str, str]:
    """Load tables.json files and build DDL mapping for all databases."""
    db_schemas = {}
    for path in tables_json_paths:
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            tables_data = json.load(f)
        for db_entry in tables_data:
            db_id = db_entry["db_id"]
            if db_id not in db_schemas:
                db_schemas[db_id] = build_schema_ddl(db_entry)
    return db_schemas


def format_problem_text(schema_ddl: str, question: str, evidence: str | None = None) -> str:
    """Format the text input to the model."""
    parts = [
        "Database Schema:\n" + schema_ddl.strip(),
        f"Question:\n{question.strip()}"
    ]
    if evidence and evidence.strip():
        parts.append(f"Evidence:\n{evidence.strip()}")
    return "\n\n".join(parts)


def process_spider(raw_dir: Path) -> dict[str, list[dict]]:
    """Process Spider train, train_others, and dev sets."""
    spider_dir = raw_dir / "spider" / "spider_data"
    schemas = load_schemas([spider_dir / "tables.json"])

    splits = {
        "spider_train": spider_dir / "train_spider.json",
        "spider_train_others": spider_dir / "train_others.json",
        "spider_dev": spider_dir / "dev.json",
    }

    result = {}
    for split_name, file_path in splits.items():
        if not file_path.exists():
            logger.warning(f"File {file_path} not found, skipping.")
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        primitives = []
        for idx, item in enumerate(data):
            db_id = item["db_id"]
            question = item["question"]
            sql = item.get("query", "").strip()
            schema_ddl = schemas.get(db_id, "")
            orig_level = eval_spider_hardness(item.get("sql"))
            uniform_level = compute_uniform_spider_hardness(item.get("sql"), sql)

            problem_text = format_problem_text(schema_ddl, question)
            primitives.append({
                "index": idx,
                "variant": db_id,
                "level": uniform_level,
                "problem": problem_text,
                "answer": sql,
                "prefix_hints": {},
                "metadata": {
                    "source": "spider",
                    "split": split_name.replace("spider_", ""),
                    "db_id": db_id,
                    "question": question,
                    "evidence": None,
                    "difficulty": uniform_level,
                    "original_difficulty": orig_level,
                    "multi_step": is_multi_step_query(sql),
                    "is_complex": is_complex_query(sql),
                }
            })
        result[split_name] = primitives
    return result


def process_bird(raw_dir: Path) -> dict[str, list[dict]]:
    """Process BIRD train and dev sets."""
    bird_dir = raw_dir / "bird"
    schemas = load_schemas([
        bird_dir / "train" / "train_tables.json",
        bird_dir / "dev" / "dev_tables.json"
    ])

    splits = {
        "bird_train": bird_dir / "train" / "train.json",
        "bird_dev": bird_dir / "dev" / "dev.json",
    }

    result = {}
    for split_name, file_path in splits.items():
        if not file_path.exists():
            logger.warning(f"File {file_path} not found, skipping.")
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        primitives = []
        for idx, item in enumerate(data):
            db_id = item["db_id"]
            question = item["question"]
            evidence = item.get("evidence", "").strip() or None
            sql = item.get("SQL", "").strip()
            orig_difficulty = item.get("difficulty")
            uniform_level = compute_uniform_spider_hardness(None, sql)
            schema_ddl = schemas.get(db_id, "")

            problem_text = format_problem_text(schema_ddl, question, evidence=evidence)
            primitives.append({
                "index": idx,
                "variant": db_id,
                "level": uniform_level,
                "problem": problem_text,
                "answer": sql,
                "prefix_hints": {},
                "metadata": {
                    "source": "bird",
                    "split": split_name.replace("bird_", ""),
                    "db_id": db_id,
                    "question": question,
                    "evidence": evidence,
                    "difficulty": uniform_level,
                    "original_difficulty": orig_difficulty,
                    "question_id": item.get("question_id", idx),
                    "multi_step": is_multi_step_query(sql),
                    "is_complex": is_complex_query(sql),
                }
            })
        result[split_name] = primitives
    return result


def process_sparc(raw_dir: Path) -> dict[str, list[dict]]:
    """Process SParC train and dev sets (converting multi-turn to single-turn with intermediate turns metadata)."""
    sparc_dir = raw_dir / "sparc" / "sparc"
    schemas = load_schemas([sparc_dir / "tables.json"])

    splits = {
        "sparc_train": sparc_dir / "train.json",
        "sparc_dev": sparc_dir / "dev.json",
    }

    result = {}
    for split_name, file_path in splits.items():
        if not file_path.exists():
            logger.warning(f"File {file_path} not found, skipping.")
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        primitives = []
        for idx, item in enumerate(data):
            db_id = item["database_id"]
            final_obj = item.get("final", {})
            question = final_obj.get("utterance", "")
            sql = final_obj.get("query", "").strip()
            schema_ddl = schemas.get(db_id, "")

            interaction = item.get("interaction", [])
            intermediate_turns = [
                {
                    "turn_idx": t_idx,
                    "question": turn.get("utterance", ""),
                    "sql": turn.get("query", "").strip(),
                }
                for t_idx, turn in enumerate(interaction)
            ]

            # Calculate difficulty from final turn AST, fallback to text parsing of final SQL
            last_turn_sql = interaction[-1].get("sql") if interaction else None
            orig_level = eval_spider_hardness(last_turn_sql)
            uniform_level = compute_uniform_spider_hardness(last_turn_sql, sql)

            problem_text = format_problem_text(schema_ddl, question)
            primitives.append({
                "index": idx,
                "variant": db_id,
                "level": uniform_level,
                "problem": problem_text,
                "answer": sql,
                "prefix_hints": {},
                "metadata": {
                    "source": "sparc",
                    "split": split_name.replace("sparc_", ""),
                    "db_id": db_id,
                    "question": question,
                    "evidence": None,
                    "difficulty": uniform_level,
                    "original_difficulty": orig_level,
                    "interaction_length": len(interaction),
                    "intermediate_turns": intermediate_turns,
                    "multi_step": is_multi_step_query(sql),
                    "is_complex": is_complex_query(sql),
                }
            })
        result[split_name] = primitives
    return result


def process_cosql(raw_dir: Path) -> dict[str, list[dict]]:
    """Process CoSQL train and dev sets (sql_state_tracking)."""
    cosql_dir = raw_dir / "cosql" / "cosql_dataset"
    schemas = load_schemas([cosql_dir / "tables.json"])

    splits = {
        "cosql_train": cosql_dir / "sql_state_tracking" / "cosql_train.json",
        "cosql_dev": cosql_dir / "sql_state_tracking" / "cosql_dev.json",
    }

    result = {}
    for split_name, file_path in splits.items():
        if not file_path.exists():
            logger.warning(f"File {file_path} not found, skipping.")
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        primitives = []
        for idx, item in enumerate(data):
            db_id = item["database_id"]
            final_obj = item.get("final", {})
            question = final_obj.get("utterance", "")
            sql = final_obj.get("query", "").strip()
            schema_ddl = schemas.get(db_id, "")

            interaction = item.get("interaction", [])
            intermediate_turns = [
                {
                    "turn_idx": t_idx,
                    "question": turn.get("utterance", ""),
                    "sql": turn.get("query", "").strip(),
                }
                for t_idx, turn in enumerate(interaction)
            ]

            last_turn_sql = interaction[-1].get("sql") if interaction else None
            orig_level = eval_spider_hardness(last_turn_sql)
            uniform_level = compute_uniform_spider_hardness(last_turn_sql, sql)

            problem_text = format_problem_text(schema_ddl, question)
            primitives.append({
                "index": idx,
                "variant": db_id,
                "level": uniform_level,
                "problem": problem_text,
                "answer": sql,
                "prefix_hints": {},
                "metadata": {
                    "source": "cosql",
                    "split": split_name.replace("cosql_", ""),
                    "db_id": db_id,
                    "question": question,
                    "evidence": None,
                    "difficulty": uniform_level,
                    "original_difficulty": orig_level,
                    "interaction_length": len(interaction),
                    "intermediate_turns": intermediate_turns,
                    "multi_step": is_multi_step_query(sql),
                    "is_complex": is_complex_query(sql),
                }
            })
        result[split_name] = primitives
    return result


def main():
    parser = argparse.ArgumentParser(description="Create primitives JSON files for Text-to-SQL benchmarks.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("/home/tanyagoyal/abstention-reasoning/data/sql/raw"),
        help="Path to raw datasets directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/tanyagoyal/abstention-reasoning/data/sql"),
        help="Path to directory where primitives files will be written",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["spider", "bird", "sparc", "cosql"],
        help="Datasets to process (spider, bird, sparc, cosql)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    handlers = {
        "spider": process_spider,
        "bird": process_bird,
        "sparc": process_sparc,
        "cosql": process_cosql,
    }

    for ds_name in args.datasets:
        if ds_name not in handlers:
            logger.error(f"Unknown dataset '{ds_name}'. Available: {list(handlers.keys())}")
            continue
        logger.info(f"Processing dataset: {ds_name}...")
        splits_dict = handlers[ds_name](args.raw_dir)

        for split_key, primitives in splits_dict.items():
            out_filename = f"primitives_{split_key}.json"
            out_path = args.output_dir / out_filename
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(primitives, f, indent=2)
            logger.info(f"Wrote {len(primitives)} primitives to {out_path}")


if __name__ == "__main__":
    main()
