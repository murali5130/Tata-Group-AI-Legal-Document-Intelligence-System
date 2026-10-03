import os
import glob


def parse_clause_file(filepath: str) -> list[dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    records = []
    # Files use \r\n line endings — normalize before splitting on blank lines
    content = content.replace("\r\n", "\n")
    blocks = [b.strip() for b in content.strip().split("\n\n") if b.strip()]

    for block in blocks:
        record = {}
        for line in block.split("\n"):
            if ":" in line:
                key, _, value = line.partition(":")
                record[key.strip().lower()] = value.strip()
        if record:
            records.append(record)

    return records


def load_all_clause_records(clause_library_dir: str = "data/clause_library") -> list[dict]:
    all_records = []
    for filepath in sorted(glob.glob(os.path.join(clause_library_dir, "*.txt"))):
        source_file = os.path.basename(filepath)
        records = parse_clause_file(filepath)
        for r in records:
            r["source_file"] = source_file
        all_records.extend(records)
    return all_records


if __name__ == "__main__":
    # Quick manual test: `python ai_workflow/rag_retrieval/clause_library_loader.py`
    records = load_all_clause_records()
    print(f"Loaded {len(records)} clause records total.\n")
    print("--- First record ---")
    for k, v in records[0].items():
        print(f"{k}: {v}")
