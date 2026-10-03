from tqdm import tqdm
from ai_workflow.rag_retrieval.clause_library_loader import load_all_clause_records
from ai_workflow.rag_retrieval.vector_store import store_clause


def populate():
    records = load_all_clause_records()
    print(f"Loaded {len(records)} clause records. Embedding and storing...")

    for record in tqdm(records):
        # document_id here identifies this as APPROVED reference material,
        # not an uploaded contract — this distinction matters when you
        # later store actual uploaded-contract clauses in the same table.
        store_clause(
            document_id=f"KNOWLEDGE_BASE::{record.get('source_file', 'unknown')}",
            clause_type=record.get("category", "UNKNOWN"),
            clause_text=(
                f"{record.get('title', '')}. {record.get('guidance', '')}"
            ),
            risk_flag=None,
        )

    print("Done. Knowledge base populated in pgvector.")


if __name__ == "__main__":
    populate()
