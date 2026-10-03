import os
import psycopg2
from pgvector.psycopg2 import register_vector
from pgvector import Vector
from dotenv import load_dotenv
from ai_workflow.rag_retrieval.embeddings import embed_text

load_dotenv()

_connection = None


def get_connection():
    global _connection
    if _connection is None or _connection.closed:
        _connection = psycopg2.connect(os.getenv("SUPABASE_DB_URL"))
        register_vector(_connection)
    return _connection


def store_clause(document_id: str, clause_type: str, clause_text: str, risk_flag: str = None):
    """Embeds one piece of text and inserts it as a row in the clauses table."""
    embedding = embed_text(clause_text)
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO clauses (document_id, clause_type, clause_text, embedding, risk_flag)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (document_id, clause_type, clause_text, embedding, risk_flag),
        )
    conn.commit()


def search_clauses(
    question: str,
    k: int = 5,
    only_knowledge_base: bool = False,
    document_id: str = None,
):
    # question_embedding = embed_text(question)
    question_embedding = Vector(embed_text(question))
    conn = get_connection()
    with conn.cursor() as cur:
        if document_id:
            cur.execute(
                """
                SELECT clause_text, clause_type, document_id, risk_flag
                FROM clauses
                WHERE document_id = %s
                ORDER BY embedding <-> %s
                LIMIT %s
                """,
                (document_id, question_embedding, k),
            )
        elif only_knowledge_base:
            cur.execute(
                """
                SELECT clause_text, clause_type, document_id, risk_flag
                FROM clauses
                WHERE document_id LIKE 'KNOWLEDGE_BASE::%%'
                ORDER BY embedding <-> %s
                LIMIT %s
                """,
                (question_embedding, k),
            )
        else:
            cur.execute(
                """
                SELECT clause_text, clause_type, document_id, risk_flag
                FROM clauses
                ORDER BY embedding <-> %s
                LIMIT %s
                """,
                (question_embedding, k),
            )
        rows = cur.fetchall()
    return rows


def delete_document(document_id: str):
    """
    Removes all stored rows for one uploaded document. Called before
    re-ingesting the same filename, so re-uploading a file replaces its
    chunks instead of duplicating them (duplicates would skew retrieval).
    """
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("DELETE FROM clauses WHERE document_id = %s", (document_id,))
    conn.commit()
