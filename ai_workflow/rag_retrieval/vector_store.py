import os
import psycopg2

from pgvector import Vector
from pgvector.psycopg2 import register_vector

from dotenv import load_dotenv

from ai_workflow.rag_retrieval.embeddings import embed_text


load_dotenv()

_connection = None


def get_connection():
    global _connection

    if _connection is None or _connection.closed:
        _connection = psycopg2.connect(
            os.getenv("SUPABASE_DB_URL")
        )

        register_vector(_connection)

    return _connection


def store_clause(
    document_id: str,
    clause_type: str,
    clause_text: str,
    risk_flag: str = None,
    page_number: int = None,
):
    """
    Embeds one piece of text and inserts it into the clauses table.

    page_number stores the PDF page number so that retrieved
    clauses can later be shown with page-level citations.
    """

    # Convert the Python list into a pgvector Vector object.
    embedding = Vector(embed_text(clause_text))

    conn = get_connection()

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO clauses (
                document_id,
                clause_type,
                clause_text,
                embedding,
                risk_flag,
                page_number
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                document_id,
                clause_type,
                clause_text,
                embedding,
                risk_flag,
                page_number,
            ),
        )

    conn.commit()


def search_clauses(
    question: str,
    k: int = 5,
    only_knowledge_base: bool = False,
    document_id: str = None,
):
    """
    Returns the k closest rows by vector distance.

    The returned rows include page_number so the calling layer
    can display PDF page-level citations.
    """

    # Convert the Python list into a pgvector Vector object.
    question_embedding = Vector(embed_text(question))

    conn = get_connection()

    with conn.cursor() as cur:

        if document_id:
            cur.execute(
                """
                SELECT
                    clause_text,
                    clause_type,
                    document_id,
                    risk_flag,
                    page_number
                FROM clauses
                WHERE document_id = %s
                ORDER BY embedding <-> %s
                LIMIT %s
                """,
                (
                    document_id,
                    question_embedding,
                    k,
                ),
            )

        elif only_knowledge_base:
            cur.execute(
                """
                SELECT
                    clause_text,
                    clause_type,
                    document_id,
                    risk_flag,
                    page_number
                FROM clauses
                WHERE document_id LIKE 'KNOWLEDGE_BASE::%%'
                ORDER BY embedding <-> %s
                LIMIT %s
                """,
                (
                    question_embedding,
                    k,
                ),
            )

        else:
            cur.execute(
                """
                SELECT
                    clause_text,
                    clause_type,
                    document_id,
                    risk_flag,
                    page_number
                FROM clauses
                ORDER BY embedding <-> %s
                LIMIT %s
                """,
                (
                    question_embedding,
                    k,
                ),
            )

        rows = cur.fetchall()

    return rows


def delete_document(document_id: str):
    """
    Removes all stored rows for one uploaded document.
    """

    conn = get_connection()

    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM clauses WHERE document_id = %s",
            (document_id,),
        )

    conn.commit()


def search_clauses_multi(
    question: str,
    document_ids: list[str],
    k: int = 5,
):
    """
    Searches across several specific documents at once.

    Used for cross-document questions such as:
    "Compare the termination clauses in these contracts."
    """

    # Convert the Python list into a pgvector Vector object.
    question_embedding = Vector(embed_text(question))

    conn = get_connection()

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                clause_text,
                clause_type,
                document_id,
                risk_flag,
                page_number
            FROM clauses
            WHERE document_id = ANY(%s)
            ORDER BY embedding <-> %s
            LIMIT %s
            """,
            (
                document_ids,
                question_embedding,
                k,
            ),
        )

        rows = cur.fetchall()

    return rows