import os
import sys
from datetime import datetime

# Make `ai_workflow` / `document_pipeline` importable when Streamlit runs
# this file from the frontend/ folder instead of the project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Tata Legal AI",
    page_icon="⚖️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Fail early with a readable message if credentials are missing.
# ---------------------------------------------------------------------------
_missing = [k for k in ("GROQ_API_KEY", "SUPABASE_DB_URL") if not os.getenv(k)]

if _missing:
    st.error(
        f"Missing environment variables in your .env file: {', '.join(_missing)}"
    )
    st.stop()

from ai_workflow.analyze_document import analyze_document
from ai_workflow.legal_reasoning.qa import (
    answer_question,
    answer_question_multi,
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "documents" not in st.session_state:
    st.session_state.documents = {}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}

if "audit_log" not in st.session_state:
    st.session_state.audit_log = []

STATUS_PENDING = "Pending legal review"


def build_report(filename: str, doc: dict) -> str:
    """Build a shareable markdown review record for one document."""

    lines = [
        f"# Legal Review Record: {filename}",
        "",
        f"**Status:** {doc.get('review_status', STATUS_PENDING)}",
        "**Note:** System-generated analysis. Not a final legal determination "
        "until approved by an authorized reviewer.",
        "",
        "## Summary",
        doc.get("summary") or "_Not available._",
        "",
        "## Extracted Clauses",
        doc.get("extracted_clauses") or "_Not available._",
        "",
        "## Risk Flags",
        doc.get("risk_flags") or "_Not available._",
        "",
        "## Reviewer Decisions",
    ]

    entries = [
        e
        for e in st.session_state.audit_log
        if e["document"] == filename
    ]

    if entries:
        for e in entries:
            lines.append(
                f"- {e['timestamp']} | {e['reviewer']} | "
                f"{e['decision']} | {e['comment'] or '-'}"
            )
    else:
        lines.append("_No reviewer decisions recorded yet._")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sidebar: upload + document selection
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚖️ Tata Legal AI")
    st.caption("Contract summaries and risk flags, with human approval")

    uploaded_files = st.file_uploader(
        "Upload contracts",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if st.button(
        "Analyze uploaded documents",
        type="primary",
        disabled=not uploaded_files,
    ):
        for f in uploaded_files:
            with st.status(
                f"Analyzing {f.name}...",
                expanded=False,
            ) as status:

                try:
                    result = analyze_document(f)

                    result["review_status"] = STATUS_PENDING

                    st.session_state.documents[f.name] = result
                    st.session_state.chat_history.setdefault(
                        f.name,
                        [],
                    )

                    if result["blocked"]:
                        status.update(
                            label=f"{f.name}: completed with a warning",
                            state="error",
                        )
                    else:
                        status.update(
                            label=f"{f.name}: done",
                            state="complete",
                        )

                except Exception as e:
                    status.update(
                        label=f"{f.name}: failed",
                        state="error",
                    )

                    st.error(
                        f"Could not analyze {f.name}: {e}"
                    )

    selected = None

    if st.session_state.documents:
        st.divider()

        selected = st.selectbox(
            "Active document",
            list(st.session_state.documents.keys()),
        )


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
if not selected:
    st.header("Contract review workspace")

    st.info(
        "Upload one or more contracts in the sidebar and click "
        "**Analyze uploaded documents**."
    )

    st.stop()


doc = st.session_state.documents[selected]

head_col, status_col = st.columns([3, 1])

head_col.header(selected)

status_col.metric(
    "Review status",
    doc.get("review_status", STATUS_PENDING),
)

st.caption(
    "System-generated analysis. Pending approval by an authorized legal reviewer."
)

if doc["blocked"]:
    st.warning(doc["block_reason"])


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_summary, tab_clauses, tab_risk, tab_qa, tab_review = st.tabs(
    [
        "Summary",
        "Clauses",
        "Risk Flags",
        "Ask Questions",
        "Review & Audit",
    ]
)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
with tab_summary:

    if doc["summary"]:
        st.markdown(doc["summary"])
    else:
        st.write("No summary available for this document.")

    st.caption(
        f"{doc.get('chunks_stored', 0)} "
        "text chunks stored for question answering."
    )


# ---------------------------------------------------------------------------
# Clauses
# ---------------------------------------------------------------------------
with tab_clauses:

    if doc["extracted_clauses"]:
        st.markdown(doc["extracted_clauses"])
    else:
        st.write("No clauses extracted.")


# ---------------------------------------------------------------------------
# Risk Flags
# ---------------------------------------------------------------------------
with tab_risk:

    if doc["risk_flags"]:
        st.markdown(doc["risk_flags"])
    else:
        st.write("No risk flags available.")


# ---------------------------------------------------------------------------
# Q&A
# ---------------------------------------------------------------------------
with tab_qa:

    compare_mode = False
    compare_docs = []

    # Multi-PDF option remains unchanged
    if len(st.session_state.documents) > 1:

        compare_mode = st.checkbox(
            "Ask across multiple documents instead of just this one",
            key=f"compare_{selected}",
        )

        if compare_mode:
            compare_docs = st.multiselect(
                "Documents to include",
                options=list(st.session_state.documents.keys()),
                default=list(st.session_state.documents.keys()),
                key=f"compare_docs_{selected}",
            )

    if compare_mode:
        st.write(
            f"Ask a question across **{len(compare_docs)} documents**. "
            "Answers will attribute facts to the source document."
        )
    else:
        st.write(
            f"Ask a question about **{selected}**. "
            "Answers use only this document's text."
        )

    history = st.session_state.chat_history[selected]

    # Display previous chat messages
    for msg in history:

        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # New question
    question = st.chat_input(
        "e.g. What is the termination notice period?"
    )

    if question:

        history.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner(
                "Running security checks and searching..."
            ):

                try:

                    if compare_mode and compare_docs:

                        # Multi-PDF Q&A
                        res = answer_question_multi(
                            document_ids=compare_docs,
                            question=question,
                        )

                    else:

                        # Single-PDF Q&A
                        res = answer_question(
                            document_id=selected,
                            question=question,
                        )

                except Exception as e:

                    res = {
                        "blocked": True,
                        "block_reason": f"Error: {e}",
                        "answer": "",
                        "sources": [],
                    }

            # ---------------------------------------------------------------
            # Blocked response
            # ---------------------------------------------------------------
            if res["blocked"]:

                reply = f"🛑 {res['block_reason']}"

                st.markdown(reply)

            # ---------------------------------------------------------------
            # Successful response
            # ---------------------------------------------------------------
            else:

                reply = res["answer"]

                st.markdown(reply)

                # -----------------------------------------------------------
                # Page-level source citations
                # -----------------------------------------------------------
                with st.expander("📚 Source passages used"):

                    sources = res.get("sources", [])

                    if not sources:

                        st.info(
                            "No source passages were returned."
                        )

                    else:

                        for i, source_data in enumerate(
                            sources,
                            start=1,
                        ):

                            source = source_data.get(
                                "source",
                                "unknown",
                            )

                            page = source_data.get(
                                "page_number"
                            )

                            clause_type = source_data.get(
                                "clause_type",
                                "Source",
                            )

                            # Multi-PDF source display
                            if compare_mode:

                                if page is not None:

                                    st.markdown(
                                        f"**{i}. 📄 {source} "
                                        f"— Page {page} "
                                        f"— {clause_type}**"
                                    )

                                else:

                                    st.markdown(
                                        f"**{i}. 📄 {source} "
                                        f"— {clause_type}**"
                                    )

                            # Single-PDF source display
                            else:

                                if page is not None:

                                    st.markdown(
                                        f"**{i}. 📄 Page {page} "
                                        f"— {clause_type}**"
                                    )

                                else:

                                    st.markdown(
                                        f"**{i}. {clause_type}**"
                                    )

                            st.write(
                                source_data.get(
                                    "text",
                                    "",
                                )
                            )

        history.append(
            {
                "role": "assistant",
                "content": reply,
            }
        )


# ---------------------------------------------------------------------------
# Review & Audit
# ---------------------------------------------------------------------------
with tab_review:

    st.subheader("Reviewer decision")

    reviewer = st.text_input(
        "Reviewer name",
        key=f"reviewer_{selected}",
    )

    comment = st.text_area(
        "Comment (optional)",
        key=f"comment_{selected}",
    )

    b1, b2, b3 = st.columns(3)

    decision = None

    if b1.button(
        "✅ Accept",
        width="stretch",
    ):
        decision = "Accepted"

    if b2.button(
        "⬆️ Escalate to senior reviewer",
        width="stretch",
    ):
        decision = "Escalated"

    if b3.button(
        "❌ Reject",
        width="stretch",
    ):
        decision = "Rejected"

    if decision:

        if not reviewer.strip():

            st.error(
                "Enter a reviewer name so the decision can be recorded."
            )

        else:

            st.session_state.audit_log.append(
                {
                    "timestamp": datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "document": selected,
                    "reviewer": reviewer.strip(),
                    "decision": decision,
                    "comment": comment.strip(),
                }
            )

            doc["review_status"] = decision

            st.rerun()

    st.subheader("Audit trail")

    entries = [
        e
        for e in st.session_state.audit_log
        if e["document"] == selected
    ]

    if entries:

        st.dataframe(
            entries,
            width="stretch",
            hide_index=True,
        )

    else:

        st.write(
            "No decisions recorded yet."
        )

    st.download_button(
        "Download review record (.md)",
        data=build_report(selected, doc),
        file_name=f"{selected}_review_record.md",
        mime="text/markdown",
    )