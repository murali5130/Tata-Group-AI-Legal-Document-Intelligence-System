
"""
frontend/streamlit_app.py

Streamlit UI for the Tata AI Legal Document Intelligence System.

Run from the PROJECT ROOT with:
    uv run streamlit run frontend/streamlit_app.py

Screens (as tabs, per the project spec):
    Summary | Clauses | Risk Flags | Ask Questions | Review & Audit

This file contains UI code only. All real work happens in:
    ai_workflow.analyze_document.analyze_document()
    ai_workflow.legal_reasoning.qa.answer_question()
"""

import os
import sys
import traceback
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Tata Legal AI", page_icon="⚖️", layout="wide")

_missing = [k for k in ("GROQ_API_KEY", "SUPABASE_DB_URL") if not os.getenv(k)]
if _missing:
    st.error(f"Missing environment variables in your .env file: {', '.join(_missing)}")
    st.stop()

from ai_workflow.analyze_document import analyze_document
from ai_workflow.legal_reasoning.qa import answer_question

if "documents" not in st.session_state:
    st.session_state.documents = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}
if "audit_log" not in st.session_state:
    st.session_state.audit_log = []

STATUS_PENDING = "Pending legal review"


def build_report(filename: str, doc: dict) -> str:
    """Builds a shareable markdown review record for one document."""
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
    entries = [e for e in st.session_state.audit_log if e["document"] == filename]
    if entries:
        for e in entries:
            lines.append(
                f"- {e['timestamp']} | {e['reviewer']} | {e['decision']} | {e['comment'] or '-'}"
            )
    else:
        lines.append("_No reviewer decisions recorded yet._")
    return "\n".join(lines)


with st.sidebar:
    st.title("⚖️ Tata Legal AI")
    st.caption("Contract summaries and risk flags, with human approval")

    uploaded_files = st.file_uploader(
        "Upload contracts", type=["pdf", "txt"], accept_multiple_files=True
    )

    if st.button("Analyze uploaded documents", type="primary", disabled=not uploaded_files):
        for f in uploaded_files:
            with st.status(f"Analyzing {f.name}...", expanded=False) as status:
                try:
                    result = analyze_document(f)
                    result["review_status"] = STATUS_PENDING
                    st.session_state.documents[f.name] = result
                    st.session_state.chat_history.setdefault(f.name, [])

                    if result["blocked"]:
                        status.update(label=f"{f.name}: completed with a warning", state="error")
                    else:
                        status.update(label=f"{f.name}: done", state="complete")

                except Exception as e:
                    error_details = traceback.format_exc()

                    status.update(label=f"{f.name}: failed", state="error")

                    st.error(
                        f"Could not analyze {f.name}: "
                        f"{type(e).__name__}: {e}"
                    )
                    st.code(error_details, language="text")
                    print(f"\n[DOCUMENT ANALYSIS ERROR: {f.name}]\n{error_details}")

    selected = None
    if st.session_state.documents:
        st.divider()
        selected = st.selectbox("Active document", list(st.session_state.documents.keys()))


if not selected:
    st.header("Contract review workspace")
    st.info("Upload one or more contracts in the sidebar and click **Analyze uploaded documents**.")
    st.stop()

doc = st.session_state.documents[selected]

head_col, status_col = st.columns([3, 1])
head_col.header(selected)
status_col.metric("Review status", doc.get("review_status", STATUS_PENDING))
st.caption("System-generated analysis. Pending approval by an authorized legal reviewer.")

if doc["blocked"]:
    st.warning(doc["block_reason"])

tab_summary, tab_clauses, tab_risk, tab_qa, tab_review = st.tabs(
    ["Summary", "Clauses", "Risk Flags", "Ask Questions", "Review & Audit"]
)

with tab_summary:
    if doc["summary"]:
        st.markdown(doc["summary"])
    else:
        st.write("No summary available for this document.")
    st.caption(f"{doc.get('chunks_stored', 0)} text chunks stored for question answering.")

with tab_clauses:
    if doc["extracted_clauses"]:
        st.markdown(doc["extracted_clauses"])
    else:
        st.write("No clauses extracted.")

with tab_risk:
    if doc["risk_flags"]:
        st.markdown(doc["risk_flags"])
    else:
        st.write("No risk flags available.")

with tab_qa:
    st.write(f"Ask a question about **{selected}**. Answers use only this document's text.")
    history = st.session_state.chat_history[selected]

    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if question := st.chat_input("e.g. What is the termination notice period?"):
        history.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Running security checks and searching the document..."):
                try:
                    res = answer_question(document_id=selected, question=question)

                except Exception as e:
                    error_details = traceback.format_exc()
                    print(f"\n[QUESTION ANSWER ERROR]\n{error_details}")

                    res = {
                        "blocked": True,
                        "block_reason": f"{type(e).__name__}: {e}",
                        "answer": "",
                        "sources": [],
                    }

            if res["blocked"]:
                reply = f"🛑 {res['block_reason']}"
                st.markdown(reply)
            else:
                reply = res["answer"]
                st.markdown(reply)
                with st.expander("Source passages used"):
                    for i, s in enumerate(res["sources"], 1):
                        st.markdown(f"**{i}.** {s['text']}")

        history.append({"role": "assistant", "content": reply})

with tab_review:
    st.subheader("Reviewer decision")
    reviewer = st.text_input("Reviewer name", key=f"reviewer_{selected}")
    comment = st.text_area("Comment (optional)", key=f"comment_{selected}")

    b1, b2, b3 = st.columns(3)
    decision = None

    if b1.button("✅ Accept", width="stretch"):
        decision = "Accepted"
    if b2.button("⬆️ Escalate to senior reviewer", width="stretch"):
        decision = "Escalated"
    if b3.button("❌ Reject", width="stretch"):
        decision = "Rejected"

    if decision:
        if not reviewer.strip():
            st.error("Enter a reviewer name so the decision can be recorded.")
        else:
            st.session_state.audit_log.append(
                {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "document": selected,
                    "reviewer": reviewer.strip(),
                    "decision": decision,
                    "comment": comment.strip(),
                }
            )
            doc["review_status"] = decision
            st.rerun()

    st.subheader("Audit trail")
    entries = [e for e in st.session_state.audit_log if e["document"] == selected]

    if entries:
        st.dataframe(entries, width="stretch", hide_index=True)
    else:
        st.write("No decisions recorded yet.")

    st.download_button(
        "Download review record (.md)",
        data=build_report(selected, doc),
        file_name=f"{selected}_review_record.md",
        mime="text/markdown",
    )