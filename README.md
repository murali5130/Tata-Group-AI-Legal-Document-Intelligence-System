
# Tata Group: AI Legal Document Intelligence System

An AI-powered system that reads legal contracts (MOA/AOA, vendor agreements), matches them against an approved clause library, and flags risks with page-level citations.

**Live Demo:** https://tata-group-legal-ai.streamlit.app/

**Demo Video:** https://drive.google.com/file/d/1YV6oscsqT1TOLkb-rQ1Wo33hHohwcZ5o/view?usp=sharing

**Sample Legal Document:** https://drive.google.com/file/d/1RV7TpoDgQQZ_tLYjxuEvtKqvkyzuwUdC/view?usp=sharing

## Problem Statement
Legal teams spend hours manually reviewing contracts to check whether clauses such as indemnity, liability caps, termination and data protection follow company policy. This project automates the first-pass review: it extracts clauses, compares them with an approved knowledge base, and highlights risks.

## Key Features
- Upload a PDF contract, including scanned documents (OCR fallback per page)
- Clause extraction and comparison against an approved clause library
- RAG-based retrieval over 22 clause categories (indemnity, limitation of liability, termination, payment, data protection, etc.)
- Risk flagging that adapts to document type (MOA/AOA vs vendor contracts)
- Answers with page citations, so every finding can be traced to the source
- Guardrails to keep outputs grounded in the approved knowledge base

## Tech Stack
| Layer | Tools |
|---|---|
| Language | Python |
| LLM | Groq SDK (gpt-oss-120b) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector DB | pgvector on Supabase |
| Document processing | pypdf, Tesseract OCR (pytesseract, pdf2image) |
| Frontend | Streamlit |
| Package management | uv |

## Architecture
```
PDF upload -> Parsing / OCR -> Clause extraction -> Embedding
          -> Vector search (clause library) -> LLM risk analysis -> Streamlit UI
```

## Knowledge Base
The clause library in `data/clause_library/` is organised into three groups:
- **Core approved knowledge:** indemnity, liability, confidentiality, termination, renewal, payment, data protection, governing law, dispute resolution, audit rights, compliance
- **Supporting knowledge:** IP, vendor management, insurance, performance, force majeure, trade and sanctions, general provisions, brand, workplace governance
- **Regulatory knowledge:** DPDP Act, Prevention of Corruption Act, FCPA

## Project Structure
```
ai_workflow/        # RAG retrieval, legal reasoning, risk flagging
document_pipeline/  # parsing, OCR, clause extraction
guardrails/         # output safety and grounding checks
data/clause_library/  # approved clause knowledge base
main2.py            # app entry point
```

## How to Run
```bash
git clone https://github.com/murali5130/Tata-Group-AI-Legal-Document-Intelligence-System.git
cd Tata-Group-AI-Legal-Document-Intelligence-System
uv sync
# add GROQ_API_KEY and database URL in a .env file
streamlit run main2.py
```

## Screenshots
<!-- Add 2-3 screenshots of the app in an images/ folder -->
![Upload screen](images/upload.png)
![Risk analysis output](images/risk-analysis.png)

## My Contribution
<!-- 2-3 lines about what YOU built, e.g. clause library, OCR pipeline, prompts -->

## Team
Murali Krishna ([LinkedIn](your-link)) and teammate

## License
MIT
