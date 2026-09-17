# Daraz Knowledge Base

Directory layout:

daraz_knowledge_base/
├── return/
├── delivery/
├── refund/
├── seller/
├── pyments/
├── customer_support/
├── faiss_index/              # created by ingest.py
├── ingest.py
├── download_official_policy_snapshots.py
├── requirements.txt
├── source_manifest.csv
└── COLAB_CELLS.md

The included PDFs are concise knowledge-base summaries. The script
download_official_policy_snapshots.py can create fresh PDF snapshots from
Daraz Pakistan's public policy/help pages in an internet-enabled environment.

FAISS metadata fields:
- id
- department
- source_file
- chunk_index
- text

The FAISS index uses normalized Sentence-Transformer embeddings and
IndexFlatIP, which gives cosine similarity when vectors are normalized.
