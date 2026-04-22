#!/usr/bin/env python
"""
Full-text chunk viewer showing complete chunk content and all metadata.
"""

from db import DocumentDB
import json

db = DocumentDB()

docs = db.list_documents()
if not docs:
    print("No documents in database.")
    exit()

print("\n📚 Available Documents:\n")
for i, (doc_id, strategy, doc_type) in enumerate(docs):
    print(f"[{i}] {doc_id[:8]}... | Strategy: {strategy} | Type: {doc_type}")

choice = int(input("\nSelect document (number): "))
doc_id, strategy, doc_type = docs[choice]

print(f"\n{'='*80}")
print(f"DOCUMENT: {doc_id}")
print(f"Strategy: {strategy} | Type: {doc_type}")
print(f"{'='*80}\n")

cursor = db.conn.execute(
    "SELECT chunk_id, metadata FROM chunks WHERE doc_id=?",
    (doc_id,)
)

chunks = cursor.fetchall()
print(f"Total chunks: {len(chunks)}\n")

for row_idx, (chunk_id, metadata_json) in enumerate(chunks, 1):
    meta = json.loads(metadata_json)
    print(f"[{row_idx}/{len(chunks)}] {chunk_id}")
    print(f"     Topic: {meta.get('topic', 'N/A')}")
    print(f"     Pages: {meta.get('pages', [])}")
    
    is_fallback = meta.get('boundary_source') == 'llm_uncovered_fallback'
    if is_fallback:
        print(f"     🟡 FALLBACK CHUNK")
    print()

choice = int(input("Select chunk number to view (or 0 to exit): "))

if choice < 1 or choice > len(chunks):
    print("Exiting.")
    exit()

choice -= 1
chunk_id, metadata_json = chunks[choice]
meta = json.loads(metadata_json)

# Get the actual chunk text
cursor = db.conn.execute(
    "SELECT content FROM chunks WHERE chunk_id=?",
    (chunk_id,)
)
content_result = cursor.fetchone()
chunk_text = content_result[0] if content_result else ""

print(f"\n{'='*80}")
print(f"FULL CHUNK VIEW")
print(f"{'='*80}\n")

print(f"Chunk ID: {chunk_id}")
print(f"Length: {len(chunk_text)} characters\n")

print(f"{'─'*80}")
print("METADATA:")
print(f"{'─'*80}")

for key in sorted(meta.keys()):
    value = meta[key]
    
    if key == "coverage" and isinstance(value, dict):
        print(f"{key}:")
        for sub_key, sub_val in sorted(value.items()):
            print(f"  {sub_key}: {sub_val}")
    elif isinstance(value, (list, dict)):
        print(f"{key}: {json.dumps(value, indent=2)}")
    else:
        print(f"{key}: {value}")

print(f"\n{'─'*80}")
print("FULL TEXT:")
print(f"{'─'*80}\n")
print(chunk_text)
print(f"\n{'='*80}")
print(f"END OF CHUNK")
print(f"{'='*80}\n")
