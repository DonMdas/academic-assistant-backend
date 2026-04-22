# inspect.py

from db import DocumentDB
import json

db = DocumentDB()

doc_id = "a94a76e5-d4e7-4a2d-b33a-d08fb3298b04"  # your ID

cursor = db.conn.execute(
    "SELECT chunk_id, metadata FROM chunks WHERE doc_id=?",
    (doc_id,)
)

print("\n📚 Chunk Details:\n")

for row in cursor.fetchall():
    chunk_id, metadata = row
    meta = json.loads(metadata)

    print(f"🧩 {chunk_id}")
    print(f"   Topic: {meta.get('topic')}")
    print(f"   Subtopics: {meta.get('subtopics')}")
    print(f"   Complexity: {meta.get('complexity')}")
    print(f"   Time: {meta.get('estimated_time')}")
    print("-" * 50)
    
    