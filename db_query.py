# import sqlite3

# db_path = "documents.db"

# conn = sqlite3.connect(db_path)
# cursor = conn.cursor()

# # Disable foreign keys temporarily (optional but safer)
# cursor.execute("PRAGMA foreign_keys = OFF;")

# # Get all user tables
# cursor.execute("""
# SELECT name FROM sqlite_master 
# WHERE type='table' AND name NOT LIKE 'sqlite_%';
# """)
# tables = cursor.fetchall()

# # Delete all rows
# for table in tables:
#     cursor.execute(f"DELETE FROM {table[0]};")

# # Check if sqlite_sequence exists
# cursor.execute("""
# SELECT name FROM sqlite_master 
# WHERE type='table' AND name='sqlite_sequence';
# """)

# if cursor.fetchone():
#     cursor.execute("DELETE FROM sqlite_sequence;")

# conn.commit()

# # Re-enable foreign keys
# cursor.execute("PRAGMA foreign_keys = ON;")

# conn.close()

# print("Database cleared successfully.")



import sqlite3 
db_path = "backend_app.db" 
conn = sqlite3.connect(db_path) 
cursor = conn.cursor() 
# FIX: wrap UUID in quotes 
query = "DROP TABLE study_plans_api;"
cursor.execute(query) 
rows = cursor.fetchall() 
for row in rows: 
    print(row) 

conn.close()