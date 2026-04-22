import sqlite3

# Path to your database file
db_path = "backend_app.db"

# Connect to the database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get all table names
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

print("Tables in database:\n")

for table in tables:
    table_name = table[0]
    print(f"Table: {table_name}")
    
    # Get column info
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns = cursor.fetchall()
    
    print("Columns:")
    for col in columns:
        col_id, name, dtype, notnull, default, pk = col
        print(f"  - {name} ({dtype})")
    
    print("-" * 40)

# Close connection
conn.close()