import argparse
import os
import sqlite3
from pathlib import Path


def _confirm() -> bool:
    answer = input("This will permanently clear documents/chunks and index files. Continue? [y/N]: ").strip().lower()
    return answer in {"y", "yes"}


def _collect_index_files(db_conn: sqlite3.Connection, indexes_dir: Path) -> list[Path]:
    files_to_delete: list[Path] = []

    cursor = db_conn.cursor()
    cursor.execute("SELECT index_path FROM documents WHERE index_path IS NOT NULL")
    rows = cursor.fetchall()

    for (index_path,) in rows:
        if not index_path:
            continue

        base = Path(index_path)
        candidates = [
            Path(f"{base}.index"),
            Path(f"{base}.pkl"),
        ]

        for candidate in candidates:
            if candidate.exists():
                files_to_delete.append(candidate)
                continue

            fallback = indexes_dir / candidate.name
            if fallback.exists():
                files_to_delete.append(fallback)

    if indexes_dir.exists():
        for file_path in indexes_dir.glob("*.index"):
            if file_path not in files_to_delete:
                files_to_delete.append(file_path)
        for file_path in indexes_dir.glob("*.pkl"):
            if file_path not in files_to_delete:
                files_to_delete.append(file_path)

    return files_to_delete


def clear_databases(db_path: Path, indexes_dir: Path, force: bool = False) -> None:
    if not force and not _confirm():
        print("Cancelled.")
        return

    if not db_path.exists():
        print(f"Database file not found: {db_path}")
        print("Index cleanup will still be attempted.")
        conn = None
    else:
        conn = sqlite3.connect(db_path)

    files_to_delete: list[Path] = []

    try:
        if conn is not None:
            files_to_delete = _collect_index_files(conn, indexes_dir)

            cursor = conn.cursor()
            cursor.execute("DELETE FROM chunks")
            cursor.execute("DELETE FROM documents")
            conn.commit()

            print("Cleared SQLite tables: documents, chunks")
        else:
            if indexes_dir.exists():
                files_to_delete.extend(indexes_dir.glob("*.index"))
                files_to_delete.extend(indexes_dir.glob("*.pkl"))

        deleted_count = 0
        for file_path in files_to_delete:
            try:
                file_path.unlink(missing_ok=True)
                deleted_count += 1
            except Exception as exc:
                print(f"Could not delete {file_path}: {exc}")

        print(f"Deleted index files: {deleted_count}")

    finally:
        if conn is not None:
            conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clear project databases (SQLite rows and saved index files)."
    )
    parser.add_argument("--db-path", default="documents.db", help="Path to SQLite database file")
    parser.add_argument("--indexes-dir", default="indexes", help="Directory containing saved index files")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")

    args = parser.parse_args()

    clear_databases(
        db_path=Path(args.db_path),
        indexes_dir=Path(args.indexes_dir),
        force=args.yes,
    )


if __name__ == "__main__":
    main()
