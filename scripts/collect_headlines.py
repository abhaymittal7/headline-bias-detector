"""
Grow the headline dataset to a target size, then print dataset statistics.

Usage (from the project root):
    python -m scripts.collect_headlines              (target 5000)
    python -m scripts.collect_headlines 8000         (custom target)
    python -m scripts.collect_headlines --stats      (statistics only, offline)

After collecting, run `python main.py` (or run.bat) so the new headlines go
through NLP, sentiment, features, bias scoring and clustering.
"""
import sys

from src.utils.db import create_table, get_connection
from src.ingestion.fetch_headlines import run_ingestion
from src.ingestion.fetch_gnews import run_gnews_ingestion


def print_stats():
    conn = get_connection()
    total = conn.execute("SELECT COUNT(*) FROM headlines").fetchone()[0]
    processed = conn.execute("SELECT COUNT(*) FROM headlines WHERE bias IS NOT NULL").fetchone()[0]
    by_source = conn.execute("""
        SELECT source_name, source_known_lean, COUNT(*) FROM headlines
        GROUP BY source_name ORDER BY COUNT(*) DESC
    """).fetchall()
    by_lean = conn.execute("""
        SELECT source_known_lean, COUNT(*) FROM headlines
        GROUP BY source_known_lean ORDER BY COUNT(*) DESC
    """).fetchall()
    conn.close()

    print("\n" + "=" * 55)
    print(" DATASET STATISTICS")
    print("=" * 55)
    print(f"Total headlines        : {total}")
    print(f"Processed by pipeline  : {processed}")
    if processed < total:
        print(f"Not yet processed      : {total - processed}  -> run main.py")
    print(f"\n{'Source':<20} {'Lean':<14} {'Headlines':>9}")
    print("-" * 45)
    for name, lean, n in by_source:
        print(f"{name:<20} {lean or '-':<14} {n:>9}")
    print(f"\n{'Lean':<14} {'Headlines':>9}")
    print("-" * 25)
    for lean, n in by_lean:
        print(f"{lean or '-':<14} {n:>9}")
    print("=" * 55)


def main():
    create_table()

    if "--stats" in sys.argv:
        print_stats()
        return

    target = 5000
    for arg in sys.argv[1:]:
        if arg.isdigit():
            target = int(arg)

    print("[1/2] Regular RSS feeds...")
    run_ingestion()

    print("\n[2/2] Google News search (same outlets, more headlines)...")
    run_gnews_ingestion(target=target)

    print_stats()


if __name__ == "__main__":
    main()
