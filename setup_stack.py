#!/usr/bin/env python
"""
Launch the CO2M dashboard after validating the local data and database stack.

This project uses an embedded Chroma database persisted on disk, so there is no
separate database server to start. This launcher verifies the required files,
opens the Chroma connection, reports optional RAG import issues, and then runs
the Streamlit app.
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path
from subprocess import Popen, PIPE


# Suppress torchaudio DLL load errors on Windows
# This allows sentence_transformers to import even when the native extension is missing
if sys.platform == "win32":
    import os
    os.environ["TORCHAUDIO_USE_TORCH_ONLY"] = "1"


PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_ROOT / "co2m" / "artifacts"
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db" / "data"

CORE_MODULES = [
    "streamlit",
    "chromadb",
    "pandas",
    "plotly",
    "numpy",
    "sklearn",
]

OPTIONAL_RAG_MODULES = [
    "sentence_transformers",
    "torch",
]

REQUIRED_ARTIFACTS = [
    "documents.parquet",
    "chunks.parquet",
    "concepts.parquet",
    "summaries.parquet",
    "concept_counts.parquet",
    "top_documents_by_concept.parquet",
    "keywords_by_concept.parquet",
]


def print_header(title: str) -> None:
    print("=" * 70)
    print(title)
    print("=" * 70)


def load_module(module_name: str) -> tuple[bool, str | None]:
    try:
        importlib.import_module(module_name)
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def check_modules(module_names: list[str], heading: str, required: bool) -> tuple[bool, dict[str, str]]:
    print(f"\n{heading}")
    failures: dict[str, str] = {}

    for module_name in module_names:
        ok, error = load_module(module_name)
        if ok:
            print(f"  OK  {module_name}")
        else:
            level = "ERROR" if required else "WARN"
            print(f"  {level} {module_name}")
            failures[module_name] = error or "Unknown import failure"

    return not failures if required else True, failures


def check_artifacts() -> bool:
    print("\nArtifacts")

    if not ARTIFACTS_DIR.exists():
        print(f"  ERROR Missing artifacts directory: {ARTIFACTS_DIR}")
        return False

    missing = [name for name in REQUIRED_ARTIFACTS if not (ARTIFACTS_DIR / name).exists()]
    if missing:
        print("  ERROR Missing required artifact files:")
        for name in missing:
            print(f"    - {name}")
        return False

    try:
        import pandas as pd

        sample = pd.read_parquet(ARTIFACTS_DIR / "documents.parquet")
        print(f"  OK  documents.parquet readable ({len(sample):,} rows)")
    except Exception as exc:
        print(f"  ERROR Unable to read parquet artifacts: {type(exc).__name__}: {exc}")
        return False

    return True


def build_database() -> bool:
    print("\nDatabase Build")
    command = [sys.executable, "ingest_enhanced.py"]
    print(f"  Running: {' '.join(command)}")
    result = subprocess.run(command, cwd=PROJECT_ROOT)
    return result.returncode == 0


def check_vector_db() -> bool:
    print("\nDatabase Connection")

    if not VECTOR_DB_DIR.exists():
        print(f"  ERROR Missing vector database directory: {VECTOR_DB_DIR}")
        return False

    try:
        from vector_db.vector_store import VectorStore

        store = VectorStore(persist_directory=str(VECTOR_DB_DIR))
        count = store.collection.count()
        print(f"  OK  Chroma persistent store opened at {VECTOR_DB_DIR}")
        print(f"  OK  Connected to collection '{store.collection_name}' with {count:,} documents")
        if count == 0:
            print("  ERROR Vector database is empty")
            return False
        return True
    except Exception as exc:
        print(f"  ERROR Failed to connect to vector database: {type(exc).__name__}: {exc}")
        return False


def check_rag_connection() -> bool:
    print("\nOptional RAG Connection")

    try:
        from rag_pipeline.query_service import RAGQueryService

        service = RAGQueryService(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            vector_store_path=str(VECTOR_DB_DIR),
            collection_name="co2m_corpus",
            device="cpu",
        )
        stats = service.get_stats()
        print(f"  OK  RAG query service initialized ({stats.get('total_documents', 0):,} indexed documents)")
        return True
    except OSError as exc:
        # Handle Windows DLL/native extension errors from torchaudio
        if "WinError" in str(exc) or "procedure could not be found" in str(exc):
            print(f"  WARN RAG service unavailable due to missing native dependency (Windows)")
            print(f"  WARN The dashboard can still start, but semantic search may be disabled.")
            print(f"  INFO To fix: pip install --upgrade --force-reinstall torchaudio")
            return False
        raise
    except Exception as exc:
        print(f"  WARN RAG service unavailable: {type(exc).__name__}: {exc}")
        print("  WARN The dashboard can still start, but semantic search may be disabled until the embedding stack is repaired.")
        return False


def launch_dashboard(host: str, port: int) -> int:
    print("\nServer Launch")
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "dashboard/app.py",
        "--server.address",
        host,
        "--server.port",
        str(port),
        "--server.headless",
        "true",
    ]
    print(f"  Running: {' '.join(command)}")
    print(f"  Open: http://{host}:{port}")
    print(f"\n  Dashboard is running.")
    print(f"  Press Ctrl+C in the terminal to stop.\n")
    # Use Popen to start as a background process with proper output handling
    try:
        process = Popen(
            command, 
            cwd=PROJECT_ROOT,
            stdout=None,  # Inherit stdout to see Streamlit logs
            stderr=None   # Inherit stderr to see Streamlit errors
        )
        # Wait for the process, allowing Ctrl+C to interrupt
        return process.wait()
    except KeyboardInterrupt:
        print("\n\nStopping dashboard...")
        try:
            process.terminate()
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the CO2M dashboard stack and launch the Streamlit server."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host address for the Streamlit server")
    parser.add_argument("--port", type=int, default=8501, help="Port for the Streamlit server")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Run validation checks without launching the Streamlit server",
    )
    parser.add_argument(
        "--build-db",
        action="store_true",
        help="Run ingest_enhanced.py before validating the vector database",
    )
    parser.add_argument(
        "--build-db-if-missing",
        action="store_true",
        help="Run ingest_enhanced.py only when the vector database directory is missing",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print_header("CO2M Stack Setup")
    print(f"Project root: {PROJECT_ROOT}")
    print("Database mode: Embedded Chroma persistent store")

    core_ok, core_failures = check_modules(CORE_MODULES, "Core Dependencies", required=True)
    if not core_ok:
        print("\nInstall the missing core dependencies before launching the dashboard.")
        for module_name, error in core_failures.items():
            print(f"  - {module_name}: {error}")
        print("\nSuggested command: pip install -r requirements-vector-db.txt")
        return 1

    _, optional_failures = check_modules(OPTIONAL_RAG_MODULES, "Optional RAG Dependencies", required=False)
    if optional_failures:
        for module_name, error in optional_failures.items():
            print(f"  WARN {module_name}: {error}")

    artifacts_ok = check_artifacts()
    if not artifacts_ok:
        print("\nBuild the artifact set before launching the dashboard.")
        return 1

    if args.build_db or (args.build_db_if_missing and not VECTOR_DB_DIR.exists()):
        if not build_database():
            print("\nDatabase build failed.")
            return 1

    vector_db_ok = check_vector_db()
    rag_ok = check_rag_connection()

    print("\nSummary")
    print(f"  Artifacts: {'OK' if artifacts_ok else 'ERROR'}")
    print(f"  Vector DB: {'OK' if vector_db_ok else 'ERROR'}")
    print(f"  RAG Service: {'OK' if rag_ok else 'WARN'}")

    if not vector_db_ok:
        print("\nThe dashboard was not launched because the vector database connection failed.")
        return 1

    if args.check_only:
        print("\nCheck-only mode complete. Server not started.")
        return 0

    return launch_dashboard(args.host, args.port)


if __name__ == "__main__":
    sys.exit(main())