#!/usr/bin/env python
"""
RAG Dashboard Quick Start
One-step initialization and verification of the RAG pipeline.
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check required packages are installed."""
    print("\n📦 Checking dependencies...")
    
    required = [
        "chromadb",
        "sentence_transformers",
        "streamlit",
        "pandas",
        "plotly",
        "sklearn",
        "numpy",
    ]
    
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ✗ {pkg}")
            missing.append(pkg)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print(f"   Install with: pip install -r requirements-vector-db.txt")
        return False
    
    return True


def check_data():
    """Check if data artifacts exist."""
    print("\n📂 Checking data...")
    
    artifacts_dir = Path("co2m/artifacts")
    required_files = [
        "documents.parquet",
        "chunks.parquet",
        "concepts.parquet",
    ]
    
    if not artifacts_dir.exists():
        print(f"  ✗ Artifacts directory not found: {artifacts_dir}")
        return False
    
    missing = []
    for fname in required_files:
        fpath = artifacts_dir / fname
        if fpath.exists():
            size_mb = fpath.stat().st_size / (1024 * 1024)
            print(f"  ✓ {fname} ({size_mb:.1f} MB)")
        else:
            print(f"  ✗ {fname}")
            missing.append(fname)
    
    if missing:
        print(f"\n❌ Missing artifacts: {', '.join(missing)}")
        print("   Run: python ingest_enhanced.py")
        return False
    
    return True


def check_vector_db():
    """Check if vector database is populated."""
    print("\n🗄️  Checking vector database...")
    
    db_path = Path("vector_db/data")
    
    if not db_path.exists():
        print(f"  ⚠️  Vector database not initialized: {db_path}")
        print("   Run: python ingest_enhanced.py")
        return False
    
    print(f"  ✓ Vector database exists at {db_path}")
    
    # Try to connect
    try:
        from vector_db.vector_store import VectorStore
        store = VectorStore()
        count = store.collection.count()
        print(f"  ✓ Connected to Chroma collection")
        print(f"  ✓ Documents indexed: {count:,}")
        
        if count == 0:
            print(f"\n❌ Vector database is empty!")
            print("   Run: python ingest_enhanced.py")
            return False
        
        return True
    
    except Exception as e:
        print(f"  ✗ Error connecting to vector database: {e}")
        return False


def test_rag():
    """Test RAG pipeline with a sample query."""
    print("\n🧪 Testing RAG pipeline...")
    
    try:
        from rag_pipeline import RAGQueryService
        
        service = RAGQueryService()
        stats = service.get_stats()
        print(f"  ✓ RAG Service initialized")
        print(f"  ✓ Documents: {stats.get('total_documents', 0):,}")
        
        # Test query
        print(f"  ⏳ Executing test query...")
        results = service.query(
            question="CO2 capture materials",
            top_k=3,
            include_context=False,
        )
        
        if results["num_results"] > 0:
            print(f"  ✓ Query successful")
            print(f"  ✓ Found {results['num_results']} results")
            print(f"  ✓ Top result: {results['results'][0]['filename']} ({results['results'][0]['similarity']:.1%})")
            return True
        else:
            print(f"  ⚠️  Query returned no results")
            return False
    
    except Exception as e:
        print(f"  ✗ RAG test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all checks and show status."""
    print("=" * 70)
    print("RAG PIPELINE QUICK START")
    print("=" * 70)
    
    all_ok = True
    
    # Run checks
    all_ok = check_dependencies() and all_ok
    all_ok = check_data() and all_ok
    all_ok = check_vector_db() and all_ok
    all_ok = test_rag() and all_ok
    
    # Summary
    print("\n" + "=" * 70)
    
    if all_ok:
        print("✅ ALL CHECKS PASSED")
        print("\n🚀 Ready to use RAG Dashboard!")
        print("\nStart dashboard with:")
        print("   streamlit run dashboard/app.py")
        print("\nThen navigate to the '🔍 RAG Search' tab")
    else:
        print("❌ SOME CHECKS FAILED")
        print("\nPlease fix the issues above and try again.")
        print("\nFull setup:")
        print("   1. pip install -r requirements-vector-db.txt")
        print("   2. python ingest_enhanced.py")
        print("   3. python rag_dashboard_quickstart.py")
        print("   4. streamlit run dashboard/app.py")
        return 1
    
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
