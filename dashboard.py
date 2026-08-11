"""
Simple Vector Database Dashboard
Flask app for viewing and searching Chroma database

Run: python dashboard.py
Then visit: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lazy-load to avoid blocking startup
store = None
embedding_gen = None

def get_store():
    """Lazy-load vector store."""
    global store
    if store is None:
        logger.info("Initializing vector store...")
        from vector_db.vector_store import VectorStore
        store = VectorStore(persist_directory="./vector_db/data")
        logger.info("Vector store loaded")
    return store

def get_embedding_gen():
    """Lazy-load embedding generator."""
    global embedding_gen
    if embedding_gen is None:
        logger.info("Loading embedding model (this may take a moment)...")
        from embeddings.embeddings import EmbeddingGenerator
        embedding_gen = EmbeddingGenerator()
        logger.info("Embedding model loaded")
    return embedding_gen


# =====================================================================
# Routes
# =====================================================================

@app.route("/")
def index():
    """Home page."""
    return render_template("index.html")


@app.route("/api/stats")
def get_stats():
    """Get database statistics."""
    stats = get_store().get_stats()
    return jsonify(stats)


@app.route("/api/search", methods=["POST"])
def search():
    """Search database."""
    data = request.json
    query = data.get("query", "").strip()
    top_k = int(data.get("top_k", 5))

    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    try:
        # Generate embedding
        query_embedding = get_embedding_gen().embed([query])[0].tolist()

        # Search
        results = get_store().search(query_embedding, top_k=top_k)

        # Format results
        formatted_results = []
        for doc_id, distance, text, metadata in zip(
            results["ids"],
            results["distances"],
            results["documents"],
            results["metadatas"],
        ):
            similarity = 1 - distance
            formatted_results.append({
                "id": doc_id,
                "similarity": round(similarity, 3),
                "filename": metadata["filename"],
                "page": metadata["page_number"],
                "concepts": metadata["concept_names"],
                "confidence": round(metadata["max_confidence"], 3),
                "text": text[:200] + "..." if len(text) > 200 else text,
            })

        return jsonify({
            "query": query,
            "count": len(formatted_results),
            "results": formatted_results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/samples")
def get_samples():
    """Get sample documents."""
    limit = request.args.get("limit", 10, type=int)

    try:
        results = get_store().collection.get(
            limit=limit,
            include=["documents", "metadatas"]
        )

        formatted_samples = []
        for text, metadata in zip(results["documents"], results["metadatas"]):
            formatted_samples.append({
                "filename": metadata["filename"],
                "page": metadata["page_number"],
                "concepts": metadata["concept_names"],
                "confidence": round(metadata["max_confidence"], 3),
                "text": text[:150] + "..." if len(text) > 150 else text,
            })

        return jsonify({
            "count": len(formatted_samples),
            "samples": formatted_samples
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/files")
def get_files():
    """Get list of files."""
    try:
        results = get_store().collection.get(
            limit=10000,
            include=["metadatas"]
        )

        file_counts = {}
        for metadata in results["metadatas"]:
            filename = metadata["filename"]
            file_counts[filename] = file_counts.get(filename, 0) + 1

        files_list = [
            {"name": f, "chunks": c}
            for f, c in sorted(file_counts.items())
        ]

        return jsonify({
            "count": len(files_list),
            "files": files_list
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/file/<filename>")
def get_file_chunks(filename):
    """Get all chunks from a file."""
    try:
        results = get_store().collection.get(
            limit=10000,
            include=["documents", "metadatas"]
        )

        file_chunks = []
        for text, metadata in zip(results["documents"], results["metadatas"]):
            if metadata["filename"] == filename:
                file_chunks.append({
                    "page": metadata["page_number"],
                    "concepts": metadata["concept_names"],
                    "confidence": round(metadata["max_confidence"], 3),
                    "text": text[:150] + "..." if len(text) > 150 else text,
                })

        return jsonify({
            "filename": filename,
            "count": len(file_chunks),
            "chunks": file_chunks
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/concept/<concept>")
def filter_by_concept(concept):
    """Filter documents by concept."""
    limit = request.args.get("limit", 10, type=int)

    try:
        results = get_store().collection.get(
            limit=10000,
            include=["documents", "metadatas"]
        )

        concept_docs = []
        for text, metadata in zip(results["documents"], results["metadatas"]):
            if concept.lower() in metadata["concept_names"].lower():
                concept_docs.append({
                    "filename": metadata["filename"],
                    "page": metadata["page_number"],
                    "concepts": metadata["concept_names"],
                    "confidence": round(metadata["max_confidence"], 3),
                    "text": text[:150] + "..." if len(text) > 150 else text,
                })

        concept_docs = concept_docs[:limit]

        return jsonify({
            "concept": concept,
            "count": len(concept_docs),
            "documents": concept_docs
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/health")
def health():
    """Health check."""
    return jsonify({"status": "ok", "total_docs": get_store().collection.count()})


# =====================================================================
# Error handlers
# =====================================================================

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Server error"}), 500


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Vector Database Dashboard")
    print("=" * 70)
    print("\nStarting server...")
    print("Visit: http://localhost:5000")
    print("\nPress Ctrl+C to stop")
    print("=" * 70 + "\n")

    app.run(debug=True, host="0.0.0.0", port=5000)
