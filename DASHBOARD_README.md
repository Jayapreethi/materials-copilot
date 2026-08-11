# Vector Database Dashboard

A simple, interactive web dashboard for viewing and searching the CO2M Chroma vector database.

## Features

- **📊 Database Statistics** - View total documents, files, and collection info
- **🔍 Semantic Search** - Search documents by meaning with similarity scores
- **📁 File Browser** - Browse and view all documents in the database
- **🏷️ Concept Filtering** - Filter documents by CO2M concepts
- **⭐ Sample Viewer** - See random sample documents
- **📱 Responsive Design** - Works on desktop and mobile

## Installation

### 1. Install Flask (if not already installed)

```bash
pip install flask
```

Or with the existing environment:

```bash
# Inside venv
pip install flask
```

### 2. Add Flask to requirements

Add this to `requirements-vector-db.txt`:
```
flask>=2.3.0
```

## Running the Dashboard

### From command line:

```bash
# Activate venv first
python dashboard.py
```

Then open your browser: **http://localhost:5000**

### From VS Code:

1. Open terminal
2. Run: `python dashboard.py`
3. Click the URL in the terminal or navigate to `http://localhost:5000`

## Dashboard Sections

### Left Column: Search
- Enter semantic queries
- Adjust number of results (1-50)
- View results with similarity scores and file info

### Right Column: Navigation

#### Samples Tab
- View 10 random documents
- See file name, page, concepts, and text preview

#### Files Tab
- Browse all files in database
- See chunk count for each file
- Click to view all chunks from that file

#### Concepts Tab
- Enter a concept name (e.g., "Carbon capture")
- Filter documents containing that concept
- View up to 15 matching documents

## API Endpoints

The dashboard uses these API endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/stats` | GET | Database statistics |
| `/api/search` | POST | Semantic search |
| `/api/samples` | GET | Random sample documents |
| `/api/files` | GET | List all files |
| `/api/file/<filename>` | GET | Get chunks from specific file |
| `/api/concept/<concept>` | GET | Filter by concept |
| `/api/health` | GET | Health check |

### Example API Usage

```bash
# Get stats
curl http://localhost:5000/api/stats

# Search
curl -X POST http://localhost:5000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "carbon capture", "top_k": 5}'

# Get samples
curl http://localhost:5000/api/samples?limit=10

# Get files
curl http://localhost:5000/api/files

# Filter by concept
curl http://localhost:5000/api/concept/Carbon%20capture?limit=10
```

## File Structure

```
project-root/
├── dashboard.py              # Flask app
├── templates/
│   └── index.html           # Main HTML
└── static/
    ├── style.css            # Styling
    └── script.js            # Frontend logic
```

## Customization

### Change Port

Edit `dashboard.py` line at the bottom:

```python
app.run(debug=True, host="0.0.0.0", port=8000)  # Change 5000 to 8000
```

### Change Host

For network access:

```python
app.run(debug=True, host="0.0.0.0", port=5000)  # Access from other machines
```

For localhost only:

```python
app.run(debug=True, host="127.0.0.1", port=5000)
```

## Troubleshooting

### Dashboard won't load
- Check if Flask is installed: `pip install flask`
- Ensure port 5000 is not in use
- Try a different port: change in `dashboard.py`

### Search not working
- Ensure vector database is built: `python examples/example_rag_pipeline.py`
- Check that `./vector_db/data` directory exists
- Verify embeddings are loaded correctly

### Slow performance
- Reduce `top_k` value for faster results
- Limit concept filter results
- Use smaller sample sizes

## Integration with Existing Dashboard

To add this to the CO2M dashboard:

```python
# In co2m_lit_dashboard/dashboard/app.py
from dashboard import app as vector_db_app

# Register blueprint
app.register_blueprint(vector_db_app)
```

Or run both dashboards on different ports:

```bash
# Terminal 1
python dashboard.py

# Terminal 2
cd co2m_lit_dashboard
python -m src.dashboard
```

## Performance Notes

- Database has 3,689 documents
- Searches return results in < 500ms
- Concept filtering may be slower for large datasets
- Sample loading is instant

## Future Enhancements

- [ ] Batch document upload
- [ ] Export search results
- [ ] Advanced filtering options
- [ ] Query history
- [ ] Batch operations
- [ ] Document management UI
- [ ] Real-time indexing status

## License

Same as parent project
