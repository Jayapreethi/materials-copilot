// Vector Database Dashboard JavaScript

const API_BASE = "/api";
const loading = document.getElementById("loading");

// =====================================================================
// Utility Functions
// =====================================================================

function showLoading(show = true) {
    loading.classList.toggle("active", show);
}

function formatSimilarity(score) {
    if (score >= 0.7) return "high";
    if (score >= 0.5) return "medium";
    return "low";
}

// =====================================================================
// Initialize Dashboard
// =====================================================================

async function initializeDashboard() {
    try {
        // Load stats
        const statsRes = await fetch(`${API_BASE}/stats`);
        const stats = await statsRes.json();
        
        document.getElementById("total-docs").textContent = stats.total_documents.toLocaleString();
        
        // Load files count
        const filesRes = await fetch(`${API_BASE}/files`);
        const filesData = await filesRes.json();
        document.getElementById("total-files").textContent = filesData.count;
        
        // Load samples
        loadSamples();
        
        // Load files
        loadFiles();
        
    } catch (error) {
        console.error("Error initializing dashboard:", error);
    }
}

// =====================================================================
// Search Functionality
// =====================================================================

document.getElementById("search-btn").addEventListener("click", performSearch);
document.getElementById("search-input").addEventListener("keypress", (e) => {
    if (e.key === "Enter") performSearch();
});

async function performSearch() {
    const query = document.getElementById("search-input").value.trim();
    const topK = parseInt(document.getElementById("top-k").value);
    
    if (!query) {
        alert("Please enter a search query");
        return;
    }
    
    showLoading(true);
    
    try {
        const response = await fetch(`${API_BASE}/search`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                query: query,
                top_k: topK
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            displaySearchResults(data);
        } else {
            showError(data.error || "Search failed");
        }
    } catch (error) {
        showError(error.message);
    } finally {
        showLoading(false);
    }
}

function displaySearchResults(data) {
    const container = document.getElementById("results-container");
    const header = document.getElementById("results-header");
    
    header.innerHTML = `<i class="fas fa-list"></i> Results (${data.count})`;
    
    if (data.count === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-search"></i>
                <p>No results found for "${data.query}"</p>
                <small>Try a different query</small>
            </div>
        `;
        return;
    }
    
    let html = "";
    data.results.forEach((result, index) => {
        const similarityClass = formatSimilarity(result.similarity);
        const conceptBadges = result.concepts
            .split(",")
            .map(c => `<span class="concept-badge">${c.trim()}</span>`)
            .join("");
        
        html += `
            <div class="result-item">
                <div>
                    <span class="result-similarity ${similarityClass}">
                        ★ ${(result.similarity * 100).toFixed(1)}% Match
                    </span>
                </div>
                <div class="result-filename">
                    <i class="fas fa-file-pdf"></i> ${result.filename}
                </div>
                <div class="result-page">
                    Page ${result.page} • Confidence: ${(result.confidence * 100).toFixed(0)}%
                </div>
                <div class="result-concepts">
                    ${conceptBadges}
                </div>
                <div class="result-text">
                    "${result.text}"
                </div>
            </div>
        `;
    });
    
    container.innerHTML = html;
}

// =====================================================================
// Samples Tab
// =====================================================================

async function loadSamples() {
    try {
        showLoading(true);
        const response = await fetch(`${API_BASE}/samples?limit=10`);
        const data = await response.json();
        
        const container = document.getElementById("samples-list");
        
        if (data.count === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-inbox"></i>
                    <p>No samples available</p>
                </div>
            `;
            return;
        }
        
        let html = "";
        data.samples.forEach((sample) => {
            html += `
                <div class="sample-item">
                    <div class="sample-filename">
                        <i class="fas fa-file"></i> ${sample.filename}
                    </div>
                    <div class="sample-page">
                        Page ${sample.page} • Confidence: ${(sample.confidence * 100).toFixed(0)}%
                    </div>
                    <div style="margin-top: 6px; font-size: 11px;">
                        <span class="concept-badge">${sample.concepts.split(",")[0].trim()}</span>
                    </div>
                    <div class="result-text" style="margin-top: 8px;">
                        "${sample.text}"
                    </div>
                </div>
            `;
        });
        
        container.innerHTML = html;
    } catch (error) {
        document.getElementById("samples-list").innerHTML = `
            <div class="empty-state">
                <i class="fas fa-exclamation-circle"></i>
                <p>Error loading samples</p>
                <small>${error.message}</small>
            </div>
        `;
    } finally {
        showLoading(false);
    }
}

// =====================================================================
// Files Tab
// =====================================================================

async function loadFiles() {
    try {
        showLoading(true);
        const response = await fetch(`${API_BASE}/files`);
        const data = await response.json();
        
        const container = document.getElementById("files-list");
        
        if (data.count === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-folder-open"></i>
                    <p>No files found</p>
                </div>
            `;
            return;
        }
        
        let html = "";
        data.files.forEach((file) => {
            html += `
                <div class="file-item" onclick="viewFileChunks('${file.name}')">
                    <span class="file-name">
                        <i class="fas fa-file-pdf"></i> ${file.name}
                    </span>
                    <span class="file-count">${file.chunks}</span>
                </div>
            `;
        });
        
        container.innerHTML = html;
    } catch (error) {
        document.getElementById("files-list").innerHTML = `
            <div class="empty-state">
                <i class="fas fa-exclamation-circle"></i>
                <p>Error loading files</p>
            </div>
        `;
    } finally {
        showLoading(false);
    }
}

async function viewFileChunks(filename) {
    try {
        showLoading(true);
        const response = await fetch(`${API_BASE}/file/${filename}`);
        const data = await response.json();
        
        const container = document.getElementById("results-container");
        const header = document.getElementById("results-header");
        
        header.innerHTML = `<i class="fas fa-file"></i> ${filename} (${data.count} chunks)`;
        
        let html = "";
        data.chunks.forEach((chunk) => {
            const conceptBadges = chunk.concepts
                .split(",")
                .map(c => `<span class="concept-badge">${c.trim()}</span>`)
                .join("");
            
            html += `
                <div class="result-item">
                    <div class="result-page">
                        Page ${chunk.page} • Confidence: ${(chunk.confidence * 100).toFixed(0)}%
                    </div>
                    <div class="result-concepts">
                        ${conceptBadges}
                    </div>
                    <div class="result-text">
                        "${chunk.text}"
                    </div>
                </div>
            `;
        });
        
        container.innerHTML = html || '<p class="text-muted">No chunks found</p>';
    } catch (error) {
        showError(error.message);
    } finally {
        showLoading(false);
    }
}

// =====================================================================
// Concepts Tab
// =====================================================================

document.getElementById("concept-btn").addEventListener("click", filterByConcept);
document.getElementById("concept-input").addEventListener("keypress", (e) => {
    if (e.key === "Enter") filterByConcept();
});

async function filterByConcept() {
    const concept = document.getElementById("concept-input").value.trim();
    
    if (!concept) {
        alert("Please enter a concept");
        return;
    }
    
    showLoading(true);
    
    try {
        const response = await fetch(`${API_BASE}/concept/${concept}?limit=15`);
        const data = await response.json();
        
        const container = document.getElementById("concept-results");
        
        if (data.count === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-search"></i>
                    <p>No documents found for "${concept}"</p>
                </div>
            `;
            return;
        }
        
        let html = `<p style="font-size: 12px; color: #666; margin-bottom: 10px;">
            Found ${data.count} documents
        </p>`;
        
        data.documents.forEach((doc) => {
            const conceptBadges = doc.concepts
                .split(",")
                .map(c => `<span class="concept-badge">${c.trim()}</span>`)
                .join("");
            
            html += `
                <div class="result-item">
                    <div class="result-filename">
                        <i class="fas fa-file"></i> ${doc.filename}
                    </div>
                    <div class="result-page">
                        Page ${doc.page} • Confidence: ${(doc.confidence * 100).toFixed(0)}%
                    </div>
                    <div class="result-concepts">
                        ${conceptBadges}
                    </div>
                </div>
            `;
        });
        
        container.innerHTML = html;
    } catch (error) {
        showError(error.message);
    } finally {
        showLoading(false);
    }
}

// =====================================================================
// Error Handling
// =====================================================================

function showError(message) {
    alert(`Error: ${message}`);
}

// =====================================================================
// Initialize on Load
// =====================================================================

document.addEventListener("DOMContentLoaded", initializeDashboard);
