"""
dashboard/app.py
CO2M Literature Intelligence Dashboard
Run: streamlit run co2m_lit_dashboard/dashboard/app.py
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
ARTIFACTS_DIR = _HERE.parent.parent / "co2m" / "artifacts"

# ---------------------------------------------------------------------------
# Page config  (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CO2M Literature Intelligence",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
_ACCENT  = "#1D4ED8"
_ACCENT2 = "#0284C7"
_MUTED   = "#6B7280"
_SURFACE = "#F9FAFB"
_BORDER  = "#E5E7EB"
_TEXT    = "#111827"
_GOOD    = "#065F46"
_FONT    = "IBM Plex Sans, ui-sans-serif, system-ui, sans-serif"

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {{
    font-family: {_FONT} !important;
    color: {_TEXT};
}}
h1, h2, h3, h4 {{ font-family: {_FONT} !important; letter-spacing: -0.01em; }}

section[data-testid="stSidebar"] {{
    background: {_SURFACE};
    border-right: 1px solid {_BORDER};
}}

[data-testid="metric-container"] {{
    background: {_SURFACE};
    border: 1px solid {_BORDER};
    border-radius: 8px;
    padding: 0.9rem 1.1rem !important;
}}
[data-testid="metric-container"] label {{
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: .08em;
    color: {_MUTED} !important;
}}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    font-size: 1.9rem !important;
    font-weight: 700 !important;
    color: {_TEXT} !important;
}}

.co2m-summary {{
    border-left: 3px solid {_ACCENT};
    background: {_SURFACE};
    padding: 0.85rem 1.1rem;
    border-radius: 0 6px 6px 0;
    font-size: 0.92rem;
    line-height: 1.7;
    color: #1F2937;
    margin: 0.5rem 0 1.5rem 0;
}}

.co2m-snippet {{
    border: 1px solid {_BORDER};
    border-radius: 6px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    background: #FFFFFF;
}}
.co2m-snippet .meta {{
    font-size: 0.68rem;
    font-weight: 600;
    color: {_MUTED};
    text-transform: uppercase;
    letter-spacing: .07em;
    margin-bottom: 0.3rem;
}}
.co2m-snippet .body {{
    font-size: 0.875rem;
    color: #374151;
    line-height: 1.65;
}}
.co2m-snippet mark {{
    background: #FEF08A;
    color: {_TEXT};
    border-radius: 2px;
    padding: 0 2px;
}}

/* Streamlit tab strip */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0;
    border-bottom: 2px solid {_BORDER};
}}
.stTabs [data-baseweb="tab"] {{
    font-family: {_FONT} !important;
    font-size: 0.82rem;
    font-weight: 500;
    padding: 0.55rem 1.1rem;
    color: {_MUTED};
    border-radius: 0;
    border-bottom: 2px solid transparent;
    margin-bottom: -2px;
}}
.stTabs [aria-selected="true"] {{
    color: {_ACCENT} !important;
    border-bottom: 2px solid {_ACCENT} !important;
    font-weight: 600 !important;
}}

hr {{ border: none; border-top: 1px solid {_BORDER}; margin: 1.25rem 0; }}
.stDataFrame {{ border: 1px solid {_BORDER}; border-radius: 6px; overflow: hidden; }}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Layout helper for Plotly (no duplicate key errors)
# ---------------------------------------------------------------------------
def _layout(**extra) -> dict:
    base = dict(
        font=dict(family=_FONT, size=12, color=_TEXT),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        margin=dict(l=10, r=10, t=36, b=10),
        title_font=dict(family=_FONT, size=13, color=_MUTED),
        title_x=0,
        xaxis=dict(showgrid=True, gridcolor=_BORDER, zeroline=False, tickfont=dict(size=11)),
        yaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=11)),
        hoverlabel=dict(font_family=_FONT),
    )
    for axis_key in ("xaxis", "yaxis"):
        if axis_key in extra:
            base[axis_key] = {**base[axis_key], **extra.pop(axis_key)}
    base.update(extra)
    return base


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading artifacts …")
def load_all():
    names = [
        "documents", "chunks", "concepts", "summaries",
        "concept_counts", "top_documents_by_concept", "keywords_by_concept",
    ]
    data = {n: pd.read_parquet(ARTIFACTS_DIR / f"{n}.parquet")
            if (ARTIFACTS_DIR / f"{n}.parquet").exists()
            else pd.DataFrame()
            for n in names}
    manifest: dict = {}
    mp = ARTIFACTS_DIR / "manifest.json"
    if mp.exists():
        manifest = json.loads(mp.read_text(encoding="utf-8"))
    return data, manifest


if not ARTIFACTS_DIR.exists():
    st.error(
        f"**Artifacts not found:** `{ARTIFACTS_DIR}`\n\n"
        "Run the build script first:\n```\n"
        "python co2m_lit_dashboard/src/build_corpus.py "
        "--pdf-dir co2m/pdfs --metadata-dir co2m/metadata --out co2m/artifacts\n```"
    )
    st.stop()

data, manifest = load_all()
documents_df     = data["documents"]
chunks_df        = data["chunks"]
concepts_df      = data["concepts"]
summaries_df     = data["summaries"]
concept_counts_df= data["concept_counts"]
top_docs_df      = data["top_documents_by_concept"]
keywords_df      = data["keywords_by_concept"]

has_year = (
    not documents_df.empty
    and "year" in documents_df.columns
    and documents_df["year"].notna().any()
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
hcol, dcol = st.columns([7, 1])
with hcol:
    st.title("CO2M Literature Intelligence")
with dcol:
    if manifest:
        ts = manifest.get("build_timestamp", "")
        if ts:
            st.caption(f"Built {ts[:10]}")
st.markdown("---")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown("### Filters")

all_concepts: list[str] = (
    concept_counts_df["concept_name"].tolist()
    if not concept_counts_df.empty and "concept_name" in concept_counts_df.columns
    else summaries_df["concept_name"].tolist()
    if not summaries_df.empty
    else []
)
if not all_concepts:
    st.warning("No concept data. Run the build script first.")
    st.stop()

selected_concept: str = st.sidebar.selectbox("Scientific Concept", all_concepts)

year_filter: list[int] = []
if has_year:
    years = sorted(documents_df["year"].dropna().astype(int).unique().tolist())
    if len(years) > 1:
        year_filter = st.sidebar.multiselect("Year", options=years, default=[])

keyword_search: str = st.sidebar.text_input("Snippet keyword", placeholder="filter snippets …")

st.sidebar.markdown("---")
if manifest:
    st.sidebar.markdown("**Corpus**")
    st.sidebar.markdown(
        f"PDFs: **{manifest.get('pdf_count','—')}**  \n"
        f"Chunks: **{manifest.get('chunk_count','—')}**  \n"
        f"Assignments: **{manifest.get('concept_assignment_count','—')}**"
    )

# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------
concept_rows = (
    concepts_df[concepts_df["concept_name"] == selected_concept].copy()
    if not concepts_df.empty else pd.DataFrame()
)
concept_chunk_ids: set[int] = set(concept_rows["chunk_id"].unique()) if not concept_rows.empty else set()
concept_doc_ids: set[int]   = set(concept_rows["doc_id"].unique())   if not concept_rows.empty else set()

if year_filter and has_year and not documents_df.empty:
    year_doc_ids = set(documents_df.loc[documents_df["year"].isin(year_filter), "doc_id"])
    concept_doc_ids &= year_doc_ids
    concept_rows = concept_rows[concept_rows["doc_id"].isin(concept_doc_ids)]
    concept_chunk_ids = set(concept_rows["chunk_id"].unique())

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
total_pdfs = len(documents_df) if not documents_df.empty else 0
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total PDFs", f"{total_pdfs:,}")
m2.metric("Matching PDFs", f"{len(concept_doc_ids):,}")
m3.metric("Matching Chunks", f"{len(concept_chunk_ids):,}")
if has_year and concept_doc_ids and not documents_df.empty:
    yr_s = documents_df.loc[documents_df["doc_id"].isin(concept_doc_ids), "year"].dropna()
    m4.metric("Peak Year", str(int(yr_s.mode().iloc[0])) if not yr_s.empty else "—")
else:
    m4.metric("Total Chunks", f"{len(chunks_df):,}" if not chunks_df.empty else "0")

# ---------------------------------------------------------------------------
# Concept summary
# ---------------------------------------------------------------------------
st.markdown(f"#### {selected_concept}")
summary_text = "No summary available for this concept."
if not summaries_df.empty:
    _r = summaries_df[summaries_df["concept_name"] == selected_concept]
    if not _r.empty:
        summary_text = _r.iloc[0]["summary_text"]
st.markdown(f'<div class="co2m-summary">{summary_text}</div>', unsafe_allow_html=True)

# ===========================================================================
# TABBED NAVIGATION
# ===========================================================================
tab_overview, tab_science, tab_deepdive, tab_domain, tab_compare, tab_docs, tab_snippets = st.tabs([
    "Corpus Overview",
    "Scientific Insights",
    "Concept Deep Dive",
    "Domain Map",
    "Cross-Concept",
    "Documents",
    "Source Snippets",
])

# ============================================================
# TAB 1 — Corpus Overview
# ============================================================
with tab_overview:
    st.markdown("#### Concept Coverage")
    ov1, ov2 = st.columns(2)

    with ov1:
        if not concept_counts_df.empty:
            df_p = concept_counts_df.sort_values("chunk_count", ascending=True).copy()
            df_p["_s"] = df_p["concept_name"].apply(lambda c: "Selected" if c == selected_concept else "Other")
            fig = px.bar(df_p, x="chunk_count", y="concept_name", orientation="h",
                         color="_s", color_discrete_map={"Selected": _ACCENT, "Other": "#D1D5DB"},
                         labels={"chunk_count": "Chunks", "concept_name": ""},
                         title="Coverage by chunks")
            fig.update_traces(showlegend=False)
            fig.update_layout(**_layout(height=360))
            st.plotly_chart(fig, use_container_width=True)

    with ov2:
        if not concept_counts_df.empty:
            df_d = concept_counts_df.sort_values("doc_count", ascending=True).copy()
            fig = px.bar(df_d, x="doc_count", y="concept_name", orientation="h",
                         color="doc_count",
                         color_continuous_scale=[[0, "#BFDBFE"], [1, _ACCENT]],
                         labels={"doc_count": "Documents", "concept_name": ""},
                         title="Coverage by documents")
            fig.update_layout(**_layout(height=360, coloraxis_showscale=False))
            st.plotly_chart(fig, use_container_width=True)

    if not concepts_df.empty:
        st.markdown("#### Concept Co-occurrence")
        st.caption("Number of documents shared between each pair of concepts.")
        dc_uniq = concepts_df[["doc_id", "concept_name"]].drop_duplicates()
        clist = sorted(dc_uniq["concept_name"].unique())
        comat = pd.DataFrame(0, index=clist, columns=clist)
        for _, grp in dc_uniq.groupby("doc_id"):
            names = grp["concept_name"].tolist()
            for i, a in enumerate(names):
                for b in names[i:]:
                    comat.loc[a, b] += 1
                    comat.loc[b, a] += 1
        fig = px.imshow(comat, color_continuous_scale="Blues",
                        labels=dict(color="Docs"),
                        title="Co-occurrence (shared documents)", aspect="auto")
        fig.update_layout(**_layout(height=430, margin=dict(l=10, r=10, t=40, b=10)))
        fig.update_xaxes(tickangle=-40, tickfont=dict(size=10))
        fig.update_yaxes(tickfont=dict(size=10))
        st.plotly_chart(fig, use_container_width=True)

    if has_year and not concepts_df.empty and not documents_df.empty:
        st.markdown("#### Activity Over Time")
        yc = (concepts_df[["doc_id", "concept_name"]].drop_duplicates()
              .merge(documents_df[["doc_id", "year"]].dropna(subset=["year"]),
                     on="doc_id", how="inner"))
        if not yc.empty:
            yc["year"] = yc["year"].astype(int)
            ycp = (yc.groupby(["year", "concept_name"]).size()
                   .reset_index(name="docs")
                   .pivot(index="concept_name", columns="year", values="docs")
                   .fillna(0))
            fig = px.imshow(ycp, color_continuous_scale="Blues",
                            labels=dict(x="Year", y="", color="Docs"),
                            title="Documents per concept per year", aspect="auto")
            fig.update_layout(**_layout(height=370, margin=dict(l=10, r=10, t=40, b=10)))
            fig.update_xaxes(tickangle=-45, tickfont=dict(size=10))
            fig.update_yaxes(tickfont=dict(size=10))
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 2 — Scientific Insights  (all concepts at once)
# ============================================================
with tab_science:
    st.markdown("#### Scientific Insights — All Concepts")
    st.caption(
        "A holistic view of every concept in the corpus. "
        "Scroll through each panel to understand the full research landscape."
    )

    # ── A. Summary cards: doc count + chunk count per concept ────────────────
    if not summaries_df.empty and not concept_counts_df.empty:
        st.markdown("##### Concept Summaries")
        merged_summary = summaries_df.merge(
            concept_counts_df[["concept_name", "doc_count", "chunk_count"]],
            on="concept_name", how="left",
        )
        for _, sr in merged_summary.iterrows():
            is_sel = sr["concept_name"] == selected_concept
            border_color = _ACCENT if is_sel else _BORDER
            bg = "#EFF6FF" if is_sel else "#FFFFFF"
            doc_c  = int(sr["doc_count"])   if pd.notna(sr.get("doc_count"))   else 0
            chk_c  = int(sr["chunk_count"]) if pd.notna(sr.get("chunk_count")) else 0
            st.markdown(
                f'<div style="border:1px solid {border_color};background:{bg};'
                f'border-radius:7px;padding:.75rem 1rem;margin-bottom:.5rem">'
                f'<div style="display:flex;justify-content:space-between;align-items:baseline">'
                f'<span style="font-weight:600;font-size:.95rem;color:{_ACCENT if is_sel else _TEXT}">{sr["concept_name"]}</span>'
                f'<span style="font-size:.75rem;color:{_MUTED}">{doc_c} docs &nbsp;·&nbsp; {chk_c:,} chunks</span>'
                f'</div>'
                f'<div style="font-size:.85rem;color:#374151;line-height:1.6;margin-top:.35rem">{sr["summary_text"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── B. Side-by-side concept size comparison (treemap) ────────────────────
    if not concept_counts_df.empty:
        st.markdown("##### Research Volume Treemap")
        st.caption("Area is proportional to total chunks assigned to each concept.")
        fig_tm = px.treemap(
            concept_counts_df,
            path=["concept_name"],
            values="chunk_count",
            color="doc_count",
            color_continuous_scale=[[0, "#BFDBFE"], [0.5, _ACCENT2], [1, _ACCENT]],
            labels={"concept_name": "Concept", "chunk_count": "Chunks", "doc_count": "Docs"},
            title="Research volume per concept (chunks)",
        )
        fig_tm.update_traces(
            textinfo="label+value",
            textfont=dict(family=_FONT, size=12),
            hovertemplate="<b>%{label}</b><br>Chunks: %{value}<br>Docs: %{color:.0f}<extra></extra>",
        )
        fig_tm.update_layout(
            font=dict(family=_FONT, size=12, color=_TEXT),
            paper_bgcolor="#FFFFFF",
            height=420,
            margin=dict(l=10, r=10, t=40, b=10),
            title_font=dict(family=_FONT, size=13, color=_MUTED),
            title_x=0,
            coloraxis_colorbar=dict(title="Docs", tickfont=dict(size=10)),
        )
        st.plotly_chart(fig_tm, use_container_width=True)

    # ── C. All-concept keyword heatmap ───────────────────────────────────────
    if not keywords_df.empty:
        st.markdown("##### Keyword Presence Across All Concepts")
        st.caption(
            "Top 15 keywords (by total TF-IDF weight across corpus) × all concepts. "
            "Brighter = stronger association."
        )
        top15_kws = (
            keywords_df.groupby("keyword")["weight"]
            .sum().nlargest(15).index.tolist()
        )
        kw_heat = (
            keywords_df[keywords_df["keyword"].isin(top15_kws)]
            .pivot_table(index="keyword", columns="concept_name",
                         values="weight", aggfunc="mean")
            .fillna(0)
        )
        fig_kh = px.imshow(
            kw_heat,
            color_continuous_scale=[[0, "#F1F5F9"], [0.4, "#93C5FD"], [1, _ACCENT]],
            labels=dict(x="Concept", y="Keyword", color="TF-IDF"),
            title="Top keywords across all concepts",
            aspect="auto",
            text_auto=".3f",
        )
        fig_kh.update_layout(
            font=dict(family=_FONT, size=11, color=_TEXT),
            paper_bgcolor="#FFFFFF",
            height=480,
            margin=dict(l=10, r=10, t=40, b=10),
            title_font=dict(family=_FONT, size=13, color=_MUTED),
            title_x=0,
        )
        fig_kh.update_xaxes(tickangle=-35, tickfont=dict(size=10))
        fig_kh.update_yaxes(tickfont=dict(size=10))
        fig_kh.update_traces(textfont=dict(size=8))
        st.plotly_chart(fig_kh, use_container_width=True)

    # ── D. All-concept year trend (area) ─────────────────────────────────────
    if has_year and not concepts_df.empty and not documents_df.empty:
        st.markdown("##### Research Activity Over Time — All Concepts")
        st.caption("Stacked area shows the total chunk volume per year, coloured by concept.")
        yc_all = (
            concepts_df[["doc_id", "concept_name", "chunk_id"]]
            .drop_duplicates(subset=["doc_id", "concept_name", "chunk_id"])
            .merge(
                documents_df[["doc_id", "year"]].dropna(subset=["year"]),
                on="doc_id", how="inner",
            )
        )
        if not yc_all.empty:
            yc_all["year"] = yc_all["year"].astype(int)
            yc_agg = (
                yc_all.groupby(["year", "concept_name"])["chunk_id"]
                .count()
                .reset_index(name="chunks")
                .sort_values("year")
            )
            fig_area = px.area(
                yc_agg,
                x="year", y="chunks",
                color="concept_name",
                line_group="concept_name",
                labels={"chunks": "Chunks", "year": "Year", "concept_name": "Concept"},
                title="Research activity (all concepts) — stacked area",
            )
            fig_area.update_layout(
                font=dict(family=_FONT, size=12, color=_TEXT),
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                height=380,
                margin=dict(l=10, r=10, t=44, b=60),
                title_font=dict(family=_FONT, size=13, color=_MUTED),
                title_x=0,
                xaxis=dict(showgrid=True, gridcolor=_BORDER, zeroline=False,
                           tickmode="linear", dtick=2, tickfont=dict(size=10)),
                yaxis=dict(showgrid=True, gridcolor=_BORDER, zeroline=False,
                           tickfont=dict(size=10)),
                legend=dict(
                    title=None,
                    orientation="h",
                    yanchor="top", y=-0.2,
                    xanchor="left", x=0,
                    font=dict(size=10),
                ),
            )
            st.plotly_chart(fig_area, use_container_width=True)

    # ── E. Top 5 docs per concept (grouped bar) ───────────────────────────────
    if not top_docs_df.empty and not documents_df.empty:
        st.markdown("##### Top Document per Concept")
        st.caption("The single best-matching document (most chunks) for each concept.")
        lc2 = "title" if "title" in top_docs_df.columns else "filename"
        top1 = (
            top_docs_df[top_docs_df["rank"] == 1][["concept_name", "chunk_count", lc2, "filename"]]
            .copy()
        )
        top1["label"] = top1[lc2].fillna(top1["filename"]).astype(str).str[:48]
        top1 = top1.sort_values("chunk_count", ascending=False)
        fig_top1 = px.bar(
            top1,
            x="chunk_count",
            y="concept_name",
            orientation="h",
            color="concept_name",
            text="label",
            labels={"chunk_count": "Chunks", "concept_name": ""},
            title="Best-matched document per concept",
        )
        fig_top1.update_traces(
            showlegend=False,
            textposition="inside",
            textfont=dict(size=9, color="#FFFFFF"),
            insidetextanchor="start",
        )
        fig_top1.update_layout(**_layout(height=360))
        st.plotly_chart(fig_top1, use_container_width=True)

# ============================================================
# TAB 3 — Concept Deep Dive
# ============================================================
with tab_deepdive:
    st.markdown(f"#### {selected_concept} — Analysis")
    dd1, dd2 = st.columns(2)

    with dd1:
        if not top_docs_df.empty:
            ct = top_docs_df[top_docs_df["concept_name"] == selected_concept].head(10).copy()
            if not ct.empty:
                lc = "title" if "title" in ct.columns else "filename"
                ct["label"] = ct[lc].fillna(ct.get("filename", "")).astype(str).str[:52]
                fig = px.bar(ct.sort_values("chunk_count", ascending=True),
                             x="chunk_count", y="label", orientation="h",
                             color="chunk_count",
                             color_continuous_scale=[[0, "#BFDBFE"], [1, _ACCENT]],
                             labels={"chunk_count": "Chunks", "label": ""},
                             title="Top documents")
                fig.update_layout(**_layout(height=370, coloraxis_showscale=False))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No documents matched this concept.")

    with dd2:
        if not concept_rows.empty and "confidence_score" in concept_rows.columns:
            fig = px.histogram(concept_rows, x="confidence_score", nbins=20,
                               color_discrete_sequence=[_ACCENT2],
                               labels={"confidence_score": "Confidence score"},
                               title="Confidence score distribution")
            fig.update_layout(**_layout(height=370))
            fig.update_traces(marker_line_color="#FFFFFF", marker_line_width=1)
            st.plotly_chart(fig, use_container_width=True)

    if not keywords_df.empty:
        kw = keywords_df[keywords_df["concept_name"] == selected_concept].head(20)
        if not kw.empty:
            st.markdown("#### Top TF-IDF Keywords")
            fig = px.bar(kw.sort_values("weight", ascending=True),
                         x="weight", y="keyword", orientation="h",
                         color="weight",
                         color_continuous_scale=[[0, "#D1FAE5"], [1, _GOOD]],
                         labels={"weight": "TF-IDF weight", "keyword": ""},
                         title=f"Keywords — {selected_concept}")
            fig.update_layout(**_layout(height=430, coloraxis_showscale=False))
            st.plotly_chart(fig, use_container_width=True)

    if has_year and not documents_df.empty and concept_doc_ids:
        yd = documents_df[
            documents_df["doc_id"].isin(concept_doc_ids) & documents_df["year"].notna()
        ].copy()
        if not yd.empty:
            st.markdown("#### Publications by Year")
            yc2 = (yd.groupby(yd["year"].astype(int)).size()
                   .reset_index(name="Documents")
                   .rename(columns={"year": "Year"})
                   .sort_values("Year"))
            fig = px.bar(yc2, x="Year", y="Documents",
                         color_discrete_sequence=[_ACCENT2],
                         title=f"Publications per year — {selected_concept}")
            fig.update_layout(**_layout(height=260, xaxis=dict(tickmode="linear", dtick=1)))
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 3 — Domain Map
# ============================================================
with tab_domain:
    st.markdown("#### Domain Map — What Each Concept Talks About")
    st.caption(
        "Computed from TF-IDF keyword weights. "
        "Similar concepts appear close together; colour encodes concept family."
    )

    # ── Build concept × keyword TF-IDF matrix from precomputed keywords ──────
    def _build_concept_matrix(kw_df: pd.DataFrame):
        """
        Pivot keywords_by_concept into a concept × keyword matrix.
        Returns (concept_names, numpy matrix) or ([], None) if data missing.
        """
        if kw_df.empty or not {"concept_name", "keyword", "weight"}.issubset(kw_df.columns):
            return [], None
        pivot = (kw_df.pivot_table(index="concept_name", columns="keyword",
                                   values="weight", aggfunc="mean")
                 .fillna(0))
        concepts = pivot.index.tolist()
        mat = normalize(pivot.values, norm="l2")
        return concepts, mat

    concepts_list, kw_matrix = _build_concept_matrix(keywords_df)

    # ── 3a. Concept Similarity Heatmap ───────────────────────────────────────
    if kw_matrix is not None and len(concepts_list) > 1:
        sim = cosine_similarity(kw_matrix)
        sim_df = pd.DataFrame(sim, index=concepts_list, columns=concepts_list)

        st.markdown("##### Concept Similarity Heatmap")
        st.caption(
            "Cosine similarity of TF-IDF keyword profiles. "
            "1.0 = identical vocabulary, 0.0 = no overlap."
        )
        fig_sim = px.imshow(
            sim_df,
            color_continuous_scale="RdBu_r",
            range_color=[0, 1],
            labels=dict(color="Similarity"),
            title="Keyword similarity between concepts",
            aspect="auto",
            text_auto=".2f",
        )
        fig_sim.update_layout(**_layout(height=460, margin=dict(l=10, r=10, t=40, b=10)))
        fig_sim.update_xaxes(tickangle=-40, tickfont=dict(size=10))
        fig_sim.update_yaxes(tickfont=dict(size=10))
        fig_sim.update_traces(textfont=dict(size=9))
        st.plotly_chart(fig_sim, use_container_width=True)

    # ── 3b. PCA Cluster Scatter ───────────────────────────────────────────────
    if kw_matrix is not None and len(concepts_list) >= 3:
        st.markdown("##### Concept Cluster Map (PCA)")
        st.caption(
            "Each point is a concept projected onto its two principal keyword axes. "
            "Concepts near each other share vocabulary."
        )
        n_comp = min(2, kw_matrix.shape[1], kw_matrix.shape[0])
        pca = PCA(n_components=n_comp)
        coords = pca.fit_transform(kw_matrix)
        var = pca.explained_variance_ratio_

        pca_df = pd.DataFrame({
            "Concept": concepts_list,
            "PC1": coords[:, 0],
            "PC2": coords[:, 1] if n_comp > 1 else [0.0] * len(concepts_list),
            "highlight": ["▶ " + c if c == selected_concept else c for c in concepts_list],
        })

        # Add chunk counts as marker size
        if not concept_counts_df.empty:
            pca_df = pca_df.merge(
                concept_counts_df[["concept_name", "chunk_count"]].rename(
                    columns={"concept_name": "Concept"}
                ),
                on="Concept", how="left",
            )
            pca_df["chunk_count"] = pca_df["chunk_count"].fillna(1)
        else:
            pca_df["chunk_count"] = 10

        fig_pca = px.scatter(
            pca_df,
            x="PC1", y="PC2",
            size="chunk_count",
            color="Concept",
            size_max=46,
            labels={
                "PC1": f"PC 1 ({var[0]*100:.0f}% var)",
                "PC2": f"PC 2 ({var[1]*100:.0f}% var)" if n_comp > 1 else "PC 2",
            },
            title="Concept cluster map — keyword space",
        )
        fig_pca.update_traces(
            marker=dict(opacity=0.75, line=dict(width=1.5, color="#FFFFFF")),
            showlegend=False,
        )
        # Add per-point annotations offset away from the bubble centre
        # so labels never sit inside a bubble
        for _, r in pca_df.iterrows():
            is_sel = r["Concept"] == selected_concept
            fig_pca.add_annotation(
                x=r["PC1"],
                y=r["PC2"],
                text=f"<b>{r['Concept']}</b>" if is_sel else r["Concept"],
                showarrow=True,
                arrowhead=0,
                arrowwidth=0.8,
                arrowcolor="#9CA3AF",
                ax=0,
                ay=-36,
                font=dict(family=_FONT, size=10 if not is_sel else 11,
                          color=_ACCENT if is_sel else _TEXT),
                bgcolor="rgba(255,255,255,0.85)",
                borderpad=2,
            )
        fig_pca.update_layout(**_layout(
            height=520,
            xaxis=dict(showgrid=True, gridcolor=_BORDER, zeroline=True,
                       zerolinecolor=_BORDER, tickfont=dict(size=10)),
            yaxis=dict(showgrid=True, gridcolor=_BORDER, zeroline=True,
                       zerolinecolor=_BORDER, tickfont=dict(size=10)),
        ))
        st.plotly_chart(fig_pca, use_container_width=True)

    # ── 3c. Multi-line concept trend over years ───────────────────────────────
    if has_year and not concepts_df.empty and not documents_df.empty:
        st.markdown("##### Concept Activity Over Time (Multi-line)")
        st.caption(
            "Chunk count per concept per year. "
            "Shows which topics gained or lost research attention over time."
        )
        yc_trend = (
            concepts_df[["doc_id", "concept_name", "chunk_id"]].drop_duplicates(
                subset=["doc_id", "concept_name", "chunk_id"]
            )
            .merge(
                documents_df[["doc_id", "year"]].dropna(subset=["year"]),
                on="doc_id", how="inner",
            )
        )
        if not yc_trend.empty:
            yc_trend["year"] = yc_trend["year"].astype(int)
            trend_agg = (
                yc_trend.groupby(["year", "concept_name"])["chunk_id"]
                .count()
                .reset_index(name="chunks")
                .sort_values("year")
            )
            # highlight selected concept with a thicker line
            trend_agg["_lw"] = trend_agg["concept_name"].apply(
                lambda c: 3 if c == selected_concept else 1.2
            )
            fig_trend = px.line(
                trend_agg,
                x="year", y="chunks",
                color="concept_name",
                line_group="concept_name",
                markers=True,
                labels={"chunks": "Chunks", "year": "Year", "concept_name": "Concept"},
                title="Research activity trend by concept",
            )
            # Bold the selected concept
            for trace in fig_trend.data:
                if trace.name == selected_concept:
                    trace.line = dict(width=3)
                    trace.marker = dict(size=7)
                else:
                    trace.line = dict(width=1.5)
                    trace.marker = dict(size=4)
            fig_trend.update_layout(**_layout(
                height=420,
                xaxis=dict(tickmode="linear", dtick=1),
                legend=dict(
                    title=None,          # removes the "Concept" legend title
                    orientation="h",
                    yanchor="top",
                    y=-0.18,             # push legend below the x-axis
                    xanchor="left",
                    x=0,
                    font=dict(size=10),
                ),
                margin=dict(l=10, r=10, t=44, b=10),
            ))
            st.plotly_chart(fig_trend, use_container_width=True)

    # ── 3d. Per-concept keyword fingerprint (small multiples) ────────────────
    if not keywords_df.empty:
        st.markdown("##### Keyword Fingerprint per Concept")
        st.caption(
            "Top 8 TF-IDF keywords for each concept shown as individual panels. "
            "Each bar is that keyword's weight within that concept's corpus."
        )
        # Top-8 keywords per concept, ordered by weight
        fp_df = (
            keywords_df.sort_values(["concept_name", "weight"], ascending=[True, False])
            .groupby("concept_name")
            .head(8)
            .copy()
        )
        if not fp_df.empty:
            # colour the selected concept's panel bars differently
            fp_df["bar_color"] = fp_df["concept_name"].apply(
                lambda c: _ACCENT if c == selected_concept else _ACCENT2
            )
            n_concepts = fp_df["concept_name"].nunique()
            # two columns of subplots via facet_col
            fig_fp = px.bar(
                fp_df.sort_values("weight", ascending=True),
                x="weight",
                y="keyword",
                facet_col="concept_name",
                facet_col_wrap=2,
                orientation="h",
                color="concept_name",
                labels={"weight": "", "keyword": "", "concept_name": ""},
                title="Keyword fingerprint per concept",
                height=max(320, n_concepts * 150),
            )
            # Clean up facet titles — strip the "concept_name=" prefix
            fig_fp.for_each_annotation(
                lambda a: a.update(
                    text=a.text.split("=")[-1],
                    font=dict(size=11, family=_FONT, color=_TEXT),
                )
            )
            fig_fp.update_traces(showlegend=False)
            fig_fp.update_xaxes(showgrid=True, gridcolor=_BORDER,
                                zeroline=False, tickfont=dict(size=9))
            fig_fp.update_yaxes(tickfont=dict(size=9), matches=None)
            fig_fp.update_layout(
                font=dict(family=_FONT, size=11, color=_TEXT),
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                margin=dict(l=10, r=10, t=48, b=10),
                title_font=dict(family=_FONT, size=13, color=_MUTED),
                title_x=0,
                showlegend=False,
            )
            st.plotly_chart(fig_fp, use_container_width=True)

# ============================================================
# TAB 4 — Cross-Concept Comparisons
# ============================================================
with tab_compare:
    cc1, cc2 = st.columns(2)

    with cc1:
        if not concepts_df.empty and "confidence_score" in concepts_df.columns:
            avg_c = (concepts_df.groupby("concept_name")["confidence_score"]
                     .mean().reset_index(name="avg_confidence")
                     .sort_values("avg_confidence", ascending=True))
            avg_c["_s"] = avg_c["concept_name"].apply(lambda c: "Selected" if c == selected_concept else "Other")
            fig = px.bar(avg_c, x="avg_confidence", y="concept_name", orientation="h",
                         color="_s", color_discrete_map={"Selected": _ACCENT, "Other": "#D1D5DB"},
                         labels={"avg_confidence": "Avg confidence", "concept_name": ""},
                         title="Average classification confidence")
            fig.update_traces(showlegend=False)
            fig.update_layout(**_layout(height=370))
            st.plotly_chart(fig, use_container_width=True)

    with cc2:
        if (not concepts_df.empty and not documents_df.empty
                and "page_count" in documents_df.columns):
            dcu = (concepts_df[["concept_name", "doc_id"]].drop_duplicates()
                   .merge(documents_df[["doc_id", "page_count"]], on="doc_id", how="left"))
            ap = (dcu.groupby("concept_name")["page_count"]
                  .mean().reset_index(name="avg_pages")
                  .sort_values("avg_pages", ascending=True))
            fig = px.bar(ap, x="avg_pages", y="concept_name", orientation="h",
                         color="avg_pages",
                         color_continuous_scale=[[0, "#FEF3C7"], [1, "#B45309"]],
                         labels={"avg_pages": "Avg pages", "concept_name": ""},
                         title="Average document length (pages)")
            fig.update_layout(**_layout(height=370, coloraxis_showscale=False))
            st.plotly_chart(fig, use_container_width=True)

    if not concepts_df.empty and not documents_df.empty:
        st.markdown("#### Concept Composition per Document")
        st.caption("Top 15 most-covered documents — coloured by concept contribution.")
        dcc = (concepts_df.groupby(["doc_id", "concept_name"])["chunk_id"]
               .nunique().reset_index(name="chunks"))
        top15 = dcc.groupby("doc_id")["chunks"].sum().nlargest(15).index.tolist()
        dct = dcc[dcc["doc_id"].isin(top15)].copy()
        ls = documents_df[["doc_id", "title", "filename"]].copy()
        ls["label"] = ls["title"].fillna(ls["filename"]).astype(str).str[:45]
        dct = dct.merge(ls[["doc_id", "label"]], on="doc_id", how="left")
        fig = px.bar(dct, x="chunks", y="label", color="concept_name",
                     orientation="h", barmode="stack",
                     labels={"chunks": "Chunks", "label": "", "concept_name": "Concept"},
                     title="Concept mix per document")
        fig.update_layout(**_layout(
            height=430,
            legend=dict(orientation="h", yanchor="bottom", y=1.01,
                        xanchor="left", x=0, font=dict(size=10)),
        ))
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 5 — Documents
# ============================================================
with tab_docs:
    st.markdown(f"#### Related Documents — {selected_concept}")
    st.caption(f"{len(concept_doc_ids)} document(s) matched")

    if not concept_doc_ids:
        st.info("No documents matched this concept with the current filters.")
    elif not documents_df.empty:
        rel = documents_df[documents_df["doc_id"].isin(concept_doc_ids)].copy()
        if not concept_rows.empty:
            dcc2 = (concept_rows.groupby("doc_id")["chunk_id"]
                    .nunique().reset_index(name="matching_chunks"))
            rel = rel.merge(dcc2, on="doc_id", how="left").sort_values(
                "matching_chunks", ascending=False)
        dcols = ["filename", "title", "authors", "year", "page_count", "matching_chunks"]
        dcols = [c for c in dcols if c in rel.columns]
        st.dataframe(rel[dcols].reset_index(drop=True),
                     use_container_width=True, hide_index=True)

# ============================================================
# TAB 6 — Source Snippets  (doc-level selection + keyword filter)
# ============================================================
with tab_snippets:
    st.markdown(f"#### Source Snippets — {selected_concept}")

    if chunks_df.empty or not concept_chunk_ids:
        st.info("No snippets available for this concept.")
    else:
        # Build a doc → filename lookup for the selectbox
        doc_options: dict[str, list[int]] = {}   # label → list of chunk_ids
        doc_options["All documents"] = list(concept_chunk_ids)

        if not concept_rows.empty and not documents_df.empty:
            # per-doc chunk counts for the selector label
            per_doc = (concept_rows.groupby("doc_id")["chunk_id"]
                       .nunique().reset_index(name="n"))
            per_doc = per_doc.merge(
                documents_df[["doc_id", "filename"]].drop_duplicates(),
                on="doc_id", how="left"
            ).sort_values("n", ascending=False)
            for _, r in per_doc.iterrows():
                label = f"{r['filename']}  ({int(r['n'])} chunks)"
                doc_chunk_ids = set(
                    concept_rows.loc[concept_rows["doc_id"] == r["doc_id"], "chunk_id"]
                )
                doc_options[label] = list(doc_chunk_ids)

        sc1, sc2 = st.columns([3, 1])
        with sc1:
            doc_sel = st.selectbox("Filter by document", options=list(doc_options.keys()))
        with sc2:
            conf_threshold = st.slider("Min confidence", 0.0, 1.0, 0.0, 0.05)

        active_ids: set[int] = set(doc_options[doc_sel])

        # Pull and filter snippets
        snip = chunks_df[chunks_df["chunk_id"].isin(active_ids)].copy()

        if not concept_rows.empty:
            cf = concept_rows[["chunk_id", "confidence_score"]].drop_duplicates("chunk_id")
            snip = snip.merge(cf, on="chunk_id", how="left")
            if conf_threshold > 0:
                snip = snip[snip["confidence_score"].fillna(0) >= conf_threshold]
            snip = snip.sort_values("confidence_score", ascending=False)

        if keyword_search.strip():
            snip = snip[snip["chunk_text"].str.contains(
                re.escape(keyword_search.strip()), case=False, na=False)]

        total = len(snip)
        limit = 30
        st.caption(
            f"{total:,} snippet(s) matched"
            + (f" — showing top {limit}" if total > limit else "")
        )

        if snip.empty:
            st.info("No snippets match the current filters.")
        else:
            for _, row in snip.head(limit).iterrows():
                cv = row.get("confidence_score")
                badge = (
                    f'&nbsp;<span style="background:#EFF6FF;color:{_ACCENT};'
                    f'font-size:.67rem;padding:1px 5px;border-radius:999px;font-weight:600">'
                    f'{cv:.3f}</span>'
                    if cv is not None and pd.notna(cv) else ""
                )
                text = str(row["chunk_text"])
                if keyword_search.strip():
                    text = re.sub(
                        f"({re.escape(keyword_search.strip())})",
                        r"<mark>\1</mark>", text, flags=re.IGNORECASE,
                    )
                st.markdown(
                    f'<div class="co2m-snippet">'
                    f'<div class="meta">'
                    f'{row["filename"]}&nbsp;·&nbsp;p.{row["page_number"]}{badge}'
                    f'</div>'
                    f'<div class="body">{text}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
