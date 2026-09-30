| Proxy band        | Interpretation for this knowledge base                       | Typical evidence                                                                                                                             |
| ----------------- | ------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------- |
| **TRL 2–3 proxy** | Laboratory concept / proof-of-concept                        | mg–100 g materials; mL–~1 L liquids; small flask/beaker/autoclave/cell; laboratory batch tests; coupon-scale experiments                     |
| **TRL 4–5 proxy** | Integrated laboratory / bench / engineering-scale validation | ~0.1–10 kg material; ~1–100 L vessels; continuous-flow bench systems; kg/day-scale processing; integrated skid; longer operation             |
| **TRL 6–7 proxy** | Pilot / demonstration-relevant operation                     | tens–thousands kg; >100 L systems; pilot plants; slipstreams; kg/h or ton/day processing; field/site systems; sustained integrated operation |

Use two passes.

Pass 1 — Extract factual scale evidence

Qwen should only extract what the article actually says.

Pass 2 — Assign the proxy band

A second prompt receives the extracted evidence and performs the classification.

This separates information extraction from judgment, which will make auditing much easier.




PROMPT 1 — Experimental scale evidence extractor

You are a scientific information-extraction model supporting the development
of a curated process-engineering scale-up knowledge base.

Your task is NOT to determine whether the research is good or bad and NOT
to guess the Technology Readiness Level.

Your task is to identify factual evidence describing the physical scale,
equipment scale, processing capacity, and experimental maturity of the work
actually performed in the supplied document.

IMPORTANT RULES

1. Extract evidence describing experiments or systems actually performed,
   constructed, tested, or operated in THIS DOCUMENT.

2. Do NOT use quantities, equipment, or TRL values that merely describe:
   - cited previous studies,
   - background literature,
   - future proposed work,
   - commercial technologies mentioned for comparison,
   - supplier specifications unrelated to the experiment,
   - analytical characterization equipment such as SEM, TEM, XRD, GC,
     HPLC, ICP-MS, TGA, BET, etc.

3. Distinguish process-scale quantities from analytical/sample-preparation
   quantities.

Example:
"10 mg was taken for TGA analysis" is NOT evidence that the process itself
operated at 10 mg scale.

4. Determine whether the document is:
   EXPERIMENTAL_RESEARCH
   TECHNICAL_REPORT
   PATENT_WITH_EXAMPLES
   REVIEW
   PERSPECTIVE
   MODELING_ONLY
   OTHER

5. If the document is primarily a REVIEW, systematic review, perspective,
   commentary, or literature survey and reports no original experimental
   process-scale work, set:

   exclude_from_scale_classification = true

6. If an explicit Technology Readiness Level is stated for the technology
   investigated in this document, extract it verbatim. Do NOT infer an
   explicit TRL.

7. Extract all relevant quantitative scale indicators where available:

   - material mass
   - reagent/feed mass
   - liquid/reagent volume
   - reactor/vessel volume
   - reactor dimensions
   - electrode area
   - membrane area
   - sorbent/catalyst amount
   - gas flow rate
   - liquid flow rate
   - feed rate
   - throughput
   - production capacity
   - energy/power rating
   - batch size
   - duration of continuous operation
   - number of integrated processing units

8. Extract qualitative scale indicators such as:

   vial
   tube
   flask
   beaker
   laboratory reactor
   autoclave
   Parr reactor
   batch reactor
   continuous reactor
   packed column
   bench-scale unit
   skid
   prototype
   pilot unit
   pilot plant
   slipstream
   demonstration plant
   field test
   industrial facility
   commercial plant

9. Identify the largest PRIMARY process experiment actually performed.
   Do not choose an analytical measurement or peripheral experiment.

10. Never invent a quantity or convert an unspecified amount into an
    estimated amount.

Return ONLY valid JSON.


As below

Analyze the following scientific document.

Extract evidence related to the physical/process scale of the work actually
performed in this document.


##output
DOCUMENT TEXT:
{{DOCUMENT_TEXT}}

Return this JSON structure:

{
  "document_type": "",
  "exclude_from_scale_classification": false,

  "explicit_trl": {
    "reported": false,
    "trl_value": null,
    "evidence": null
  },

  "primary_process": "",

  "quantitative_scale_evidence": [
    {
      "parameter": "",
      "value": "",
      "unit": "",
      "context": "",
      "is_primary_process_scale": true,
      "evidence_quote": ""
    }
  ],

  "equipment_scale_evidence": [
    {
      "equipment": "",
      "size_or_capacity": "",
      "context": "",
      "evidence_quote": ""
    }
  ],

  "operating_scale_evidence": {
    "batch_or_continuous": "",
    "throughput": "",
    "operating_duration": "",
    "power_or_capacity": "",
    "integration_description": "",
    "operating_environment": ""
  },

  "scale_terms_found": [],

  "largest_demonstrated_scale": {
    "description": "",
    "evidence_quote": ""
  },

  "evidence_sufficient_for_scale_classification": true,

  "extraction_notes": ""
}




PROMPT 2 — Scale-derived TRL proxy classifier

You are classifying scientific documents according to the physical and
engineering scale demonstrated by the work.

The classification is called a SCALE-DERIVED TRL PROXY.

It is NOT an official Technology Readiness Level.

Use only the evidence supplied in the extracted record. Do not introduce
outside knowledge about the technology or authors.

Assign one of:

TRL_2_3_PROXY
TRL_4_5_PROXY
TRL_6_7_PROXY
INSUFFICIENT_EVIDENCE
REVIEW_EXCLUDED

Use the following decision framework.

A. TRL_2_3_PROXY — Laboratory / proof-of-concept scale

Typical evidence may include:

- milligram to approximately 100 gram-scale primary material quantities
- milliliter to approximately 1 liter-scale primary process volumes
- small laboratory vessels, vials, beakers, flasks, tube reactors,
  laboratory cells, small autoclaves
- coupon-scale or single-component experiments
- small laboratory batch experiments
- low-throughput laboratory flow experiments
- primary purpose is demonstrating feasibility, mechanism, material
  behavior, or proof of concept

Examples of evidence:
10 mg catalyst
25 g sorbent
250 mL reaction mixture
500 mL flask
1 L laboratory reactor
small electrochemical cell

These quantities are guidance rather than absolute thresholds.


B. TRL_4_5_PROXY — Integrated laboratory / bench / engineering scale

Typical evidence may include:

- approximately 0.1–10 kg-scale material processing
- approximately 1–100 L primary process vessels
- meaningful continuous feed or product throughput
- bench-scale continuous reactors
- integrated process units
- engineering-scale prototypes
- laboratory pilot skids
- multi-component systems operating together
- sustained operation demonstrating process performance
- kg/day-scale processing

An experimental system may qualify even if one individual material
quantity is small when the overall integrated system clearly operates at
bench or engineering scale.


C. TRL_6_7_PROXY — Pilot / demonstration-relevant scale

Typical evidence may include:

- tens of kilograms or more of material processing
- approximately 100 L or larger primary systems
- substantial continuous throughput
- kg/hour, tens or hundreds of kg/day, ton/day, or larger processing
- pilot plant or demonstration system
- industrial slipstream testing
- field/site operation
- operation using industrially relevant feeds in an integrated system
- significant power rating such as multi-kW to MW systems when appropriate
  to the technology
- sustained pilot operation
- integrated prototype demonstrated outside simple laboratory conditions

Explicit use of "pilot", "pilot plant", "demonstration", "slipstream",
or equivalent terms is strong evidence ONLY when the authors actually
operated or tested such a system.


D. INSUFFICIENT_EVIDENCE

Use when the paper does not provide enough information to estimate physical
or engineering scale reliably.


E. REVIEW_EXCLUDED

Use for review articles, perspectives, commentaries, meta-analyses, or other
documents without original experimental process-scale work.

## Applying the scale tags

Save the Prompt 1 JSON objects as JSONL (one extracted record per line) or as
a JSON array, then run:

```powershell
python co2m/metadata/classify_scale_trl.py --input co2m/metadata/scale_evidence.jsonl
```

The script writes `co2m/metadata/scale_trl_tags.json`. Each record includes a
`scale_derived_trl_proxy` object with a classification tag, the verbatim
explicit-TRL field, reasons, and the original extracted evidence. The top-level
`coverage` object counts each proxy band, insufficient-evidence records, and
review exclusions. The proxy is not an official TRL; it does not replace or
modify existing metadata.

To run both prompts against the 3,899-PDF target list, use:

```powershell
python co2m/metadata/run_scale_trl_pipeline.py
```

The default target list is `qwen80b_10903_unique_pdfs_metadata.json`, with
filenames resolved against `all_pdfs/`. The resumable outputs are
`scale_trl_two_pass_3899.jsonl` (per-document evidence and tags),
`scale_trl_two_pass_3899_coverage.json` (coverage counts), and an issues JSONL.
Run a small smoke test first with `--limit 2`; subsequent runs skip documents
already completed in the output JSONL.