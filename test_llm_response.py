#!/usr/bin/env python
"""Test LLM response generation."""

from rag_pipeline import RAGQueryService
import json

# Initialize service
query_service = RAGQueryService()

# Test query with LLM
result = query_service.generate_with_llm(
    question='What materials capture CO2?',
    top_k=5,
    concept_filter=None,
    temperature=0.7,
    max_tokens=500
)

print('=== RESULT STRUCTURE ===')
print(f'Keys: {list(result.keys())}')
print(f'LLM Response (raw): {repr(result.get("llm_response"))}')
print(f'LLM Response length: {len(result.get("llm_response", ""))}')
print(f'LLM Provider: {result.get("llm_provider")}')
print(f'LLM Model: {result.get("llm_model")}')
print(f'Error: {result.get("error")}')
print()

# Show first 500 chars of response
resp = result.get("llm_response", "")
if resp:
    print('=== RESPONSE START ===')
    print(resp[:500])
    print('=== RESPONSE END ===')
else:
    print('=== NO RESPONSE GENERATED ===')
    
print()
print('=== ALL RESPONSE KEYS ===')
for k, v in result.items():
    if k == 'retrieved_results':
        print(f'{k}: {len(v)} items')
    elif k == 'llm_response':
        print(f'{k}: {len(v)} chars - {repr(v[:100])}')
    else:
        print(f'{k}: {repr(v)[:150]}')
