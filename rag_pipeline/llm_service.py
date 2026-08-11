"""
llm_service.py
LLM integration for RAG pipeline - generates responses using retrieved context.
"""

import logging
import os
from typing import Dict, Any, List, Optional
import requests

logger = logging.getLogger(__name__)


class LLMService:
    """LLM service for generating responses from RAG context."""

    def __init__(self, provider: str = "auto", model: str = None, api_key: str = None):
        """Initialize LLM service with auto-detection of available providers.

        Args:
            provider: LLM provider ('auto', 'openai', 'anthropic', 'ollama', 'huggingface', 'together')
                     'auto' attempts: ollama → together → huggingface → openai
            model: Model name (e.g., 'gpt-3.5-turbo', 'llama2')
            api_key: API key for the provider (defaults to env variable)
        """
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key or os.getenv(f"{provider.upper()}_API_KEY")
        self.client = None
        self.available = False
        self.auto_provider = None

        # Auto-detect available provider if 'auto' is specified
        if self.provider == "auto":
            self.provider, self.auto_provider = self._auto_detect_provider()

        self._initialize_client()

    def _auto_detect_provider(self) -> tuple:
        """Auto-detect available LLM provider (free options first).
        
        Returns:
            Tuple of (selected_provider, display_name)
        """
        # Try Ollama first (free, local)
        if self._check_ollama_available():
            logger.info("Auto-detected: Ollama (free local inference)")
            return "ollama", "ollama"
        
        # Try HuggingFace (free tier available, no key needed for basic inference)
        try:
            from huggingface_hub import InferenceClient
            logger.info("HuggingFace inference available as fallback")
            return "huggingface_fallback", "huggingface"
        except ImportError:
            pass
        
        # Try HuggingFace with API key if set
        hf_key = os.getenv("HUGGINGFACE_API_KEY")
        if hf_key:
            logger.info("Auto-detected: HuggingFace (using API key)")
            return "huggingface", "huggingface"
        
        # Fall back to OpenAI if key is set
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            logger.info("Auto-detected: OpenAI (using API key)")
            return "openai", "openai"
        
        # Default to Ollama (will show as unavailable if not installed)
        logger.warning("No available LLM provider detected, defaulting to Ollama")
        return "ollama", "ollama"

    def _check_ollama_available(self) -> bool:
        """Check if Ollama is running locally."""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    def _check_together_ai_available(self) -> bool:
        """Check if Together.ai endpoint is accessible (free tier)."""
        try:
            response = requests.head("https://api.together.xyz", timeout=2)
            return True
        except Exception:
            return False

    def _initialize_client(self):
        """Initialize the appropriate LLM client based on provider."""
        try:
            if self.provider == "openai":
                import openai
                self.client = openai.OpenAI(api_key=self.api_key)
                self.model = self.model or "gpt-3.5-turbo"
                self.available = bool(self.api_key)
                logger.info(f"OpenAI client initialized with model: {self.model}")

            elif self.provider == "anthropic":
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
                self.model = self.model or "claude-3-sonnet-20240229"
                self.available = bool(self.api_key)
                logger.info(f"Anthropic client initialized with model: {self.model}")

            elif self.provider == "ollama":
                import ollama
                self.client = ollama
                self.model = self.model or "llama2"
                # Test if Ollama is actually running
                self.available = self._check_ollama_available()
                if self.available:
                    logger.info(f"Ollama client initialized with model: {self.model}")
                else:
                    logger.warning("Ollama not running - install and start with: ollama serve")

            elif self.provider == "together":
                # Together.ai free tier (no key needed)
                self.client = "together"
                self.model = self.model or "meta-llama/Llama-2-7b-chat-hf"
                self.available = self._check_together_ai_available()
                if self.available:
                    logger.info(f"Together.ai free tier initialized with model: {self.model}")
                else:
                    logger.warning("Together.ai endpoint not available")

            elif self.provider == "huggingface" or self.provider == "huggingface_fallback":
                try:
                    from huggingface_hub import InferenceClient
                    self.client = InferenceClient(api_key=self.api_key) if self.api_key else InferenceClient()
                    self.model = self.model or "mistralai/Mistral-7B-Instruct-v0.1"
                    self.available = True
                    logger.info(f"Hugging Face client initialized with model: {self.model}")
                except Exception as e:
                    logger.warning(f"Failed to initialize HuggingFace: {e}")
                    self.available = False

            else:
                logger.warning(f"Unknown provider: {self.provider}")
                self.available = False

        except ImportError as e:
            logger.error(f"Failed to import {self.provider} library: {e}")
            logger.error(f"Install with: pip install {self._get_install_package()}")
            self.available = False
        except Exception as e:
            logger.error(f"Failed to initialize {self.provider} client: {e}")
            self.available = False

    def _get_install_package(self) -> str:
        """Get the pip package name for the provider."""
        packages = {
            "openai": "openai",
            "anthropic": "anthropic",
            "ollama": "ollama",
            "together": "together",
            "huggingface": "huggingface-hub",
        }
        return packages.get(self.provider, self.provider)

    def generate(
        self,
        question: str,
        context: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
        system_prompt: str = None,
    ) -> Dict[str, Any]:
        """Generate LLM response using retrieved context.

        Args:
            question: Original user question
            context: Retrieved context from RAG pipeline
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum response length
            system_prompt: Custom system prompt

        Returns:
            Dict with generated response and metadata
        """
        result = {
            "question": question,
            "provider": self.provider,
            "model": self.model,
            "response": "",
            "usage": {},
            "error": None,
        }

        if not self.client:
            # If no client, use fallback response generation
            if self.provider == "huggingface_fallback":
                result.update(self._generate_fallback_response(question, context))
                return result
            result["error"] = f"LLM client not initialized for {self.provider}"
            logger.error(result["error"])
            return result

        # Check API key for paid providers only
        if not self.api_key and self.provider not in ["ollama", "together", "huggingface_fallback"]:
            result["error"] = f"API key not configured for {self.provider}"
            logger.error(result["error"])
            return result

        try:
            if self.provider == "openai":
                result.update(self._generate_openai(question, context, temperature, max_tokens, system_prompt))
            elif self.provider == "anthropic":
                result.update(self._generate_anthropic(question, context, temperature, max_tokens, system_prompt))
            elif self.provider == "ollama":
                result.update(self._generate_ollama(question, context, temperature, max_tokens, system_prompt))
            elif self.provider == "together":
                result.update(self._generate_together(question, context, temperature, max_tokens, system_prompt))
            elif self.provider == "huggingface" or self.provider == "huggingface_fallback":
                # Try HuggingFace, fall back to structured response if it fails
                try:
                    result.update(self._generate_huggingface(question, context, temperature, max_tokens, system_prompt))
                except Exception as hf_error:
                    logger.warning(f"HuggingFace generation failed: {str(hf_error)}, using fallback")
                    result.update(self._generate_fallback_response(question, context))

            result["status"] = "success" if not result.get("error") else "failed"

        except Exception as e:
            result["error"] = str(e)
            result["status"] = "failed"
            logger.error(f"Generation failed: {e}")
            # Try fallback response as last resort
            try:
                result.update(self._generate_fallback_response(question, context))
                result["error"] = None
            except:
                pass

        return result

    def _generate_openai(
        self,
        question: str,
        context: str,
        temperature: float,
        max_tokens: int,
        system_prompt: str,
    ) -> Dict[str, Any]:
        """Generate using OpenAI API."""
        system_prompt = system_prompt or (
            "You are an expert scientific assistant specializing in CO2 capture and carbon sequestration. "
            "Provide comprehensive, well-structured answers based on the provided research context. "
            "Format your response with:\n"
            "- Clear main points using bullet points\n"
            "- Bold text (**text**) for important terms and categories\n"
            "- Organized sections like Pros, Cons, Applications, etc.\n"
            "- Specific numbers and data when available\n"
            "- Citations to the source documents provided\n"
            "Always cite sources and acknowledge limitations in the data."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {question}\n\nResearch Context:\n{context}"},
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return {
            "response": response.choices[0].message.content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        }

    def _generate_anthropic(
        self,
        question: str,
        context: str,
        temperature: float,
        max_tokens: int,
        system_prompt: str,
    ) -> Dict[str, Any]:
        """Generate using Anthropic Claude API."""
        system_prompt = system_prompt or (
            "You are an expert scientific assistant specializing in CO2 capture and carbon sequestration. "
            "Provide comprehensive, well-structured answers based on the provided research context. "
            "Format your response with:\n"
            "- Clear main points using bullet points\n"
            "- Bold text (**text**) for important terms and categories\n"
            "- Organized sections like Pros, Cons, Applications, etc.\n"
            "- Specific numbers and data when available\n"
            "- Citations to the source documents provided\n"
            "Always cite sources and acknowledge limitations in the data."
        )

        message = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nResearch Context:\n{context}",
                }
            ],
            temperature=temperature,
        )

        return {
            "response": message.content[0].text,
            "usage": {
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens,
            },
        }

    def _generate_ollama(
        self,
        question: str,
        context: str,
        temperature: float,
        max_tokens: int,
        system_prompt: str,
    ) -> Dict[str, Any]:
        """Generate using Ollama (local LLM)."""
        system_prompt = system_prompt or (
            "You are an expert scientific assistant specializing in CO2 capture. "
            "Provide comprehensive, well-structured answers. Use bullet points, bold text, "
            "and organize information clearly. Cite sources when possible."
        )

        prompt = f"System: {system_prompt}\n\nQuestion: {question}\n\nContext:\n{context}\n\nAnswer:"

        response = self.client.generate(
            model=self.model,
            prompt=prompt,
            stream=False,
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        )

        return {
            "response": response.get("response", ""),
            "usage": {
                "tokens": response.get("eval_count", 0),
            },
        }

    def _generate_together(
        self,
        question: str,
        context: str,
        temperature: float,
        max_tokens: int,
        system_prompt: str,
    ) -> Dict[str, Any]:
        """Generate using Together.ai free tier or fallback to simple response."""
        system_prompt = system_prompt or (
            "You are an expert scientific assistant specializing in CO2 capture and carbon sequestration. "
            "Provide comprehensive, well-structured answers based on the provided research context. "
            "Format your response with:\n"
            "- Clear main points using bullet points\n"
            "- Bold text (**text**) for important terms and categories\n"
            "- Organized sections like Pros, Cons, Applications, etc.\n"
            "- Specific numbers and data when available\n"
            "- Citations to the source documents provided\n"
            "Always cite sources and acknowledge limitations in the data."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {question}\n\nResearch Context:\n{context}"},
        ]

        headers = {
            "Content-Type": "application/json",
        }
        
        # Add auth token if available (for paid/free tier with account)
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            response = requests.post(
                "https://api.together.xyz/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=20,
            )
            
            if response.status_code == 200:
                data = response.json()
                if "choices" in data and len(data["choices"]) > 0:
                    return {
                        "response": data["choices"][0]["message"]["content"],
                        "usage": data.get("usage", {"total_tokens": 0}),
                    }
                else:
                    logger.warning("Unexpected Together.ai response format")
                    return self._generate_fallback_response(question, context)
            else:
                logger.warning(f"Together.ai API error {response.status_code}, using fallback")
                return self._generate_fallback_response(question, context)
                
        except Exception as e:
            logger.warning(f"Together.ai request failed: {str(e)}, using fallback")
            return self._generate_fallback_response(question, context)

    def _generate_fallback_response(self, question: str, context: str) -> Dict[str, Any]:
        """Generate a structured, synthesized summary response from RAG context."""
        import re

        # ── 1. Split context into labelled source blocks ──────────────────────
        source_blocks = re.split(r'\[Source \d+[^\]]*\]', context)
        source_headers = re.findall(r'\[Source \d+[^\]]*\]', context)

        # ── 2. Filter out reference-list-heavy blocks ─────────────────────────
        def is_reference_dump(text: str) -> bool:
            """True when the passage is mostly bibliography entries."""
            ref_tags = len(re.findall(r'\[\d+\]', text))
            words = max(len(text.split()), 1)
            # Lower threshold — even 3 refs per 100 words flags a bibliography block
            return (ref_tags / words) >= 0.03

        def extract_content(block: str) -> str:
            """Return the text content portion of a block (before File: line)."""
            lines = block.split('\n')
            content_lines = [l for l in lines if not l.strip().startswith('File:')]
            return ' '.join(content_lines).strip()

        # Extract filenames from File: lines in each block
        def extract_file_info(block: str) -> str:
            m = re.search(r'File:\s*(.+?),\s*Page:\s*(\S+)', block)
            if m:
                return f"{m.group(1)} (Page {m.group(2)})"
            return None

        source_file_infos = []
        seen_filenames = set()
        clean_blocks = []
        seen_texts = set()
        for header, block in zip(source_headers, source_blocks[1:]):
            block = block.strip()
            if not block:
                continue
            finfo = extract_file_info(block)
            if finfo:
                fname = finfo.split(' (Page')[0]
                if fname not in seen_filenames:
                    seen_filenames.add(fname)
                    source_file_infos.append(finfo)
            content = extract_content(block)
            # De-duplicate identical source blocks
            key = content[:120]
            if key in seen_texts:
                continue
            seen_texts.add(key)
            # Skip paper-header blocks (author lists, institution lines)
            is_header_block = bool(re.search(
                r'\baDept\b|\bbDept\b|\bCorresponding author\b|\bNumber of pages\b|\bNumber of figures\b', content))
            if not is_reference_dump(content) and not is_header_block:
                clean_blocks.append((header, content))

        # Fall back to stripped passages if all were reference-heavy
        if not clean_blocks:
            seen_texts2 = set()
            for header, block in zip(source_headers, source_blocks[1:]):
                block = block.strip()
                content = extract_content(block)
                # Strip inline citations [1], [2], …
                content = re.sub(r'\[\d+\]', '', content).strip()
                key = content[:120]
                if key in seen_texts2 or len(content) < 60:
                    continue
                seen_texts2.add(key)
                clean_blocks.append((header, content))

        # ── 3. Extract structured entities ────────────────────────────────────
        full_text = context

        # Materials / chemical names
        mat_patterns = [
            r'\bmonoethanolamine\b', r'\bdiethanolamine\b', r'\bpiperazine\b',
            r'\bMEA\b', r'\bDEA\b', r'\bMDEA\b', r'\bPZ\b',
            r'\bzeolite[s]?\b', r'\bMOF[s]?\b', r'\bactivated carbon\b',
            r'\bamine[s]?\b', r'\bsolvent[s]?\b', r'\bsorbent[s]?\b',
            r'\badsorbent[s]?\b', r'\babsorbent[s]?\b',
            r'\bcalcium oxide\b', r'\bCaO\b', r'\bpotassium carbonate\b',
            r'\bK2CO3\b', r'\bsodium hydroxide\b', r'\bNaOH\b',
            r'\bsilica[te]?\b', r'\bcement\b', r'\bbasalt\b', r'\bperidotite[s]?\b',
            r'\bmineral[s]?\b', r'\bcarbonate[s]?\b', r'\bwollastonite\b',
            r'\bserpentine\b', r'\bolivine\b',
        ]
        found_materials = []
        for pat in mat_patterns:
            matches = re.findall(pat, full_text, re.IGNORECASE)
            if matches:
                canonical = matches[0].strip().title()
                if canonical not in found_materials:
                    found_materials.append(canonical)

        # Performance numbers / percentages
        pct_matches = re.findall(r'(\d+(?:\.\d+)?)\s*%', full_text)
        pcts = sorted(set(float(p) for p in pct_matches if 10 <= float(p) <= 100), reverse=True)

        # Temperature / pressure values
        temps = re.findall(r'(\d+(?:\.\d+)?)\s*°[CF]', full_text)
        pressures = re.findall(r'(\d+(?:\.\d+)?)\s*(?:bar|MPa|kPa|atm)', full_text, re.IGNORECASE)

        # Pros-like language — filter out short or bibliography-style fragments
        def is_bib_fragment(s: str) -> bool:
            """Detect bibliography entries masquerading as sentences."""
            return bool(re.search(r'\bvol\b|\bpp\b|\bDOE\b|\bAgency\b|\bpress\b', s, re.I)) \
                or len(s.split()) < 12

        pro_sentences, con_sentences = [], []
        for sent in re.split(r'[.!?]', full_text):
            s = sent.strip()
            if not s or len(s) < 40 or is_bib_fragment(s):
                continue
            if re.search(r'\b(high|efficient|effective|commercial|proven|low cost|fast|selective)\b', s, re.I):
                pro_sentences.append(s)
            if re.search(r'\b(energy.intensive|degradation|corros|toxic|expensive|limitation|drawback|challenge)\b', s, re.I):
                con_sentences.append(s)

        # ── 4. Build formatted response ────────────────────────────────────────
        q_lower = question.lower()
        is_material_q = any(w in q_lower for w in ['material', 'compound', 'chemical', 'what', 'which', 'type', 'kind', 'sorbent', 'solvent', 'capture'])

        lines_out = []
        lines_out.append(f"**Summary: {question}**\n")

        # Materials section
        if is_material_q and found_materials:
            lines_out.append("**Materials / Compounds Identified in Literature:**\n")
            for mat in found_materials[:8]:
                lines_out.append(f"- **{mat}**")
            lines_out.append("")

        # Key findings from clean passages
        lines_out.append("**Key Findings from Research Literature:**\n")
        shown = 0
        for header, block in clean_blocks[:5]:
            excerpt = block[:400].replace('\n', ' ').strip()
            if len(block) > 400:
                excerpt += "…"
            lines_out.append(f"- {excerpt}")
            shown += 1
        if shown == 0:
            lines_out.append("- No content passages were available; see source excerpts in the Detailed Results below.")
        lines_out.append("")

        # Pros / Cons if sentences found
        if pro_sentences or con_sentences:
            if pro_sentences:
                lines_out.append("**Advantages noted:**\n")
                for s in pro_sentences[:3]:
                    lines_out.append(f"- {s.strip()}")
                lines_out.append("")
            if con_sentences:
                lines_out.append("**Challenges / Limitations noted:**\n")
                for s in con_sentences[:3]:
                    lines_out.append(f"- {s.strip()}")
                lines_out.append("")

        # Quantitative data
        quant_items = []
        if pcts:
            quant_items.append(f"Efficiencies up to **{pcts[0]:.0f}%** mentioned in the literature")
        if temps:
            quant_items.append(f"Operating temperatures: **{', '.join(temps[:3])}°**")
        if pressures:
            quant_items.append(f"Pressures: **{', '.join(pressures[:3])}**")
        if quant_items:
            lines_out.append("**Quantitative Data:**\n")
            for item in quant_items:
                lines_out.append(f"- {item}")
            lines_out.append("")

        # Sources cited
        if source_file_infos:
            lines_out.append("**Sources consulted:**\n")
            for finfo in source_file_infos[:5]:
                lines_out.append(f"- {finfo}")
            lines_out.append("")

        lines_out.append("*See Detailed Results below for full source excerpts.*")

        fallback_response = "\n".join(lines_out)

        return {
            "response": fallback_response,
            "usage": {"total_tokens": len(fallback_response.split())},
        }

    def _generate_huggingface(
        self,
        question: str,
        context: str,
        temperature: float,
        max_tokens: int,
        system_prompt: str,
    ) -> Dict[str, Any]:
        """Generate using Hugging Face Inference API."""
        system_prompt = system_prompt or (
            "You are an expert scientific assistant specializing in CO2 capture. "
            "Provide comprehensive, well-structured answers. Use bullet points, bold text, "
            "and organize information clearly. Cite sources when possible."
        )
        
        prompt = f"{system_prompt}\n\nQuestion: {question}\n\nContext:\n{context}\n\nAnswer:"

        response = self.client.text_generation(
            prompt,
            model=self.model,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )

        return {
            "response": response,
            "usage": {},
        }

    def generate_with_rag(
        self,
        question: str,
        retrieved_results: List[Dict],
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> Dict[str, Any]:
        """Generate response using RAG retrieved results.

        Args:
            question: User question
            retrieved_results: List of retrieved document chunks
            temperature: Sampling temperature
            max_tokens: Maximum response tokens

        Returns:
            Dict with generated response, sources, and metadata
        """
        # Format context from retrieved results
        context_parts = []
        sources = []

        for i, result in enumerate(retrieved_results, 1):
            context_parts.append(
                f"[Source {i}] {result.get('text', '')}"
                f"\n  File: {result.get('filename', 'Unknown')}, Page: {result.get('page_number', 'N/A')}"
            )
            sources.append({
                "rank": result.get("rank", i),
                "file": result.get("filename", "Unknown"),
                "page": result.get("page_number", "N/A"),
                "similarity": result.get("similarity", 0),
            })

        context = "\n\n".join(context_parts)

        # Generate response
        generation_result = self.generate(question, context, temperature, max_tokens)

        # Add source information
        generation_result["sources"] = sources
        generation_result["num_sources"] = len(sources)

        return generation_result

    def is_available(self) -> bool:
        """Check if LLM service is available."""
        return self.available

    def get_status(self) -> Dict[str, Any]:
        """Get LLM service status."""
        return {
            "provider": self.provider,
            "model": self.model,
            "available": self.available,
            "auto_detected": self.auto_provider is not None,
            "has_api_key": bool(self.api_key),
            "client_initialized": self.client is not None,
        }
