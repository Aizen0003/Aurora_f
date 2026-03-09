"""
Knowledge Bank Engine - Extracts company intelligence from documents

This engine is domain-agnostic and adaptively extracts intelligence from
various knowledge bank formats (PDFs, docs, structured/unstructured text).

Supports two extraction modes:
  1. RAG-lite: PDF → Semantic Chunking → LLM (Groq) → TF-IDF Cosine Ranking → Structured Extraction
  2. Regex Fallback: Pattern-matching heuristics for when no LLM is available
"""

import json
import re
import os
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
import yaml

# Optional RAG-lite dependencies
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


class KnowledgeBankEngine:
    """Extracts and structures company knowledge from text documents.

    Supports RAG-lite mode (PDF + LLM) and regex fallback mode.
    """

    def __init__(self, config_path: str = 'config/config.yaml'):
        self.north_star = None
        self.feature_goal_map = None
        self.tone_hook_matrix = None
        self.detected_domain = None
        self.rag_mode_used = False

        # Load config for tones and other configurable values
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            self.config = {}

        # Get communication config with defaults
        comm_config = self.config.get('communication', {})
        self.allowed_tones = comm_config.get('allowed_tones', [
            'encouraging', 'friendly', 'aspirational', 'urgent',
            'celebratory', 'motivational', 'supportive'
        ])
        self.forbidden_tones = comm_config.get('forbidden_tones', [
            'aggressive', 'desperate', 'salesy', 'guilt-tripping',
            'condescending', 'pushy'
        ])
        self.tone_by_lifecycle = comm_config.get('tone_by_lifecycle', {
            'trial': ['encouraging', 'friendly', 'aspirational', 'supportive'],
            'paid': ['celebratory', 'friendly', 'aspirational', 'motivational'],
            'churned': ['friendly', 'urgent', 'aspirational', 'curious'],
            'inactive': ['curious', 'friendly', 'inviting']
        })

        # KB / RAG config
        kb_config = self.config.get('knowledge_bank', {})
        self.kb_mode = kb_config.get('mode', 'auto')  # auto | rag | regex
        self.llm_model = kb_config.get('llm_model', 'llama-3.1-8b-instant')
        self.rag_top_chunks = kb_config.get('rag_top_chunks', 15)
        self.rag_vocab_size = kb_config.get('rag_vocab_size', 25)

        # Initialize Groq client if available
        self.groq_client = None
        api_key = os.environ.get("GROQ_API_KEY", "")
        if api_key and GROQ_AVAILABLE:
            try:
                self.groq_client = Groq(api_key=api_key)
            except Exception:
                pass

    # ===================================================================
    # PUBLIC API
    # ===================================================================

    def process_knowledge_bank(self, kb_text_or_pdf: str) -> Dict[str, Any]:
        """
        Process knowledge bank text/PDF and extract all intelligence.

        If a PDF path is provided and RAG-lite dependencies are available,
        uses the RAG-lite pipeline. Otherwise falls back to regex extraction.

        Args:
            kb_text_or_pdf: Raw text OR path to a PDF file

        Returns:
            dict: Structured knowledge bank data
        """
        kb_text = kb_text_or_pdf
        pdf_path = None

        # Check if input is a PDF path
        if kb_text_or_pdf and Path(kb_text_or_pdf).suffix.lower() == '.pdf' and Path(kb_text_or_pdf).exists():
            pdf_path = kb_text_or_pdf

        # Decide mode
        use_rag = False
        if self.kb_mode == 'rag':
            use_rag = True
        elif self.kb_mode == 'auto':
            use_rag = (self.groq_client is not None and SKLEARN_AVAILABLE)
        # mode == 'regex' → use_rag stays False

        if use_rag and pdf_path and PYMUPDF_AVAILABLE:
            print("\n   [RAG] Using RAG-lite pipeline (PDF → LLM → TF-IDF)")
            return self._process_rag_lite(pdf_path)
        elif use_rag and not pdf_path and self.groq_client:
            # Have LLM but no PDF — extract text, still try RAG on raw text
            print("\n   [RAG] Using RAG-lite pipeline (Text → LLM → TF-IDF)")
            return self._process_rag_lite_from_text(kb_text)
        else:
            print("\n   [Regex] Using regex-based extraction (no LLM)")
            return self._process_regex(kb_text)

    # ===================================================================
    # RAG-LITE PIPELINE
    # ===================================================================

    def _process_rag_lite(self, pdf_path: str) -> Dict[str, Any]:
        """Full RAG-lite pipeline from PDF."""
        self.rag_mode_used = True

        # Step 1: Extract text from PDF
        raw_text = self._extract_text_from_pdf(pdf_path)
        return self._process_rag_lite_from_text(raw_text)

    def _process_rag_lite_from_text(self, raw_text: str) -> Dict[str, Any]:
        """RAG-lite pipeline from raw text."""
        self.rag_mode_used = True

        # Step 2: Semantic chunking
        chunks = self._semantic_chunking(raw_text)
        print(f"   [RAG] Created {len(chunks)} semantic chunks")

        # Step 3: LLM Call 1 — Extract domain + dynamic vocabulary
        domain, vocab = self._llm_extract_domain_vocab(raw_text)
        self.detected_domain = domain
        print(f"   [RAG] Detected domain: {domain}")
        print(f"   [RAG] Extracted {len(vocab)} vocabulary terms")

        # Step 4: TF-IDF cosine ranking
        top_chunks = self._rank_chunks_cosine(chunks, vocab)
        print(f"   [RAG] Selected top {len(top_chunks)} relevant chunks")

        # Step 5: LLM Call 2 — Structured intelligence extraction
        extracted = self._llm_extract_intelligence(top_chunks, domain)

        # Build structured outputs from LLM extraction
        self.north_star = self._build_north_star_from_rag(extracted)
        self.feature_goal_map = self._build_feature_goal_map_from_rag(extracted)
        self.tone_hook_matrix = self._extract_tone_hook_matrix(raw_text)

        return {
            'north_star': self.north_star,
            'feature_goal_map': self.feature_goal_map,
            'tone_hook_matrix': self.tone_hook_matrix,
            'detected_domain': self.detected_domain
        }

    def _extract_text_from_pdf(self, path: str) -> str:
        """Extract text from PDF using PyMuPDF."""
        print(f"   [RAG] Extracting text from PDF: {path}")
        try:
            doc = fitz.open(path)
            text = " ".join([page.get_text("text") for page in doc])
            doc.close()
            return text
        except Exception as e:
            print(f"   [WARN] PDF extraction failed: {e}, using fallback text")
            return "Company provides core platform features for user engagement and retention."

    def _semantic_chunking(self, text: str, n_sents: int = 5) -> List[str]:
        """Split text into overlapping semantic chunks."""
        sents = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        step = max(1, n_sents - 2)  # Overlap of 2 sentences
        for i in range(0, len(sents), step):
            chunk_txt = " ".join(sents[i:i + n_sents]).strip()
            if len(chunk_txt.split()) > 15:
                chunks.append(chunk_txt)
        return chunks

    def _llm_extract_domain_vocab(self, text: str) -> tuple:
        """LLM Call 1: Extract domain and dynamic vocabulary."""
        if not self.groq_client:
            return self._detect_domain(text), []

        prompt = f'''Extract JSON from this document text:
1. "domain": Specific business domain (1-3 words, e.g. "EdTech", "FinTech", "Health & Fitness").
2. "vocabulary": Array of exactly {self.rag_vocab_size} business/KPI phrases specific to this document.

Return ONLY valid JSON.
TEXT: {text[:2000]}'''

        try:
            resp = self.groq_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            data = json.loads(resp.choices[0].message.content)
            domain = data.get('domain', 'Generic')
            vocab = data.get('vocabulary', [])
            # Normalize domain to our standard categories
            domain_normalized = self._normalize_domain(domain)
            return domain_normalized, vocab[:self.rag_vocab_size]
        except Exception as e:
            print(f"   [WARN] LLM vocab extraction failed: {e}")
            return self._detect_domain(text), []

    def _normalize_domain(self, domain_str: str) -> str:
        """Normalize LLM-returned domain string to standard category."""
        d = domain_str.lower()
        if any(kw in d for kw in ['edtech', 'education', 'learning', 'tutor']):
            return 'edtech'
        elif any(kw in d for kw in ['fintech', 'finance', 'banking', 'payment']):
            return 'fintech'
        elif any(kw in d for kw in ['health', 'fitness', 'wellness', 'medical']):
            return 'health'
        elif any(kw in d for kw in ['entertainment', 'streaming', 'media', 'video']):
            return 'entertainment'
        elif any(kw in d for kw in ['ecommerce', 'commerce', 'shopping', 'retail']):
            return 'ecommerce'
        elif any(kw in d for kw in ['social', 'community', 'network']):
            return 'social'
        return 'generic'

    def _rank_chunks_cosine(self, chunks: List[str], vocab: List[str]) -> List[Dict]:
        """Rank chunks by TF-IDF cosine similarity against vocabulary."""
        if not chunks or not vocab or not SKLEARN_AVAILABLE:
            # Fallback: return first N chunks
            return [{"text": c, "score": 0.5} for c in chunks[:self.rag_top_chunks]]

        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
        try:
            X_chunks = vectorizer.fit_transform(chunks)
            X_vocab = vectorizer.transform(vocab)
            sim_scores = sklearn_cosine_similarity(X_chunks, X_vocab).mean(axis=1)
            top_idx = np.argsort(sim_scores)[::-1][:self.rag_top_chunks]
            return [{"text": chunks[i], "score": float(sim_scores[i])} for i in top_idx]
        except Exception:
            return [{"text": c, "score": 0.5} for c in chunks[:self.rag_top_chunks]]

    def _llm_extract_intelligence(self, top_chunks: List[Dict], domain: str) -> Dict:
        """LLM Call 2: Extract structured intelligence from top-ranked chunks."""
        if not self.groq_client:
            return {}

        context = "\n---\n".join([c['text'] for c in top_chunks])

        system_prompt = f"Domain: {domain}. Extract intelligence ONLY from the following text chunks."
        user_prompt = f'''Extract JSON with these fields:
{{
  "north_star_metric": {{"name": "...", "definition": "..."}},
  "feature_goal_mapping": [{{"feature": "...", "goal": "..."}}],
  "allowed_tones": ["tone1", "tone2", "tone3"],
  "behavioral_hooks": [
    {{"hook": "MUST be one of: Epic Meaning, Accomplishment, Empowerment, Ownership, Social Influence, Scarcity, Unpredictability, Loss Avoidance",
      "trigger": "...", "reward": "..."}}
  ]
}}
TEXT CHUNKS:
{context}'''

        try:
            resp = self.groq_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            print(f"   [WARN] LLM intelligence extraction failed: {e}")
            return {}

    def _build_north_star_from_rag(self, extracted: Dict) -> Dict:
        """Build north_star dict from RAG-extracted data."""
        ns = extracted.get('north_star_metric', {})
        name = ns.get('name', 'Daily Active Users')
        definition = ns.get('definition', 'Primary success metric measuring product-market fit')

        return {
            "north_star_metric": name,
            "definition": definition,
            "why_it_matters": f"Core indicator of {self.detected_domain or 'product'} success",
            "measurement": f"Metric tracking for {name}",
            "key_drivers": [f.get('feature', 'Core Feature') for f in extracted.get('feature_goal_mapping', [])[:5]]
        }

    def _build_feature_goal_map_from_rag(self, extracted: Dict) -> Dict:
        """Build feature_goal_map dict from RAG-extracted data."""
        features = []
        for idx, item in enumerate(extracted.get('feature_goal_mapping', [])):
            fname = item.get('feature', f'Feature_{idx}')
            fgoal = item.get('goal', 'user_engagement')
            fid = re.sub(r'[^a-z0-9_]', '', fname.lower().replace(' ', '_'))
            features.append({
                "feature_name": fname,
                "feature_id": fid,
                "primary_goal": fgoal,
                "secondary_goals": ["engagement", "retention", "satisfaction"],
                "user_segments_most_relevant": ["all"],
                "engagement_driver_score": round(0.75 + (idx * 0.03), 2),
                "description": f"Feature enabling {fname.lower()} functionality"
            })

        if not features:
            features = self._get_default_features("")

        return {"features": features}

    # ===================================================================
    # REGEX FALLBACK PIPELINE (original logic)
    # ===================================================================

    def _process_regex(self, kb_text: str) -> Dict[str, Any]:
        """Process KB using regex/heuristic extraction (original approach)."""
        self.detected_domain = self._detect_domain(kb_text)

        self.north_star = self._extract_north_star(kb_text)
        self.feature_goal_map = self._extract_feature_goal_map(kb_text)
        self.tone_hook_matrix = self._extract_tone_hook_matrix(kb_text)

        return {
            'north_star': self.north_star,
            'feature_goal_map': self.feature_goal_map,
            'tone_hook_matrix': self.tone_hook_matrix,
            'detected_domain': self.detected_domain
        }

    def _detect_domain(self, text: str) -> str:
        """
        Detect the domain/industry from KB text for context-aware extraction.
        Returns: 'edtech', 'fintech', 'entertainment', 'ecommerce', 'health', 'social', or 'generic'
        """
        text_lower = text.lower()

        domain_indicators = {
            'edtech': ['learn', 'education', 'course', 'lesson', 'practice', 'exercise', 'quiz', 'tutor', 'student', 'teacher', 'skill'],
            'fintech': ['payment', 'transaction', 'wallet', 'money', 'bank', 'finance', 'loan', 'credit', 'invest', 'transfer'],
            'entertainment': ['watch', 'stream', 'video', 'movie', 'show', 'content', 'music', 'play', 'episode', 'series'],
            'ecommerce': ['shop', 'buy', 'cart', 'order', 'product', 'delivery', 'price', 'sale', 'discount', 'checkout'],
            'health': ['health', 'fitness', 'workout', 'exercise', 'diet', 'wellness', 'medical', 'doctor', 'patient', 'prescription'],
            'social': ['friend', 'follow', 'post', 'share', 'comment', 'like', 'message', 'connect', 'network', 'community']
        }

        scores = {}
        for domain, keywords in domain_indicators.items():
            scores[domain] = sum(1 for kw in keywords if kw in text_lower)

        best_domain = max(scores, key=scores.get)
        return best_domain if scores[best_domain] >= 2 else 'generic'

    def _extract_north_star(self, text: str) -> Dict[str, Any]:
        """Extract North Star metric from KB using pattern matching"""

        metric_name = None

        key_metrics_match = re.search(r"(?:key metrics?|north star)[\s\S]{0,500}?([A-Z][^.\n]{10,80}(?:retention|conversion|engagement|revenue|growth|active users))", text, re.IGNORECASE)

        if key_metrics_match:
            candidate = key_metrics_match.group(1).strip()
            candidate = re.sub(r'^[*•\-\d\.\s]+', '', candidate)
            candidate = re.sub(r'\s+', ' ', candidate)
            if len(candidate) > 10 and len(candidate) < 100:
                metric_name = candidate

        if not metric_name:
            patterns = [
                r"north star(?:\s+metric)?(?:\s+is)?:\s*([A-Z][^.\n]{10,80})",
                r"primary metric:\s*([A-Z][^.\n]{10,80})",
                r"goal:\s*\*?\*?([^*\n]{10,80}(?:retention|conversion|engagement))\*?\*?",
            ]

            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    candidate = match.group(1).strip()
                    candidate = re.sub(r'^[*•\-\s]+', '', candidate)
                    if len(candidate) > 10 and len(candidate) < 100:
                        metric_name = candidate
                        break

        if not metric_name or len(metric_name) < 10:
            text_lower = text.lower()
            domain = self.detected_domain or 'generic'

            domain_metrics = {
                'edtech': "Weekly Active Learners",
                'fintech': "Transaction Volume",
                'health': "Daily Active Health Users",
                'entertainment': "Content Engagement Hours",
                'ecommerce': "Purchase Conversion Rate",
                'social': "Daily Active Connections",
                'generic': "Daily Active Users"
            }
            metric_name = domain_metrics.get(domain, "Daily Active Users")

        definition = self._extract_definition(text, metric_name)
        why_matters = self._extract_rationale(text, metric_name)

        north_star = {
            "north_star_metric": metric_name,
            "definition": definition,
            "why_it_matters": why_matters,
            "measurement": f"Metric tracking for {metric_name}",
            "key_drivers": self._extract_key_drivers(text)
        }

        return north_star

    def _extract_definition(self, text: str, metric: str) -> str:
        """Extract definition of the metric"""
        metrics_section = re.search(r"(?:key metrics?)([\s\S]{0,800}?)(?:\n#{1,3}\s|\Z)", text, re.IGNORECASE)

        if metrics_section:
            section = metrics_section.group(1)
            metric_pattern = rf"{re.escape(metric[:20])}[^\n]*?\n\s*[*•\-]?\s*([A-Z][^.\n]{{20,150}})"
            match = re.search(metric_pattern, section, re.IGNORECASE)
            if match:
                defn = match.group(1).strip('* \t')
                if len(defn) > 20:
                    return defn

        if 'retention' in metric.lower():
            return "Percentage of users who remain active and engaged over time"
        elif 'conversion' in metric.lower():
            return "Rate at which trial users convert to paying customers"
        elif 'engagement' in metric.lower():
            return "Users who actively interact with core features daily"
        else:
            return "Primary success metric measuring product-market fit"

    def _extract_rationale(self, text: str, metric: str) -> str:
        """Extract why the metric matters"""
        goal_match = re.search(r"goal:?\s*\*?\*?([^*\n]{15,100})\*?\*?", text, re.IGNORECASE)
        if goal_match:
            return f"Drives {goal_match.group(1).strip().lower()}"

        if 'retention' in metric.lower():
            return "Indicates product stickiness and long-term value creation"
        elif 'conversion' in metric.lower():
            return "Validates product-market fit and monetization effectiveness"
        elif 'engagement' in metric.lower():
            return "Measures active usage and habit formation, leading to retention"
        elif 'revenue' in metric.lower():
            return "Direct indicator of business sustainability and growth"
        else:
            return "Core indicator of product success and user satisfaction"

    def _extract_key_drivers(self, text: str) -> List[str]:
        """Extract key drivers from text"""
        drivers = []

        metrics_section = re.search(r"(?:key metrics?|metrics?|kpis?)[\s:]*\n([\s\S]{0,500}?)(?:\n#{1,3}\s|\n\*\*[A-Z]|\Z)", text, re.IGNORECASE)

        if metrics_section:
            section_text = metrics_section.group(1)
            driver_patterns = [
                r"[*•\-]\s+\*?\*?([^*\n]{15,}?)(?:\*?\*?\s*$|\*?\*?\n)",
                r"^\d+\.\s+([^.\n]{15,})$"
            ]

            for pattern in driver_patterns:
                matches = re.findall(pattern, section_text, re.MULTILINE)
                for match in matches:
                    cleaned = match.strip().strip('*').strip()
                    if len(cleaned) > 15 and len(cleaned) < 100:
                        drivers.append(cleaned)

        if not drivers:
            goal_match = re.search(r"goal:?\s*\*?\*?([^*\n]{15,})\*?\*?", text, re.IGNORECASE)
            if goal_match:
                drivers.append(goal_match.group(1).strip())

        if not drivers:
            text_lower = text.lower()
            if 'retention' in text_lower:
                drivers.append("User retention and engagement")
            if 'conversion' in text_lower:
                drivers.append("Trial to paid conversion")
            if 'engagement' in text_lower:
                drivers.append("Daily active engagement")

        if not drivers:
            drivers = ["User engagement", "Feature adoption", "Retention"]

        drivers = list(dict.fromkeys(drivers))[:5]
        return drivers

    def _extract_feature_goal_map(self, text: str) -> Dict[str, List[Dict]]:
        """Extract feature-to-goal mappings from KB text using multiple adaptive strategies."""
        features = []
        feature_names = []

        section_patterns = [
            r"(?:features?|achieved through|capabilities|key functionality|product offerings?)[\ s:]*\n([\s\S]{0,1200}?)(?:\n#{1,3}\s|\n\*\*[A-Z]|\Z)",
            r"(?:what we offer|how it works|our solution)[\ s:]*\n([\s\S]{0,1200}?)(?:\n#{1,3}\s|\Z)",
            r"(?:product|app|platform) (?:includes?|provides?|offers?)([\s\S]{0,800}?)(?:\n#{1,3}\s|\Z)"
        ]

        for pattern in section_patterns:
            section_match = re.search(pattern, text, re.IGNORECASE)
            if section_match:
                section_text = section_match.group(1)
                bullet_patterns = [
                    r"[*•\-]\s+([A-Z][^*\n]{5,80})",
                    r"\d+[.)\s]+([A-Z][^\n]{5,80})",
                    r"\*\*([^*]{5,50})\*\*",
                ]
                for bp in bullet_patterns:
                    matches = re.findall(bp, section_text)
                    for match in matches:
                        cleaned = match.strip().strip('*').strip()
                        if 5 < len(cleaned) < 80 and cleaned not in feature_names:
                            feature_names.append(cleaned)
                if feature_names:
                    break

        if not feature_names:
            noun_phrase_pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}(?:\s+(?:Feature|Tool|System|Module|Engine|Dashboard|Analytics|Practice|Mode))?)\b"
            matches = re.findall(noun_phrase_pattern, text)
            for match in matches:
                if 5 < len(match) < 60 and match not in feature_names:
                    feature_names.append(match)

        if not feature_names:
            feature_names = self._infer_features_from_domain(text)

        for idx, feature_name in enumerate(feature_names[:8]):
            feature_id = re.sub(r'[^a-z0-9_]', '', feature_name.lower().replace(' ', '_'))
            features.append({
                "feature_name": feature_name.strip(),
                "feature_id": feature_id,
                "primary_goal": self._infer_goal_from_feature(feature_name, text),
                "secondary_goals": self._infer_secondary_goals(feature_name, text),
                "user_segments_most_relevant": ["all"],
                "engagement_driver_score": round(0.75 + (idx * 0.03), 2),
                "description": self._generate_feature_description(feature_name, text)
            })

        if not features:
            features = self._get_default_features(text)

        return {"features": features}

    def _infer_features_from_domain(self, text: str) -> List[str]:
        """Infer likely features based on detected domain."""
        text_lower = text.lower()
        features = []
        domain = getattr(self, 'detected_domain', None) or 'generic'

        domain_features = {
            'edtech': {
                'patterns': {
                    'ai|tutor|mentor': 'AI Tutor',
                    'leaderboard|rank|compete': 'Leaderboard',
                    'streak|daily|habit': 'Streak System',
                    'coin|reward|point': 'Rewards System',
                    'exercise|practice|quiz': 'Practice Exercises',
                    'progress|report|analytics': 'Progress Reports',
                    'gamif': 'Gamification Features'
                },
                'defaults': ['Learning Modules', 'Practice Mode', 'Progress Tracking']
            },
            'fintech': {
                'patterns': {
                    'wallet|balance': 'Digital Wallet',
                    'payment|pay': 'Payments',
                    'transfer|send': 'Money Transfer',
                    'cashback|reward': 'Cashback Rewards',
                    'bill|recharge': 'Bill Payments',
                    'invest|mutual|stock': 'Investments'
                },
                'defaults': ['Wallet', 'Payments', 'Transactions']
            },
            'entertainment': {
                'patterns': {
                    'recommend|discover': 'Recommendations',
                    'watchlist|save': 'Watchlist',
                    'download|offline': 'Downloads',
                    'profile|account': 'User Profiles',
                    'search|browse': 'Content Discovery'
                },
                'defaults': ['Content Library', 'Streaming', 'Personalization']
            },
            'ecommerce': {
                'patterns': {
                    'cart|basket': 'Shopping Cart',
                    'wishlist|save': 'Wishlist',
                    'checkout|buy': 'Checkout',
                    'track|delivery': 'Order Tracking',
                    'review|rating': 'Reviews & Ratings'
                },
                'defaults': ['Product Catalog', 'Shopping Cart', 'Order Management']
            },
            'health': {
                'patterns': {
                    'workout|exercise': 'Workouts',
                    'track|log': 'Activity Tracking',
                    'goal|target': 'Goal Setting',
                    'remind|notification': 'Health Reminders',
                    'meal|diet|nutrition': 'Nutrition Tracking'
                },
                'defaults': ['Activity Tracking', 'Health Dashboard', 'Goals']
            },
            'social': {
                'patterns': {
                    'feed|timeline': 'News Feed',
                    'message|chat': 'Messaging',
                    'profile|account': 'User Profiles',
                    'friend|connect': 'Connections',
                    'group|community': 'Communities'
                },
                'defaults': ['Feed', 'Messaging', 'Profiles']
            },
            'generic': {
                'patterns': {},
                'defaults': ['Core Feature', 'User Dashboard', 'Notifications']
            }
        }

        config = domain_features.get(domain, domain_features['generic'])

        for pattern, feature_name in config['patterns'].items():
            if re.search(pattern, text_lower):
                features.append(feature_name)

        if len(features) < 3:
            for default in config['defaults']:
                if default not in features:
                    features.append(default)
                    if len(features) >= 5:
                        break

        return features

    def _infer_goal_from_feature(self, feature_name: str, text: str) -> str:
        """Infer primary goal from feature name (domain-generic)."""
        feature_lower = feature_name.lower()

        goal_mapping = {
            'tutor': 'skill_development', 'learn': 'skill_development',
            'course': 'knowledge_acquisition', 'practice': 'skill_mastery',
            'exercise': 'skill_development', 'quiz': 'knowledge_assessment',
            'leaderboard': 'competitive_engagement', 'streak': 'habit_formation',
            'coin': 'gamification_engagement', 'reward': 'motivation',
            'badge': 'achievement_recognition', 'progress': 'self_improvement',
            'report': 'performance_awareness',
            'wallet': 'financial_convenience', 'payment': 'transaction_completion',
            'transfer': 'money_movement', 'invest': 'wealth_growth',
            'cashback': 'savings_motivation', 'bill': 'utility_management',
            'content': 'entertainment', 'watch': 'content_consumption',
            'stream': 'content_access', 'recommend': 'content_discovery',
            'download': 'offline_access', 'playlist': 'content_curation',
            'cart': 'purchase_facilitation', 'wishlist': 'purchase_planning',
            'checkout': 'purchase_completion', 'track': 'order_visibility',
            'review': 'social_proof',
            'workout': 'fitness_improvement', 'diet': 'health_management',
            'goal': 'achievement_tracking', 'reminder': 'behavior_reinforcement',
            'friend': 'social_connection', 'message': 'communication',
            'feed': 'content_consumption', 'community': 'belonging',
            'dashboard': 'information_access', 'notification': 'engagement',
            'profile': 'identity_expression', 'setting': 'personalization',
            'search': 'discovery', 'ai': 'personalized_experience'
        }

        for keyword, goal in goal_mapping.items():
            if keyword in feature_lower:
                return goal

        return 'user_engagement'

    def _generate_feature_description(self, feature_name: str, text: str) -> str:
        """Generate a meaningful description for the feature based on context"""
        feature_lower = feature_name.lower()

        escaped_name = re.escape(feature_name[:15])
        context_match = re.search(rf"{escaped_name}[^.]*?([A-Z][^.]+\.)", text, re.IGNORECASE)
        if context_match:
            desc = context_match.group(1).strip()
            if 20 < len(desc) < 150:
                return desc

        description_templates = {
            'tutor': 'AI-powered tutoring for personalized learning',
            'streak': 'Daily engagement tracking to build habits',
            'leaderboard': 'Competitive ranking to motivate users',
            'reward': 'Incentive system to drive engagement',
            'wallet': 'Digital wallet for seamless transactions',
            'payment': 'Quick and secure payment processing',
            'content': 'Rich content library for user engagement',
            'workout': 'Guided workout routines for fitness',
            'feed': 'Personalized content feed',
            'message': 'Real-time messaging functionality',
            'cart': 'Shopping cart for purchase management',
            'track': 'Real-time tracking and monitoring',
        }

        for keyword, desc in description_templates.items():
            if keyword in feature_lower:
                return desc

        return f"Feature enabling {feature_name.lower()} functionality"

    def _infer_secondary_goals(self, feature_name: str, text: str) -> List[str]:
        """Infer secondary goals"""
        return ["engagement", "retention", "satisfaction"]

    def _get_default_features(self, text: str) -> List[Dict]:
        """Get default features when extraction fails"""
        return [
            {
                "feature_name": "Core Feature",
                "feature_id": "core_feature",
                "primary_goal": "user_engagement",
                "secondary_goals": ["retention", "satisfaction"],
                "user_segments_most_relevant": ["all"],
                "engagement_driver_score": 0.80,
                "description": "Primary product feature"
            }
        ]

    def _extract_tone_hook_matrix(self, text: str) -> Dict[str, Any]:
        """
        Extract allowed tones and behavioral hooks.
        Tones come from config; Octalysis 8 Core Drives are universal behavioral
        psychology principles (Yu-kai Chou, 2015) and are standardized.
        """
        domain = getattr(self, 'detected_domain', 'generic')
        hook_examples = self._generate_hook_examples(domain)

        matrix = {
            "allowed_tones": self.allowed_tones,
            "forbidden_tones": self.forbidden_tones,
            "tone_by_lifecycle": self.tone_by_lifecycle,
            "detected_domain": domain,
            "octalysis_hooks": {
                "epic_meaning": {
                    "description": "Be part of something bigger",
                    "examples": hook_examples.get('epic_meaning', ["Join millions of users", "Transform your life", "Unlock your potential"]),
                    "best_for_segments": ["aspirational", "purpose_driven"]
                },
                "accomplishment": {
                    "description": "Make progress and achieve",
                    "examples": hook_examples.get('accomplishment', ["Complete your goal", "Reach the milestone", "Earn rewards"]),
                    "best_for_segments": ["achievers", "goal_oriented"]
                },
                "empowerment": {
                    "description": "Have control and experiment",
                    "examples": hook_examples.get('empowerment', ["Choose your path", "Customize your experience", "Do it your way"]),
                    "best_for_segments": ["casual", "explorers"]
                },
                "ownership": {
                    "description": "Build something valuable",
                    "examples": hook_examples.get('ownership', ["Your progress", "Your collection", "Your achievements"]),
                    "best_for_segments": ["achievers", "collectors"]
                },
                "social_influence": {
                    "description": "Others are doing it",
                    "examples": hook_examples.get('social_influence', ["Join top users", "See what others do", "Compare with friends"]),
                    "best_for_segments": ["social", "community_driven"]
                },
                "scarcity": {
                    "description": "Limited time or availability",
                    "examples": hook_examples.get('scarcity', ["Only hours left", "Last chance", "Limited offer"]),
                    "best_for_segments": ["achievers", "fomo_driven"]
                },
                "unpredictability": {
                    "description": "What will happen next",
                    "examples": hook_examples.get('unpredictability', ["Unlock surprise", "Discover something new", "See what's next"]),
                    "best_for_segments": ["explorers", "casual"]
                },
                "loss_avoidance": {
                    "description": "Don't lose what you have",
                    "examples": hook_examples.get('loss_avoidance', ["Don't lose progress", "Maintain your status", "Keep your streak"]),
                    "best_for_segments": ["achievers", "at_risk"]
                }
            },
            "hooks_by_segment": {
                "achievers": ["accomplishment", "ownership", "loss_avoidance"],
                "social_competitors": ["social_influence", "accomplishment", "scarcity"],
                "casual_learners": ["unpredictability", "empowerment", "epic_meaning"],
                "at_risk_churners": ["loss_avoidance", "scarcity", "social_influence"],
                "dormant_users": ["unpredictability", "epic_meaning", "empowerment"]
            }
        }

        return matrix

    def _generate_hook_examples(self, domain: str) -> Dict[str, List[str]]:
        """Generate domain-specific examples for Octalysis hooks"""

        domain_examples = {
            'edtech': {
                'epic_meaning': ["Join 1M+ learners", "Transform your skills", "Unlock your potential"],
                'accomplishment': ["Complete 10 lessons", "Reach 7-day streak", "Earn 100 coins"],
                'empowerment': ["Choose your topic", "Learn at your pace", "Customize your path"],
                'ownership': ["Your 500 coins", "Your 30-day streak", "Your certificates"],
                'social_influence': ["Beat the leaderboard", "Join top learners", "Compete with friends"],
                'scarcity': ["Only 3 hours left", "Limited seats", "Today's bonus expires"],
                'unpredictability': ["Unlock surprise reward", "Daily bonus waiting", "New content unlocked"],
                'loss_avoidance': ["Streak will break", "Don't lose progress", "Maintain your rank"]
            },
            'fintech': {
                'epic_meaning': ["Join millions saving smart", "Financial freedom awaits", "Be money smart"],
                'accomplishment': ["Saved ₹10,000", "100 transactions done", "Gold member unlocked"],
                'empowerment': ["Your money, your rules", "Invest your way", "Control your finances"],
                'ownership': ["Your savings: ₹50K", "Your cashback earned", "Your portfolio"],
                'social_influence': ["Friends using this", "Top savers this month", "Trusted by millions"],
                'scarcity': ["Offer ends tonight", "Limited cashback", "Last chance for bonus"],
                'unpredictability': ["Scratch and win", "Surprise cashback", "Lucky draw entry"],
                'loss_avoidance': ["Cashback expiring", "Don't miss savings", "Bill overdue"]
            },
            'entertainment': {
                'epic_meaning': ["Join the community", "Be part of the fandom", "Experience the best"],
                'accomplishment': ["Watched 100 hours", "Completed the series", "Top fan badge"],
                'empowerment': ["Watch anywhere", "Your playlist", "Skip the ads"],
                'ownership': ["Your watchlist", "Your favorites", "Your downloads"],
                'social_influence': ["Trending now", "Friends are watching", "Most popular"],
                'scarcity': ["Leaving soon", "Limited release", "Premium only"],
                'unpredictability': ["New releases", "Surprise drop", "What's next"],
                'loss_avoidance': ["Continue watching", "Don't miss finale", "Expires soon"]
            },
            'ecommerce': {
                'epic_meaning': ["Shop smart, save big", "Join happy shoppers", "Best deals await"],
                'accomplishment': ["100 orders placed", "Platinum member", "Top reviewer"],
                'empowerment': ["Shop your way", "Easy returns", "Your choices"],
                'ownership': ["Your wishlist", "Your rewards", "Your savings"],
                'social_influence': ["Best seller", "Highly rated", "Others bought"],
                'scarcity': ["Only 3 left", "Sale ends soon", "Flash deal"],
                'unpredictability': ["Mystery discount", "Surprise offer", "Lucky coupon"],
                'loss_avoidance': ["Price drop alert", "Cart expiring", "Almost sold out"]
            },
            'health': {
                'epic_meaning': ["Join the healthy movement", "Transform your life", "Be your best self"],
                'accomplishment': ["30-day streak", "Goal achieved", "Personal best"],
                'empowerment': ["Your fitness journey", "Your pace", "Your goals"],
                'ownership': ["Your progress", "Your records", "Your achievements"],
                'social_influence': ["Top performers", "Friends active", "Community challenge"],
                'scarcity': ["Challenge ends soon", "Limited spots", "This week only"],
                'unpredictability': ["New workout unlocked", "Bonus points", "Surprise reward"],
                'loss_avoidance': ["Streak at risk", "Don't lose progress", "Stay on track"]
            }
        }

        return domain_examples.get(domain, {
            'epic_meaning': ["Join millions", "Be part of something big", "Transform your experience"],
            'accomplishment': ["Complete goals", "Earn rewards", "Level up"],
            'empowerment': ["Your way", "Your choice", "Customize"],
            'ownership': ["Your progress", "Your achievements", "Your journey"],
            'social_influence': ["Others are here", "Join top users", "Trending"],
            'scarcity': ["Limited time", "Act now", "Ends soon"],
            'unpredictability': ["Surprise waiting", "Discover more", "What's next"],
            'loss_avoidance': ["Don't miss out", "Keep going", "Stay active"]
        })

    def save_outputs(self, output_dir: str):
        """Save extracted knowledge to JSON files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        with open(output_path / 'company_north_star.json', 'w', encoding='utf-8') as f:
            json.dump(self.north_star, f, indent=2, ensure_ascii=False)

        with open(output_path / 'feature_goal_map.json', 'w', encoding='utf-8') as f:
            json.dump(self.feature_goal_map, f, indent=2, ensure_ascii=False)

        with open(output_path / 'allowed_tone_hook_matrix.json', 'w', encoding='utf-8') as f:
            json.dump(self.tone_hook_matrix, f, indent=2, ensure_ascii=False)

        mode_label = "RAG-lite" if self.rag_mode_used else "Regex"
        print(f"[OK] Knowledge Bank outputs saved to {output_dir} (mode: {mode_label})")
