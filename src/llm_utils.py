"""
LLM Utility Helpers — shared across all dynamic pipeline stages.

Provides a retry-aware Groq LLM call that automatically reduces context
on 413/rate-limit errors, plus a consistent JSON parsing helper.
"""

import json
import os
import re
import time
import yaml
from typing import Optional, Dict, Any

# Global throttle: minimum seconds between LLM calls to avoid rate limits
_LLM_MIN_INTERVAL = 2.0
_last_llm_call_time = 0.0

# Circuit breaker: after N consecutive rate-limit failures, skip LLM for the rest of the session
_CIRCUIT_BREAKER_THRESHOLD = 3
_consecutive_rate_limit_failures = 0
_circuit_open = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


def _load_llm_config(config_path: str = 'config/config.yaml') -> Dict:
    """Load LLM config from YAML."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg.get('knowledge_bank', {})


def call_llm_with_retry(
    system_prompt: str,
    user_prompt: str,
    config_path: str = 'config/config.yaml',
    max_retries: int = 5,
    temperature: float = 0.4,
) -> Optional[str]:
    """Call Groq LLM with automatic retry on context overflow.

    On 413 / rate-limit / context-too-long errors the user prompt is
    halved and the call is retried up to *max_retries* times.

    Returns the raw text response or None on failure.
    """
    if not GROQ_AVAILABLE:
        print("   [WARN] Groq SDK not available — skipping LLM call")
        return None

    global _circuit_open, _consecutive_rate_limit_failures, _last_llm_call_time
    if _circuit_open:
        return None

    api_key = os.environ.get('GROQ_API_KEY', '')
    if not api_key:
        # Try loading from .env in project root
        try:
            from pathlib import Path
            env_path = Path(__file__).parent.parent / ".env"
            if env_path.exists():
                with open(env_path, "r") as f:
                    for line in f:
                        if line.startswith("GROQ_API_KEY="):
                            api_key = line.strip().split("=")[1].strip("'").strip('"')
                            os.environ["GROQ_API_KEY"] = api_key
                            break
        except Exception:
            pass

    if not api_key:
        print("   [WARN] GROQ_API_KEY not set — skipping LLM call")
        return None

    kb_cfg = _load_llm_config(config_path)
    model = kb_cfg.get('llm_model', 'llama-3.3-70b-versatile')
    client = Groq(api_key=api_key)

    current_user_prompt = user_prompt

    for attempt in range(1, max_retries + 1):
        try:
            # Throttle: wait if needed to respect rate limits
            elapsed = time.time() - _last_llm_call_time
            if elapsed < _LLM_MIN_INTERVAL:
                time.sleep(_LLM_MIN_INTERVAL - elapsed)
            _last_llm_call_time = time.time()

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": current_user_prompt},
                ],
                temperature=temperature,
                max_tokens=4096,
            )
            _consecutive_rate_limit_failures = 0  # Reset on success
            return response.choices[0].message.content.strip()

        except Exception as e:
            err_str = str(e).lower()
            is_retryable = any(kw in err_str for kw in [
                '413', 'rate_limit', 'token', 'context_length', 'too long',
                'request too large', '429',
            ])
            is_rate_limit = '429' in err_str or 'rate_limit' in err_str

            if is_rate_limit:
                _consecutive_rate_limit_failures += 1
                if _consecutive_rate_limit_failures >= _CIRCUIT_BREAKER_THRESHOLD:
                    print(f"   [WARN] Circuit breaker OPEN — {_consecutive_rate_limit_failures} consecutive rate-limit failures. Skipping all LLM calls.")
                    _circuit_open = True
                    return None

            if is_retryable and attempt < max_retries:
                if is_rate_limit:
                    wait = 10 * attempt  # 10s, 20s, 30s... exponential backoff
                    print(f"   [WARN] Rate limited — waiting {wait}s before retry (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait)
                else:
                    # Context too long — halve the user prompt
                    half = len(current_user_prompt) // 2
                    current_user_prompt = current_user_prompt[:half]
                    print(f"   [WARN] LLM context too large — reducing prompt to {half} chars and retrying (attempt {attempt + 1}/{max_retries})")
            else:
                print(f"   [WARN] LLM call failed: {str(e)[:120]}")
                return None

    return None


def parse_json_response(text: str) -> Optional[Any]:
    """Extract and parse JSON from an LLM response that may contain markdown fences."""
    if not text:
        return None
    # Strip markdown code fences
    cleaned = re.sub(r'```(?:json)?\s*', '', text)
    cleaned = cleaned.strip()
    # Try parsing directly
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Try extracting first JSON object/array
    for pattern in [r'\{[\s\S]*\}', r'\[[\s\S]*\]']:
        match = re.search(pattern, text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                continue
    return None
