"""
Inference engine with multi-provider support.

Supports:
- Local models (HuggingFace transformers)
- Anthropic Claude
- Google Gemini (with OAuth)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from pi_sidecar.models.registry import ModelRegistry
from pi_sidecar.personality import get_personality

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Multi-provider inference engine."""

    def __init__(self, registry: ModelRegistry):
        """
        Initialize the inference engine.
        Args:
            registry: The model registry to use for text completion.
        """
        self.registry = registry
        self._embedding_model = None
        self._anthropic_client = None
        self._gemini_client = None

    async def load_model(self, model_id: str) -> dict[str, Any]:
        """
        Load a model by ID or handle API-based models.
        Args:
            model_id: The model ID to load.
        Returns:
            A dictionary containing the status of the operation.
        """
        # API models don't need local loading
        is_api = any(
            p in model_id.lower()
            for p in ["claude-", "gemini-", "gpt-", "deepseek-"]
        )

        if is_api:
            logger.info("API model selected: %s. No loading required.", model_id)
            return {"status": "ready", "model_id": model_id}

        # Local models use the registry
        try:
            await self.registry.load_model(model_id)
            return {"status": "loaded", "model_id": model_id}
        except Exception as e:
            logger.error("Failed to load local model %s: %s", model_id, e)
            raise

    async def embed(self, text: str, model_id: str = "all-MiniLM-L6-v2") -> list[float]:
        """
        Generate embeddings using sentence-transformers.
        Args:
            text: The text to generate embeddings for.
            model_id: The model ID to use for text completion.
        Returns:
            A list of floats containing the embeddings.
        """
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model: %s", model_id)
            self._embedding_model = SentenceTransformer(model_id)

        embedding = self._embedding_model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def _load_secrets(self) -> dict[str, str]:
        """Load secrets from ~/.pi-assistant/secrets.json."""
        import json
        secrets_path = Path.home() / ".pi-assistant" / "secrets.json"
        if secrets_path.exists():
            try:
                with open(secrets_path, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_secrets(self, secrets: dict[str, str]) -> None:
        """Save secrets to ~/.pi-assistant/secrets.json."""
        import json
        secrets_path = Path.home() / ".pi-assistant" / "secrets.json"
        secrets_path.parent.mkdir(parents=True, exist_ok=True)
        with open(secrets_path, "w") as f:
            json.dump(secrets, f, indent=2)

    async def complete(
        self,
        prompt: str,
        provider: str = "local",
        model_id: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        """
        Generate text completion from specified provider.
        Args:
            prompt: The prompt to use for text completion.
            provider: The provider to use for text completion.
            model_id: The model ID to use for text completion.
            max_tokens: The maximum number of tokens to generate.
            temperature: The temperature to use for text completion.
        Returns:
            A dictionary containing the text completion.
        """
        if provider == "anthropic":
            secrets = self._load_secrets()
            has_claude_max = bool(secrets.get("claude_max_oauth"))
            has_antigravity_tokens = bool(
                secrets.get("anthropic_oauth") or secrets.get("antigravity_oauth")
            )

            if has_claude_max:
                try:
                    return await self._complete_claude_max(
                        prompt, model_id, max_tokens, temperature
                    )
                except Exception as e:
                    logger.warning("Claude Max direct API failed: %s", e)
                    if has_antigravity_tokens:
                        logger.info("Falling back to Antigravity Gateway")
                        model = model_id or "claude-3-5-sonnet-v2@20241022"
                        return await self._complete_gemini(
                            prompt, model, max_tokens, temperature
                        )
                    raise
            elif has_antigravity_tokens:
                logger.info("Routing Anthropic request via Antigravity Gateway")
                model = model_id or "claude-3-5-sonnet-v2@20241022"
                return await self._complete_gemini(prompt, model, max_tokens, temperature)
            else:
                return await self._complete_anthropic(
                    prompt, model_id, max_tokens, temperature
                )
        elif provider == "google":
            return await self._complete_gemini(prompt, model_id, max_tokens, temperature)
        else:
            return await self._complete_local(prompt, model_id, max_tokens, temperature)

    async def plan(
        self,
        task: str,
        iteration: int,
        context: list[dict],
        tools: list[dict] = [],
        provider: str = "local",
    ) -> dict[str, Any]:
        """
        Generate agent plan using structured output.
        Args:
            task: The task to generate a plan for.
            iteration: The iteration number.
            context: The context to use for text completion.
            tools: The list of available tools.
            provider: The provider to use for text completion.
        Returns:
            A dictionary containing the agent plan.
        """
        # Get personality-aware base prompt
        personality = get_personality()
        personality_prompt = personality.system_prompt

        tools_list_str = "\n".join(
            f"- {t.get('name')}: {t.get('description')}" for t in tools
        )

        system_prompt = f"""{personality_prompt}

# Agent Planner Instructions

Given a task and context, decide:
1. What tools to call (if any)
2. Whether to ask the user a question
3. Whether the task is complete

Respond with JSON:
{{
    "reasoning": "your chain of thought",
    "tool_calls": [{{"tool_name": "...", "parameters": {{...}}}}],
    "question": "optional question for user",
    "is_complete": false
}}

Available tools:
{tools_list_str if tools_list_str else "shell, code, browser"}"""

        context_str = "\n".join(
            f"[{c.get('role', 'system')}]: {c.get('content', '')}" for c in context
        )

        prompt = f"""Task: {task}
            Iteration: {iteration}

            Context:
            {context_str}

            What should I do next?"""

        result = await self.complete(
            prompt=f"{system_prompt}\n\n{prompt}",
            provider=provider,
            max_tokens=1024,
            temperature=0.3,
        )

        # Parse JSON from response
        import json

        try:
            text = result.get("text", "{}")
            # Find JSON in response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

        # Fallback
        return {
            "reasoning": result.get("text", ""),
            "tool_calls": [],
            "question": None,
            "is_complete": False,
        }

    async def _complete_anthropic(
        self, prompt: str, model_id: str | None, max_tokens: int, temperature: float
    ) -> dict[str, Any]:
        """
        Complete using Anthropic Claude via standard SDK (API Key).
        Note: OAuth traffic is routed via _complete_gemini.
        """
        import os
        
        # If we got here, we are using standard API Key
        if self._anthropic_client is None:
            import anthropic
            
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                 from pathlib import Path
                 import json
                 secrets_path = Path.home() / ".pi-assistant" / "secrets.json"
                 if secrets_path.exists():
                    try:
                        with open(secrets_path, "r") as f:
                            secrets = json.load(f)
                            api_key = secrets.get("anthropic")
                    except:
                        pass
            
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")
                
            self._anthropic_client = anthropic.AsyncAnthropic(api_key=api_key)

        model = model_id or "claude-3-5-sonnet-latest"
        logger.info("Calling Anthropic SDK: %s", model)
        
        response = await self._anthropic_client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )

        return {
            "text": response.content[0].text,
            "provider": "anthropic",
            "model": model,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        }

    async def _refresh_claude_max_token(self) -> str:
        """Refresh Claude Max OAuth token and return new access token."""
        import httpx
        secrets = self._load_secrets()
        refresh_token = secrets.get("claude_max_refresh")
        if not refresh_token:
            raise ValueError("No Claude Max refresh token available")

        async with httpx.AsyncClient(trust_env=False) as client:
            response = await client.post(
                "https://console.anthropic.com/v1/oauth/token",
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "client_id": "9d1c250a-e61b-44d9-88ed-5944d1962f5e",
                },
                timeout=30.0,
            )

            if response.status_code != 200:
                raise ValueError(
                    f"Token refresh failed ({response.status_code}): {response.text}"
                )

            data = response.json()
            new_token = data["access_token"]

            # Update secrets
            secrets["claude_max_oauth"] = new_token
            if "refresh_token" in data:
                secrets["claude_max_refresh"] = data["refresh_token"]
            if "expires_in" in data:
                import time
                secrets["claude_max_expires_at"] = str(
                    int(time.time()) + data["expires_in"]
                )
            self._save_secrets(secrets)

            logger.info("Claude Max token refreshed successfully")
            return new_token

    async def _complete_claude_max(
        self, prompt: str, model_id: str | None, max_tokens: int, temperature: float
    ) -> dict[str, Any]:
        """
        Complete using Claude Pro/Max subscription via direct Anthropic API with OAuth token.
        """
        import httpx
        import time

        secrets = self._load_secrets()
        access_token = secrets.get("claude_max_oauth")
        if not access_token:
            raise ValueError("No Claude Max OAuth token found")

        # Check expiry and refresh if needed
        expires_at = secrets.get("claude_max_expires_at")
        if expires_at:
            try:
                if int(expires_at) < int(time.time()) + 60:
                    logger.info("Claude Max token expired, refreshing...")
                    access_token = await self._refresh_claude_max_token()
            except (ValueError, TypeError):
                pass

        model = model_id or "claude-sonnet-4-5-20250514"
        logger.info("Calling Claude Max API directly: %s", model)

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "oauth-2025-04-20",
        }

        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }

        async with httpx.AsyncClient(trust_env=False) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                json=payload,
                headers=headers,
                timeout=120.0,
            )

            if response.status_code == 401:
                # Token might be expired, try refresh
                logger.info("Got 401, attempting token refresh...")
                access_token = await self._refresh_claude_max_token()
                headers["Authorization"] = f"Bearer {access_token}"
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    json=payload,
                    headers=headers,
                    timeout=120.0,
                )

            if response.status_code != 200:
                error_text = response.text
                logger.error(
                    "Claude Max API error %d: %s", response.status_code, error_text
                )
                raise ValueError(
                    f"Claude Max API error ({response.status_code}): {error_text}"
                )

            data = response.json()

            # Parse Anthropic Messages API response
            content_blocks = data.get("content", [])
            text = "".join(
                block.get("text", "")
                for block in content_blocks
                if block.get("type") == "text"
            )

            usage = data.get("usage", {})
            return {
                "text": text,
                "provider": "anthropic",
                "model": model,
                "usage": {
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                },
            }

    async def _complete_gemini(
        self, prompt: str, model_id: str | None, max_tokens: int, temperature: float
    ) -> dict[str, Any]:
        """
        Complete using Google Antigravity (Cloud Code Assist API) with OAuth.
        
        This mimics the `opencode-antigravity-auth` plugin behavior:
        - Endpoint: daily-cloudcode-pa.sandbox.googleapis.com
        - Headers: Specific User-Agent and Client-Metadata
        - Payload: Uppercase types
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            # Try loading from secrets.json
            secrets_path = Path.home() / ".pi-assistant" / "secrets.json"
            if secrets_path.exists():
                import json
                try:
                    with open(secrets_path, "r") as f:
                        secrets = json.load(f)
                        api_key = (
                            secrets.get("anthropic_oauth")
                            or secrets.get("antigravity_oauth")
                            or secrets.get("google_oauth")
                            or secrets.get("gemini_oauth")
                            or secrets.get("google")
                            or secrets.get("gemini")
                        )
                except Exception as e:
                    logger.error("Failed to load secrets: %s", e)

        if not api_key:
            raise ValueError("No Google OAuth token found in secrets.json")

        model_name = model_id or "gemini-2.0-flash"
        
        # Antigravity Logic
        # See reversed logic from opencode-antigravity-auth
        base_url = "https://daily-cloudcode-pa.sandbox.googleapis.com"
        endpoint = f"{base_url}/v1internal:generateContent"
        
        # Headers from constants.js (Antigravity/1.15.8)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Antigravity/1.15.8 Chrome/138.0.7204.235 Electron/37.3.1 Safari/537.36",
            "X-Goog-Api-Client": "google-cloud-sdk vscode_cloudshelleditor/0.1",
            "Client-Metadata": '{"ideType":"IDE_UNSPECIFIED","platform":"PLATFORM_UNSPECIFIED","pluginType":"GEMINI"}',
        }

        # Antigravity Model Mapping
        model_map = {
            "gemini-3-flash": "gemini-3-pro-low",
            "gemini-2.0-flash": "gemini-3-pro-low",
            "claude-3-5-sonnet": "claude-sonnet-4-5",
            # Add other mappings as needed
        }
        
        target_model = model_map.get(model_name, model_name)
        
        # Payload construction
        # Antigravity expects a wrapped "request" object
        import time
        payload = {
            "project": "rising-fact-p41fc", # Sandbox default
            "model": target_model,
            "request": {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": prompt}]
                    }
                ],
                "generationConfig": {
                    "maxOutputTokens": max_tokens,
                    "temperature": temperature,
                    "candidateCount": 1
                }
            },
            "requestType": "agent",
            "userAgent": "antigravity",
            "requestId": f"agent-{int(time.time()*1000)}"
        }
        
        # Add system instruction if personality is available (implicit) or passed
        # Currently prompt includes system prompt, but Antigravity separates it.
        # For now, we keep it simple as prompt is merged. 
        # But if we were to split it:
        # payload["request"]["systemInstruction"] = { "parts": [{ "text": system_prompt }] }

        logger.info(f"Calling Antigravity: {endpoint} model={model_name}")

        import httpx
        async with httpx.AsyncClient(trust_env=False) as client:
            try:
                response = await client.post(endpoint, json=payload, headers=headers, timeout=60.0)
                
                if response.status_code != 200:
                    error_text = response.text
                    logger.error(f"Antigravity API Error {response.status_code}: {error_text}")
                    raise ValueError(f"Antigravity API Error {response.status_code}: {error_text}")
                
                data = response.json()
                
                # Parse response (standard Gemini format)
                # candidates[0].content.parts[0].text
                candidates = data.get("candidates", [])
                if not candidates:
                    return {"text": "", "provider": "gemini", "model": model_name}
                
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                text = "".join(part.get("text", "") for part in parts)
                
                return {
                    "text": text,
                    "provider": "gemini",
                    "model": model_name,
                    "usage": {
                        "input_tokens": 0, # Not always returned
                        "output_tokens": 0
                    }
                }
                
            except Exception as e:
                logger.error(f"Antigravity call failed: {e}")
                raise

    async def _complete_local(
        self, prompt: str, model_id: str | None, max_tokens: int, temperature: float
    ) -> dict[str, Any]:
        """
        Complete using local model (transformers or llama.cpp).
        Args:
            prompt: The prompt to use for text completion.
            model_id: The model ID to use for text completion.
            max_tokens: The maximum number of tokens to generate.
            temperature: The temperature to use for text completion.
        Returns:
            A dictionary containing the text completion.
        """
        model = self.registry.get_model(model_id or "default")
        if model is None:
            logger.warning("No local model loaded, returning placeholder")
            return {
                "text": "[Local model not loaded. Please load a model first.]",
                "provider": "local",
                "model": None,
            }

        if model.backend == "llama.cpp":
            # Use llama.cpp
            response = model.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["</s>", "User:", "Assistant:"], # Basic stop tokens
            )
            return {
                "text": response["choices"][0]["text"],
                "provider": "local",
                "model": model_id,
                "backend": "llama.cpp",
            }
        else:
            # Use transformers pipeline
            from transformers import pipeline

            generator = pipeline("text-generation", model=model.model, tokenizer=model.tokenizer)
            result = generator(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
            )

            return {
                "text": result[0]["generated_text"][len(prompt) :],
                "provider": "local",
                "model": model_id,
                "backend": "transformers",
            }
