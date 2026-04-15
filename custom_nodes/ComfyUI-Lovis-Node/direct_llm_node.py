import json
import os
import urllib.request
import urllib.error


class DirectLLMNode:
    """Call OpenAI-compatible API directly with your own API key. No ComfyUI login needed."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "tooltip": "Your OpenAI API key"}),
                "model": ("STRING", {"default": "gpt-4.1-mini", "tooltip": "Model name"}),
                "system_prompt": ("STRING", {"multiline": True, "default": "You are a helpful assistant."}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "max_tokens": ("INT", {"default": 4096, "min": 1, "max": 32768}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1}),
            },
            "optional": {
                "base_url": ("STRING", {"default": "https://api.openai.com/v1", "tooltip": "API base URL (OpenAI, Gemini, local, etc.)"}),
                "context_text": ("STRING", {"multiline": True, "default": "", "forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = "Lovis/LLM"

    def run(self, api_key, model, system_prompt, prompt, max_tokens, temperature, base_url="https://api.openai.com/v1", context_text=""):
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY", "")
        if context_text:
            full_prompt = f"{prompt}\n\n---\nContext:\n{context_text}"
        else:
            full_prompt = prompt

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_prompt}
        ]

        # GPT-5+ and o-series models have different API requirements
        is_new_model = any(x in model for x in ["gpt-5", "o1", "o3", "o4"])
        body = {"model": model, "messages": messages}
        body["max_completion_tokens" if is_new_model else "max_tokens"] = max_tokens
        if not is_new_model:
            body["temperature"] = temperature
        payload = json.dumps(body).encode("utf-8")

        url = f"{base_url.rstrip('/')}/chat/completions"
        req = urllib.request.Request(url, data=payload, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {api_key}")

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return (data["choices"][0]["message"]["content"],)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            return (f"[ERROR {e.code}] {body[:500]}",)
        except Exception as e:
            return (f"[ERROR] {str(e)}",)


class DirectLLMConfig:
    """Store API key + base URL to reuse across multiple DirectLLMNode."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
                "base_url": ("STRING", {"default": "https://api.openai.com/v1"}),
                "model": ("STRING", {"default": "gpt-4.1-mini"}),
            }
        }

    RETURN_TYPES = ("LLM_CONFIG",)
    RETURN_NAMES = ("config",)
    FUNCTION = "run"
    CATEGORY = "Lovis/LLM"

    def run(self, api_key, base_url, model):
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY", "")
        return ({"api_key": api_key, "base_url": base_url, "model": model},)


class DirectLLMChat:
    """Chat node that takes LLM_CONFIG for cleaner wiring."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": ("LLM_CONFIG",),
                "system_prompt": ("STRING", {"multiline": True, "default": "You are a helpful assistant."}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "context_text": ("STRING", {"multiline": True, "default": "", "forceInput": True}),
                "max_tokens": ("INT", {"default": 4096, "min": 1, "max": 32768}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = "Lovis/LLM"

    def run(self, config, system_prompt, prompt, context_text="", max_tokens=4096, temperature=0.7):
        node = DirectLLMNode()
        return node.run(
            api_key=config["api_key"],
            model=config["model"],
            system_prompt=system_prompt,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            base_url=config["base_url"],
            context_text=context_text
        )
