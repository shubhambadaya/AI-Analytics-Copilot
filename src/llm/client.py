import json
import os
from typing import Type, TypeVar, Optional, Any
from pydantic import BaseModel
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

# Model tier constants
MODEL_FAST = "gemini-2.5-flash"
MODEL_PRO = "gemini-2.5-pro"

class LLMClient:
    """
    A unified client wrapper for OpenAI, Anthropic, and Gemini.
    Provides structured output parsing conforming to a Pydantic model.
    """
    def __init__(self):
        self._openai_client = None
        self._anthropic_client = None
        self._gemini_configured = False
        
        self.initialize_clients()

    def initialize_clients(self):
        """Initializes the active LLM clients based on configuration."""
        if config.has_openai:
            try:
                from openai import OpenAI
                self._openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
                logger.info("OpenAI client initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {str(e)}")

        if config.has_anthropic:
            try:
                from anthropic import Anthropic
                self._anthropic_client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
                logger.info("Anthropic client initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {str(e)}")

        if config.has_gemini:
            try:
                import google.generativeai as genai
                genai.configure(api_key=config.GEMINI_API_KEY)
                self._gemini_configured = True
                logger.info("Gemini client initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {str(e)}")

    def get_available_providers(self) -> list:
        """Returns list of configured and ready LLM providers."""
        providers = []
        if self._openai_client:
            providers.append("openai")
        if self._gemini_configured:
            providers.append("gemini")
        if self._anthropic_client:
            providers.append("anthropic")
        return providers

    def generate_structured_output(
        self, 
        prompt: str, 
        response_model: Type[T], 
        system_prompt: Optional[str] = None,
        provider: Optional[str] = None,
        model_override: Optional[str] = None
    ) -> T:
        """
        Queries the preferred LLM provider and parses the result into a Pydantic model.
        
        Args:
            prompt: User message / prompt.
            response_model: Pydantic model class to validate output against.
            system_prompt: Optional instruction for the system role.
            provider: Optional override for provider selection ("openai", "gemini", "anthropic").
            model_override: Optional specific model name (e.g. 'gemini-2.5-flash').
            
        Returns:
            An instance of response_model containing parsed data.
        """
        available = self.get_available_providers()
        if not available:
            raise ValueError(
                "No LLM providers are configured. Please set OPENAI_API_KEY, "
                "ANTHROPIC_API_KEY, or GEMINI_API_KEY in your environment/.env file."
            )
            
        if provider:
            provider = provider.lower()
            if provider not in available:
                logger.warning(f"Requested provider '{provider}' is not available. Available: {available}. Falling back.")
                provider = available[0]
        else:
            provider = available[0]

        logger.info(f"Generating structured output using provider: {provider}")

        if provider == "openai":
            return self._call_openai(prompt, response_model, system_prompt)
        elif provider == "gemini":
            return self._call_gemini(prompt, response_model, system_prompt, model_override=model_override)
        elif provider == "anthropic":
            return self._call_anthropic(prompt, response_model, system_prompt)
        else:
            raise ValueError(f"Unknown provider '{provider}'")

    def _call_openai(self, prompt: str, response_model: Type[T], system_prompt: Optional[str]) -> T:
        model_name = "gpt-4o-mini"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        logger.info(f"Calling OpenAI chat completion using {model_name}...")
        response = self._openai_client.beta.chat.completions.parse(
            model=model_name,
            messages=messages,
            response_format=response_model,
            temperature=0.0
        )
        return response.choices[0].message.parsed

    def _call_gemini(self, prompt: str, response_model: Type[T], system_prompt: Optional[str], model_override: Optional[str] = None) -> T:
        import google.generativeai as genai
        model_name = model_override or MODEL_PRO
        
        logger.info(f"Calling Gemini generation using {model_name}...")
        
        # 1. Fetch standard Pydantic JSON Schema representation
        schema_dict = response_model.model_json_schema()
        
        # 2. Extract internal references
        defs = schema_dict.get("$defs", {})
        
        # 3. Recursively dereference all $ref targets and clean 'default' and 'title' properties
        # This is required because Gemini's schema parser does not support external 
        # definition blocks ($defs), default values, or 'title' metadata tags.
        def process_schema_node(node: Any) -> Any:
            if isinstance(node, dict):
                # Inline $ref targets immediately
                if "$ref" in node:
                    ref_path = node["$ref"]
                    ref_name = ref_path.split("/")[-1]
                    sub_schema = defs.get(ref_name, {})
                    return process_schema_node(sub_schema)
                
                # Flatten anyOf: Pydantic generates anyOf for Optional[X] as [X, null].
                # Gemini does not support anyOf. Extract the first non-null type instead.
                if "anyOf" in node:
                    non_null_types = [t for t in node["anyOf"] if t != {"type": "null"} and t.get("type") != "null"]
                    if non_null_types:
                        # Use the first concrete type; carry over any sibling keys (e.g. description)
                        flattened = dict(non_null_types[0])
                        for k, v in node.items():
                            if k not in ("anyOf", "default", "title"):
                                flattened[k] = v
                        return process_schema_node(flattened)
                    # All types are null — drop to string fallback
                    return {"type": "string"}
                
                # Filter out 'default' and 'title' keys and recursively process children
                return {k: process_schema_node(v) for k, v in node.items() if k not in ("default", "title")}
                
            elif isinstance(node, list):
                return [process_schema_node(x) for x in node]
            return node
            
        processed_schema = process_schema_node(schema_dict)
        
        # 4. Remove $defs from root since they are now completely inlined
        if "$defs" in processed_schema:
            del processed_schema["$defs"]
            
        # IMPORTANT: We deliberately do NOT pass `response_schema` to Gemini.
        # Native constrained decoding on a complex nested schema (e.g. AnalysisPlan
        # with VisualSpec's many free-text label fields) drives gemini-2.5 models
        # into runaway repetition loops — they echo instructions into a string field
        # until they hit max_output_tokens, producing truncated/invalid JSON.
        # Instead we embed the schema in the prompt and validate the result with
        # Pydantic below; this completes reliably in a few seconds with no loops.
        schema_instruction = (
            "Respond with ONLY a single JSON object that strictly conforms to this "
            "JSON Schema. Include every required field, and use null for unused "
            "optional fields. Do not wrap it in markdown or add any prose.\n"
            f"JSON Schema:\n{json.dumps(processed_schema)}"
        )
        prompt = f"{schema_instruction}\n\n{prompt}"

        # Gemini 2.5 are "thinking" models: reasoning tokens are deducted from
        # max_output_tokens, so keep a generous ceiling for thinking + output.
        max_output_tokens = 32768
        generation_config = genai.types.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.0,
            max_output_tokens=max_output_tokens
        )

        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_prompt
        )
        
        # Use a thread-based timeout to prevent indefinite hangs (signal.alarm
        # doesn't work in Streamlit's worker threads).
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
        
        def _generate():
            return model.generate_content(
                prompt,
                generation_config=generation_config
            )
        
        # Gemini 2.5 thinking models can take well over a minute on large structured
        # outputs; allow generous headroom so valid responses aren't cut off.
        request_timeout = 180
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_generate)
            try:
                response = future.result(timeout=request_timeout)
            except FuturesTimeoutError:
                future.cancel()
                raise TimeoutError(f"Gemini API call timed out after {request_timeout} seconds")
        
        # Detect truncation before parsing: if the model hit the token ceiling the
        # JSON is incomplete, so raise a clear error rather than a cryptic
        # "EOF while parsing a string" from Pydantic downstream.
        try:
            finish_reason = response.candidates[0].finish_reason
            finish_name = getattr(finish_reason, "name", str(finish_reason))
        except (AttributeError, IndexError):
            finish_name = None

        if finish_name == "MAX_TOKENS":
            raise ValueError(
                f"Gemini response was truncated (finish_reason=MAX_TOKENS) before completing "
                f"valid JSON for {response_model.__name__}; output exceeded "
                f"max_output_tokens={max_output_tokens}. Try a more specific question."
            )

        try:
            result_text = response.text
        except Exception as e:
            raise ValueError(
                f"Gemini returned no parseable text for {response_model.__name__} "
                f"(finish_reason={finish_name}): {e}"
            )

        # Cleanup markdown and preamble text that Gemini sometimes returns
        result_text = result_text.strip()
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        elif result_text.startswith("```"):
            result_text = result_text[3:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]
        result_text = result_text.strip()
        
        # Fallback to finding the first `{` and last `}`
        if not result_text.startswith("{") and "{" in result_text:
            result_text = result_text[result_text.find("{"):]
        if not result_text.endswith("}") and "}" in result_text:
            result_text = result_text[:result_text.rfind("}")+1]
            
        return response_model.model_validate_json(result_text)

    def _call_anthropic(self, prompt: str, response_model: Type[T], system_prompt: Optional[str]) -> T:
        model_name = "claude-3-haiku-20240307"
        logger.info(f"Calling Anthropic using {model_name}...")
        
        schema = response_model.model_json_schema()
        tool_name = f"record_{response_model.__name__.lower()}"
        tools = [
            {
                "name": tool_name,
                "description": f"Record structured data conforming to {response_model.__name__}",
                "input_schema": schema
            }
        ]
        
        system = system_prompt if system_prompt else ""
        
        response = self._anthropic_client.messages.create(
            model=model_name,
            max_tokens=4000,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            tools=tools,
            tool_choice={"type": "tool", "name": tool_name},
            temperature=0.0
        )
        
        tool_use = None
        for content_block in response.content:
            if content_block.type == "tool_use":
                tool_use = content_block
                break
                
        if not tool_use:
            raise ValueError("Anthropic API failed to call the specified tool.")
            
        return response_model.model_validate(tool_use.input)

# Shared client instance
llm_client = LLMClient()
