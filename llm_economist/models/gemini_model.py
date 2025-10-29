"""
Gemini model implementation for the LLM Economist framework.
"""

from typing import Tuple, Optional
import os
import json
from time import sleep
from .base import BaseLLMModel


class GeminiModel(BaseLLMModel):
    """Gemini model implementation using Google's Gemini API."""
    
    def __init__(self, model_name: str = "gemini-2.5-flash", 
                 api_key: Optional[str] = None,
                 max_tokens: int = 1000, temperature: float = 0.7,
                 project: Optional[str] = None, location: Optional[str] = None):
        """
        Initialize the Gemini model.
        
        Args:
            model_name: Name of the Gemini model to use
            api_key: Google API key (if None, will look for GOOGLE_API_KEY env var)
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling
        """
        super().__init__(model_name, max_tokens, temperature)
        
        # Get API key from parameter or environment
        api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        
        self.model_name = model_name
        
        if api_key:
            # Do api_key setup
            try:
                import google.generativeai as genai
            except ImportError as e:
                raise ImportError("Please `pip install google-generativeai`") from e
            genai.configure(api_key=api_key)
            self.api_key = api_key
            self.client = genai
            self.model = self.client.GenerativeModel(self.model_name)
            self.mode = "studio"  # for distinction between api_key and vertex ai setup
        else:
            # Do vertex ai setup
            try:
                from google import genai
            except ImportError as e:
                raise ImportError(
                    "Vertex path needs `pip install google-genai` (new SDK)."
                ) from e
            _project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
            _location = location or os.environ.get("VERTEX_LOCATION", "us-central1")
            if not _project:
                raise ValueError(
                    "Vertex AI requires GOOGLE_CLOUD_PROJECT set, "
                    "and ADC via `gcloud auth application-default login` "
                    "or GOOGLE_APPLICATION_CREDENTIALS."
                )
            self.genai = genai
            self.client = genai.Client(vertexai=True, project=_project, location=_location)
            self.mode = "vertexai"
            
        	
        
    def send_msg(self, system_prompt: str, user_prompt: str, 
                 temperature: Optional[float] = None, 
                 json_format: bool = False) -> Tuple[str, bool]:
        """
        Send a message to the Gemini API and get a response.
        
        Args:
            system_prompt: System prompt to set the context
            user_prompt: User prompt/question
            temperature: Temperature override for this call
            json_format: Whether to request JSON format response
            
        Returns:
            Tuple of (response_text, is_json_valid)
        """
        if temperature is None:
            temperature = self.temperature
            
        retry_count = 0
        max_retries = 3
        
        # Combine system and user prompts
        combined_prompt = f"{system_prompt}\n\n{user_prompt}"
        # Add JSON format instruction if requested
        if json_format:
            combined_prompt += "\nRespond ONLY with valid JSON, like this: {key: value, key2: value2, etc.}. Be sure to include all the keys and values."

        while retry_count < max_retries:
            try:
                if self.mode == "studio":
                    # Configure generation parameters
                    generation_config = self.client.types.GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=self.max_tokens,
                        candidate_count=1
                    )
                    # Generate response
                    response = self.model.generate_content(
                        combined_prompt,
                        generation_config=generation_config
                    )
                    message = response.text
                else: # vertex ai mode
                    # Configure generation parameters
                    config = {
                        "temperature": temperature,
                        "maxOutputTokens": self.max_tokens,
                        "candidateCount": 1
                    }
                    # Generate response
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=combined_prompt,
                        config=config
                    )
                    message = response.text
                
                if not self._validate_response(message):
                    self.logger.warning(f"Invalid response received: {message}")
                    retry_count += 1
                    continue
                
                # Extract JSON if requested
                if json_format:
                    return self._extract_json(message)
                
                return message, False
                
            except Exception as e:
                if "quota" in str(e).lower() or "rate" in str(e).lower():
                    self.logger.warning(f"Rate limit or quota exceeded: {e}")
                    self._handle_rate_limit(retry_count, max_retries)
                    retry_count += 1
                else:
                    self.logger.error(f"Error calling Gemini: {e}")
                    retry_count += 1
                    if retry_count >= max_retries:
                        raise
                    sleep(1)
        
        raise Exception(f"Failed to get response after {max_retries} retries")
    
    @classmethod
    def get_available_models(cls):
        """Get list of available Gemini models."""
        return [
        	"gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-1.0-pro",
            "gemini-1.0-pro-latest"
        ]
    
    def list_models(self):
        """List all available models dynamically."""
        try:
            models = list(self.client.list_models())
            return [model.name for model in models if 'generateContent' in model.supported_generation_methods]
        except Exception as e:
            self.logger.error(f"Error listing models: {e}")
            return self.get_available_models()


class GeminiModelViaOpenRouter(BaseLLMModel):
    """Gemini model implementation using OpenRouter as a proxy."""
    
    def __init__(self, model_name: str = "google/gemini-flash-1.5", 
                 api_key: Optional[str] = None,
                 max_tokens: int = 1000, temperature: float = 0.7):
        """
        Initialize the Gemini model via OpenRouter.
        
        Args:
            model_name: Name of the Gemini model on OpenRouter
            api_key: OpenRouter API key (if None, will look for OPENROUTER_API_KEY env var)
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling
        """
        super().__init__(model_name, max_tokens, temperature)
        
        # Import OpenRouter model
        from .openrouter_model import OpenRouterModel
        
        self.openrouter_client = OpenRouterModel(
            model_name=model_name,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
    def send_msg(self, system_prompt: str, user_prompt: str, 
                 temperature: Optional[float] = None, 
                 json_format: bool = False) -> Tuple[str, bool]:
        """
        Send a message to the Gemini API via OpenRouter and get a response.
        
        Args:
            system_prompt: System prompt to set the context
            user_prompt: User prompt/question
            temperature: Temperature override for this call
            json_format: Whether to request JSON format response
            
        Returns:
            Tuple of (response_text, is_json_valid)
        """
        return self.openrouter_client.send_msg(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            json_format=json_format
        )
    
    @classmethod
    def get_available_models(cls):
        """Get list of available Gemini models on OpenRouter."""
        return [
            "google/gemini-pro-1.5",
            "google/gemini-flash-1.5",
            "google/gemini-pro"
        ] 
