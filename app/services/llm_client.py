"""
LLM client service for interacting with language models.

This module provides a wrapper around the OpenAI-compatible chat completion API
for Llama 3 models via ollama or vllm.
"""

import json
import time
import httpx
from typing import List, Dict, Any, Optional, Union, Tuple
from app import settings

class Message:
    """Class representing a chat message."""
    
    def __init__(self, role: str, content: str):
        """Initialize a chat message."""
        self.role = role
        self.content = content
        
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary representation."""
        return {"role": self.role, "content": self.content}


class Tool:
    """Class representing a function-calling tool."""
    
    def __init__(
        self, 
        name: str, 
        description: str, 
        parameters: Dict[str, Any]
    ):
        """Initialize a tool definition."""
        self.name = name
        self.description = description
        self.parameters = parameters
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }


class LLMClient:
    """Client for interacting with LLMs via OpenAI-compatible API."""
    
    _instance = None
    
    def __new__(cls):
        """Ensure only one instance of the LLM client exists."""
        if cls._instance is None:
            cls._instance = super(LLMClient, cls).__new__(cls)
        return cls._instance
    
    def chat(
        self,
        system: str,
        user: str,
        tools: Optional[List[Tool]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
        retry_count: int = 3,
        retry_delay: float = 2.0
    ) -> Union[str, Dict[str, Any]]:
        """
        Send a chat completion request to the LLM.
        
        Args:
            system: System message content
            user: User message content
            tools: Optional list of function-calling tools
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream the response
            retry_count: Number of retries on failure
            retry_delay: Delay between retries in seconds
            
        Returns:
            Text response from the LLM, or full response dict if tools are used
        """
        messages = [
            Message("system", system).to_dict(),
            Message("user", user).to_dict()
        ]
        
        payload = {
            "model": settings.LLM_MODEL,
            "messages": messages,
            "temperature": temperature or settings.LLM_TEMPERATURE,
            "max_tokens": max_tokens or settings.LLM_MAX_TOKENS,
            "stream": stream
        }
        
        if tools:
            payload["tools"] = [tool.to_dict() for tool in tools]
        
        headers = {
            "Content-Type": "application/json"
        }
        
        if settings.LLM_API_KEY:
            headers["Authorization"] = f"Bearer {settings.LLM_API_KEY}"
        
        url = f"{settings.LLM_API_BASE}/chat/completions"
        
        # Retry loop
        for attempt in range(retry_count):
            try:
                with httpx.Client(timeout=120) as client:
                    response = client.post(
                        url,
                        json=payload,
                        headers=headers
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Handle tool calls if present
                        if tools and "tool_calls" in result.get("choices", [{}])[0].get("message", {}):
                            return result
                        
                        # Regular text response
                        return result["choices"][0]["message"]["content"]
                    else:
                        print(f"Request failed: HTTP {response.status_code}")
                        print(f"Response: {response.text}")
                        
                        # Retry only on server errors or rate limits
                        if response.status_code < 500 and response.status_code != 429:
                            return f"Error: HTTP {response.status_code}"
                
            except Exception as e:
                print(f"Request error on attempt {attempt+1}/{retry_count}: {str(e)}")
            
            # Wait before retrying
            if attempt < retry_count - 1:
                time.sleep(retry_delay)
        
        return "Error: Failed to get response from LLM after retries"
    
    def stream_chat(
        self,
        system: str,
        user: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ):
        """
        Stream a chat completion from the LLM.
        
        Args:
            system: System message content
            user: User message content
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Yields:
            Chunks of the generated text
        """
        messages = [
            Message("system", system).to_dict(),
            Message("user", user).to_dict()
        ]
        
        payload = {
            "model": settings.LLM_MODEL,
            "messages": messages,
            "temperature": temperature or settings.LLM_TEMPERATURE,
            "max_tokens": max_tokens or settings.LLM_MAX_TOKENS,
            "stream": True
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        if settings.LLM_API_KEY:
            headers["Authorization"] = f"Bearer {settings.LLM_API_KEY}"
        
        url = f"{settings.LLM_API_BASE}/chat/completions"
        
        with httpx.Client(timeout=120) as client:
            with client.stream("POST", url, json=payload, headers=headers) as response:
                if response.status_code != 200:
                    yield f"Error: HTTP {response.status_code}"
                    return
                
                buffer = ""
                for chunk in response.iter_lines():
                    if not chunk or chunk.startswith("data: [DONE]"):
                        continue
                    
                    if chunk.startswith("data: "):
                        try:
                            chunk_data = json.loads(chunk[6:])
                            content = chunk_data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if content:
                                buffer += content
                                yield content
                        except json.JSONDecodeError:
                            continue
        
        if not buffer:
            yield "Error: No content received from LLM"


# Export singleton instance
llm_client = LLMClient() 