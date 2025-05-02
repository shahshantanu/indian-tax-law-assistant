"""
Drafter agent module for generating legal responses with citations.

This module provides functionality to generate formal legal responses
based on relevant context from the vector store.
"""

import os
import re
from typing import List, Dict, Any, Tuple, Set, Optional
from app.services.llm_client import llm_client
from app.agents.retriever_agent import ContextChunk

# Check if embeddings are disabled
EMBEDDINGS_DISABLED = os.getenv("DISABLE_EMBEDDINGS", "false").lower() == "true"

class DrafterAgent:
    """Agent for drafting formal responses with citations."""
    
    def __init__(self):
        """Initialize the drafter agent."""
        pass
    
    def draft_response(
        self, 
        query: str, 
        context_chunks: List[ContextChunk],
        max_tokens: int = 2048
    ) -> Dict[str, Any]:
        """
        Generate a formal response to a query with context and citations.
        
        Args:
            query: User query string
            context_chunks: List of relevant context chunks
            max_tokens: Maximum tokens for the response
            
        Returns:
            Dictionary with analysis, answer, and citations
        """
        if not context_chunks:
            if EMBEDDINGS_DISABLED:
                # Provide a demo mode response when running without embeddings
                return self._generate_demo_response(query)
            else:
                return {
                    "analysis": "Insufficient context available to answer the query.",
                    "answer": "I don't have enough information to answer this question.",
                    "citations": []
                }
        
        # Format the context with citation markers
        formatted_context = self._format_context(context_chunks)
        
        # Build the prompt for the LLM
        system_prompt = """You are an expert Indian tax law professional specializing in detailed legal analysis and formal drafting. 
Your task is to answer legal tax questions with authoritative, comprehensive analysis based on the Indian Income Tax Act.

APPROACH:
1. Carefully analyze the provided context from the Indian tax code
2. Identify all relevant sections, provisions, and legal interpretations
3. Provide structured legal reasoning with proper citations
4. Present a formal, precise, and authoritative analysis
5. When referencing legislation or legal provisions, always cite sources using [n] tags
6. Every citation [n] MUST correspond to a specific source in the context provided

Your response must follow this exact format:

Analysis
[Your detailed legal analysis including:
- Relevant statutory provisions with precise citations
- Interpretation of applicable sections with legal reasoning
- Discussion of relevant conditions, exemptions, and requirements
- Clear explanation of tax implications based on the cited provisions]

Final Answer
[Your definitive, authoritative answer to the query in formal legal language]

Citations
[1] [First citation source]
[2] [Second citation source]
...

Ensure every citation mentioned in your analysis and answer appears in the Citations section.
Your analysis must be formal, precise, and demonstrate deep expertise in Indian tax law."""

        user_prompt = f"""[CONTEXT]
{formatted_context}

[REQUEST]
{query}"""

        # Get response from LLM
        response = llm_client.chat(
            system=system_prompt,
            user=user_prompt,
            max_tokens=max_tokens,
            temperature=0.1  # Lower temperature for more factual responses
        )
        
        # Post-process the response
        return self._process_response(response, context_chunks)
    
    def _generate_demo_response(self, query: str) -> Dict[str, Any]:
        """
        Generate a demonstration response when running in demo mode.
        
        Args:
            query: User query string
            
        Returns:
            Mock structured response
        """
        system_prompt = """You are an Indian tax-law assistant. 
The system is currently running in DEMO MODE, and you don't have access to the tax law database.
Create a simulated response that includes:
1. A detailed analysis section that would be typical for this type of tax question
2. A concise final answer
3. Mention that this is a demonstration and would typically include actual citations to the Indian Income Tax Act

Your response must follow this format:

Analysis
[Your simulated analysis of what you would provide if you had access to the tax law database]

Final Answer
[Your simulated concise answer]"""

        user_prompt = f"""[REQUEST]
{query}

Remember this is a DEMO response. Make it clear that in normal operation, this would include actual citations to specific sections of tax law."""

        # Get response from LLM
        response = llm_client.chat(
            system=system_prompt,
            user=user_prompt,
            max_tokens=2048,
            temperature=0.7  # More creative for demo responses
        )
        
        # Process the demo response
        analysis = self._extract_section(response, "Analysis", "Final Answer")
        answer = self._extract_section(response, "Final Answer", None)
        
        if not analysis:
            analysis = "This is a demonstration of the Indian Tax Law Assistant running in low-memory mode. In normal operation, this would provide detailed analysis with citations to specific sections of the Income Tax Act. Vector search is currently disabled."
        
        if not answer:
            answer = "This is a demonstration response. For accurate tax information, please consult the system when running in normal mode with full vector search capabilities."
        
        return {
            "analysis": analysis.strip(),
            "answer": answer.strip() + "\n\n(DEMO MODE - Vector search disabled)",
            "citations": ["[1] Demo citation - In normal operation, this would reference specific sections of the Income Tax Act"]
        }
    
    def _format_context(self, chunks: List[ContextChunk]) -> str:
        """
        Format context chunks for LLM input with citation markers.
        
        Args:
            chunks: List of context chunks
            
        Returns:
            Formatted context string
        """
        formatted_chunks = []
        
        for i, chunk in enumerate(chunks):
            # Format with citation marker [n]
            citation_number = i + 1
            formatted_chunk = f"[{citation_number}] {chunk.text}"
            formatted_chunks.append(formatted_chunk)
        
        return "\n\n".join(formatted_chunks)
    
    def _process_response(
        self, 
        response: str, 
        context_chunks: List[ContextChunk]
    ) -> Dict[str, Any]:
        """
        Process and structure the raw LLM response.
        
        Args:
            response: Raw LLM response
            context_chunks: Original context chunks
            
        Returns:
            Structured response with analysis, answer, and citations
        """
        # Extract sections
        analysis = self._extract_section(response, "Analysis", "Final Answer")
        answer = self._extract_section(response, "Final Answer", "Citations")
        citations_text = self._extract_section(response, "Citations", None)
        
        # Create citations map
        citations_map = {}
        for i, chunk in enumerate(context_chunks):
            citation_number = i + 1
            
            # Format the full citation
            section = chunk.metadata.get("section_number", "")
            heading = chunk.metadata.get("heading", "")
            jurisdiction = chunk.metadata.get("jurisdiction", "")
            
            if section:
                full_citation = f"Section {section}"
                if heading and heading != section:
                    full_citation += f" ({heading})"
            else:
                full_citation = heading or "Reference"
            
            if jurisdiction and jurisdiction not in full_citation:
                full_citation += f", {jurisdiction}"
            
            citations_map[citation_number] = full_citation
        
        # Extract all citation references from the response
        citation_refs = set(re.findall(r'\[(\d+)\]', analysis + " " + answer))
        citation_refs = {int(ref) for ref in citation_refs if ref.isdigit()}
        
        # Build the formatted citations list
        citations = []
        for ref in sorted(citation_refs):
            if ref <= len(context_chunks):
                citations.append(f"[{ref}] {citations_map.get(ref, 'Reference')}")
        
        # Ensure all references are valid
        if not citations:
            # If no valid citations, add a generic one
            citations.append("[1] General reference to tax law provisions")
        
        return {
            "analysis": analysis.strip(),
            "answer": answer.strip(),
            "citations": citations
        }
    
    def _extract_section(
        self, 
        text: str, 
        start_marker: str, 
        end_marker: Optional[str]
    ) -> str:
        """
        Extract a section from the response text.
        
        Args:
            text: Full response text
            start_marker: Section start marker
            end_marker: Section end marker (or None for last section)
            
        Returns:
            Extracted section text
        """
        if start_marker not in text:
            return ""
        
        # Find the start of the section
        start_pos = text.find(start_marker) + len(start_marker)
        
        # Find the end of the section
        if end_marker and end_marker in text[start_pos:]:
            end_pos = text.find(end_marker, start_pos)
            return text[start_pos:end_pos].strip()
        else:
            return text[start_pos:].strip()


# Export agent instance
drafter_agent = DrafterAgent() 