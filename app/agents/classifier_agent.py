"""
Classifier agent module for categorizing legal documents.

This module provides functionality to classify documents into predefined
categories using LLM with few-shot prompting.
"""

from typing import Dict, Any, List, Optional
from app.services.llm_client import llm_client

# Valid document types
DOCUMENT_TYPES = [
    "tax_notice",
    "audit_report",
    "it_return",
    "tribunal_ruling",
    "other"
]

# Few-shot examples for the classifier
FEW_SHOT_EXAMPLES = """
Example 1:
```
NOTICE UNDER SECTION 143(2) OF THE INCOME TAX ACT, 1961
Ref: AX20221117100A
Date: 01.04.2023

To,
Mr. JOHN DOE
123 MAIN STREET
MUMBAI - 400001

Subject: Notice for providing further information for assessment year 2022-23

Dear Sir/Madam,
Your return of income for Assessment Year 2022-23 has been selected for scrutiny...
```
Document Type: tax_notice

Example 2:
```
INCOME TAX DEPARTMENT
AUDIT REPORT
Assessment Year: 2022-23
PAN: ABCDE1234F
Taxpayer Name: XYZ INDUSTRIES LTD.

1. INTRODUCTION
This audit was conducted to verify the claims made in the income tax return of XYZ Industries Ltd...

2. AUDIT FINDINGS
a) Discrepancies found in depreciation claims on plant and machinery...
```
Document Type: audit_report

Example 3:
```
FORM ITR-2
INDIAN INCOME TAX RETURN
Assessment Year 2022-23

PART A - GENERAL INFORMATION
Name: JANE DOE
PAN: FGHIJ5678K
Status: Individual
Address: 456 Park Avenue, Delhi - 110001

PART B - GROSS TOTAL INCOME
1. Income from salary: ₹12,50,000
2. Income from house property: ₹3,25,000
...
```
Document Type: it_return
"""

class ClassifierAgent:
    """Agent for classifying document types."""
    
    def __init__(self):
        """Initialize the classifier agent."""
        pass
    
    def classify_document(
        self, 
        metadata: Dict[str, Any], 
        content_preview: str,
        custom_types: Optional[List[str]] = None
    ) -> str:
        """
        Classify a document based on its metadata and content preview.
        
        Args:
            metadata: Document metadata (filename, date, etc.)
            content_preview: First 2K characters of the document
            custom_types: Optional list of custom document types
            
        Returns:
            Classification label
        """
        valid_types = custom_types or DOCUMENT_TYPES
        valid_types_str = ", ".join([f'"{t}"' for t in valid_types])
        
        system_prompt = f"""You are an expert legal document classifier for Indian tax documents.
Your task is to classify the document into one of the following categories: {valid_types_str}.
Analyze both the metadata and content preview to determine the document type.
Respond ONLY with the document type as your classification."""
        
        user_prompt = f"""I need to classify this tax-related document:

METADATA:
{metadata}

CONTENT PREVIEW:
{content_preview}

Based on the examples below, classify this document into one of these types: {valid_types_str}.

{FEW_SHOT_EXAMPLES}

Return ONLY the document type label, nothing else."""
        
        classification = llm_client.chat(
            system=system_prompt,
            user=user_prompt,
            temperature=0.1,  # Low temperature for more deterministic outputs
        )
        
        # Clean up the response
        classification = classification.strip().lower()
        
        # Ensure it's a valid type
        if classification not in valid_types:
            # Try to find the closest match
            for valid_type in valid_types:
                if valid_type in classification:
                    return valid_type
            return "other"  # Default to "other" if no match
        
        return classification


# Export agent instance
classifier_agent = ClassifierAgent() 