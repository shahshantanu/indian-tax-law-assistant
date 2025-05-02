#!/usr/bin/env python3
# PDF Processing Pipeline for Indian Tax Law Documents

import pymupdf
import pymupdf4llm
import tiktoken
import ftfy
import unicodedata
import json
import re
import os
import sys
import io
import shutil
from collections import Counter

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF, handling both text-based and scanned pages."""
    print(f"Opening PDF: {pdf_path}")
    doc = pymupdf.open(pdf_path)
    print(f"Processing {doc.page_count} pages...")
    
    page_texts = []        # will hold extracted text (Markdown) for each page
    ocr_pages = set()      # indices of pages where OCR was used
    page_mapping = {}      # maps page content to page numbers
    
    # Check if Tesseract is available
    tesseract_available = False
    if shutil.which("tesseract"):
        tesseract_available = True
        print("Tesseract OCR is available")
    else:
        print("WARNING: Tesseract OCR is not available. Image-only pages will be marked but text cannot be extracted.")
        print("Install Tesseract OCR and make sure it's in your PATH to enable OCR functionality.")
    
    for i in range(doc.page_count):
        if i % 50 == 0:
            print(f"Processing page {i+1}/{doc.page_count}...")
            
        page = doc[i]
        # Try extracting text directly
        text = page.get_text("text")  # plain text extraction
        
        if text is None or text.strip() == "":
            # No text found – likely a scanned page
            print(f"Page {i+1} appears to be image-only")
            ocr_pages.add(i)
            
            # Check for images on the page that might contain text
            image_count = 0
            for img in page.get_images():
                image_count += 1
            
            if image_count > 0:
                print(f"  Found {image_count} images on page {i+1}")
            
            if tesseract_available:
                try:
                    # Use PyMuPDF's integrated OCR (Tesseract)
                    print(f"  Attempting OCR on page {i+1}")
                    tp = page.get_textpage_ocr(language="eng", dpi=300, full=True)
                    ocr_text = page.get_text(textpage=tp)  # extract OCR text from TextPage
                    
                    if ocr_text.strip():
                        print(f"  OCR successful: extracted {len(ocr_text)} characters")
                        page_texts.append(ocr_text)
                    else:
                        print(f"  OCR returned empty result")
                        ocr_text = f"[IMAGE-ONLY PAGE {i+1} - OCR RETURNED EMPTY RESULT]"
                        page_texts.append(ocr_text)
                except Exception as e:
                    print(f"  Warning: OCR failed on page {i+1}: {e}")
                    ocr_text = f"[OCR FAILED ON PAGE {i+1}: {str(e)}]"
                    page_texts.append(ocr_text)
            else:
                # Fallback for when OCR is not available
                ocr_text = f"[IMAGE-ONLY PAGE {i+1} - TESSERACT OCR NOT AVAILABLE]"
                page_texts.append(ocr_text)
            
            page_mapping[len(page_texts)-1] = i
        else:
            # Text content exists; use basic text extraction
            try:
                # Format the text with markdown
                text_with_formatting = format_text_as_markdown(text)
                page_texts.append(text_with_formatting)
            except Exception as e:
                print(f"Error formatting page {i+1}: {e}")
                # Fallback to plain text
                page_texts.append(text)
            
            page_mapping[len(page_texts)-1] = i
    
    print(f"Text extraction complete. {len(ocr_pages)} pages required OCR.")
    return page_texts, ocr_pages, page_mapping, doc.page_count

def format_text_as_markdown(text):
    """Format text as markdown with simple rules."""
    lines = text.split('\n')
    result = []
    
    for i, line in enumerate(lines):
        # Simple heuristics for headings
        line_stripped = line.strip()
        
        if line_stripped.startswith("Chapter ") and len(line_stripped) < 30:
            result.append(f"# {line_stripped}")
        elif line_stripped.startswith("Section ") and len(line_stripped) < 50:
            result.append(f"## {line_stripped}")
        else:
            # Check for likely heading based on length and capitalization
            if len(line_stripped) < 60 and line_stripped.isupper() and line_stripped:
                result.append(f"### {line_stripped}")
            else:
                result.append(line)
    
    return '\n'.join(result)

def clean_and_format_text(page_texts, page_count):
    """Clean up text and format as Markdown."""
    # Combine all page texts into a single Markdown string
    full_md_text = "\n".join(page_texts)
    
    # Identify common first and last lines across pages (potential headers/footers)
    first_lines = [pt.splitlines()[0].strip() for pt in page_texts if pt.strip() and pt.splitlines()]
    last_lines = [pt.splitlines()[-1].strip() for pt in page_texts if pt.strip() and pt.splitlines()]
    
    common_first = [line for line, cnt in Counter(first_lines).items() 
                    if line and cnt > page_count * 0.3]  # appears in >30% pages
    common_last = [line for line, cnt in Counter(last_lines).items() 
                   if line and cnt > page_count * 0.3]
    
    print(f"Identified {len(common_first)} common headers and {len(common_last)} common footers")
    if common_first:
        print(f"  Headers: {common_first}")
    if common_last:
        print(f"  Footers: {common_last}")
    
    # Remove these common header/footer lines from the full text
    for hdr in common_first + common_last:
        full_md_text = re.sub(re.escape(hdr) + r'\s*\n', '', full_md_text)
    
    # Remove standalone page numbers or "Page X" lines
    full_md_text = re.sub(r'(?m)^(?:Page\s*\d+|\d+)\s*$', '', full_md_text)
    
    # Add markdown heading markers for OCR pages where headings are evident
    full_md_text = re.sub(r'(?m)^(Section\s+\d+[^:\n]*:?)', r'## \1', full_md_text)
    full_md_text = re.sub(r'(?m)^(CHAPTER\s+[IVX]+[^:\n]*:?)', r'# \1', full_md_text)
    
    # Join hyphenated words split across lines
    full_md_text = re.sub(r'(?<=\w)-\n(?=\w)', '', full_md_text)
    
    # Replace single newlines with space (preserving paragraph breaks)
    full_md_text = re.sub(r'(?<!\n)\n(?!\n)', ' ', full_md_text)
    
    # Normalize Unicode and fix any encoding issues
    full_md_text = unicodedata.normalize("NFC", full_md_text)
    full_md_text = ftfy.fix_text(full_md_text)
    
    return full_md_text

def split_into_sections(full_md_text):
    """Split text into sections based on headings."""
    sections = []
    current_section = None
    section_pages = []  # Track which pages each section spans
    current_pages = []
    
    for line in full_md_text.splitlines():
        # Identify a heading (Markdown heading or a Section line)
        if re.match(r'^(#{1,3}\s+.*|Section\s+\d+|CHAPTER\s+[IVX]+)', line):
            # If we were accumulating a section, store it
            if current_section:
                current_section["pages"] = current_pages.copy()
                sections.append(current_section)
                current_pages = []
            
            # Start a new section
            heading_text = line.lstrip('# ').strip()
            # Extract section number and title if available
            sec_num = None
            sec_title = heading_text
            m = re.match(r'(?:Section\s+)?(\d+[\.\)]?)\s*(.*)', heading_text)
            if m:
                sec_num = m.group(1).rstrip('.')  # e.g., "1" from "1." or "Section 1"
                if m.group(2):
                    sec_title = m.group(2)
            
            # Handle chapter headings
            chapter_match = re.match(r'CHAPTER\s+([IVX]+)[\.:\s]*\s*(.*)', heading_text)
            if chapter_match:
                sec_num = f"Chapter {chapter_match.group(1)}"
                if chapter_match.group(2):
                    sec_title = chapter_match.group(2)
                else:
                    sec_title = f"Chapter {chapter_match.group(1)}"
            
            current_section = {
                "section_number": sec_num or heading_text,
                "heading": sec_title if sec_num else heading_text,
                "text": line + "\n",  # Include the heading in the text
                "pages": []
            }
        elif re.match(r'^\[(?:OCR FAILED|IMAGE-ONLY) ON PAGE (\d+).*\]', line):
            # Track page numbers from OCR failure markers
            page_match = re.match(r'^\[(?:OCR FAILED|IMAGE-ONLY) ON PAGE (\d+).*\]', line)
            if page_match:
                page_num = int(page_match.group(1)) - 1
                if current_section:
                    current_pages.append(page_num)
                    
            # For image-only pages with section numbering, try to extract section info
            section_match = re.search(r'Section\s+(\d+)[\.:\s]*\s*([^:\]]+)', line)
            if section_match and not current_section:
                sec_num = section_match.group(1)
                sec_title = section_match.group(2).strip()
                
                current_section = {
                    "section_number": sec_num,
                    "heading": sec_title,
                    "text": f"## Section {sec_num}. {sec_title}\n",
                    "pages": [page_num] if 'page_num' in locals() else []
                }
                
            if current_section:
                current_section["text"] += line + "\n"
        else:
            # Append line to current section text (if a section has started)
            if current_section:
                current_section["text"] += line + "\n"
            else:
                # Lines before the first identified section (could be preamble)
                if not sections or sections[-1]["heading"] != "Introduction":
                    sections.append({
                        "section_number": "Intro",
                        "heading": "Introduction",
                        "text": "",
                        "pages": []
                    })
                sections[-1]["text"] += line + "\n"
    
    # Append the last section
    if current_section:
        current_section["pages"] = current_pages.copy()
        sections.append(current_section)
    
    print(f"Identified {len(sections)} sections")
    return sections

def create_chunks(sections, ocr_pages, max_tokens=500, overlap=50):
    """Create chunks from sections with token count control and overlap."""
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    chunk_dicts = []
    
    for sec in sections:
        sec_text = sec["text"].strip()
        if not sec_text:
            continue  # skip empty sections
        
        tokens = enc.encode(sec_text)
        print(f"Section '{sec['heading']}': {len(tokens)} tokens")
        
        if len(tokens) <= max_tokens:
            # Section fits in one chunk
            is_ocr = any(p in ocr_pages for p in sec.get("pages", []))
            chunk_dicts.append({
                "section_number": sec["section_number"],
                "heading": sec["heading"],
                "jurisdiction": "India",
                "ocr": is_ocr,
                "text": sec_text,
                "token_count": len(tokens),
                "pages": sec.get("pages", [])
            })
        else:
            # Split section text into overlapping chunks
            start = 0
            chunk_num = 1
            total_chunks = (len(tokens) + max_tokens - 1) // max_tokens
            
            while start < len(tokens):
                end = min(start + max_tokens, len(tokens))
                
                # Try to find a good break point (period followed by space, or newline)
                if end < len(tokens):
                    chunk_text = enc.decode(tokens[start:end])
                    # Look for the last period or newline within the last 100 characters
                    last_100_chars = chunk_text[-100:] if len(chunk_text) > 100 else chunk_text
                    period_pos = last_100_chars.rfind('. ')
                    newline_pos = last_100_chars.rfind('\n')
                    
                    # Use the later of period or newline as break point
                    break_pos = max(period_pos, newline_pos)
                    if break_pos != -1:
                        # Adjust end to this breakpoint
                        adjusted_end = len(chunk_text) - (len(last_100_chars) - break_pos)
                        if period_pos > newline_pos:  # If period is the break point, include the period
                            adjusted_end += 1
                        chunk_text = chunk_text[:adjusted_end]
                        # Recalculate end in tokens
                        end = start + len(enc.encode(chunk_text))
                
                # Decode the final chunk
                chunk_text = enc.decode(tokens[start:end])
                
                # Estimate which pages this chunk covers
                # (simplified - distribute pages proportionally across chunks)
                all_pages = sec.get("pages", [])
                if all_pages:
                    chunk_ratio = (end - start) / len(tokens)
                    chunk_pages = all_pages[int(start/len(tokens)*len(all_pages)):
                                          min(int((end/len(tokens))*len(all_pages)) + 1, len(all_pages))]
                else:
                    chunk_pages = []
                
                # Check if any of these pages were OCR-based
                is_ocr = any(p in ocr_pages for p in chunk_pages)
                
                chunk_dicts.append({
                    "section_number": sec["section_number"],
                    "heading": sec["heading"],
                    "jurisdiction": "India",
                    "ocr": is_ocr,
                    "text": chunk_text.strip(),
                    "token_count": end - start,
                    "chunk": f"{chunk_num} of {total_chunks}",
                    "pages": chunk_pages
                })
                
                chunk_num += 1
                if end == len(tokens):
                    break  # reached end of section
                start = end - overlap  # overlap tokens with previous chunk
    
    print(f"Created {len(chunk_dicts)} chunks")
    return chunk_dicts

def write_jsonl(chunks, output_path):
    """Write chunks to JSONL file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            # Create clean record for output (remove internal tracking fields)
            record = {
                "section_number": chunk["section_number"],
                "heading": chunk["heading"],
                "jurisdiction": "India",
                "ocr": chunk["ocr"],
                "text": chunk["text"]
            }
            # Include chunk number for multi-chunk sections
            if "chunk" in chunk:
                record["chunk"] = chunk["chunk"]
            
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"Wrote {len(chunks)} chunks to {output_path}")

def process_pdf(pdf_path, output_path, max_tokens=500, overlap=50):
    """Process PDF into chunks and save as JSONL."""
    # Step 1: Extract text from PDF
    page_texts, ocr_pages, page_mapping, page_count = extract_text_from_pdf(pdf_path)
    
    # Step 2 & 3: Clean and format text
    full_md_text = clean_and_format_text(page_texts, page_count)
    
    # Step 4: Split into sections
    sections = split_into_sections(full_md_text)
    
    # Step 4 & 5: Create chunks with metadata
    chunks = create_chunks(sections, ocr_pages, max_tokens, overlap)
    
    # Step 6: Write to JSONL
    write_jsonl(chunks, output_path)
    
    return len(chunks)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process PDF into JSONL chunks for LLMs")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--output", "-o", default="output_chunks.jsonl", 
                        help="Output JSONL file path (default: output_chunks.jsonl)")
    parser.add_argument("--max-tokens", "-m", type=int, default=500,
                        help="Maximum tokens per chunk (default: 500)")
    parser.add_argument("--overlap", type=int, default=50,
                        help="Token overlap between chunks (default: 50)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Check if PDF file exists
    if not os.path.isfile(args.pdf_path):
        print(f"Error: PDF file '{args.pdf_path}' not found")
        sys.exit(1)
    
    print(f"Processing {args.pdf_path}...")
    print(f"Max tokens per chunk: {args.max_tokens}")
    print(f"Overlap between chunks: {args.overlap}")
    
    num_chunks = process_pdf(args.pdf_path, args.output, args.max_tokens, args.overlap)
    
    print(f"Done! {num_chunks} chunks written to {args.output}")
    print(f"Use these chunks for vector database ingestion or LLM fine-tuning.") 