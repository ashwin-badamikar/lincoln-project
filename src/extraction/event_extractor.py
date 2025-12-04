"""
Event Extraction Pipeline using LLM.

Extracts information about 5 key Lincoln events from historical texts.
"""

import os
import json
import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict
from openai import OpenAI
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential


# The 5 key events to extract
EVENTS = {
    "election_night_1860": {
        "name": "Election Night 1860",
        "description": "Abraham Lincoln's election as the 16th President on November 6, 1860",
        "keywords": ["election", "november 1860", "voted", "elected", "president", "republican", "votes", "electoral"],
        "date_range": "November 1860"
    },
    "fort_sumter_decision": {
        "name": "Fort Sumter Decision", 
        "description": "Lincoln's decision whether to resupply or reinforce Fort Sumter, leading to the start of the Civil War",
        "keywords": ["sumter", "fort", "charleston", "resupply", "reinforce", "anderson", "provisions", "april 1861", "bombardment"],
        "date_range": "March-April 1861"
    },
    "gettysburg_address": {
        "name": "Gettysburg Address",
        "description": "Lincoln's famous speech at the dedication of the Soldiers' National Cemetery",
        "keywords": ["gettysburg", "cemetery", "dedicated", "fourscore", "four score", "november 1863", "consecrate", "brave men"],
        "date_range": "November 19, 1863"
    },
    "second_inaugural_address": {
        "name": "Second Inaugural Address",
        "description": "Lincoln's second inaugural speech with 'malice toward none'",
        "keywords": ["inaugural", "second", "malice", "charity", "march 1865", "bind up", "wounds", "sworn"],
        "date_range": "March 4, 1865"
    },
    "fords_theatre_assassination": {
        "name": "Ford's Theatre Assassination",
        "description": "Lincoln's assassination by John Wilkes Booth at Ford's Theatre",
        "keywords": ["ford", "theatre", "theater", "booth", "assassin", "shot", "april 1865", "killed", "pistol", "our american cousin"],
        "date_range": "April 14-15, 1865"
    }
}


@dataclass
class ExtractedEvent:
    """Represents extracted information about an event."""
    event_id: str
    event_name: str
    source_id: str
    source_title: str
    author: str
    claims: list[str]
    temporal_details: dict
    tone: str
    direct_quotes: list[str]
    confidence: str


class EventExtractor:
    """Extracts event information from texts using LLM."""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Initialize the extractor.
        
        Args:
            model: OpenAI model to use (gpt-4o, gpt-4o-mini, etc.)
        """
        self.client = OpenAI()  # Uses OPENAI_API_KEY env var
        self.model = model
        self.output_dir = Path("data/extracted")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def chunk_text(self, text: str, chunk_size: int = 12000, overlap: int = 1000) -> list[str]:
        """
        Split text into overlapping chunks to handle context window limits.
        
        Args:
            text: Full text to chunk
            chunk_size: Maximum characters per chunk
            overlap: Overlap between chunks to avoid cutting context
            
        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at paragraph boundary
            if end < len(text):
                # Look for paragraph break near the end
                para_break = text.rfind("\n\n", start + chunk_size - 500, end)
                if para_break > start:
                    end = para_break
            
            chunks.append(text[start:end])
            start = end - overlap
        
        return chunks
    
    def is_relevant_chunk(self, chunk: str, event_id: str) -> bool:
        """
        Quick keyword check to filter irrelevant chunks before LLM call.
        
        Args:
            chunk: Text chunk to check
            event_id: Event identifier
            
        Returns:
            True if chunk might contain event information
        """
        event_info = EVENTS.get(event_id, {})
        keywords = event_info.get("keywords", [])
        
        chunk_lower = chunk.lower()
        
        # Check if any keyword appears
        for keyword in keywords:
            if keyword.lower() in chunk_lower:
                return True
        
        return False
    
    def build_extraction_prompt(self, chunk: str, event_id: str) -> str:
        """
        Build the prompt for event extraction.
        
        Uses Chain-of-Thought prompting for better quality extraction.
        """
        event_info = EVENTS[event_id]
        
        prompt = f"""You are a meticulous historical analyst extracting information about a specific event from primary and secondary sources about Abraham Lincoln.

## Event to Extract
**{event_info['name']}**
{event_info['description']}
Time period: {event_info['date_range']}

## Source Text
{chunk}

## Your Task
Extract ALL factual claims, details, and information about "{event_info['name']}" from this text.

Think step by step:
1. First, identify any mentions of this event or related circumstances
2. Note specific factual claims (dates, times, names, numbers, locations, actions)
3. Identify any direct quotes from Lincoln or others about this event
4. Assess the author's tone/attitude toward Lincoln regarding this event
5. Rate your confidence in the extraction

## Output Format
Return a valid JSON object with these fields:

{{
    "event_mentioned": true/false,
    "reasoning": "Brief explanation of what you found",
    "claims": [
        "Specific factual claim 1",
        "Specific factual claim 2"
    ],
    "temporal_details": {{
        "date": "If mentioned",
        "time": "If mentioned",
        "duration": "If mentioned"
    }},
    "people_mentioned": ["List of people involved"],
    "locations_mentioned": ["List of places"],
    "direct_quotes": [
        "Any direct quotes from Lincoln about this event"
    ],
    "tone": "Sympathetic | Critical | Neutral | Admiring | Mixed",
    "confidence": "High | Medium | Low"
}}

If the event is NOT mentioned in this text, return:
{{
    "event_mentioned": false,
    "reasoning": "Event not discussed in this passage",
    "claims": [],
    "temporal_details": {{}},
    "people_mentioned": [],
    "locations_mentioned": [],
    "direct_quotes": [],
    "tone": "N/A",
    "confidence": "High"
}}

Important: Only include information EXPLICITLY stated in the text. Do not infer or add external knowledge."""

        return prompt
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def call_llm(self, prompt: str, temperature: float = 0.1) -> dict:
        """
        Make LLM API call with retry logic.
        
        Args:
            prompt: The extraction prompt
            temperature: Sampling temperature (low for consistency)
            
        Returns:
            Parsed JSON response
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            response_format={"type": "json_object"},
            max_tokens=2000
        )
        
        content = response.choices[0].message.content
        return json.loads(content)
    
    def extract_from_chunk(self, chunk: str, event_id: str) -> Optional[dict]:
        """
        Extract event information from a single chunk.
        
        Args:
            chunk: Text chunk to process
            event_id: Event to extract
            
        Returns:
            Extraction result dict, or None if event not found
        """
        # Quick relevance check
        if not self.is_relevant_chunk(chunk, event_id):
            return None
        
        prompt = self.build_extraction_prompt(chunk, event_id)
        
        try:
            result = self.call_llm(prompt)
            
            if result.get("event_mentioned", False) and result.get("claims"):
                return result
            
        except Exception as e:
            print(f"    LLM call failed: {e}")
        
        return None
    
    def merge_extractions(self, extractions: list[dict]) -> dict:
        """
        Merge multiple extraction results for the same event.
        
        Args:
            extractions: List of extraction dicts from different chunks
            
        Returns:
            Merged extraction dict
        """
        if not extractions:
            return {}
        
        if len(extractions) == 1:
            return extractions[0]
        
        # Merge claims (deduplicate)
        all_claims = []
        seen_claims = set()
        for ext in extractions:
            for claim in ext.get("claims", []):
                claim_lower = claim.lower().strip()
                if claim_lower not in seen_claims:
                    all_claims.append(claim)
                    seen_claims.add(claim_lower)
        
        # Merge temporal details
        temporal = {}
        for ext in extractions:
            td = ext.get("temporal_details", {})
            for key, value in td.items():
                if value and key not in temporal:
                    temporal[key] = value
        
        # Merge direct quotes
        all_quotes = []
        seen_quotes = set()
        for ext in extractions:
            for quote in ext.get("direct_quotes", []):
                quote_key = quote.lower()[:50]
                if quote_key not in seen_quotes:
                    all_quotes.append(quote)
                    seen_quotes.add(quote_key)
        
        # Merge people and locations
        people = list(set(
            person for ext in extractions 
            for person in ext.get("people_mentioned", [])
        ))
        locations = list(set(
            loc for ext in extractions
            for loc in ext.get("locations_mentioned", [])
        ))
        
        # Determine overall tone (most common)
        tones = [ext.get("tone", "Neutral") for ext in extractions if ext.get("tone") != "N/A"]
        tone = max(set(tones), key=tones.count) if tones else "Neutral"
        
        return {
            "event_mentioned": True,
            "claims": all_claims,
            "temporal_details": temporal,
            "people_mentioned": people,
            "locations_mentioned": locations,
            "direct_quotes": all_quotes,
            "tone": tone,
            "confidence": "High" if len(extractions) > 1 else extractions[0].get("confidence", "Medium"),
            "num_sources": len(extractions)
        }
    
    def extract_events_from_document(self, document: dict) -> list[ExtractedEvent]:
        """
        Extract all events from a single document.
        
        Args:
            document: Normalized document dict
            
        Returns:
            List of ExtractedEvent objects
        """
        results = []
        content = document.get("content", "")
        source_id = document.get("id", "unknown")
        source_title = document.get("title", "Unknown")
        author = document.get("from", "Unknown")
        
        # Chunk the document
        chunks = self.chunk_text(content)
        print(f"    Split into {len(chunks)} chunks")
        
        for event_id, event_info in EVENTS.items():
            print(f"    Checking for: {event_info['name']}")
            
            # Extract from each relevant chunk
            event_extractions = []
            for i, chunk in enumerate(chunks):
                extraction = self.extract_from_chunk(chunk, event_id)
                if extraction:
                    event_extractions.append(extraction)
                    print(f"      Found in chunk {i+1}: {len(extraction.get('claims', []))} claims")
            
            # Merge extractions
            if event_extractions:
                merged = self.merge_extractions(event_extractions)
                
                result = ExtractedEvent(
                    event_id=event_id,
                    event_name=event_info["name"],
                    source_id=source_id,
                    source_title=source_title,
                    author=author,
                    claims=merged.get("claims", []),
                    temporal_details=merged.get("temporal_details", {}),
                    tone=merged.get("tone", "Neutral"),
                    direct_quotes=merged.get("direct_quotes", []),
                    confidence=merged.get("confidence", "Medium")
                )
                results.append(result)
                print(f"      Total: {len(result.claims)} claims extracted")
        
        return results
    
    def process_dataset(self, dataset_path: str, dataset_name: str) -> list[ExtractedEvent]:
        """
        Process an entire dataset and extract events.
        
        Args:
            dataset_path: Path to normalized JSON dataset
            dataset_name: Name for output file
            
        Returns:
            List of all extracted events
        """
        print(f"\n{'='*60}")
        print(f"Processing: {dataset_path}")
        print(f"{'='*60}")
        
        with open(dataset_path, "r", encoding="utf-8") as f:
            documents = json.load(f)
        
        print(f"Found {len(documents)} documents")
        
        all_extractions = []
        
        for doc in tqdm(documents, desc="Extracting events"):
            print(f"\n  Document: {doc.get('title', 'Unknown')}")
            extractions = self.extract_events_from_document(doc)
            all_extractions.extend(extractions)
            
            # Small delay between documents
            time.sleep(0.5)
        
        # Save results
        output_path = self.output_dir / f"{dataset_name}_events.json"
        output_data = [asdict(e) for e in all_extractions]
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nSaved {len(all_extractions)} event extractions to {output_path}")
        
        return all_extractions
    
    def generate_summary(self, extractions: list[ExtractedEvent]) -> dict:
        """Generate a summary of extracted events."""
        summary = {event_id: {"name": info["name"], "sources": [], "total_claims": 0} 
                   for event_id, info in EVENTS.items()}
        
        for ext in extractions:
            event_id = ext.event_id
            if event_id in summary:
                summary[event_id]["sources"].append({
                    "source": ext.source_title,
                    "author": ext.author,
                    "claims_count": len(ext.claims),
                    "tone": ext.tone
                })
                summary[event_id]["total_claims"] += len(ext.claims)
        
        return summary


def main():
    """Main entry point for event extraction."""
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set!")
        print("Set it with: $env:OPENAI_API_KEY='your-key-here'")
        return
    
    extractor = EventExtractor(model="gpt-4o-mini")  # Use gpt-4o for better quality
    
    all_extractions = []
    
    # Process Gutenberg dataset (other authors)
    gutenberg_path = Path("data/processed/gutenberg.json")
    if gutenberg_path.exists():
        extractions = extractor.process_dataset(str(gutenberg_path), "gutenberg")
        all_extractions.extend(extractions)
    else:
        print(f"Warning: {gutenberg_path} not found. Run scrapers first!")
    
    # Process LoC dataset (Lincoln's own words)
    loc_path = Path("data/processed/loc.json")
    if loc_path.exists():
        extractions = extractor.process_dataset(str(loc_path), "loc")
        all_extractions.extend(extractions)
    else:
        print(f"Warning: {loc_path} not found. Run scrapers first!")
    
    # Generate and save summary
    if all_extractions:
        summary = extractor.generate_summary(all_extractions)
        
        summary_path = Path("data/extracted/extraction_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "="*60)
        print("EXTRACTION SUMMARY")
        print("="*60)
        for event_id, data in summary.items():
            print(f"\n{data['name']}:")
            print(f"  Total claims: {data['total_claims']}")
            print(f"  Sources: {len(data['sources'])}")


if __name__ == "__main__":
    main()

