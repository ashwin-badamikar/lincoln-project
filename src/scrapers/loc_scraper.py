"""
Library of Congress Scraper for Lincoln first-person documents.

Downloads and processes Lincoln's letters, speeches, and notes from LoC.
"""

import re
import time
import json
import requests
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from bs4 import BeautifulSoup
from tqdm import tqdm


@dataclass
class LocDocument:
    """Represents a document from Library of Congress."""
    id: str
    title: str
    reference: str
    document_type: str
    date: str = ""
    place: str = ""
    from_person: str = "Abraham Lincoln"
    to_person: str = ""
    content: str = ""


# Documents to download - Lincoln's first-person accounts
LOC_DOCUMENTS = [
    {
        "id": "election_night_1860",
        "title": "Letter about Election Night 1860",
        "url": "https://www.loc.gov/item/mal0440500/",
        "resource_url": "https://www.loc.gov/resource/mal.0440500/",
        "type": "Letter",
        "date": "November 1860"
    },
    {
        "id": "fort_sumter_decision",
        "title": "Fort Sumter Decision",
        "url": "https://www.loc.gov/resource/mal.0882800",
        "resource_url": "https://www.loc.gov/resource/mal.0882800/",
        "type": "Note",
        "date": "April 1861"
    },
    {
        "id": "gettysburg_address",
        "title": "Gettysburg Address (Nicolay Copy)",
        "url": "https://www.loc.gov/exhibits/gettysburg-address/ext/trans-nicolay-copy.html",
        "type": "Speech",
        "date": "November 19, 1863",
        "place": "Gettysburg, Pennsylvania"
    },
    {
        "id": "second_inaugural",
        "title": "Second Inaugural Address",
        "url": "https://www.loc.gov/resource/mal.4361300",
        "resource_url": "https://www.loc.gov/resource/mal.4361300/",
        "type": "Speech",
        "date": "March 4, 1865",
        "place": "Washington, D.C."
    },
    {
        "id": "last_public_address",
        "title": "Last Public Address",
        "url": "https://www.loc.gov/resource/mal.4361800/",
        "resource_url": "https://www.loc.gov/resource/mal.4361800/",
        "type": "Speech",
        "date": "April 11, 1865",
        "place": "Washington, D.C."
    },
]


class LocScraper:
    """Scraper for Library of Congress Lincoln documents."""
    
    BASE_URL = "https://www.loc.gov"
    
    def __init__(self, output_dir: str = "data/raw/loc"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Lincoln-Project-Research/1.0",
            "Accept": "application/json, text/html, */*"
        })
    
    def fetch_json_api(self, url: str) -> Optional[dict]:
        """
        Fetch data from LoC JSON API.
        
        Args:
            url: The resource URL
            
        Returns:
            JSON response dict, or None if failed
        """
        # LoC JSON API: append ?fo=json
        json_url = url.rstrip("/") + "/?fo=json"
        
        try:
            response = self.session.get(json_url, timeout=30)
            if response.ok and response.headers.get("content-type", "").startswith("application/json"):
                return response.json()
        except Exception as e:
            print(f"  JSON API failed: {e}")
        
        return None
    
    def fetch_html_page(self, url: str) -> Optional[str]:
        """
        Fetch HTML content from a page.
        
        Args:
            url: The page URL
            
        Returns:
            HTML content string, or None if failed
        """
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"  HTML fetch failed: {e}")
            return None
    
    def extract_text_from_loc_json(self, data: dict) -> str:
        """
        Extract text content from LoC JSON response.
        
        The LoC API structure varies - we try multiple paths.
        """
        text_parts = []
        
        # Try different paths where text might be stored
        
        # Check for 'text' field directly
        if "text" in data:
            return data["text"]
        
        # Check for content in 'resources' or 'resource'
        if "resource" in data:
            resource = data["resource"]
            if isinstance(resource, dict) and "text" in resource:
                return resource["text"]
        
        # Check for item/content structure
        if "item" in data:
            item = data["item"]
            if isinstance(item, dict):
                if "notes" in item:
                    text_parts.extend(item["notes"])
                if "contents" in item:
                    text_parts.append(item["contents"])
        
        # Check for 'content' field
        if "content" in data:
            content = data["content"]
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                text_parts.extend([str(c) for c in content])
        
        return "\n".join(text_parts)
    
    def extract_text_from_exhibit_html(self, html: str) -> str:
        """
        Extract transcript text from LoC exhibit page (like Gettysburg Address).
        
        Args:
            html: HTML content of the exhibit page
            
        Returns:
            Extracted text content
        """
        soup = BeautifulSoup(html, "lxml")
        
        # Try different selectors for exhibit pages
        # The Gettysburg Address page has the transcript in specific divs
        
        # Look for main content area
        content_selectors = [
            "div.content",
            "div.main-content", 
            "div#content",
            "article",
            "div.transcript",
            "div.document-text",
            "main",
        ]
        
        for selector in content_selectors:
            content = soup.select_one(selector)
            if content:
                # Get text, preserving some structure
                text = content.get_text(separator="\n", strip=True)
                if len(text) > 100:  # Reasonable content found
                    return text
        
        # Fallback: get body text
        body = soup.find("body")
        if body:
            # Remove script and style tags
            for tag in body.find_all(["script", "style", "nav", "header", "footer"]):
                tag.decompose()
            return body.get_text(separator="\n", strip=True)
        
        return ""
    
    def fetch_document_content(self, doc_info: dict) -> Optional[str]:
        """
        Fetch document content using appropriate method.
        
        Args:
            doc_info: Document metadata dict
            
        Returns:
            Document text content
        """
        url = doc_info.get("url", "")
        resource_url = doc_info.get("resource_url", url)
        
        # Special handling for exhibit pages (Gettysburg Address)
        if "/exhibits/" in url:
            print(f"  Fetching exhibit page...")
            html = self.fetch_html_page(url)
            if html:
                # Save raw HTML
                raw_path = self.output_dir / f"{doc_info['id']}_raw.html"
                raw_path.write_text(html, encoding="utf-8")
                return self.extract_text_from_exhibit_html(html)
        
        # Try JSON API first for resource URLs
        print(f"  Trying JSON API...")
        json_data = self.fetch_json_api(resource_url)
        if json_data:
            # Save raw JSON
            raw_path = self.output_dir / f"{doc_info['id']}_raw.json"
            raw_path.write_text(json.dumps(json_data, indent=2), encoding="utf-8")
            
            text = self.extract_text_from_loc_json(json_data)
            if text:
                return text
        
        # Try item page JSON API
        if "/item/" in url:
            print(f"  Trying item JSON API...")
            json_data = self.fetch_json_api(url)
            if json_data:
                raw_path = self.output_dir / f"{doc_info['id']}_item.json"
                raw_path.write_text(json.dumps(json_data, indent=2), encoding="utf-8")
                
                text = self.extract_text_from_loc_json(json_data)
                if text:
                    return text
        
        # Fallback: fetch HTML and parse
        print(f"  Falling back to HTML parsing...")
        html = self.fetch_html_page(url)
        if html:
            raw_path = self.output_dir / f"{doc_info['id']}_raw.html"
            raw_path.write_text(html, encoding="utf-8")
            return self.extract_text_from_exhibit_html(html)
        
        return None
    
    def process_document(self, doc_info: dict) -> Optional[LocDocument]:
        """
        Process a single LoC document.
        
        Args:
            doc_info: Document metadata dict
            
        Returns:
            LocDocument object, or None if processing failed
        """
        print(f"\n  Fetching: {doc_info['title']}")
        
        content = self.fetch_document_content(doc_info)
        
        if not content:
            print(f"  ✗ Could not extract content")
            return None
        
        print(f"  ✓ Extracted {len(content):,} chars")
        
        # Create document object
        doc = LocDocument(
            id=f"loc_{doc_info['id']}",
            title=doc_info["title"],
            reference=doc_info["url"],
            document_type=doc_info["type"],
            date=doc_info.get("date", ""),
            place=doc_info.get("place", ""),
            from_person="Abraham Lincoln",
            to_person=doc_info.get("to", ""),
            content=content
        )
        
        return doc
    
    def scrape_all(self, delay: float = 2.0) -> list[LocDocument]:
        """
        Scrape all configured Lincoln documents.
        
        Args:
            delay: Seconds to wait between requests (respect rate limits!)
            
        Returns:
            List of LocDocument objects
        """
        documents = []
        
        print(f"\n{'='*60}")
        print("Library of Congress Scraper - Lincoln First-Person Documents")
        print(f"{'='*60}")
        print(f"Downloading {len(LOC_DOCUMENTS)} documents...")
        print("(Respecting rate limits with delays)")
        
        for doc_info in tqdm(LOC_DOCUMENTS, desc="Downloading documents"):
            doc = self.process_document(doc_info)
            if doc:
                documents.append(doc)
            
            # Respect rate limits - LoC is stricter
            time.sleep(delay)
        
        print(f"\n{'='*60}")
        print(f"Successfully downloaded {len(documents)}/{len(LOC_DOCUMENTS)} documents")
        print(f"{'='*60}\n")
        
        return documents
    
    def save_normalized_dataset(self, documents: list[LocDocument], output_path: str = "data/processed/loc.json"):
        """
        Save documents as normalized JSON dataset.
        
        Args:
            documents: List of LocDocument objects
            output_path: Path for output JSON file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to list of dicts matching required schema
        dataset = []
        for doc in documents:
            item = {
                "id": doc.id,
                "title": doc.title,
                "reference": doc.reference,
                "document_type": doc.document_type,
                "date": doc.date,
                "place": doc.place,
                "from": doc.from_person,
                "to": doc.to_person,
                "content": doc.content
            }
            dataset.append(item)
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        print(f"Saved normalized dataset to {output_file}")
        print(f"Total items: {len(dataset)}")


# Well-known transcripts for fallback (these are public domain)
KNOWN_TRANSCRIPTS = {
    "gettysburg_address": """Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal.

Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure. We are met on a great battle-field of that war. We have come to dedicate a portion of that field, as a final resting place for those who here gave their lives that that nation might live. It is altogether fitting and proper that we should do this.

But, in a larger sense, we can not dedicate -- we can not consecrate -- we can not hallow -- this ground. The brave men, living and dead, who struggled here, have consecrated it, far above our poor power to add or detract. The world will little note, nor long remember what we say here, but it can never forget what they did here. It is for us the living, rather, to be dedicated here to the unfinished work which they who fought here have thus far so nobly advanced. It is rather for us to be here dedicated to the great task remaining before us -- that from these honored dead we take increased devotion to that cause for which they gave the last full measure of devotion -- that we here highly resolve that these dead shall not have died in vain -- that this nation, under God, shall have a new birth of freedom -- and that government of the people, by the people, for the people, shall not perish from the earth.

Abraham Lincoln
November 19, 1863""",
}


def main():
    """Main entry point for LoC scraper."""
    # Initialize scraper
    scraper = LocScraper(output_dir="data/raw/loc")
    
    # Scrape all documents
    documents = scraper.scrape_all(delay=2.0)
    
    # Save normalized dataset
    if documents:
        scraper.save_normalized_dataset(documents, "data/processed/loc.json")
        
        # Print summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        for doc in documents:
            print(f"\n{doc.title}")
            print(f"  Type: {doc.document_type}")
            print(f"  Date: {doc.date}")
            print(f"  Length: {len(doc.content):,} characters")


if __name__ == "__main__":
    main()

