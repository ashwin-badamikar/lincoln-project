"""
Project Gutenberg Scraper for Lincoln biographies.

Downloads and processes books from Project Gutenberg programmatically.
"""

import re
import time
import json
import requests
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm


@dataclass
class GutenbergBook:
    """Represents a book from Project Gutenberg."""
    id: str
    title: str
    author: str
    reference: str
    document_type: str = "Book"
    date: str = ""
    place: str = ""
    content: str = ""
    

# Books to download - Lincoln biographies by other authors
GUTENBERG_BOOKS = [
    {"id": "6812", "url": "https://www.gutenberg.org/ebooks/6812"},
    {"id": "6811", "url": "https://www.gutenberg.org/ebooks/6811"},
    {"id": "12801", "url": "https://www.gutenberg.org/ebooks/12801"},
    {"id": "14004", "url": "https://www.gutenberg.org/ebooks/14004"},
    {"id": "18379", "url": "https://www.gutenberg.org/ebooks/18379"},
]


class GutenbergScraper:
    """Scraper for Project Gutenberg books."""
    
    BASE_URL = "https://www.gutenberg.org"
    CACHE_URL = "https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    
    def __init__(self, output_dir: str = "data/raw/gutenberg"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Lincoln-Project-Research/1.0"
        })
    
    def get_plain_text_url(self, book_id: str) -> str:
        """Get the plain text download URL for a book."""
        return self.CACHE_URL.format(book_id=book_id)
    
    def download_book(self, book_id: str) -> Optional[str]:
        """
        Download a book's plain text content.
        
        Args:
            book_id: The Gutenberg book ID
            
        Returns:
            The book's text content, or None if download failed
        """
        url = self.get_plain_text_url(book_id)
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Save raw file
            raw_path = self.output_dir / f"{book_id}_raw.txt"
            raw_path.write_text(response.text, encoding="utf-8")
            print(f"  ✓ Downloaded book {book_id} ({len(response.text):,} chars)")
            
            return response.text
            
        except requests.RequestException as e:
            print(f"  ✗ Failed to download book {book_id}: {e}")
            return None
    
    def extract_metadata(self, text: str) -> dict:
        """
        Extract metadata from Gutenberg header.
        
        Args:
            text: Raw book text
            
        Returns:
            Dictionary with title, author, release_date, language
        """
        metadata = {
            "title": "Unknown",
            "author": "Unknown", 
            "release_date": "",
            "language": "English"
        }
        
        # Extract title
        title_match = re.search(r"Title:\s*(.+?)(?:\r?\n|$)", text)
        if title_match:
            metadata["title"] = title_match.group(1).strip()
        
        # Extract author
        author_match = re.search(r"Author:\s*(.+?)(?:\r?\n|$)", text)
        if author_match:
            metadata["author"] = author_match.group(1).strip()
        
        # Extract release date
        date_match = re.search(r"Release [Dd]ate:\s*(.+?)(?:\[|$|\r?\n)", text)
        if date_match:
            metadata["release_date"] = date_match.group(1).strip()
        
        # Extract language
        lang_match = re.search(r"Language:\s*(.+?)(?:\r?\n|$)", text)
        if lang_match:
            metadata["language"] = lang_match.group(1).strip()
            
        return metadata
    
    def strip_boilerplate(self, text: str) -> str:
        """
        Remove Project Gutenberg header and footer.
        
        Args:
            text: Raw book text with boilerplate
            
        Returns:
            Clean book content
        """
        # Find start of actual content
        start_markers = [
            "*** START OF THE PROJECT GUTENBERG",
            "*** START OF THIS PROJECT GUTENBERG",
            "***START OF THE PROJECT GUTENBERG",
            "***START OF THIS PROJECT GUTENBERG",
        ]
        
        end_markers = [
            "*** END OF THE PROJECT GUTENBERG",
            "*** END OF THIS PROJECT GUTENBERG", 
            "***END OF THE PROJECT GUTENBERG",
            "***END OF THIS PROJECT GUTENBERG",
            "End of the Project Gutenberg",
            "End of Project Gutenberg",
        ]
        
        start_idx = 0
        end_idx = len(text)
        
        # Find start
        for marker in start_markers:
            if marker in text:
                idx = text.find(marker)
                # Move past the marker line
                next_newline = text.find("\n", idx)
                if next_newline != -1:
                    start_idx = next_newline + 1
                break
        
        # Find end
        for marker in end_markers:
            if marker in text:
                end_idx = text.find(marker)
                break
        
        content = text[start_idx:end_idx].strip()
        return content
    
    def process_book(self, book_id: str) -> Optional[GutenbergBook]:
        """
        Download and process a single book.
        
        Args:
            book_id: The Gutenberg book ID
            
        Returns:
            GutenbergBook object, or None if processing failed
        """
        # Download
        raw_text = self.download_book(book_id)
        if not raw_text:
            return None
        
        # Extract metadata
        metadata = self.extract_metadata(raw_text)
        
        # Strip boilerplate
        content = self.strip_boilerplate(raw_text)
        
        # Create book object
        book = GutenbergBook(
            id=f"gutenberg_{book_id}",
            title=metadata["title"],
            author=metadata["author"],
            reference=f"https://www.gutenberg.org/ebooks/{book_id}",
            document_type="Book",
            date=metadata["release_date"],
            content=content
        )
        
        return book
    
    def scrape_all(self, delay: float = 1.0) -> list[GutenbergBook]:
        """
        Scrape all configured Lincoln books.
        
        Args:
            delay: Seconds to wait between requests (be polite!)
            
        Returns:
            List of GutenbergBook objects
        """
        books = []
        
        print(f"\n{'='*60}")
        print("Project Gutenberg Scraper - Lincoln Biographies")
        print(f"{'='*60}")
        print(f"Downloading {len(GUTENBERG_BOOKS)} books...\n")
        
        for book_info in tqdm(GUTENBERG_BOOKS, desc="Downloading books"):
            book_id = book_info["id"]
            print(f"\nProcessing book {book_id}...")
            
            book = self.process_book(book_id)
            if book:
                books.append(book)
                print(f"  Title: {book.title}")
                print(f"  Author: {book.author}")
                print(f"  Content length: {len(book.content):,} chars")
            
            # Be polite - wait between requests
            time.sleep(delay)
        
        print(f"\n{'='*60}")
        print(f"Successfully downloaded {len(books)}/{len(GUTENBERG_BOOKS)} books")
        print(f"{'='*60}\n")
        
        return books
    
    def save_normalized_dataset(self, books: list[GutenbergBook], output_path: str = "data/processed/gutenberg.json"):
        """
        Save books as normalized JSON dataset.
        
        Args:
            books: List of GutenbergBook objects
            output_path: Path for output JSON file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to list of dicts matching required schema
        dataset = []
        for book in books:
            item = {
                "id": book.id,
                "title": book.title,
                "reference": book.reference,
                "document_type": book.document_type,
                "date": book.date,
                "place": book.place,
                "from": book.author,  # Author of the book
                "to": "",  # Not applicable for books
                "content": book.content
            }
            dataset.append(item)
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        print(f"Saved normalized dataset to {output_file}")
        print(f"Total items: {len(dataset)}")


def main():
    """Main entry point for Gutenberg scraper."""
    # Initialize scraper
    scraper = GutenbergScraper(output_dir="data/raw/gutenberg")
    
    # Scrape all books
    books = scraper.scrape_all(delay=1.0)
    
    # Save normalized dataset
    if books:
        scraper.save_normalized_dataset(books, "data/processed/gutenberg.json")
        
        # Print summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        for book in books:
            print(f"\n{book.title}")
            print(f"  Author: {book.author}")
            print(f"  Length: {len(book.content):,} characters")
            print(f"  ~Words: {len(book.content.split()):,}")


if __name__ == "__main__":
    main()

