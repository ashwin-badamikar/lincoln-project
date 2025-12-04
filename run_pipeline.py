"""
Lincoln Project - Main Pipeline Runner

Run this script to execute the complete pipeline:
1. Scrape data from Project Gutenberg and Library of Congress
2. Extract events using LLM
3. Run LLM Judge for consistency analysis
4. Perform statistical validation

Usage:
    python run_pipeline.py --all           # Run entire pipeline
    python run_pipeline.py --scrape        # Only scrape data
    python run_pipeline.py --extract       # Only extract events
    python run_pipeline.py --judge         # Only run judge
    python run_pipeline.py --validate      # Only run validation
"""

import os
import sys
import argparse
from pathlib import Path

# Load .env file from project root
from dotenv import load_dotenv
env_path = Path(__file__).parent / ".env"

if env_path.exists():
    load_dotenv(env_path)
    print(f"✓ Loaded API key from .env")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def check_api_key():
    """Check if OpenAI API key is set."""
    if not os.environ.get("OPENAI_API_KEY"):
        print("="*60)
        print("ERROR: OPENAI_API_KEY not set!")
        print("="*60)
        print("\nSet it using one of these methods:")
        print("\n  Windows PowerShell:")
        print('    $env:OPENAI_API_KEY="your-key-here"')
        print("\n  Windows Command Prompt:")
        print('    set OPENAI_API_KEY=your-key-here')
        print("\n  Linux/Mac:")
        print('    export OPENAI_API_KEY="your-key-here"')
        print("\nOr create a .env file with:")
        print('    OPENAI_API_KEY=your-key-here')
        return False
    return True


def run_scrapers():
    """Run data scrapers."""
    print("\n" + "="*60)
    print("PHASE 1: DATA ACQUISITION")
    print("="*60)
    
    # Change to project directory
    os.chdir(Path(__file__).parent)
    
    # Run Gutenberg scraper
    print("\n--- Project Gutenberg Scraper ---")
    from scrapers.gutenberg_scraper import GutenbergScraper
    
    gutenberg = GutenbergScraper(output_dir="data/raw/gutenberg")
    books = gutenberg.scrape_all(delay=1.0)
    if books:
        gutenberg.save_normalized_dataset(books, "data/processed/gutenberg.json")
    
    # Run LoC scraper
    print("\n--- Library of Congress Scraper ---")
    from scrapers.loc_scraper import LocScraper
    
    loc = LocScraper(output_dir="data/raw/loc")
    documents = loc.scrape_all(delay=2.0)
    if documents:
        loc.save_normalized_dataset(documents, "data/processed/loc.json")
    
    print("\n✓ Data acquisition complete!")
    print(f"  - Gutenberg books: {len(books)}")
    print(f"  - LoC documents: {len(documents)}")


def run_extraction():
    """Run event extraction."""
    print("\n" + "="*60)
    print("PHASE 2: EVENT EXTRACTION")
    print("="*60)
    
    if not check_api_key():
        return False
    
    os.chdir(Path(__file__).parent)
    
    from extraction.event_extractor import EventExtractor
    
    extractor = EventExtractor(model="gpt-4o-mini")
    
    # Check for data files
    gutenberg_path = Path("data/processed/gutenberg.json")
    loc_path = Path("data/processed/loc.json")
    
    if not gutenberg_path.exists() or not loc_path.exists():
        print("ERROR: Run scrapers first (--scrape)")
        return False
    
    # Process both datasets
    extractor.process_dataset(str(gutenberg_path), "gutenberg")
    extractor.process_dataset(str(loc_path), "loc")
    
    print("\n✓ Event extraction complete!")
    return True


def run_judge():
    """Run LLM Judge."""
    print("\n" + "="*60)
    print("PHASE 3: LLM JUDGE")
    print("="*60)
    
    if not check_api_key():
        return False
    
    os.chdir(Path(__file__).parent)
    
    from judge.llm_judge import LLMJudge, load_extractions
    
    # Check for extraction files
    loc_events = Path("data/extracted/loc_events.json")
    gutenberg_events = Path("data/extracted/gutenberg_events.json")
    
    if not loc_events.exists() or not gutenberg_events.exists():
        print("ERROR: Run extraction first (--extract)")
        return False
    
    judge = LLMJudge(model="gpt-4o")
    
    lincoln_extractions = load_extractions(str(loc_events))
    other_extractions = load_extractions(str(gutenberg_events))
    
    results = judge.compare_all_pairs(
        lincoln_extractions,
        other_extractions,
        strategy="chain_of_thought"
    )
    
    judge.save_results(results)
    
    print("\n✓ LLM Judge complete!")
    return True


def run_validation():
    """Run statistical validation."""
    print("\n" + "="*60)
    print("PHASE 4: STATISTICAL VALIDATION")
    print("="*60)
    
    if not check_api_key():
        return False
    
    os.chdir(Path(__file__).parent)
    
    from validation.statistical_tests import StatisticalValidator
    from judge.llm_judge import load_extractions
    
    # Check for extraction files
    loc_events = Path("data/extracted/loc_events.json")
    gutenberg_events = Path("data/extracted/gutenberg_events.json")
    
    if not loc_events.exists() or not gutenberg_events.exists():
        print("ERROR: Run extraction first (--extract)")
        return False
    
    validator = StatisticalValidator(model="gpt-4o")
    
    lincoln_extractions = load_extractions(str(loc_events))
    other_extractions = load_extractions(str(gutenberg_events))
    
    # Run ablation study
    print("\n--- Ablation Study ---")
    ablation_results = validator.run_ablation_study(
        lincoln_extractions,
        other_extractions
    )
    
    # Run self-consistency
    print("\n--- Self-Consistency Test ---")
    consistency_results = validator.run_self_consistency_batch(
        lincoln_extractions,
        other_extractions,
        n_samples=3,  # Reduced for speed
        n_runs=5
    )
    
    # Create manual labeling template
    print("\n--- Creating Manual Labeling Template ---")
    validator.create_manual_labeling_template(
        lincoln_extractions,
        other_extractions,
        n_samples=10
    )
    
    # Save report
    validator.save_validation_report(
        ablation_results=ablation_results,
        consistency_results=consistency_results
    )
    
    print("\n✓ Validation complete!")
    print("\nNext steps:")
    print("  1. Fill in data/validation/manual_labels_template.json")
    print("  2. Run Kappa calculation to get human alignment score")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Lincoln Project Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_pipeline.py --all        # Run everything
    python run_pipeline.py --scrape     # Just scrape data
    python run_pipeline.py --extract    # Just extract events
    python run_pipeline.py --judge      # Just run LLM judge
    python run_pipeline.py --validate   # Just run validation
        """
    )
    
    parser.add_argument("--all", action="store_true", help="Run entire pipeline")
    parser.add_argument("--scrape", action="store_true", help="Run data scrapers")
    parser.add_argument("--extract", action="store_true", help="Run event extraction")
    parser.add_argument("--judge", action="store_true", help="Run LLM judge")
    parser.add_argument("--validate", action="store_true", help="Run statistical validation")
    
    args = parser.parse_args()
    
    # If no args, show help
    if not any([args.all, args.scrape, args.extract, args.judge, args.validate]):
        parser.print_help()
        return
    
    print("="*60)
    print("LINCOLN PROJECT - ML EVALUATION PIPELINE")
    print("="*60)
    
    if args.all or args.scrape:
        run_scrapers()
    
    if args.all or args.extract:
        run_extraction()
    
    if args.all or args.judge:
        run_judge()
    
    if args.all or args.validate:
        run_validation()
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()

