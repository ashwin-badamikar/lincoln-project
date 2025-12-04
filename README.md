# Lincoln Project - ML Evaluation Pipeline

An automated system to analyze historiographical divergence in accounts of Abraham Lincoln's life, comparing first-person accounts (Lincoln's own words) with secondary sources (biographies by other authors).

## 🎯 Project Overview

This project implements a complete ML evaluation pipeline that:

1. **Scrapes data** from Project Gutenberg (5 Lincoln biographies) and Library of Congress (5 first-person documents)
2. **Extracts events** using LLM to identify information about 5 key historical events
3. **Compares accounts** using an LLM Judge to detect inconsistencies between Lincoln's words and biographers' accounts
4. **Validates statistically** through ablation studies, self-consistency tests, and Cohen's Kappa

## 📁 Project Structure

```
lincoln-project/
├── data/
│   ├── raw/                    # Downloaded source files
│   │   ├── gutenberg/          # Gutenberg book texts
│   │   └── loc/                # LoC document files
│   ├── processed/              # Normalized JSON datasets
│   │   ├── gutenberg.json
│   │   └── loc.json
│   ├── extracted/              # LLM extraction results
│   │   ├── gutenberg_events.json
│   │   ├── loc_events.json
│   │   └── judgment_results.json
│   └── validation/             # Statistical validation results
├── src/
│   ├── scrapers/
│   │   ├── gutenberg_scraper.py
│   │   └── loc_scraper.py
│   ├── extraction/
│   │   └── event_extractor.py
│   ├── judge/
│   │   └── llm_judge.py
│   └── validation/
│       └── statistical_tests.py
├── notebooks/
│   └── analysis.ipynb
├── run_pipeline.py             # Main entry point
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd lincoln-project
pip install -r requirements.txt
```

### 2. Set OpenAI API Key

**Windows PowerShell:**
```powershell
$env:OPENAI_API_KEY="your-api-key-here"
```

**Windows Command Prompt:**
```cmd
set OPENAI_API_KEY=your-api-key-here
```

**Linux/Mac:**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 3. Run the Pipeline

**Run everything:**
```bash
python run_pipeline.py --all
```

**Or run steps individually:**
```bash
python run_pipeline.py --scrape      # Phase 1: Download data
python run_pipeline.py --extract     # Phase 2: Extract events
python run_pipeline.py --judge       # Phase 3: Run LLM judge
python run_pipeline.py --validate    # Phase 4: Statistical validation
```

## 📊 The 5 Key Events

The system extracts and compares information about:

1. **Election Night 1860** - Lincoln's election as President
2. **Fort Sumter Decision** - The decision to resupply the fort
3. **Gettysburg Address** - The famous cemetery dedication speech
4. **Second Inaugural Address** - "With malice toward none..."
5. **Ford's Theatre Assassination** - Lincoln's death

## 🔬 Statistical Validation

### 1. Prompt Ablation Study
Compares three prompting strategies:
- **Zero-shot**: Direct instructions, no examples
- **Chain-of-Thought**: Step-by-step reasoning
- **Few-shot**: Includes examples

### 2. Self-Consistency Test
- Runs the same comparison 5 times with temperature > 0
- Measures score variance to assess reliability
- A reliable judge should have low standard deviation

### 3. Cohen's Kappa (Human Alignment)
- Manually label 10 comparison pairs
- Calculate inter-rater agreement between LLM and human
- Kappa interpretation:
  - 0.81-1.00: Almost perfect agreement
  - 0.61-0.80: Substantial agreement
  - 0.41-0.60: Moderate agreement

## 📝 Data Schema

### Normalized Document Schema
```json
{
    "id": "unique_identifier",
    "title": "Human-readable title",
    "reference": "URL or citation",
    "document_type": "Letter | Speech | Book | Note",
    "date": "As in source",
    "place": "Location if known",
    "from": "Author/Writer",
    "to": "Recipient if applicable",
    "content": "Full text content"
}
```

### Event Extraction Schema
```json
{
    "event_id": "fort_sumter_decision",
    "event_name": "Fort Sumter Decision",
    "source_id": "gutenberg_6812",
    "author": "Author Name",
    "claims": ["List of factual claims"],
    "temporal_details": {"date": "...", "time": "..."},
    "tone": "Sympathetic | Critical | Neutral",
    "direct_quotes": ["Any direct quotes"]
}
```

### Judgment Result Schema
```json
{
    "event_id": "fort_sumter_decision",
    "consistency_score": 75,
    "contradictions": [
        {
            "type": "Factual | Interpretive | Omission",
            "lincoln_claim": "What Lincoln said",
            "other_claim": "What author said",
            "explanation": "Why it's a contradiction",
            "severity": "High | Medium | Low"
        }
    ],
    "alignments": ["Claims that match"]
}
```

## 💡 Key Design Decisions

### Handling Long Documents
Books can be 200,000+ characters. The pipeline:
1. Chunks text into ~12,000 character segments with overlap
2. Pre-filters chunks using keyword matching (saves API costs)
3. Merges extractions from multiple chunks

### Prompt Engineering
Chain-of-Thought prompting provides:
- More stable scores (lower variance)
- Better reasoning explanations
- Clearer contradiction classification

### Rate Limiting
- Project Gutenberg: 1 second delay between requests
- Library of Congress: 2 second delay (stricter)
- OpenAI API: Retry with exponential backoff

## 📈 Expected Output

After running the full pipeline, you'll have:

1. **Raw data** in `data/raw/`
2. **Normalized datasets** in `data/processed/`
3. **Event extractions** in `data/extracted/`
4. **Judgment results** with consistency scores
5. **Validation report** with:
   - Ablation study results
   - Self-consistency metrics
   - Manual labeling template for Kappa

## 🔧 Troubleshooting

### "API key not set"
Make sure you've exported the OPENAI_API_KEY environment variable in the same terminal session.

### "Rate limit exceeded"
The code includes automatic retries. If you still hit limits, increase the delays in the scrapers.

### "Module not found"
Run from the project root directory, or ensure the `src` folder is in your Python path.

## 📚 Data Sources

- **Project Gutenberg**: https://www.gutenberg.org
  - Books 6812, 6811, 12801, 14004, 18379
  
- **Library of Congress**: https://www.loc.gov
  - Lincoln Papers and Exhibits

## 🎓 Learning Resources

- [LLM as Judge - Confident AI](https://www.confident-ai.com/blog/why-llm-as-a-judge-is-the-best-llm-evaluation-method)
- [LLM Guide - Evidently AI](https://www.evidentlyai.com/llm-guide)
- [HuggingFace Cookbook - LLM Judge](https://huggingface.co/learn/cookbook)

