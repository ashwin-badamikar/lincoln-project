"""
LLM Judge for Historiographical Consistency Analysis.

Compares Lincoln's first-person accounts with secondary source claims
to detect inconsistencies and score alignment.
"""

import os
import json
import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass
class Contradiction:
    """Represents a detected contradiction between sources."""
    type: str  # Factual, Interpretive, Omission
    lincoln_claim: str
    other_claim: str
    explanation: str
    severity: str  # High, Medium, Low


@dataclass
class JudgmentResult:
    """Result of comparing two claim sets."""
    event_id: str
    event_name: str
    lincoln_source: str
    other_source: str
    other_author: str
    consistency_score: int  # 0-100
    contradictions: list[Contradiction]
    alignments: list[str]  # Claims that match
    reasoning: str
    prompt_strategy: str
    temperature: float


class LLMJudge:
    """
    LLM-based evaluator for historiographical consistency.
    
    Supports multiple prompt strategies for ablation studies.
    """
    
    PROMPT_STRATEGIES = ["zero_shot", "chain_of_thought", "few_shot"]
    
    def __init__(self, model: str = "gpt-4o"):
        """
        Initialize the judge.
        
        Args:
            model: OpenAI model to use
        """
        self.client = OpenAI()
        self.model = model
        self.output_dir = Path("data/extracted")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def build_zero_shot_prompt(self, lincoln_claims: list[str], other_claims: list[str], 
                                event_name: str, other_author: str) -> str:
        """Build a zero-shot prompt (direct instruction, no examples)."""
        
        lincoln_text = "\n".join(f"- {claim}" for claim in lincoln_claims) if lincoln_claims else "- No direct claims found"
        other_text = "\n".join(f"- {claim}" for claim in other_claims) if other_claims else "- No claims found"
        
        return f"""Compare these two sets of claims about "{event_name}".

## Lincoln's Own Words/Account:
{lincoln_text}

## {other_author}'s Account:
{other_text}

Analyze the consistency between these accounts.

Return a JSON object with:
- consistency_score: 0-100 (0=total contradiction, 100=perfect alignment)
- contradictions: array of objects with {{type, lincoln_claim, other_claim, explanation, severity}}
  - type must be one of: "Factual", "Interpretive", "Omission"
  - severity must be one of: "High", "Medium", "Low"
- alignments: array of claims that match between sources
- reasoning: brief overall analysis"""

    def build_chain_of_thought_prompt(self, lincoln_claims: list[str], other_claims: list[str],
                                       event_name: str, other_author: str) -> str:
        """Build a chain-of-thought prompt (step-by-step reasoning)."""
        
        lincoln_text = "\n".join(f"- {claim}" for claim in lincoln_claims) if lincoln_claims else "- No direct claims found"
        other_text = "\n".join(f"- {claim}" for claim in other_claims) if other_claims else "- No claims found"
        
        return f"""You are a meticulous historical analyst comparing primary and secondary sources about "{event_name}".

## Lincoln's Own Words/Account (Primary Source):
{lincoln_text}

## {other_author}'s Account (Secondary Source):
{other_text}

## Analysis Instructions
Think through this step-by-step:

1. **IDENTIFY ALIGNMENTS**: First, find claims that appear in BOTH accounts or are consistent with each other. These show agreement between sources.

2. **FIND FACTUAL CONTRADICTIONS**: Look for claims where specific facts differ:
   - Different dates, times, numbers
   - Different names or locations
   - Different sequences of events
   - Contradictory statements about what happened

3. **FIND INTERPRETIVE DIFFERENCES**: Look for claims where the interpretation differs:
   - Different motivations attributed to Lincoln
   - Different emotional states described
   - Different significance assigned to events
   - Different judgments about decisions

4. **IDENTIFY OMISSIONS**: Note significant information present in one account but absent from the other:
   - Key details Lincoln mentioned that the author omitted
   - Claims the author makes that Lincoln never addressed

5. **CALCULATE CONSISTENCY SCORE**: 
   - Start at 100 (perfect alignment)
   - Subtract points for each contradiction based on severity:
     - High severity factual: -20 points
     - Medium severity factual: -10 points
     - High severity interpretive: -15 points
     - Medium/Low interpretive: -5 points
     - Significant omissions: -5 points
   - Minimum score is 0

Return a JSON object:
{{
    "step_by_step_reasoning": "Your detailed analysis following the steps above",
    "consistency_score": <0-100>,
    "contradictions": [
        {{
            "type": "Factual|Interpretive|Omission",
            "lincoln_claim": "What Lincoln said/wrote",
            "other_claim": "What the other author said (or 'Not mentioned' for omissions)",
            "explanation": "Why this is a contradiction",
            "severity": "High|Medium|Low"
        }}
    ],
    "alignments": ["List of claims that align between sources"],
    "reasoning": "Brief summary of overall consistency"
}}"""

    def build_few_shot_prompt(self, lincoln_claims: list[str], other_claims: list[str],
                               event_name: str, other_author: str) -> str:
        """Build a few-shot prompt (with examples)."""
        
        lincoln_text = "\n".join(f"- {claim}" for claim in lincoln_claims) if lincoln_claims else "- No direct claims found"
        other_text = "\n".join(f"- {claim}" for claim in other_claims) if other_claims else "- No claims found"
        
        return f"""You are comparing historical accounts for consistency. Here are examples of how to analyze contradictions:

## Example 1: Factual Contradiction
Lincoln says: "I received the news of my election at the telegraph office around midnight"
Biographer says: "Lincoln learned of his victory at 2 AM at his home"
Analysis: FACTUAL contradiction - different time (midnight vs 2 AM) and location (telegraph office vs home). Severity: Medium.

## Example 2: Interpretive Difference  
Lincoln says: "I made the decision after careful deliberation with my cabinet"
Biographer says: "Lincoln had already decided and merely sought validation from advisors"
Analysis: INTERPRETIVE difference - same events but different interpretation of Lincoln's decision-making process. Severity: Low.

## Example 3: Omission
Lincoln says: "Secretary Seward strongly opposed my decision"
Biographer says: [No mention of Seward's opposition]
Analysis: OMISSION - Lincoln's account includes Seward's opposition, biographer omits this detail. Severity: Medium.

## Example 4: Alignment
Lincoln says: "The ceremony took place on March 4th"
Biographer says: "On March 4th, Lincoln delivered his address"
Analysis: ALIGNMENT - both agree on the date.

---

Now analyze this comparison about "{event_name}":

## Lincoln's Own Words:
{lincoln_text}

## {other_author}'s Account:
{other_text}

Return a JSON object with:
- consistency_score: 0-100
- contradictions: array with {{type, lincoln_claim, other_claim, explanation, severity}}
- alignments: array of matching claims
- reasoning: brief analysis"""

    def build_prompt(self, lincoln_claims: list[str], other_claims: list[str],
                     event_name: str, other_author: str, strategy: str) -> str:
        """
        Build prompt based on strategy.
        
        Args:
            lincoln_claims: Claims from Lincoln's first-person account
            other_claims: Claims from another author
            event_name: Name of the event being compared
            other_author: Name of the other author
            strategy: Prompt strategy (zero_shot, chain_of_thought, few_shot)
        """
        if strategy == "zero_shot":
            return self.build_zero_shot_prompt(lincoln_claims, other_claims, event_name, other_author)
        elif strategy == "chain_of_thought":
            return self.build_chain_of_thought_prompt(lincoln_claims, other_claims, event_name, other_author)
        elif strategy == "few_shot":
            return self.build_few_shot_prompt(lincoln_claims, other_claims, event_name, other_author)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def call_llm(self, prompt: str, temperature: float = 0.3) -> dict:
        """Make LLM API call with retry logic."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            response_format={"type": "json_object"},
            max_tokens=3000
        )
        
        content = response.choices[0].message.content
        return json.loads(content)
    
    def judge_consistency(
        self,
        lincoln_claims: list[str],
        other_claims: list[str],
        event_id: str,
        event_name: str,
        lincoln_source: str,
        other_source: str,
        other_author: str,
        strategy: str = "chain_of_thought",
        temperature: float = 0.3
    ) -> JudgmentResult:
        """
        Compare Lincoln's account with another author's account.
        
        Args:
            lincoln_claims: Claims from Lincoln's first-person account
            other_claims: Claims from another author's account
            event_id: Event identifier
            event_name: Human-readable event name
            lincoln_source: Source document for Lincoln's claims
            other_source: Source document for other claims
            other_author: Name of the other author
            strategy: Prompt strategy to use
            temperature: LLM temperature setting
            
        Returns:
            JudgmentResult with consistency analysis
        """
        prompt = self.build_prompt(lincoln_claims, other_claims, event_name, other_author, strategy)
        
        result = self.call_llm(prompt, temperature)
        
        # Parse contradictions
        contradictions = []
        for c in result.get("contradictions", []):
            contradictions.append(Contradiction(
                type=c.get("type", "Unknown"),
                lincoln_claim=c.get("lincoln_claim", ""),
                other_claim=c.get("other_claim", ""),
                explanation=c.get("explanation", ""),
                severity=c.get("severity", "Medium")
            ))
        
        return JudgmentResult(
            event_id=event_id,
            event_name=event_name,
            lincoln_source=lincoln_source,
            other_source=other_source,
            other_author=other_author,
            consistency_score=result.get("consistency_score", 50),
            contradictions=contradictions,
            alignments=result.get("alignments", []),
            reasoning=result.get("reasoning", result.get("step_by_step_reasoning", "")),
            prompt_strategy=strategy,
            temperature=temperature
        )
    
    def compare_all_pairs(
        self,
        lincoln_extractions: list[dict],
        other_extractions: list[dict],
        strategy: str = "chain_of_thought"
    ) -> list[JudgmentResult]:
        """
        Compare all (Lincoln, Other) pairs for the same events.
        
        Args:
            lincoln_extractions: Extractions from Lincoln's documents
            other_extractions: Extractions from other authors
            strategy: Prompt strategy to use
            
        Returns:
            List of JudgmentResult objects
        """
        results = []
        
        # Group extractions by event
        lincoln_by_event = {}
        for ext in lincoln_extractions:
            event_id = ext.get("event_id")
            if event_id not in lincoln_by_event:
                lincoln_by_event[event_id] = []
            lincoln_by_event[event_id].append(ext)
        
        # For each other author's extraction, compare with Lincoln's
        for other_ext in other_extractions:
            event_id = other_ext.get("event_id")
            event_name = other_ext.get("event_name", event_id)
            
            if event_id not in lincoln_by_event:
                print(f"  No Lincoln source for event: {event_name}")
                continue
            
            # Merge all Lincoln claims for this event
            lincoln_claims = []
            lincoln_sources = []
            for lincoln_ext in lincoln_by_event[event_id]:
                lincoln_claims.extend(lincoln_ext.get("claims", []))
                lincoln_sources.append(lincoln_ext.get("source_title", "Unknown"))
            
            other_claims = other_ext.get("claims", [])
            other_author = other_ext.get("author", "Unknown")
            other_source = other_ext.get("source_title", "Unknown")
            
            if not lincoln_claims or not other_claims:
                print(f"  Skipping {event_name}: insufficient claims")
                continue
            
            print(f"  Comparing: {event_name}")
            print(f"    Lincoln claims: {len(lincoln_claims)}")
            print(f"    {other_author} claims: {len(other_claims)}")
            
            try:
                result = self.judge_consistency(
                    lincoln_claims=lincoln_claims,
                    other_claims=other_claims,
                    event_id=event_id,
                    event_name=event_name,
                    lincoln_source=", ".join(lincoln_sources),
                    other_source=other_source,
                    other_author=other_author,
                    strategy=strategy
                )
                results.append(result)
                print(f"    Score: {result.consistency_score}")
                print(f"    Contradictions: {len(result.contradictions)}")
                
            except Exception as e:
                print(f"    Error: {e}")
            
            time.sleep(1)  # Rate limiting
        
        return results
    
    def save_results(self, results: list[JudgmentResult], filename: str = "judgment_results.json"):
        """Save judgment results to JSON."""
        output_path = self.output_dir / filename
        
        # Convert to serializable format
        data = []
        for r in results:
            item = asdict(r)
            # Convert Contradiction objects
            item["contradictions"] = [asdict(c) for c in r.contradictions]
            data.append(item)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(results)} judgments to {output_path}")


def load_extractions(filepath: str) -> list[dict]:
    """Load extraction results from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    """Main entry point for LLM Judge."""
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set!")
        return
    
    print("="*60)
    print("LLM Judge - Historiographical Consistency Analysis")
    print("="*60)
    
    judge = LLMJudge(model="gpt-4o")
    
    # Load extractions
    loc_path = Path("data/extracted/loc_events.json")
    gutenberg_path = Path("data/extracted/gutenberg_events.json")
    
    if not loc_path.exists() or not gutenberg_path.exists():
        print("Error: Run event extraction first!")
        print(f"  Missing: {loc_path if not loc_path.exists() else gutenberg_path}")
        return
    
    lincoln_extractions = load_extractions(str(loc_path))
    other_extractions = load_extractions(str(gutenberg_path))
    
    print(f"\nLoaded {len(lincoln_extractions)} Lincoln extractions")
    print(f"Loaded {len(other_extractions)} other author extractions")
    
    # Run comparison with Chain-of-Thought strategy
    print("\n" + "-"*40)
    print("Running comparisons (Chain-of-Thought)...")
    print("-"*40)
    
    results = judge.compare_all_pairs(
        lincoln_extractions,
        other_extractions,
        strategy="chain_of_thought"
    )
    
    # Save results
    judge.save_results(results, "judgment_results.json")
    
    # Print summary
    print("\n" + "="*60)
    print("JUDGMENT SUMMARY")
    print("="*60)
    
    if results:
        scores = [r.consistency_score for r in results]
        print(f"\nTotal comparisons: {len(results)}")
        print(f"Average consistency score: {sum(scores)/len(scores):.1f}")
        print(f"Score range: {min(scores)} - {max(scores)}")
        
        # By event
        by_event = {}
        for r in results:
            if r.event_name not in by_event:
                by_event[r.event_name] = []
            by_event[r.event_name].append(r.consistency_score)
        
        print("\nBy Event:")
        for event, scores in by_event.items():
            print(f"  {event}: avg={sum(scores)/len(scores):.1f}, n={len(scores)}")


if __name__ == "__main__":
    main()

