"""
Statistical Validation for LLM Judge.

Implements the three required validation experiments:
1. Prompt Robustness (Ablation Study)
2. Self-Consistency (Reliability)
3. Inter-Rater Agreement (Cohen's Kappa)
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Import our judge
import sys
sys.path.append(str(Path(__file__).parent.parent))
from judge.llm_judge import LLMJudge, load_extractions


@dataclass
class AblationResult:
    """Results from prompt ablation study."""
    strategy: str
    scores: list[float]
    mean: float
    std: float
    min: float
    max: float


@dataclass
class SelfConsistencyResult:
    """Results from self-consistency test."""
    event_id: str
    event_name: str
    comparison: str  # e.g., "Lincoln vs Goodwin"
    scores: list[int]
    mean: float
    std: float
    range: int
    is_reliable: bool  # std < threshold


@dataclass
class KappaResult:
    """Results from Cohen's Kappa calculation."""
    kappa: float
    interpretation: str
    llm_labels: list[str]
    human_labels: list[str]
    agreement_rate: float


class StatisticalValidator:
    """
    Validates LLM Judge reliability through statistical tests.
    """
    
    KAPPA_INTERPRETATIONS = [
        (0.81, 1.00, "Almost perfect agreement"),
        (0.61, 0.80, "Substantial agreement"),
        (0.41, 0.60, "Moderate agreement"),
        (0.21, 0.40, "Fair agreement"),
        (0.01, 0.20, "Slight agreement"),
        (-1.0, 0.00, "No agreement (or worse than chance)"),
    ]
    
    def __init__(self, model: str = "gpt-4o"):
        self.judge = LLMJudge(model=model)
        self.output_dir = Path("data/validation")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # 1. PROMPT ABLATION STUDY
    # =========================================================================
    
    def run_ablation_study(
        self,
        lincoln_extractions: list[dict],
        other_extractions: list[dict],
        strategies: list[str] = ["zero_shot", "chain_of_thought", "few_shot"]
    ) -> dict[str, AblationResult]:
        """
        Compare different prompt strategies on the same data.
        
        Tests which prompting approach yields more stable/better results.
        
        Args:
            lincoln_extractions: Lincoln's first-person extractions
            other_extractions: Other authors' extractions
            strategies: Prompt strategies to compare
            
        Returns:
            Dict mapping strategy name to AblationResult
        """
        print("\n" + "="*60)
        print("ABLATION STUDY: Comparing Prompt Strategies")
        print("="*60)
        
        results = {}
        
        for strategy in strategies:
            print(f"\n--- Strategy: {strategy} ---")
            
            judgments = self.judge.compare_all_pairs(
                lincoln_extractions,
                other_extractions,
                strategy=strategy
            )
            
            scores = [j.consistency_score for j in judgments]
            
            if scores:
                results[strategy] = AblationResult(
                    strategy=strategy,
                    scores=scores,
                    mean=float(np.mean(scores)),
                    std=float(np.std(scores)),
                    min=float(np.min(scores)),
                    max=float(np.max(scores))
                )
                
                print(f"  Mean score: {results[strategy].mean:.1f}")
                print(f"  Std dev: {results[strategy].std:.1f}")
                print(f"  Range: {results[strategy].min:.0f} - {results[strategy].max:.0f}")
        
        return results
    
    def visualize_ablation(self, results: dict[str, AblationResult], save_path: str = None):
        """Create visualization of ablation study results."""
        if not results:
            print("No ablation results to visualize")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Box plot of scores by strategy
        strategies = list(results.keys())
        scores_data = [results[s].scores for s in strategies]
        
        ax1 = axes[0]
        bp = ax1.boxplot(scores_data, labels=strategies, patch_artist=True)
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        for patch, color in zip(bp['boxes'], colors[:len(strategies)]):
            patch.set_facecolor(color)
        ax1.set_ylabel('Consistency Score')
        ax1.set_title('Score Distribution by Prompt Strategy')
        ax1.set_ylim(0, 100)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Mean ± Std bar chart
        ax2 = axes[1]
        means = [results[s].mean for s in strategies]
        stds = [results[s].std for s in strategies]
        
        bars = ax2.bar(strategies, means, yerr=stds, capsize=5, 
                       color=colors[:len(strategies)], edgecolor='black')
        ax2.set_ylabel('Mean Consistency Score (± Std)')
        ax2.set_title('Prompt Strategy Comparison')
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, mean, std in zip(bars, means, stds):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 2,
                    f'{mean:.1f}±{std:.1f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved ablation visualization to {save_path}")
        
        plt.show()
    
    # =========================================================================
    # 2. SELF-CONSISTENCY TEST
    # =========================================================================
    
    def run_self_consistency(
        self,
        lincoln_claims: list[str],
        other_claims: list[str],
        event_id: str,
        event_name: str,
        lincoln_source: str,
        other_source: str,
        other_author: str,
        n_runs: int = 5,
        temperature: float = 0.7,
        reliability_threshold: float = 10.0
    ) -> SelfConsistencyResult:
        """
        Run the same comparison multiple times to check reliability.
        
        A reliable judge should give similar scores even with temperature > 0.
        
        Args:
            lincoln_claims: Claims from Lincoln
            other_claims: Claims from other author
            event_id: Event identifier
            event_name: Event name
            lincoln_source: Lincoln source doc
            other_source: Other source doc
            other_author: Other author name
            n_runs: Number of repetitions
            temperature: LLM temperature (> 0 for variability)
            reliability_threshold: Max acceptable std dev
            
        Returns:
            SelfConsistencyResult with scores and statistics
        """
        scores = []
        
        for i in range(n_runs):
            result = self.judge.judge_consistency(
                lincoln_claims=lincoln_claims,
                other_claims=other_claims,
                event_id=event_id,
                event_name=event_name,
                lincoln_source=lincoln_source,
                other_source=other_source,
                other_author=other_author,
                strategy="chain_of_thought",
                temperature=temperature
            )
            scores.append(result.consistency_score)
        
        std = float(np.std(scores))
        
        return SelfConsistencyResult(
            event_id=event_id,
            event_name=event_name,
            comparison=f"Lincoln vs {other_author}",
            scores=scores,
            mean=float(np.mean(scores)),
            std=std,
            range=max(scores) - min(scores),
            is_reliable=std < reliability_threshold
        )
    
    def run_self_consistency_batch(
        self,
        lincoln_extractions: list[dict],
        other_extractions: list[dict],
        n_samples: int = 5,
        n_runs: int = 5,
        temperature: float = 0.7
    ) -> list[SelfConsistencyResult]:
        """
        Run self-consistency test on multiple comparison pairs.
        
        Args:
            lincoln_extractions: Lincoln's extractions
            other_extractions: Other extractions
            n_samples: Number of pairs to test
            n_runs: Runs per pair
            temperature: LLM temperature
            
        Returns:
            List of SelfConsistencyResult
        """
        print("\n" + "="*60)
        print(f"SELF-CONSISTENCY TEST: {n_runs} runs per comparison")
        print(f"Temperature: {temperature}")
        print("="*60)
        
        results = []
        
        # Group Lincoln extractions by event
        lincoln_by_event = {}
        for ext in lincoln_extractions:
            event_id = ext.get("event_id")
            if event_id not in lincoln_by_event:
                lincoln_by_event[event_id] = []
            lincoln_by_event[event_id].append(ext)
        
        # Sample comparisons
        comparisons = []
        for other_ext in other_extractions:
            event_id = other_ext.get("event_id")
            if event_id in lincoln_by_event:
                lincoln_claims = []
                for le in lincoln_by_event[event_id]:
                    lincoln_claims.extend(le.get("claims", []))
                
                if lincoln_claims and other_ext.get("claims"):
                    comparisons.append({
                        "lincoln_claims": lincoln_claims,
                        "other_claims": other_ext.get("claims", []),
                        "event_id": event_id,
                        "event_name": other_ext.get("event_name", event_id),
                        "lincoln_source": "LoC Documents",
                        "other_source": other_ext.get("source_title", "Unknown"),
                        "other_author": other_ext.get("author", "Unknown")
                    })
        
        # Limit to n_samples
        comparisons = comparisons[:n_samples]
        
        print(f"\nTesting {len(comparisons)} comparison pairs...")
        
        for comp in tqdm(comparisons, desc="Self-consistency tests"):
            print(f"\n  {comp['event_name']}: Lincoln vs {comp['other_author']}")
            
            result = self.run_self_consistency(
                lincoln_claims=comp["lincoln_claims"],
                other_claims=comp["other_claims"],
                event_id=comp["event_id"],
                event_name=comp["event_name"],
                lincoln_source=comp["lincoln_source"],
                other_source=comp["other_source"],
                other_author=comp["other_author"],
                n_runs=n_runs,
                temperature=temperature
            )
            
            results.append(result)
            
            print(f"    Scores: {result.scores}")
            print(f"    Mean: {result.mean:.1f}, Std: {result.std:.1f}")
            print(f"    Reliable: {'Yes' if result.is_reliable else 'No'}")
        
        return results
    
    def visualize_self_consistency(self, results: list[SelfConsistencyResult], save_path: str = None):
        """Create visualization of self-consistency results."""
        if not results:
            print("No self-consistency results to visualize")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Score distributions for each comparison
        ax1 = axes[0]
        labels = [f"{r.event_name[:20]}...\nvs {r.comparison.split('vs')[1].strip()[:15]}" 
                  for r in results]
        scores_data = [r.scores for r in results]
        
        bp = ax1.boxplot(scores_data, labels=range(1, len(results)+1), patch_artist=True)
        colors = plt.cm.Set3(np.linspace(0, 1, len(results)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax1.set_xlabel('Comparison #')
        ax1.set_ylabel('Consistency Score')
        ax1.set_title(f'Score Variability Across {len(results[0].scores)} Runs')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Standard deviations
        ax2 = axes[1]
        stds = [r.std for r in results]
        colors = ['green' if r.is_reliable else 'red' for r in results]
        
        bars = ax2.bar(range(1, len(results)+1), stds, color=colors, edgecolor='black')
        ax2.axhline(y=10, color='orange', linestyle='--', label='Reliability Threshold')
        ax2.set_xlabel('Comparison #')
        ax2.set_ylabel('Standard Deviation')
        ax2.set_title('Score Variability (Green=Reliable, Red=Unreliable)')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved self-consistency visualization to {save_path}")
        
        plt.show()
    
    # =========================================================================
    # 3. COHEN'S KAPPA (Inter-Rater Agreement)
    # =========================================================================
    
    def calculate_cohens_kappa(
        self,
        llm_scores: list[int],
        human_labels: list[str],
        threshold: int = 50
    ) -> KappaResult:
        """
        Calculate Cohen's Kappa between LLM judge and human labels.
        
        Args:
            llm_scores: Consistency scores from LLM (0-100)
            human_labels: Human labels ("consistent" or "contradictory")
            threshold: Score threshold for binary classification
            
        Returns:
            KappaResult with kappa score and interpretation
        """
        # Convert LLM scores to binary labels
        llm_labels = ["consistent" if s >= threshold else "contradictory" for s in llm_scores]
        
        # Ensure human labels are lowercase
        human_labels = [l.lower().strip() for l in human_labels]
        
        # Calculate kappa
        kappa = cohen_kappa_score(human_labels, llm_labels)
        
        # Get interpretation
        interpretation = "Unknown"
        for low, high, interp in self.KAPPA_INTERPRETATIONS:
            if low <= kappa <= high:
                interpretation = interp
                break
        
        # Calculate simple agreement rate
        agreements = sum(1 for h, l in zip(human_labels, llm_labels) if h == l)
        agreement_rate = agreements / len(human_labels) if human_labels else 0
        
        return KappaResult(
            kappa=float(kappa),
            interpretation=interpretation,
            llm_labels=llm_labels,
            human_labels=human_labels,
            agreement_rate=float(agreement_rate)
        )
    
    def create_manual_labeling_template(
        self,
        lincoln_extractions: list[dict],
        other_extractions: list[dict],
        n_samples: int = 10,
        output_path: str = "data/validation/manual_labels_template.json"
    ):
        """
        Create a template for manual labeling.
        
        You need to fill this in manually before running Kappa calculation.
        
        Args:
            lincoln_extractions: Lincoln's extractions
            other_extractions: Other extractions
            n_samples: Number of pairs to label
            output_path: Where to save template
        """
        print("\n" + "="*60)
        print("Creating Manual Labeling Template")
        print("="*60)
        
        # Group Lincoln extractions by event
        lincoln_by_event = {}
        for ext in lincoln_extractions:
            event_id = ext.get("event_id")
            if event_id not in lincoln_by_event:
                lincoln_by_event[event_id] = []
            lincoln_by_event[event_id].append(ext)
        
        # Create comparison pairs
        pairs = []
        for other_ext in other_extractions:
            event_id = other_ext.get("event_id")
            if event_id in lincoln_by_event:
                lincoln_claims = []
                for le in lincoln_by_event[event_id]:
                    lincoln_claims.extend(le.get("claims", []))
                
                if lincoln_claims and other_ext.get("claims"):
                    pairs.append({
                        "id": len(pairs) + 1,
                        "event": other_ext.get("event_name", event_id),
                        "other_author": other_ext.get("author", "Unknown"),
                        "lincoln_claims": lincoln_claims[:5],  # First 5 for readability
                        "other_claims": other_ext.get("claims", [])[:5],
                        "human_label": "",  # TO BE FILLED
                        "notes": ""  # Optional notes
                    })
        
        # Limit to n_samples
        pairs = pairs[:n_samples]
        
        template = {
            "instructions": """
MANUAL LABELING INSTRUCTIONS:
1. Read the Lincoln claims and Other Author claims for each pair
2. Determine if they are "consistent" or "contradictory"
3. Use "consistent" if the accounts generally agree or complement each other
4. Use "contradictory" if there are significant factual conflicts
5. Fill in the "human_label" field with your judgment
6. Optionally add notes explaining your reasoning
            """,
            "pairs": pairs
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(template, f, indent=2, ensure_ascii=False)
        
        print(f"\nCreated template with {len(pairs)} pairs at: {output_path}")
        print("Please fill in the 'human_label' field for each pair (consistent/contradictory)")
        print("Then run the Kappa calculation with your filled-in labels.")
    
    def run_kappa_from_file(
        self,
        labels_file: str,
        llm_results_file: str,
        threshold: int = 50
    ) -> KappaResult:
        """
        Calculate Kappa using saved labels and LLM results.
        
        Args:
            labels_file: Path to filled manual labels JSON
            llm_results_file: Path to LLM judgment results JSON
            threshold: Score threshold for binary classification
            
        Returns:
            KappaResult
        """
        # Load files
        with open(labels_file, "r", encoding="utf-8") as f:
            labels_data = json.load(f)
        
        with open(llm_results_file, "r", encoding="utf-8") as f:
            llm_results = json.load(f)
        
        # Extract human labels and match with LLM scores
        human_labels = []
        llm_scores = []
        
        for pair in labels_data.get("pairs", []):
            if pair.get("human_label"):
                human_labels.append(pair["human_label"])
                
                # Find matching LLM result
                event = pair["event"]
                author = pair["other_author"]
                
                matching_score = None
                for result in llm_results:
                    if (result.get("event_name") == event and 
                        result.get("other_author") == author):
                        matching_score = result.get("consistency_score", 50)
                        break
                
                if matching_score is not None:
                    llm_scores.append(matching_score)
                else:
                    # Use default if no match found
                    llm_scores.append(50)
        
        if len(human_labels) != len(llm_scores):
            print(f"Warning: Mismatched counts - {len(human_labels)} human labels, {len(llm_scores)} LLM scores")
        
        result = self.calculate_cohens_kappa(llm_scores, human_labels, threshold)
        
        print("\n" + "="*60)
        print("COHEN'S KAPPA RESULT")
        print("="*60)
        print(f"Kappa Score: {result.kappa:.3f}")
        print(f"Interpretation: {result.interpretation}")
        print(f"Simple Agreement Rate: {result.agreement_rate:.1%}")
        print(f"Samples: {len(human_labels)}")
        
        return result
    
    # =========================================================================
    # SAVE ALL RESULTS
    # =========================================================================
    
    def save_validation_report(
        self,
        ablation_results: dict[str, AblationResult] = None,
        consistency_results: list[SelfConsistencyResult] = None,
        kappa_result: KappaResult = None,
        output_path: str = "data/validation/validation_report.json"
    ):
        """Save all validation results to a JSON report."""
        report = {
            "ablation_study": {},
            "self_consistency": [],
            "cohens_kappa": {}
        }
        
        if ablation_results:
            report["ablation_study"] = {
                strategy: asdict(result) 
                for strategy, result in ablation_results.items()
            }
        
        if consistency_results:
            report["self_consistency"] = [asdict(r) for r in consistency_results]
        
        if kappa_result:
            report["cohens_kappa"] = asdict(kappa_result)
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\nSaved validation report to {output_path}")


def main():
    """Run all validation experiments."""
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set!")
        return
    
    validator = StatisticalValidator(model="gpt-4o")
    
    # Load extractions
    loc_path = Path("data/extracted/loc_events.json")
    gutenberg_path = Path("data/extracted/gutenberg_events.json")
    
    if not loc_path.exists() or not gutenberg_path.exists():
        print("Error: Run event extraction first!")
        return
    
    lincoln_extractions = load_extractions(str(loc_path))
    other_extractions = load_extractions(str(gutenberg_path))
    
    print(f"Loaded {len(lincoln_extractions)} Lincoln extractions")
    print(f"Loaded {len(other_extractions)} other extractions")
    
    # 1. Ablation Study
    print("\n" + "#"*60)
    print("# EXPERIMENT 1: PROMPT ABLATION STUDY")
    print("#"*60)
    
    ablation_results = validator.run_ablation_study(
        lincoln_extractions,
        other_extractions,
        strategies=["zero_shot", "chain_of_thought", "few_shot"]
    )
    
    validator.visualize_ablation(
        ablation_results, 
        save_path=str(validator.output_dir / "ablation_study.png")
    )
    
    # 2. Self-Consistency
    print("\n" + "#"*60)
    print("# EXPERIMENT 2: SELF-CONSISTENCY TEST")
    print("#"*60)
    
    consistency_results = validator.run_self_consistency_batch(
        lincoln_extractions,
        other_extractions,
        n_samples=5,
        n_runs=5,
        temperature=0.7
    )
    
    validator.visualize_self_consistency(
        consistency_results,
        save_path=str(validator.output_dir / "self_consistency.png")
    )
    
    # 3. Create Kappa Template
    print("\n" + "#"*60)
    print("# EXPERIMENT 3: COHEN'S KAPPA PREPARATION")
    print("#"*60)
    
    validator.create_manual_labeling_template(
        lincoln_extractions,
        other_extractions,
        n_samples=10
    )
    
    # Save report
    validator.save_validation_report(
        ablation_results=ablation_results,
        consistency_results=consistency_results,
        output_path="data/validation/validation_report.json"
    )
    
    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Review ablation_study.png and self_consistency.png")
    print("2. Fill in manual_labels_template.json with your labels")
    print("3. Run Kappa calculation with your filled labels")


if __name__ == "__main__":
    main()

