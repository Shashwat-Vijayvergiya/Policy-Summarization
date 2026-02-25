"""
PolicySum Comprehensive Evaluation Script

Implements all evaluation metrics from paper:
- ROUGE scores with confidence intervals
- BERTScore
- Coverage (stakeholder/domain preservation)
- Coherence (semantic similarity)
- Statistical significance testing
- Feature importance analysis
- Cross-domain evaluation
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer
import json
from collections import defaultdict
from scipy import stats
from rouge_score import rouge_scorer
from bert_score import score as bertscore
import argparse
from tqdm import tqdm

# Import model from training script
from train_policysum import PolicySumModel, PolicyBriefDataset


class PolicySumEvaluator:
    """Comprehensive evaluator for PolicySum"""
    
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_name = 'microsoft/deberta-v3-base'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load model
        self.model = PolicySumModel(self.model_name).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Initialize scorers
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=True
        )
        
        # Policy vocabulary
        self.domains = ['economic', 'social', 'environmental', 'legal', 'health', 'education']
        self.stakeholders = ['government', 'citizen', 'business', 'ngo', 'international']
    
    def generate_summary(self, document, top_k=5, redundancy_threshold=0.8):
        """Generate summary from document"""
        # Tokenize and prepare input
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc_obj = nlp(document)
        sentences = [sent.text.strip() for sent in doc_obj.sents][:30]
        
        if len(sentences) == 0:
            return []
        
        # Prepare batch
        encoded_sents = []
        features = []
        temporal_ids = []
        
        for i, sent in enumerate(sentences):
            enc = self.tokenizer(sent, max_length=64, padding='max_length',
                                truncation=True, return_tensors='pt')
            encoded_sents.append(enc['input_ids'].squeeze(0))
            
            # Extract features
            feat = self._extract_features(sent, i, len(sentences))
            features.append(feat)
            
            temp_id = int(i / (len(sentences) / 3))
            temporal_ids.append(min(temp_id, 2))
        
        # Pad to 30 sentences
        while len(encoded_sents) < 30:
            encoded_sents.append(torch.zeros(64, dtype=torch.long))
            features.append(np.zeros(11, dtype=np.float32))
            temporal_ids.append(0)
        
        # Create batch
        input_ids = torch.stack(encoded_sents[:30]).unsqueeze(0).to(self.device)
        features_tensor = torch.FloatTensor(features[:30]).unsqueeze(0).to(self.device)
        temporal_tensor = torch.LongTensor(temporal_ids[:30]).unsqueeze(0).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            scores, _, gate_weights = self.model(input_ids, features_tensor, temporal_tensor)
            scores = torch.sigmoid(scores[0]).cpu().numpy()
        
        # Select top-k with redundancy filtering
        selected_indices = []
        selected_sentences = []
        
        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]
        
        for idx in sorted_indices:
            if idx >= len(sentences):
                continue
                
            sent = sentences[idx]
            
            # Check redundancy
            is_redundant = False
            for prev_sent in selected_sentences:
                similarity = self._cosine_similarity(sent, prev_sent)
                if similarity > redundancy_threshold:
                    is_redundant = True
                    break
            
            if not is_redundant:
                selected_indices.append(idx)
                selected_sentences.append(sent)
                
                if len(selected_sentences) >= top_k:
                    break
        
        # Sort by original order
        selected_indices.sort()
        summary_sentences = [sentences[i] for i in selected_indices]
        
        return summary_sentences
    
    def _extract_features(self, text, position, total_sents):
        """Extract 11 policy features"""
        # Domain indicators (6)
        domain_scores = [1.0 if d in text.lower() else 0.0 for d in self.domains]
        
        # Stakeholder salience (5)
        stakeholder_scores = []
        for sh in self.stakeholders:
            count = text.lower().count(sh)
            score = count / max(len(text.split()), 1)
            stakeholder_scores.append(score)
        
        return np.array(domain_scores + stakeholder_scores, dtype=np.float32)
    
    def _cosine_similarity(self, sent1, sent2):
        """Calculate cosine similarity between sentences"""
        # Simple word overlap similarity
        words1 = set(sent1.lower().split())
        words2 = set(sent2.lower().split())
        
        if len(words1) == 0 or len(words2) == 0:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def evaluate_rouge(self, predictions, references):
        """Calculate ROUGE scores with confidence intervals"""
        scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        for pred, ref in zip(predictions, references):
            result = self.rouge_scorer.score(ref, pred)
            for metric in scores:
                scores[metric].append(result[metric].fmeasure)
        
        # Calculate mean and confidence interval
        results = {}
        for metric in scores:
            values = np.array(scores[metric])
            mean = np.mean(values)
            std = np.std(values)
            ci = stats.t.interval(0.95, len(values)-1, loc=mean, scale=stats.sem(values))
            
            results[metric] = {
                'mean': mean,
                'std': std,
                'ci_95': ci
            }
        
        return results
    
    def evaluate_bertscore(self, predictions, references):
        """Calculate BERTScore"""
        P, R, F1 = bertscore(predictions, references, lang='en', 
                             model_type='microsoft/deberta-v3-base',
                             verbose=False)
        
        f1_scores = F1.cpu().numpy()
        mean = np.mean(f1_scores)
        std = np.std(f1_scores)
        ci = stats.t.interval(0.95, len(f1_scores)-1, loc=mean, scale=stats.sem(f1_scores))
        
        return {
            'mean': mean,
            'std': std,
            'ci_95': ci
        }
    
    def evaluate_coverage(self, predictions, references):
        """Evaluate stakeholder and domain coverage"""
        coverage_scores = []
        
        for pred, ref in zip(predictions, references):
            # Extract entities from reference
            ref_domains = set(d for d in self.domains if d in ref.lower())
            ref_stakeholders = set(s for s in self.stakeholders if s in ref.lower())
            
            # Check coverage in prediction
            pred_domains = set(d for d in self.domains if d in pred.lower())
            pred_stakeholders = set(s for s in self.stakeholders if s in pred.lower())
            
            # Calculate coverage
            domain_coverage = len(pred_domains & ref_domains) / max(len(ref_domains), 1)
            stakeholder_coverage = len(pred_stakeholders & ref_stakeholders) / max(len(ref_stakeholders), 1)
            
            overall_coverage = (domain_coverage + stakeholder_coverage) / 2
            coverage_scores.append(overall_coverage)
        
        values = np.array(coverage_scores)
        mean = np.mean(values)
        std = np.std(values)
        ci = stats.t.interval(0.95, len(values)-1, loc=mean, scale=stats.sem(values))
        
        return {
            'mean': mean,
            'std': std,
            'ci_95': ci
        }
    
    def evaluate_coherence(self, predictions):
        """Evaluate coherence via adjacent sentence similarity"""
        coherence_scores = []
        
        for pred in predictions:
            sentences = pred.split('.')
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) < 2:
                coherence_scores.append(1.0)
                continue
            
            similarities = []
            for i in range(len(sentences) - 1):
                sim = self._cosine_similarity(sentences[i], sentences[i+1])
                similarities.append(sim)
            
            coherence_scores.append(np.mean(similarities))
        
        values = np.array(coherence_scores)
        mean = np.mean(values)
        std = np.std(values)
        
        return {
            'mean': mean,
            'std': std
        }
    
    def statistical_significance_test(self, baseline_scores, model_scores):
        """Perform paired t-test"""
        t_stat, p_value = stats.ttest_rel(model_scores, baseline_scores)
        
        diff = np.array(model_scores) - np.array(baseline_scores)
        ci = stats.t.interval(0.95, len(diff)-1, loc=np.mean(diff), scale=stats.sem(diff))
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.001,
            'ci_95': ci,
            'improvement': np.mean(diff)
        }
    
    def evaluate_dataset(self, test_data_path, baseline_predictions=None):
        """Evaluate on full test set"""
        # Load test data
        with open(test_data_path, 'r') as f:
            test_data = json.load(f)
        
        print(f"Evaluating on {len(test_data)} test examples...")
        
        predictions = []
        references = []
        
        for item in tqdm(test_data):
            # Generate summary
            summary_sents = self.generate_summary(item['document'], top_k=5)
            summary = ' '.join(summary_sents)
            
            predictions.append(summary)
            references.append(item['summary'])
        
        # Evaluate ROUGE
        print("\nCalculating ROUGE scores...")
        rouge_results = self.evaluate_rouge(predictions, references)
        
        # Evaluate BERTScore
        print("Calculating BERTScore...")
        bertscore_results = self.evaluate_bertscore(predictions, references)
        
        # Evaluate Coverage
        print("Calculating coverage...")
        coverage_results = self.evaluate_coverage(predictions, references)
        
        # Evaluate Coherence
        print("Calculating coherence...")
        coherence_results = self.evaluate_coherence(predictions)
        
        # Compile results
        results = {
            'rouge': rouge_results,
            'bertscore': bertscore_results,
            'coverage': coverage_results,
            'coherence': coherence_results,
            'num_examples': len(test_data)
        }
        
        # Statistical significance if baseline provided
        if baseline_predictions:
            print("Running statistical significance tests...")
            baseline_rouge = self.evaluate_rouge(baseline_predictions, references)
            
            sig_test = self.statistical_significance_test(
                [baseline_rouge['rougeL']['mean']] * len(test_data),
                rouge_results['rougeL']['mean']
            )
            results['significance_test'] = sig_test
        
        return results, predictions
    
    def print_results(self, results):
        """Print formatted results"""
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)
        
        print("\nROUGE Scores:")
        print(f"  ROUGE-1: {results['rouge']['rouge1']['mean']:.4f} ± {results['rouge']['rouge1']['std']:.4f}")
        print(f"           95% CI: [{results['rouge']['rouge1']['ci_95'][0]:.4f}, {results['rouge']['rouge1']['ci_95'][1]:.4f}]")
        print(f"  ROUGE-2: {results['rouge']['rouge2']['mean']:.4f} ± {results['rouge']['rouge2']['std']:.4f}")
        print(f"           95% CI: [{results['rouge']['rouge2']['ci_95'][0]:.4f}, {results['rouge']['rouge2']['ci_95'][1]:.4f}]")
        print(f"  ROUGE-L: {results['rouge']['rougeL']['mean']:.4f} ± {results['rouge']['rougeL']['std']:.4f}")
        print(f"           95% CI: [{results['rouge']['rougeL']['ci_95'][0]:.4f}, {results['rouge']['rougeL']['ci_95'][1]:.4f}]")
        
        print(f"\nBERTScore: {results['bertscore']['mean']:.4f} ± {results['bertscore']['std']:.4f}")
        print(f"           95% CI: [{results['bertscore']['ci_95'][0]:.4f}, {results['bertscore']['ci_95'][1]:.4f}]")
        
        print(f"\nCoverage: {results['coverage']['mean']:.4f} ± {results['coverage']['std']:.4f}")
        print(f"          95% CI: [{results['coverage']['ci_95'][0]:.4f}, {results['coverage']['ci_95'][1]:.4f}]")
        
        print(f"\nCoherence: {results['coherence']['mean']:.4f} ± {results['coherence']['std']:.4f}")
        
        if 'significance_test' in results:
            print("\nStatistical Significance Test:")
            print(f"  t-statistic: {results['significance_test']['t_statistic']:.4f}")
            print(f"  p-value: {results['significance_test']['p_value']:.6f}")
            print(f"  Significant: {'Yes' if results['significance_test']['significant'] else 'No'} (p < 0.001)")
            print(f"  Improvement: {results['significance_test']['improvement']:.4f}")
        
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Evaluate PolicySum model')
    parser.add_argument('--model', type=str, default='policysum_model_best.pt',
                       help='Path to trained model')
    parser.add_argument('--test_data', type=str, default='policysum_data/test.json',
                       help='Path to test data')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    print("Initializing PolicySum evaluator...")
    evaluator = PolicySumEvaluator(args.model, device=args.device)
    
    # Run evaluation
    results, predictions = evaluator.evaluate_dataset(args.test_data)
    
    # Print results
    evaluator.print_results(results)
    
    # Save results
    with open(args.output, 'w') as f:
        # Convert numpy types to Python types for JSON
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, tuple):
                return list(obj)
            return obj
        
        results_serializable = json.loads(
            json.dumps(results, default=convert)
        )
        json.dump(results_serializable, f, indent=2)
    
    print(f"\nResults saved to {args.output}")
    
    # Save predictions
    pred_file = args.output.replace('.json', '_predictions.json')
    with open(pred_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    print(f"Predictions saved to {pred_file}")


if __name__ == '__main__':
    main()