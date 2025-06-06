"""
Unmet need lookup system with fuzzy matching capabilities.
Handles CSV lookup and deterministic fuzzy matching for indication strings.
"""

import pandas as pd
from rapidfuzz import fuzz
from typing import Optional, Tuple, Dict
import logging
import numpy as np
from pathlib import Path


class UnmetNeedLookupError(Exception):
    """Custom exception for unmet need lookup errors."""
    pass


class UnmetNeedLookup:
    """Handles fuzzy matching and lookup of unmet need scores."""
    
    def __init__(self, lookup_csv_path: Optional[str] = None):
        self.lookup_df = None
        self.fuzzy_threshold = 97
        self.logger = logging.getLogger(__name__)
        
        # Score buckets based on statistical distribution
        self.score_buckets = {}
        
        if lookup_csv_path:
            self.load_lookup_table(lookup_csv_path)
    
    def load_lookup_table(self, csv_path: str) -> None:
        """Load the unmet need CSV lookup table."""
        try:
            csv_file = Path(csv_path)
            if not csv_file.exists():
                raise UnmetNeedLookupError(f"Lookup CSV not found: {csv_path}")
            
            self.lookup_df = pd.read_csv(csv_path)
            
            # Validate required columns
            required_columns = ['indication', 'unmet_need_score']
            missing_cols = [col for col in required_columns if col not in self.lookup_df.columns]
            if missing_cols:
                raise UnmetNeedLookupError(f"Missing required columns in CSV: {missing_cols}")
            
            # Clean and normalize indication names
            self.lookup_df['indication_normalized'] = (
                self.lookup_df['indication']
                .astype(str)
                .str.lower()
                .str.strip()
                .str.replace(r'[^\w\s]', '', regex=True)
            )
            
            # Calculate statistical buckets
            self._calculate_score_buckets()
            
            self.logger.info(f"Loaded {len(self.lookup_df)} indications from {csv_path}")
            
        except Exception as e:
            raise UnmetNeedLookupError(f"Failed to load lookup table: {str(e)}")
    
    def _calculate_score_buckets(self) -> None:
        """Calculate score buckets based on mean and standard deviation."""
        if self.lookup_df is None or 'unmet_need_score' not in self.lookup_df.columns:
            return
        
        scores = pd.to_numeric(self.lookup_df['unmet_need_score'], errors='coerce').dropna()
        
        if len(scores) == 0:
            self.logger.warning("No valid numeric scores found in lookup table")
            return
        
        mean_score = scores.mean()
        std_score = scores.std()
        
        # Define buckets: High (>mean+0.5*std), Medium (mean±0.5*std), Low (<mean-0.5*std)
        high_threshold = mean_score + 0.5 * std_score
        low_threshold = mean_score - 0.5 * std_score
        
        self.score_buckets = {
            'high_threshold': high_threshold,
            'low_threshold': low_threshold,
            'mean': mean_score,
            'std': std_score
        }
        
        self.logger.info(f"Score buckets calculated - High: >{high_threshold:.2f}, Low: <{low_threshold:.2f}")
    
    def lookup_indication(self, indication: str) -> Tuple[Optional[str], bool, Optional[float]]:
        """
        Look up unmet need score for indication.
        Returns (bucketed_score, match_found, z_score) tuple.
        """
        if self.lookup_df is None:
            self.logger.warning("Lookup table not loaded")
            return None, False, None
        
        # First try exact match (case-insensitive)
        normalized_indication = self._normalize_indication(indication)
        
        exact_match = self.lookup_df[
            self.lookup_df['indication_normalized'] == normalized_indication
        ]
        
        if not exact_match.empty:
            score = exact_match.iloc[0]['unmet_need_score']
            bucketed_score = self._bucket_score(score)
            z_score = self._calculate_z_score(score)
            self.logger.debug(f"Exact match found for '{indication}': {bucketed_score}")
            return bucketed_score, True, z_score
        
        # Try fuzzy matching
        fuzzy_result = self.fuzzy_match(indication)
        if fuzzy_result:
            matched_indication, match_score = fuzzy_result
            
            # Look up the matched indication's score
            matched_row = self.lookup_df[
                self.lookup_df['indication_normalized'] == self._normalize_indication(matched_indication)
            ]
            
            if not matched_row.empty:
                score = matched_row.iloc[0]['unmet_need_score']
                bucketed_score = self._bucket_score(score)
                z_score = self._calculate_z_score(score)
                self.logger.debug(f"Fuzzy match found for '{indication}' -> '{matched_indication}' (score: {match_score:.1f}): {bucketed_score}")
                return bucketed_score, True, z_score
        
        self.logger.debug(f"No match found for indication: '{indication}'")
        return None, False, None
    
    def fuzzy_match(self, indication: str) -> Optional[Tuple[str, float]]:
        """
        Perform fuzzy matching with >=97 token-sort ratio.
        Returns (matched_indication, score) or None.
        """
        if self.lookup_df is None:
            return None
        
        normalized_input = self._normalize_indication(indication)
        best_match = None
        best_score = 0
        
        for _, row in self.lookup_df.iterrows():
            target = row['indication_normalized']
            
            # Use token_sort_ratio for better matching of reordered words
            score = fuzz.token_sort_ratio(normalized_input, target)
            
            if score >= self.fuzzy_threshold and score > best_score:
                best_score = score
                best_match = row['indication']
        
        if best_match:
            return (best_match, best_score)
        
        return None
    
    def _normalize_indication(self, indication: str) -> str:
        """Normalize indication string for matching."""
        return (
            str(indication)
            .lower()
            .strip()
            .replace(r'[^\w\s]', '')
        )
    
    def _bucket_score(self, score: float) -> str:
        """Convert numeric score to bucketed category."""
        try:
            score_float = float(score)
            
            if not self.score_buckets:
                return f"Score: {score_float:.1f}"
            
            if score_float >= self.score_buckets['high_threshold']:
                return "High Unmet Need"
            elif score_float <= self.score_buckets['low_threshold']:
                return "Low Unmet Need"
            else:
                return "Medium Unmet Need"
                
        except (ValueError, TypeError):
            return f"Invalid Score: {score}"
    
    def _calculate_z_score(self, score: float) -> Optional[float]:
        """Calculate z-score relative to the mean and standard deviation."""
        try:
            score_float = float(score)
            
            if not self.score_buckets or 'mean' not in self.score_buckets or 'std' not in self.score_buckets:
                return None
            
            mean = self.score_buckets['mean']
            std = self.score_buckets['std']
            
            if std == 0:
                return None
            
            z_score = (score_float - mean) / std
            return round(z_score, 2)
            
        except (ValueError, TypeError):
            return None
    
    def get_stats(self) -> Dict:
        """Get statistics about the lookup table."""
        if self.lookup_df is None:
            return {"status": "not_loaded"}
        
        return {
            "total_indications": len(self.lookup_df),
            "score_distribution": self.score_buckets,
            "sample_indications": self.lookup_df['indication'].head(5).tolist()
        }
    
    def create_sample_lookup_table(self, output_path: str) -> None:
        """Create a sample lookup table for testing purposes."""
        sample_data = {
            'indication': [
                'Alzheimer\'s Disease',
                'Type 2 Diabetes',
                'Rheumatoid Arthritis',
                'Non-Small Cell Lung Cancer',
                'Multiple Sclerosis',
                'Parkinson\'s Disease',
                'Crohn\'s Disease',
                'Psoriasis',
                'Hepatitis C',
                'Chronic Kidney Disease',
                'Bipolar Disorder',
                'Migraine',
                'Osteoarthritis',
                'Hypertension',
                'Depression'
            ],
            'unmet_need_score': [
                8.5, 7.2, 6.8, 9.1, 8.0, 8.7, 7.5, 5.5, 4.2, 7.8, 6.9, 6.0, 5.8, 4.5, 6.5
            ]
        }
        
        df = pd.DataFrame(sample_data)
        df.to_csv(output_path, index=False)
        self.logger.info(f"Created sample lookup table at {output_path}")