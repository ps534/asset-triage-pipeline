"""
Input parsing and validation system for asset lists.
Handles Excel and CSV file parsing with deduplication.
"""

from pathlib import Path
import pandas as pd
from typing import List, Tuple, Optional
import logging


class InputParserError(Exception):
    """Custom exception for input parsing errors."""
    pass


class InputParser:
    """Handles parsing and validation of input asset files."""
    
    def __init__(self):
        self.required_columns = ['Asset Name', 'Company Name']
        self.optional_columns = ['Primary Indication', 'Mechanism of Action']
        self.logger = logging.getLogger(__name__)
    
    def parse_file(self, file_path: Path) -> pd.DataFrame:
        """Parse Excel or CSV file and return validated DataFrame."""
        if not file_path.exists():
            raise InputParserError(f"File not found: {file_path}")
        
        try:
            # Determine file type and parse accordingly
            if file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = self._parse_excel(file_path)
            elif file_path.suffix.lower() == '.csv':
                df = self._parse_csv(file_path)
            else:
                raise InputParserError(f"Unsupported file format: {file_path.suffix}")
            
            # Validate and clean
            self.validate_columns(df)
            df = self._clean_data(df)
            df = self.deduplicate(df)
            
            self.logger.info(f"Successfully parsed {len(df)} rows from {file_path}")
            return df
            
        except Exception as e:
            raise InputParserError(f"Failed to parse {file_path}: {str(e)}")
    
    def _parse_excel(self, file_path: Path) -> pd.DataFrame:
        """Parse Excel file, handling multiple sheets if necessary."""
        try:
            # Try to read the first sheet
            df = pd.read_excel(file_path, sheet_name=0)
            return df
        except Exception as e:
            raise InputParserError(f"Error reading Excel file: {str(e)}")
    
    def _parse_csv(self, file_path: Path) -> pd.DataFrame:
        """Parse CSV file with encoding detection."""
        try:
            # Try UTF-8 first, then common alternatives
            encodings = ['utf-8', 'latin-1', 'cp1252']
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    return df
                except UnicodeDecodeError:
                    continue
            raise InputParserError("Could not detect file encoding")
        except Exception as e:
            raise InputParserError(f"Error reading CSV file: {str(e)}")
    
    def validate_columns(self, df: pd.DataFrame) -> bool:
        """Validate that required columns are present."""
        missing_columns = [col for col in self.required_columns if col not in df.columns]
        if missing_columns:
            available_cols = list(df.columns)
            raise InputParserError(
                f"Missing required columns: {missing_columns}. "
                f"Available columns: {available_cols}"
            )
        return True
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize the data."""
        # Remove rows where both required columns are empty
        df = df.dropna(subset=self.required_columns, how='all')
        
        # Strip whitespace from string columns
        for col in self.required_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        # Remove rows with empty asset names or company names
        df = df[df['Asset Name'].str.len() > 0]
        df = df[df['Company Name'].str.len() > 0]
        
        return df
    
    def deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate asset/company combinations."""
        initial_count = len(df)
        
        # Create a normalized version for deduplication
        df['_asset_normalized'] = df['Asset Name'].str.lower().str.strip()
        df['_company_normalized'] = df['Company Name'].str.lower().str.strip()
        
        # Drop duplicates based on normalized values
        df = df.drop_duplicates(subset=['_asset_normalized', '_company_normalized'], keep='first')
        
        # Remove temporary columns
        df = df.drop(columns=['_asset_normalized', '_company_normalized'])
        
        duplicates_removed = initial_count - len(df)
        if duplicates_removed > 0:
            self.logger.info(f"Removed {duplicates_removed} duplicate asset/company combinations")
        
        return df
    
    def get_summary(self, df: pd.DataFrame) -> dict:
        """Get summary statistics of the parsed data."""
        return {
            'total_rows': len(df),
            'unique_assets': df['Asset Name'].nunique(),
            'unique_companies': df['Company Name'].nunique(),
            'columns': list(df.columns)
        }