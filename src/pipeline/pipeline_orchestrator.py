"""
Pipeline orchestrator for the three-stage asset filtering process.
Handles rate limiting, error handling, and progress tracking.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd
from pathlib import Path
import time

from src.llm.prompt_manager import PromptManager, LLMError
from src.data.unmet_need_lookup import UnmetNeedLookup, UnmetNeedLookupError


@dataclass
class AssetRecord:
    """Single asset record to be processed."""
    asset_name: str
    company_name: str
    primary_indication: Optional[str]
    mechanism_of_action: Optional[str]
    row_index: int
    timeline: Optional[str] = None


@dataclass
class ProcessingResult:
    """Result of processing a single asset-indication pair."""
    asset_name: str
    company_name: str
    indication: str
    pursue: Optional[bool]
    fail_reasons: List[str]
    degree_of_unmet_need: str
    is_repurposing: bool
    rationale: str
    error: Optional[str]
    info_confidence: Optional[int]
    indication_match_found: bool
    unmet_need_z_score: Optional[float]
    timeline: Optional[str] = None
    novelty_differentiation_score: Optional[int] = None
    unmet_medical_need_score: Optional[int] = None
    development_stage_score: Optional[int] = None
    capital_efficiency_score: Optional[int] = None
    peak_sales_potential_score: Optional[int] = None
    ip_strength_duration_score: Optional[int] = None
    probability_technical_success_score: Optional[int] = None
    competitive_landscape_score: Optional[int] = None
    transactability_score: Optional[int] = None
    regulatory_path_complexity_score: Optional[int] = None
    strategic_fit_score: Optional[int] = None
    novelty_differentiation_rationale: Optional[str] = None
    unmet_medical_need_rationale: Optional[str] = None
    development_stage_rationale: Optional[str] = None
    capital_efficiency_rationale: Optional[str] = None
    peak_sales_potential_rationale: Optional[str] = None
    ip_strength_duration_rationale: Optional[str] = None
    probability_technical_success_rationale: Optional[str] = None
    competitive_landscape_rationale: Optional[str] = None
    transactability_rationale: Optional[str] = None
    regulatory_path_complexity_rationale: Optional[str] = None
    strategic_fit_rationale: Optional[str] = None


class PipelineOrchestrator:
    """Orchestrates the three-stage filtering pipeline with rate limiting and batch processing."""
    
    def __init__(self, unmet_need_csv_path: Optional[str] = None, enable_batching: bool = True):
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.prompt_manager = PromptManager()
        self.unmet_need_lookup = UnmetNeedLookup(unmet_need_csv_path)
        
        # Batching configuration
        self.enable_batching = enable_batching
        self.batch_size = 50  # Optimal batch size for most operations
        self.prefer_anthropic_batch = True  # Default to Anthropic batching for Claude models
        
        # Rate limiting settings (≤20 concurrent calls as per requirements)
        self.max_concurrent = 5  # Reduced to avoid rate limiting with web search
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        
        # Progress tracking
        self.total_assets = 0
        self.processed_assets = 0
        self.errors_count = 0
        
        # Results storage
        self.results: List[ProcessingResult] = []
        
        self.logger.info(f"PipelineOrchestrator initialized with batching {'enabled' if enable_batching else 'disabled'}")
    
    async def process_assets(self, assets_df: pd.DataFrame) -> List[ProcessingResult]:
        """
        Process all assets through the three-stage pipeline.
        Uses batch processing when enabled for improved efficiency.
        Returns list of ProcessingResult objects.
        """
        self.total_assets = len(assets_df)
        self.processed_assets = 0
        self.errors_count = 0
        self.results = []
        
        self.logger.info(f"Starting pipeline processing for {self.total_assets} assets (batching {'enabled' if self.enable_batching else 'disabled'})")
        start_time = time.time()
        
        # Convert DataFrame to AssetRecord objects
        asset_records = []
        for idx, row in assets_df.iterrows():
            primary_indication = None
            if 'Primary Indication' in assets_df.columns and pd.notna(row.get('Primary Indication')):
                primary_indication = str(row['Primary Indication']).strip()
                if primary_indication == "" or primary_indication.lower() in ['nan', 'none', 'null']:
                    primary_indication = None
            
            mechanism_of_action = None
            if 'Mechanism of Action' in assets_df.columns and pd.notna(row.get('Mechanism of Action')):
                mechanism_of_action = str(row['Mechanism of Action']).strip()
                if mechanism_of_action == "" or mechanism_of_action.lower() in ['nan', 'none', 'null']:
                    mechanism_of_action = None
                    
            asset_records.append(AssetRecord(
                asset_name=row['Asset Name'],
                company_name=row['Company Name'],
                primary_indication=primary_indication,
                mechanism_of_action=mechanism_of_action,
                row_index=idx
            ))
        
        # Choose processing strategy based on batching setting
        # Temporarily disable batch processing due to SSL issues with batch manager
        if self.enable_batching and len(asset_records) > 100:  # Increased threshold to effectively disable batching
            results = await self._process_assets_batched(asset_records)
        else:
            results = await self._process_assets_individual(asset_records)
        
        # Flatten results (each asset can generate multiple indication results)
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Asset processing failed: {result}")
                self.errors_count += 1
            elif isinstance(result, list):
                self.results.extend(result)
            else:
                self.results.append(result)
        
        processing_time = time.time() - start_time
        self.logger.info(
            f"Pipeline completed in {processing_time:.1f}s. "
            f"Generated {len(self.results)} asset-indication pairs. "
            f"Errors: {self.errors_count}/{self.total_assets}"
        )
        
        return self.results
    
    async def _process_assets_individual(self, asset_records: List[AssetRecord]) -> List[Any]:
        """Process assets individually (original implementation)."""
        tasks = [self._process_single_asset(asset) for asset in asset_records]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _process_assets_batched(self, asset_records: List[AssetRecord]) -> List[Any]:
        """Process assets using batch operations for improved efficiency."""
        self.logger.info(f"Using batch processing for {len(asset_records)} assets")
        
        try:
            # Stage 1: Batch lookups for missing data
            await self._batch_stage_1_lookups(asset_records)
            
            # Stage 2: Batch repurposing generation
            all_asset_indications = await self._batch_stage_2_repurposing(asset_records)
            
            # Stage 3: Batch asset screening
            return await self._batch_stage_3_screening(all_asset_indications)
            
        except Exception as e:
            self.logger.error(f"Batch processing failed, falling back to individual processing: {e}")
            # Fallback to individual processing
            return await self._process_assets_individual(asset_records)
    
    async def _batch_stage_1_lookups(self, asset_records: List[AssetRecord]) -> None:
        """Stage 1: Batch lookup missing MOA, primary indication data, and timeline research."""
        # All assets need timeline research
        all_assets = [
            {"asset_name": asset.asset_name, "company_name": asset.company_name}
            for asset in asset_records
        ]
        
        # Find assets missing MOA
        assets_missing_moa = [
            {"asset_name": asset.asset_name, "company_name": asset.company_name}
            for asset in asset_records 
            if not asset.mechanism_of_action
        ]
        
        # Find assets missing primary indication
        assets_missing_primary = [
            {"asset_name": asset.asset_name, "company_name": asset.company_name}
            for asset in asset_records 
            if not asset.primary_indication
        ]
        
        timeline_results = {}
        moa_results = {}
        primary_indication_results = {}
        
        # Batch timeline research for all assets
        if all_assets:
            self.logger.info(f"Batch timeline research for {len(all_assets)} assets")
            try:
                timeline_results = await self.prompt_manager.prompt_timeline_research_batch(
                    all_assets, prefer_anthropic_batch=self.prefer_anthropic_batch
                )
            except Exception as e:
                self.logger.warning(f"Batch timeline research failed: {e}")
        
        # Batch MOA lookups
        if assets_missing_moa:
            self.logger.info(f"Batch lookup MOA for {len(assets_missing_moa)} assets")
            try:
                moa_results = await self.prompt_manager.prompt_mechanism_of_action_lookup_batch(
                    assets_missing_moa, prefer_anthropic_batch=self.prefer_anthropic_batch
                )
            except Exception as e:
                self.logger.warning(f"Batch MOA lookup failed: {e}")
        
        # Batch primary indication lookups
        if assets_missing_primary:
            self.logger.info(f"Batch lookup primary indication for {len(assets_missing_primary)} assets")
            try:
                primary_indication_results = await self.prompt_manager.prompt_primary_indication_lookup_batch(
                    assets_missing_primary, prefer_anthropic_batch=self.prefer_anthropic_batch
                )
            except Exception as e:
                self.logger.warning(f"Batch primary indication lookup failed: {e}")
        
        # Update asset records with lookup results
        for asset in asset_records:
            asset_key = f"{asset.asset_name}_{asset.company_name}"
            
            # Update timeline
            if asset_key in timeline_results:
                timeline = timeline_results[asset_key]
                if timeline:
                    asset.timeline = timeline
                    self.logger.debug(f"Updated timeline for {asset.asset_name}")
            
            if not asset.mechanism_of_action and asset_key in moa_results:
                moa = moa_results[asset_key]
                if moa and moa != "Unknown":
                    asset.mechanism_of_action = moa
                    self.logger.debug(f"Updated MOA for {asset.asset_name}: {moa}")
            
            if not asset.primary_indication and asset_key in primary_indication_results:
                primary = primary_indication_results[asset_key]
                if primary and primary != "Unknown":
                    asset.primary_indication = primary
                    self.logger.debug(f"Updated primary indication for {asset.asset_name}: {primary}")
    
    async def _batch_stage_2_repurposing(self, asset_records: List[AssetRecord]) -> List[Dict[str, Any]]:
        """Stage 2: Batch generate repurposing indications."""
        # Prepare data for batch repurposing
        assets_for_repurposing = [
            {
                "asset_name": asset.asset_name,
                "company_name": asset.company_name,
                "mechanism_of_action": asset.mechanism_of_action,
                "primary_indication": asset.primary_indication,
                "timeline": asset.timeline
            }
            for asset in asset_records
        ]
        
        # Batch repurposing generation
        repurposing_results = {}
        if assets_for_repurposing:
            self.logger.info(f"Batch repurposing generation for {len(assets_for_repurposing)} assets")
            try:
                repurposing_results = await self.prompt_manager.prompt_a_repurposing_batch(
                    assets_for_repurposing, prefer_anthropic_batch=self.prefer_anthropic_batch
                )
            except Exception as e:
                self.logger.warning(f"Batch repurposing failed: {e}")
        
        # Compile all asset-indication pairs
        all_asset_indications = []
        for asset in asset_records:
            asset_key = f"{asset.asset_name}_{asset.company_name}"
            
            # Get all indications for this asset
            indications = []
            
            # Add primary indication if available
            if asset.primary_indication:
                indications.append({
                    "indication": asset.primary_indication,
                    "source": "primary",
                    "plausibility": "High"
                })
            
            # Add repurposing indications
            repurposing_indications = repurposing_results.get(asset_key, [])
            indications.extend(repurposing_indications)
            
            # Create asset-indication pairs
            for indication_data in indications:
                all_asset_indications.append({
                    "asset": asset,
                    "indication_data": indication_data,
                    "asset_key": asset_key
                })
        
        return all_asset_indications
    
    async def _batch_stage_3_screening(self, all_asset_indications: List[Dict[str, Any]]) -> List[Any]:
        """Stage 3: Batch asset screening with unmet need lookup."""
        results = []
        
        # Process unmet need lookups (local CSV - fast)
        screening_requests = []
        for item in all_asset_indications:
            asset = item["asset"]
            indication_data = item["indication_data"]
            indication = indication_data["indication"]
            is_repurposing = indication_data["source"] == "repurposing"
            
            # Stage 2: Unmet need lookup (local)
            unmet_need_score, match_found, z_score = self.unmet_need_lookup.lookup_indication(indication)
            if unmet_need_score is None:
                unmet_need_score = "Unknown"
            
            # Prepare for batch screening
            screening_requests.append({
                "asset_name": asset.asset_name,
                "company_name": asset.company_name,
                "indication": indication,
                "unmet_need_score": unmet_need_score,
                "is_repurposing": is_repurposing,
                "mechanism_of_action": asset.mechanism_of_action,
                "match_found": match_found,
                "z_score": z_score,
                "indication_data": indication_data,
                "timeline": asset.timeline
            })
        
        # Batch asset screening
        screening_results = {}
        if screening_requests:
            self.logger.info(f"Batch asset screening for {len(screening_requests)} asset-indication pairs")
            try:
                screening_results = await self.prompt_manager.prompt_c_asset_screen_batch(
                    screening_requests, prefer_anthropic_batch=self.prefer_anthropic_batch
                )
            except Exception as e:
                self.logger.error(f"Batch asset screening failed: {e}")
                # Fall back to individual processing for screening
                return await self._fallback_individual_screening(screening_requests)
        
        # Convert to ProcessingResult objects
        for req in screening_requests:
            request_key = f"{req['asset_name']}_{req['company_name']}_{req['indication']}"
            screening_result = screening_results.get(request_key)
            
            if screening_result:
                # Extract scoring data and rationales
                scoring = screening_result.get("scoring", {})
                scoring_rationale = screening_result.get("scoring_rationale", {})
                
                result = ProcessingResult(
                    asset_name=req["asset_name"],
                    company_name=req["company_name"],
                    indication=req["indication"],
                    pursue=screening_result.get("pursue"),
                    fail_reasons=screening_result.get("fail_reasons", []),
                    degree_of_unmet_need=req["unmet_need_score"],
                    is_repurposing=req["is_repurposing"],
                    rationale=screening_result.get("rationale", ""),
                    error=None,
                    info_confidence=screening_result.get("info_confidence"),
                    indication_match_found=req["match_found"],
                    unmet_need_z_score=req["z_score"],
                    timeline=req.get("timeline"),  # Timeline from batch processing
                    novelty_differentiation_score=scoring.get("novelty_differentiation_score"),
                    unmet_medical_need_score=scoring.get("unmet_medical_need_score"),
                    development_stage_score=scoring.get("development_stage_score"),
                    capital_efficiency_score=scoring.get("capital_efficiency_score"),
                    peak_sales_potential_score=scoring.get("peak_sales_potential_score"),
                    ip_strength_duration_score=scoring.get("ip_strength_duration_score"),
                    probability_technical_success_score=scoring.get("probability_technical_success_score"),
                    competitive_landscape_score=scoring.get("competitive_landscape_score"),
                    transactability_score=scoring.get("transactability_score"),
                    regulatory_path_complexity_score=scoring.get("regulatory_path_complexity_score"),
                    strategic_fit_score=scoring.get("strategic_fit_score"),
                    novelty_differentiation_rationale=scoring_rationale.get("novelty_differentiation_rationale"),
                    unmet_medical_need_rationale=scoring_rationale.get("unmet_medical_need_rationale"),
                    development_stage_rationale=scoring_rationale.get("development_stage_rationale"),
                    capital_efficiency_rationale=scoring_rationale.get("capital_efficiency_rationale"),
                    peak_sales_potential_rationale=scoring_rationale.get("peak_sales_potential_rationale"),
                    ip_strength_duration_rationale=scoring_rationale.get("ip_strength_duration_rationale"),
                    probability_technical_success_rationale=scoring_rationale.get("probability_technical_success_rationale"),
                    competitive_landscape_rationale=scoring_rationale.get("competitive_landscape_rationale"),
                    transactability_rationale=scoring_rationale.get("transactability_rationale"),
                    regulatory_path_complexity_rationale=scoring_rationale.get("regulatory_path_complexity_rationale"),
                    strategic_fit_rationale=scoring_rationale.get("strategic_fit_rationale")
                )
                results.append(result)
            else:
                # Create error result for missing screening result
                error_result = ProcessingResult(
                    asset_name=req["asset_name"],
                    company_name=req["company_name"],
                    indication=req["indication"],
                    pursue=None,
                    fail_reasons=["batch_processing_error"],
                    degree_of_unmet_need=req["unmet_need_score"],
                    is_repurposing=req["is_repurposing"],
                    rationale="",
                    error="Batch screening failed",
                    info_confidence=None,
                    indication_match_found=req["match_found"],
                    unmet_need_z_score=req["z_score"],
                    timeline=req.get("timeline")  # Timeline from batch processing
                )
                results.append(error_result)
        
        # Update processed assets count
        unique_assets = set((req["asset_name"], req["company_name"]) for req in screening_requests)
        self.processed_assets += len(unique_assets)
        
        if self.processed_assets % 10 == 0:
            progress = (self.processed_assets / self.total_assets) * 100
            self.logger.info(f"Progress: {self.processed_assets}/{self.total_assets} ({progress:.1f}%)")
        
        return results
    
    async def _fallback_individual_screening(self, screening_requests: List[Dict[str, Any]]) -> List[ProcessingResult]:
        """Fallback to individual screening when batch processing fails."""
        self.logger.info("Falling back to individual screening")
        
        results = []
        for req in screening_requests:
            try:
                # Individual screening call
                screening_result = await self.prompt_manager.prompt_c_asset_screen(
                    req["asset_name"],
                    req["company_name"],
                    req["indication"],
                    req["unmet_need_score"],
                    req["is_repurposing"],
                    req["mechanism_of_action"],
                    req.get("timeline")
                )
                
                # Extract scoring data and rationales
                scoring = screening_result.get("scoring", {})
                scoring_rationale = screening_result.get("scoring_rationale", {})
                
                result = ProcessingResult(
                    asset_name=req["asset_name"],
                    company_name=req["company_name"],
                    indication=req["indication"],
                    pursue=screening_result.get("pursue"),
                    fail_reasons=screening_result.get("fail_reasons", []),
                    degree_of_unmet_need=req["unmet_need_score"],
                    is_repurposing=req["is_repurposing"],
                    rationale=screening_result.get("rationale", ""),
                    error=None,
                    info_confidence=screening_result.get("info_confidence"),
                    indication_match_found=req["match_found"],
                    unmet_need_z_score=req["z_score"],
                    timeline=req.get("timeline"),  # Timeline from batch processing
                    novelty_differentiation_score=scoring.get("novelty_differentiation_score"),
                    unmet_medical_need_score=scoring.get("unmet_medical_need_score"),
                    development_stage_score=scoring.get("development_stage_score"),
                    capital_efficiency_score=scoring.get("capital_efficiency_score"),
                    peak_sales_potential_score=scoring.get("peak_sales_potential_score"),
                    ip_strength_duration_score=scoring.get("ip_strength_duration_score"),
                    probability_technical_success_score=scoring.get("probability_technical_success_score"),
                    competitive_landscape_score=scoring.get("competitive_landscape_score"),
                    transactability_score=scoring.get("transactability_score"),
                    regulatory_path_complexity_score=scoring.get("regulatory_path_complexity_score"),
                    strategic_fit_score=scoring.get("strategic_fit_score"),
                    novelty_differentiation_rationale=scoring_rationale.get("novelty_differentiation_rationale"),
                    unmet_medical_need_rationale=scoring_rationale.get("unmet_medical_need_rationale"),
                    development_stage_rationale=scoring_rationale.get("development_stage_rationale"),
                    capital_efficiency_rationale=scoring_rationale.get("capital_efficiency_rationale"),
                    peak_sales_potential_rationale=scoring_rationale.get("peak_sales_potential_rationale"),
                    ip_strength_duration_rationale=scoring_rationale.get("ip_strength_duration_rationale"),
                    probability_technical_success_rationale=scoring_rationale.get("probability_technical_success_rationale"),
                    competitive_landscape_rationale=scoring_rationale.get("competitive_landscape_rationale"),
                    transactability_rationale=scoring_rationale.get("transactability_rationale"),
                    regulatory_path_complexity_rationale=scoring_rationale.get("regulatory_path_complexity_rationale"),
                    strategic_fit_rationale=scoring_rationale.get("strategic_fit_rationale")
                )
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Individual screening failed for {req['asset_name']}/{req['indication']}: {e}")
                
                error_result = ProcessingResult(
                    asset_name=req["asset_name"],
                    company_name=req["company_name"],
                    indication=req["indication"],
                    pursue=None,
                    fail_reasons=["individual_screening_error"],
                    degree_of_unmet_need=req["unmet_need_score"],
                    is_repurposing=req["is_repurposing"],
                    rationale="",
                    error=str(e),
                    info_confidence=None,
                    indication_match_found=req["match_found"],
                    unmet_need_z_score=req["z_score"],
                    timeline=req.get("timeline")  # Timeline from batch processing
                )
                results.append(error_result)
        
        return results
    
    async def _process_single_asset(self, asset: AssetRecord) -> List[ProcessingResult]:
        """
        Process a single asset through all stages:
        1a. Determine mechanism of action (from input or web lookup)
        1b. Determine primary indication (from input or web lookup)  
        1c. Generate repurposing indications using MOA + primary indication context (Prompt A)
        2. Process each indication through unmet need lookup
        3. Screen each indication using MOA context (Prompt C)
        """
        async with self.semaphore:  # Rate limiting
            try:
                results = []
                
                # Stage 0: Timeline research (first step)
                timeline = None
                try:
                    timeline = await self.prompt_manager.prompt_timeline_research(
                        asset.asset_name, asset.company_name
                    )
                    self.logger.info(f"Generated timeline for {asset.asset_name}")
                except LLMError as e:
                    self.logger.warning(f"Timeline research failed for {asset.asset_name}: {str(e)}")
                    timeline = f"Timeline research unavailable: {str(e)}"
                
                # Stage 1a: Determine mechanism of action
                mechanism_of_action = asset.mechanism_of_action
                
                # If no MOA provided, look it up via web search
                if not mechanism_of_action:
                    try:
                        mechanism_of_action = await self.prompt_manager.prompt_mechanism_of_action_lookup(
                            asset.asset_name, asset.company_name
                        )
                        self.logger.info(f"Found mechanism of action for {asset.asset_name}: {mechanism_of_action}")
                    except LLMError as e:
                        self.logger.warning(f"Could not determine mechanism of action for {asset.asset_name}: {str(e)}")
                        mechanism_of_action = None
                
                # Stage 1b: Determine primary indication (moved before repurposing)
                primary_indication = asset.primary_indication
                
                # If no primary indication provided, look it up via web search
                if not primary_indication:
                    try:
                        primary_indication = await self.prompt_manager.prompt_primary_indication_lookup(
                            asset.asset_name, asset.company_name
                        )
                        self.logger.info(f"Found primary indication for {asset.asset_name}: {primary_indication}")
                    except LLMError as e:
                        self.logger.warning(f"Could not determine primary indication for {asset.asset_name}: {str(e)}")
                        primary_indication = None
                
                # Stage 1c: Generate repurposing indications with primary indication context (Prompt A)
                repurposing_indications = await self._get_repurposing_indications(asset, mechanism_of_action, primary_indication, timeline)
                
                # Compile all indications for processing
                all_indications = []
                
                # Add primary indication if available
                if primary_indication:
                    all_indications.append({"indication": primary_indication, "source": "primary", "plausibility": "High"})
                
                all_indications.extend(repurposing_indications)
                
                # Stage 2 & 3: Process each indication through unmet need lookup and screening
                if not all_indications:
                    # No indications to process (no primary indication and repurposing failed)
                    self.logger.warning(f"No indications found for asset {asset.asset_name}")
                    return []
                
                indication_tasks = [
                    self._process_single_indication(asset, indication, mechanism_of_action, timeline)
                    for indication in all_indications
                ]
                
                indication_results = await asyncio.gather(*indication_tasks, return_exceptions=True)
                
                for result in indication_results:
                    if isinstance(result, Exception):
                        # Create error result for failed indication
                        error_result = ProcessingResult(
                            asset_name=asset.asset_name,
                            company_name=asset.company_name,
                            indication="Unknown",
                            pursue=None,
                            fail_reasons=[],
                            degree_of_unmet_need="Unknown",
                            is_repurposing=False,
                            rationale="",
                            error=str(result),
                            info_confidence=None,
                            indication_match_found=False,
                            unmet_need_z_score=None,
                            timeline=timeline,
                            novelty_differentiation_score=None,
                            unmet_medical_need_score=None,
                            development_stage_score=None,
                            capital_efficiency_score=None,
                            peak_sales_potential_score=None,
                            ip_strength_duration_score=None,
                            probability_technical_success_score=None,
                            competitive_landscape_score=None,
                            transactability_score=None,
                            regulatory_path_complexity_score=None,
                            strategic_fit_score=None,
                            novelty_differentiation_rationale=None,
                            unmet_medical_need_rationale=None,
                            development_stage_rationale=None,
                            capital_efficiency_rationale=None,
                            peak_sales_potential_rationale=None,
                            ip_strength_duration_rationale=None,
                            probability_technical_success_rationale=None,
                            competitive_landscape_rationale=None,
                            transactability_rationale=None,
                            regulatory_path_complexity_rationale=None,
                            strategic_fit_rationale=None
                        )
                        results.append(error_result)
                    else:
                        results.append(result)
                
                self.processed_assets += 1
                if self.processed_assets % 10 == 0:
                    progress = (self.processed_assets / self.total_assets) * 100
                    self.logger.info(f"Progress: {self.processed_assets}/{self.total_assets} ({progress:.1f}%)")
                
                return results
                
            except Exception as e:
                self.logger.error(f"Failed to process asset {asset.asset_name}: {str(e)}")
                
                # Return error result
                error_result = ProcessingResult(
                    asset_name=asset.asset_name,
                    company_name=asset.company_name,
                    indication=asset.asset_name,
                    pursue=None,
                    fail_reasons=[],
                    degree_of_unmet_need="Unknown",
                    is_repurposing=False,
                    rationale="",
                    error=str(e),
                    info_confidence=None,
                    indication_match_found=False,
                    unmet_need_z_score=None,
                    timeline=timeline if 'timeline' in locals() else None,
                    novelty_differentiation_score=None,
                    unmet_medical_need_score=None,
                    development_stage_score=None,
                    capital_efficiency_score=None,
                    peak_sales_potential_score=None,
                    ip_strength_duration_score=None,
                    probability_technical_success_score=None,
                    competitive_landscape_score=None,
                    transactability_score=None,
                    regulatory_path_complexity_score=None,
                    strategic_fit_score=None,
                    novelty_differentiation_rationale=None,
                    unmet_medical_need_rationale=None,
                    development_stage_rationale=None,
                    capital_efficiency_rationale=None,
                    peak_sales_potential_rationale=None,
                    ip_strength_duration_rationale=None,
                    probability_technical_success_rationale=None,
                    competitive_landscape_rationale=None,
                    transactability_rationale=None,
                    regulatory_path_complexity_rationale=None,
                    strategic_fit_rationale=None
                )
                return [error_result]
    
    async def _get_repurposing_indications(self, asset: AssetRecord, mechanism_of_action: Optional[str] = None, primary_indication: Optional[str] = None, timeline: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get repurposing indications for an asset (Stage 1)."""
        try:
            return await self.prompt_manager.prompt_a_repurposing(
                asset.asset_name, 
                asset.company_name,
                mechanism_of_action,
                primary_indication,
                timeline
            )
        except LLMError as e:
            self.logger.warning(f"Repurposing prompt failed for {asset.asset_name}: {str(e)}")
            return []  # Return empty list if repurposing fails
    
    async def _process_single_indication(self, asset: AssetRecord, indication_data: Dict[str, Any], mechanism_of_action: Optional[str] = None, timeline: Optional[str] = None) -> ProcessingResult:
        """Process a single asset-indication pair through stages 2 and 3."""
        indication = indication_data["indication"]
        is_repurposing = indication_data["source"] == "repurposing"
        
        try:
            # Stage 2: Unmet need lookup
            unmet_need_score, match_found, z_score = self.unmet_need_lookup.lookup_indication(indication)
            if unmet_need_score is None:
                unmet_need_score = "Unknown"
            
            # Stage 3: Asset screening (Prompt C)
            screening_result = await self.prompt_manager.prompt_c_asset_screen(
                asset.asset_name,
                asset.company_name,
                indication,
                unmet_need_score,
                is_repurposing,
                mechanism_of_action,
                timeline
            )
            
            # Extract scoring data and rationales
            scoring = screening_result.get("scoring", {})
            scoring_rationale = screening_result.get("scoring_rationale", {})
            
            # Create result
            result = ProcessingResult(
                asset_name=asset.asset_name,
                company_name=asset.company_name,
                indication=indication,
                pursue=screening_result["pursue"],
                fail_reasons=screening_result["fail_reasons"],
                degree_of_unmet_need=unmet_need_score,
                is_repurposing=is_repurposing,
                rationale=screening_result["rationale"],
                error=None,
                info_confidence=screening_result["info_confidence"],
                indication_match_found=match_found,
                unmet_need_z_score=z_score,
                timeline=timeline,
                novelty_differentiation_score=scoring.get("novelty_differentiation_score"),
                unmet_medical_need_score=scoring.get("unmet_medical_need_score"),
                development_stage_score=scoring.get("development_stage_score"),
                capital_efficiency_score=scoring.get("capital_efficiency_score"),
                peak_sales_potential_score=scoring.get("peak_sales_potential_score"),
                ip_strength_duration_score=scoring.get("ip_strength_duration_score"),
                probability_technical_success_score=scoring.get("probability_technical_success_score"),
                competitive_landscape_score=scoring.get("competitive_landscape_score"),
                transactability_score=scoring.get("transactability_score"),
                regulatory_path_complexity_score=scoring.get("regulatory_path_complexity_score"),
                strategic_fit_score=scoring.get("strategic_fit_score"),
                novelty_differentiation_rationale=scoring_rationale.get("novelty_differentiation_rationale"),
                unmet_medical_need_rationale=scoring_rationale.get("unmet_medical_need_rationale"),
                development_stage_rationale=scoring_rationale.get("development_stage_rationale"),
                capital_efficiency_rationale=scoring_rationale.get("capital_efficiency_rationale"),
                peak_sales_potential_rationale=scoring_rationale.get("peak_sales_potential_rationale"),
                ip_strength_duration_rationale=scoring_rationale.get("ip_strength_duration_rationale"),
                probability_technical_success_rationale=scoring_rationale.get("probability_technical_success_rationale"),
                competitive_landscape_rationale=scoring_rationale.get("competitive_landscape_rationale"),
                transactability_rationale=scoring_rationale.get("transactability_rationale"),
                regulatory_path_complexity_rationale=scoring_rationale.get("regulatory_path_complexity_rationale"),
                strategic_fit_rationale=scoring_rationale.get("strategic_fit_rationale")
            )
            
            self.logger.debug(f"Created ProcessingResult with scoring: novelty={result.novelty_differentiation_score}, unmet_need={result.unmet_medical_need_score}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to process indication {indication} for {asset.asset_name}: {str(e)}")
            
            # Return error result
            return ProcessingResult(
                asset_name=asset.asset_name,
                company_name=asset.company_name,
                indication=indication,
                pursue=None,
                fail_reasons=[],
                degree_of_unmet_need=unmet_need_score if 'unmet_need_score' in locals() else "Unknown",
                is_repurposing=is_repurposing,
                rationale="",
                error=str(e),
                info_confidence=None,
                indication_match_found=match_found if 'match_found' in locals() else False,
                unmet_need_z_score=z_score if 'z_score' in locals() else None,
                timeline=timeline,
                novelty_differentiation_score=None,
                unmet_medical_need_score=None,
                development_stage_score=None,
                capital_efficiency_score=None,
                peak_sales_potential_score=None,
                ip_strength_duration_score=None,
                probability_technical_success_score=None,
                competitive_landscape_score=None,
                transactability_score=None,
                regulatory_path_complexity_score=None,
                strategic_fit_score=None,
                novelty_differentiation_rationale=None,
                unmet_medical_need_rationale=None,
                development_stage_rationale=None,
                capital_efficiency_rationale=None,
                peak_sales_potential_rationale=None,
                ip_strength_duration_rationale=None,
                probability_technical_success_rationale=None,
                competitive_landscape_rationale=None,
                transactability_rationale=None,
                regulatory_path_complexity_rationale=None,
                strategic_fit_rationale=None
            )
    
    def get_progress_stats(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        if self.total_assets == 0:
            return {"status": "not_started"}
        
        pursue_count = sum(1 for r in self.results if r.pursue is True)
        total_pairs = len(self.results)
        
        return {
            "processed_assets": self.processed_assets,
            "total_assets": self.total_assets,
            "progress_percentage": (self.processed_assets / self.total_assets) * 100,
            "total_asset_indication_pairs": total_pairs,
            "pursue_count": pursue_count,
            "pursue_rate": (pursue_count / total_pairs * 100) if total_pairs > 0 else 0,
            "error_count": self.errors_count,
            "error_rate": (self.errors_count / self.total_assets * 100) if self.total_assets > 0 else 0
        }
    
    async def close(self):
        """Clean up resources."""
        if hasattr(self.prompt_manager, 'close_batch_manager'):
            await self.prompt_manager.close_batch_manager()