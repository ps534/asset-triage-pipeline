"""
LLM prompt management for the three-stage filtering pipeline.
Handles Prompt A (repurposing), Prompt B (unmet need), and Prompt C (asset screening).
"""

import asyncio
import json
import logging
import os
from typing import Dict, List, Any, Optional
import litellm
import anthropic
import httpx
from litellm.utils import supports_prompt_caching
from .batch_manager import BatchManager, BatchRequest, BatchResponse


class LLMError(Exception):
    """Custom exception for LLM-related errors."""
    pass


class PromptManager:
    """Manages the three LLM prompts in the filtering pipeline."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Configure LiteLLM for your gateway
        self.gateway_url = os.getenv("LITELLM_GATEWAY_URL", "https://llmgateway.experiment.trialspark.com/")
        
        # Ensure gateway URL ends with / for proper path construction
        if not self.gateway_url.endswith('/'):
            self.gateway_url += '/'
        
        # Set the base URL for all LiteLLM calls
        litellm.api_base = self.gateway_url
        
        # Initialize Anthropic client for direct API access (if needed)
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_api_key:
            self.anthropic_client = anthropic.AsyncAnthropic(api_key=anthropic_api_key)
            self.logger.info("Anthropic direct client initialized")
        else:
            self.anthropic_client = None
            self.logger.info("No Anthropic API key provided - will use gateway only")
        
        # Model configuration for each prompt
        # Can be overridden via environment variables
        self.models = {
            "prompt_a": os.getenv("PROMPT_A_MODEL", "o3"),  # Repurposing
            "prompt_c": os.getenv("PROMPT_C_MODEL", "claude-sonnet-4-20250514"),  # Asset screening
            "moa_lookup": os.getenv("MOA_LOOKUP_MODEL", "claude-sonnet-4-20250514"),  # MOA lookup
            "primary_indication_lookup": os.getenv("PRIMARY_INDICATION_MODEL", "claude-sonnet-4-20250514"),  # Primary indication lookup
        }
        
        # Determine which models should use direct API vs gateway
        self.use_direct_anthropic = os.getenv("USE_DIRECT_ANTHROPIC", "true").lower() == "true"
        
        self.logger.info(f"Using LiteLLM gateway at: {self.gateway_url}")
        self.logger.info(f"Model configuration: Prompt A={self.models['prompt_a']}, Prompt C={self.models['prompt_c']}")
        self.logger.info(f"Direct Anthropic API: {'Enabled' if self.use_direct_anthropic and self.anthropic_client else 'Disabled'}")
        
        # Rate limiting and retry settings
        self.max_retries = 5
        
        # Prompt caching configuration
        self.enable_prompt_caching = os.getenv("ENABLE_PROMPT_CACHING", "true").lower() == "true"
        self.logger.info(f"Prompt caching: {'Enabled' if self.enable_prompt_caching else 'Disabled'}")
        
        # Extended reasoning configuration
        self.o3_effort = os.getenv("O3_EFFORT", "high").lower()
        self.claude_extended_thinking = os.getenv("CLAUDE_EXTENDED_THINKING", "true").lower() == "true"
        self.logger.info(f"o3 effort level: {self.o3_effort}")
        self.logger.info(f"Claude extended thinking: {'Enabled' if self.claude_extended_thinking else 'Disabled'}")
        
        # Initialize batch manager for efficient batching
        self.batch_manager = BatchManager(
            gateway_url=self.gateway_url,
            anthropic_api_key=anthropic_api_key or "",
            max_batch_size=100,
            max_concurrent_batches=5,
            anthropic_batch_enabled=True,
            litellm_multi_model_enabled=True
        )
        self.logger.info("Batch manager initialized")
        
    async def prompt_a_repurposing(self, asset_name: str, company_name: str, mechanism_of_action: Optional[str] = None, primary_indication: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Prompt A: Generate up to 5 plausible repurposing indications.
        Returns list of indications with plausibility ratings.
        """
        prompt = self._build_repurposing_prompt(asset_name, company_name, mechanism_of_action, primary_indication)
        
        try:
            response = await self._call_llm_with_retry(
                model=self.models["prompt_a"],
                prompt=prompt,
                use_web_search=False
            )
            
            # Parse JSON with better error handling
            try:
                result = json.loads(response)
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON decode error: {e}. Raw response: {response[:500]}")
                raise LLMError(f"Invalid JSON response: {str(e)}")
            indications = result.get("repurposing_indications", [])
            
            # Validate and format the response
            formatted_indications = []
            for indication in indications[:5]:  # Limit to 5 as per requirements
                if isinstance(indication, dict) and "indication" in indication and "plausibility" in indication:
                    formatted_indications.append({
                        "indication": indication["indication"],
                        "plausibility": indication["plausibility"],
                        "source": "repurposing"
                    })
            
            self.logger.debug(f"Generated {len(formatted_indications)} repurposing indications for {asset_name}")
            return formatted_indications
            
        except Exception as e:
            self.logger.error(f"Prompt A failed for {asset_name}: {str(e)}")
            raise LLMError(f"Repurposing prompt failed: {str(e)}")
    
    def _build_repurposing_prompt(self, asset_name: str, company_name: str, mechanism_of_action: Optional[str] = None, primary_indication: Optional[str] = None) -> str:
        """Build the repurposing enumeration prompt."""
        moa_context = f"\nMechanism of Action: {mechanism_of_action}" if mechanism_of_action else ""
        primary_context = f"\nPrimary Indication: {primary_indication}" if primary_indication else ""
        
        return f\"\"\"You are an expert in drug repurposing and clinical pharmacology. Given a clinical-stage asset, identify up to 5 plausible repurposing indications beyond its primary indication.

Asset: {asset_name}
Company: {company_name}{primary_context}{moa_context}

For each repurposing indication, assess the plausibility based on:
- Mechanism of action compatibility{" (considering the known MOA above)" if mechanism_of_action else ""}
- Primary indication insights{" (considering how the mechanism might work in other diseases)" if primary_indication else ""}
- Published research or clinical evidence
- Similar successful repurposing cases
- Biological pathway relevance

Return your response as JSON in this exact format:
{{
  "repurposing_indications": [
    {{
      "indication": "specific medical condition or disease",
      "plausibility": "High|Medium|Low"
    }}
  ]
}}

Guidelines:
- Focus on scientifically plausible repurposing opportunities DISTINCT from the primary indication{" (" + primary_indication + ")" if primary_indication else ""}
- Consider off-label uses that have clinical evidence
- Prioritize indications with unmet medical needs
- Be specific about the medical condition (e.g., "Type 2 diabetes" not "metabolic disorders")
- Do not consider or recommend repurposing for oncology diseases, as this is not a focus of the company.
- Make sure to consider rare diseases, particularly those that are not well-served by existing therapies.
- Include only serious consideration-worthy repurposing opportunities
- Avoid suggesting the same disease category as the primary indication unless it's a very different sub-condition\"\"\"

    def _build_primary_indication_prompt(self, asset_name: str, company_name: str) -> str:
        """Build the primary indication lookup prompt."""
        return f\"\"\"You are a pharmaceutical research analyst. Use web search to find the primary medical indication for this clinical-stage asset.

Asset: {asset_name}
Company: {company_name}

SEARCH STRATEGY:
1. Search for "{asset_name} {company_name} indication"
2. Search for "{asset_name} {company_name} clinical trial"
3. Search for "{asset_name} phase 1 phase 2 phase 3"
4. Search for "{company_name} pipeline {asset_name}"
5. Search for "{asset_name} FDA" or "{asset_name} regulatory"

Look for information from:
- ClinicalTrials.gov database (most authoritative)
- Company press releases and investor presentations
- SEC filings and annual reports
- FDA communications and regulatory documents
- PubMed/scientific literature
- Pharmaceutical industry databases
- Biotech industry news sources

Based on your web search, identify the PRIMARY medical indication that this asset is being developed to treat. This should be the main disease or condition, not secondary indications or repurposing opportunities.

Return your response as JSON in this exact format:
{{
  "primary_indication": "Specific medical condition (e.g., Type 2 Diabetes, Alzheimer's Disease)"
}}

Guidelines:
- Be specific (e.g., "Type 2 diabetes" not "diabetes")
- Use standard medical terminology
- If multiple indications exist, choose the primary/lead indication
- If truly unclear from web search, respond with "Unknown"
- Do not guess - only report what you find from web research\"\"\"

    def _build_mechanism_of_action_prompt(self, asset_name: str, company_name: str) -> str:
        """Build the mechanism of action lookup prompt."""
        return f\"\"\"You are a pharmaceutical research analyst. Use web search to find the mechanism of action for this clinical-stage asset.

Asset: {asset_name}
Company: {company_name}

SEARCH STRATEGY:
1. Search for "{asset_name} {company_name} mechanism of action"
2. Search for "{asset_name} {company_name} drug development"
3. Search for "{asset_name} clinical trial"
4. Search for "{company_name} pipeline {asset_name}"
5. Search for "{asset_name} patent" or "{asset_name} regulatory filing"

Look for information from:
- Company press releases and investor presentations
- ClinicalTrials.gov database
- PubMed/scientific literature
- SEC filings and annual reports
- Pharmaceutical industry databases (DrugBank, etc.)
- Patent databases (USPTO, Google Patents)
- Biotech industry news sources

Based on your web search, identify the MECHANISM OF ACTION - how this drug works at the molecular/cellular level to produce its therapeutic effect.

Return your response as JSON in this exact format:
{{
  "mechanism_of_action": "Specific mechanism description (e.g., GLP-1 receptor agonist, PD-1 checkpoint inhibitor)"
}}

Guidelines:
- Be specific and scientific (e.g., "SGLT2 inhibitor" not "diabetes drug")
- Use standard pharmacological terminology
- Include target/pathway if known (e.g., "mTOR pathway inhibitor")
- If multiple mechanisms, focus on the primary one
- If truly unclear from web search, respond with "Unknown"
- Do not guess - only report what you find from scientific sources\"\"\"

    async def prompt_mechanism_of_action_lookup(self, asset_name: str, company_name: str) -> str:
        """
        New prompt to determine mechanism of action via web search.
        Returns the mechanism of action for the asset.
        """
        prompt = self._build_mechanism_of_action_prompt(asset_name, company_name)
        
        try:
            response = await self._call_llm_with_retry(
                model=self.models["moa_lookup"],
                prompt=prompt,
                use_web_search=True
            )
            
            # Parse JSON with better error handling
            try:
                result = json.loads(response)
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON decode error: {e}. Raw response: {response[:500]}")
                raise LLMError(f"Invalid JSON response: {str(e)}")
            
            # Validate required field
            if "mechanism_of_action" not in result:
                raise LLMError("Missing required field: mechanism_of_action")
            
            mechanism_of_action = result["mechanism_of_action"].strip()
            if not mechanism_of_action or mechanism_of_action.lower() in ['unknown', 'not found', 'unclear']:
                raise LLMError("Could not determine mechanism of action from web search")
            
            self.logger.debug(f"Found mechanism of action for {asset_name}: {mechanism_of_action}")
            return mechanism_of_action
            
        except Exception as e:
            self.logger.error(f"Mechanism of action lookup failed for {asset_name}: {str(e)}")
            raise LLMError(f"Mechanism of action lookup failed: {str(e)}")

    async def prompt_primary_indication_lookup(self, asset_name: str, company_name: str) -> str:
        """
        New prompt to determine primary indication via web search.
        Returns the primary medical indication for the asset.
        """
        prompt = self._build_primary_indication_prompt(asset_name, company_name)
        
        try:
            response = await self._call_llm_with_retry(
                model=self.models["primary_indication_lookup"],
                prompt=prompt,
                use_web_search=True
            )
            
            # Parse JSON with better error handling
            try:
                result = json.loads(response)
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON decode error: {e}. Raw response: {response[:500]}")
                raise LLMError(f"Invalid JSON response: {str(e)}")
            
            # Validate required field
            if "primary_indication" not in result:
                raise LLMError("Missing required field: primary_indication")
            
            primary_indication = result["primary_indication"].strip()
            if not primary_indication or primary_indication.lower() in ['unknown', 'not found', 'unclear']:
                raise LLMError("Could not determine primary indication from web search")
            
            self.logger.debug(f"Found primary indication for {asset_name}: {primary_indication}")
            return primary_indication
            
        except Exception as e:
            self.logger.error(f"Primary indication lookup failed for {asset_name}: {str(e)}")
            raise LLMError(f"Primary indication lookup failed: {str(e)}")

    async def prompt_c_asset_screen(self, asset_name: str, company_name: str, indication: str, unmet_need_score: str, is_repurposing: bool = False, mechanism_of_action: Optional[str] = None) -> Dict[str, Any]:
        """
        Prompt C: Screen asset-indication pair for pursue/not-pursue decision.
        Returns JSON with pursue boolean, fail_reasons, and confidence.
        """
        # Check if the model supports prompt caching and we have it enabled
        model = self.models["prompt_c"]
        supports_caching = (self.enable_prompt_caching and 
                          supports_prompt_caching(f"anthropic/{model}") and 
                          "claude" in model.lower())
        
        try:
            if supports_caching:
                # Use messages format with caching for Claude models
                messages = self._build_screening_prompt_messages(asset_name, company_name, indication, unmet_need_score, is_repurposing, mechanism_of_action)
                response = await self._call_llm_with_retry(
                    model=model,
                    messages=messages,
                    use_web_search=True
                )
            else:
                # Fall back to original prompt format for other models
                prompt = self._build_screening_prompt(asset_name, company_name, indication, unmet_need_score, is_repurposing, mechanism_of_action)
                response = await self._call_llm_with_retry(
                    model=model,
                    prompt=prompt,
                    use_web_search=True
                )
            
            # Parse JSON with better error handling
            try:
                result = json.loads(response)
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON decode error: {e}. Raw response: {response[:500]}")
                raise LLMError(f"Invalid JSON response: {str(e)}")
            
            # Check if this is an insufficient_data response
            if result.get("insufficient_data", False):
                # Handle insufficient data case
                if "reason" not in result or "info_confidence" not in result:
                    raise LLMError("Insufficient data response missing required fields: reason, info_confidence")
                
                # Create a standardized insufficient data response
                standardized_result = {
                    "pursue": False,
                    "fail_reasons": ["insufficient_data"],
                    "info_confidence": int(result["info_confidence"]),
                    "rationale": result["reason"],
                    "scoring": {},
                    "scoring_rationale": {}
                }
                
                # Add None values for all scoring fields to maintain data structure
                scoring_fields = [
                    "novelty_differentiation_score", "unmet_medical_need_score", "development_stage_score",
                    "capital_efficiency_score", "peak_sales_potential_score", "ip_strength_duration_score",
                    "probability_technical_success_score", "competitive_landscape_score", "transactability_score",
                    "regulatory_path_complexity_score", "strategic_fit_score"
                ]
                rationale_fields = [
                    "novelty_differentiation_rationale", "unmet_medical_need_rationale", "development_stage_rationale",
                    "capital_efficiency_rationale", "peak_sales_potential_rationale", "ip_strength_duration_rationale",
                    "probability_technical_success_rationale", "competitive_landscape_rationale", "transactability_rationale",
                    "regulatory_path_complexity_rationale", "strategic_fit_rationale"
                ]
                
                for field in scoring_fields:
                    standardized_result["scoring"][field] = None
                for field in rationale_fields:
                    standardized_result["scoring_rationale"][field] = None
                
                self.logger.info(f"Insufficient data response for {asset_name}/{indication}: {result['reason']}")
                return standardized_result
            
            # Standard response validation - ALL fields are mandatory for complete assessments
            required_fields = ["pursue", "fail_reasons", "info_confidence", "rationale", "scoring", "scoring_rationale"]
            for field in required_fields:
                if field not in result:
                    raise LLMError(f"Missing required field: {field}")
            
            # Validate scoring fields - ALL scoring fields are mandatory
            required_scoring_fields = [
                "novelty_differentiation_score", "unmet_medical_need_score", "development_stage_score",
                "capital_efficiency_score", "peak_sales_potential_score", "ip_strength_duration_score",
                "probability_technical_success_score", "competitive_landscape_score", "transactability_score",
                "regulatory_path_complexity_score", "strategic_fit_score"
            ]
            for field in required_scoring_fields:
                if field not in result["scoring"]:
                    raise LLMError(f"Missing required scoring field: {field}")
                # Ensure scores are integers between 0-10
                try:
                    score = int(result["scoring"][field])
                    result["scoring"][field] = max(0, min(10, score))
                except (ValueError, TypeError):
                    raise LLMError(f"Invalid score value for {field}: {result['scoring'][field]}")
            
            # Validate scoring rationale fields - ALL rationale fields are mandatory
            required_rationale_fields = [
                "novelty_differentiation_rationale", "unmet_medical_need_rationale", "development_stage_rationale",
                "capital_efficiency_rationale", "peak_sales_potential_rationale", "ip_strength_duration_rationale",
                "probability_technical_success_rationale", "competitive_landscape_rationale", "transactability_rationale",
                "regulatory_path_complexity_rationale", "strategic_fit_rationale"
            ]
            for field in required_rationale_fields:
                if field not in result["scoring_rationale"]:
                    raise LLMError(f"Missing required rationale field: {field}")
                if not result["scoring_rationale"][field] or result["scoring_rationale"][field].strip() == "":
                    raise LLMError(f"Empty rationale for {field}")
            
            # Ensure proper data types
            result["pursue"] = bool(result["pursue"])
            result["fail_reasons"] = result["fail_reasons"] if isinstance(result["fail_reasons"], list) else []
            result["info_confidence"] = int(result["info_confidence"])
            
            self.logger.debug(f"Screening result for {asset_name}/{indication}: pursue={result['pursue']}")
            self.logger.debug(f"Scoring data included: novelty={result['scoring']['novelty_differentiation_score']}, unmet_need={result['scoring']['unmet_medical_need_score']}")
            return result
            
        except Exception as e:
            self.logger.error(f"Prompt C failed for {asset_name}/{indication}: {str(e)}")
            raise LLMError(f"Asset screening prompt failed: {str(e)}")
    
    def _get_screening_framework(self) -> str:
        """Get the cacheable scoring framework for asset screening."""
        return \"\"\"You are a business development analyst evaluating clinical-stage assets for potential acquisition or partnership. You will use web search to gather available information, but recognize that many valuable assets have limited public disclosures due to confidentiality or early development stage.

## Key Principles:
1. Limited public information does NOT automatically disqualify an asset - it may indicate a proprietary opportunity
2. Early-stage assets (Phase 1/2) can be highly valuable if they address unmet needs
3. Recommend "pursue" for assets that warrant deeper investigation through direct company contact
4. Consider the strategic value of learning more about assets with novel mechanisms or high unmet need indications

## Scoring Framework (use 0–10 scale; include one crisp sentence of rationale)

| Criterion                               | 0                                           | 3-4                                    | 5-6                                    | 7-8                                       | 9-10                                               |
|-----------------------------------------|---------------------------------------------|----------------------------------------|----------------------------------------|--------------------------------------------|----------------------------------------------------|
| **Novelty & Differentiation**           | Me-too / well-trodden target               | Incremental tweak                     | New in this indication                | First-in-class or new modality            | First-ever target / modality with no peers        |
| **Unmet Medical Need**                  | Effective therapies plentiful              | Modest gap                            | Partial efficacy or QoL gap           | Few/no options; high morbidity           | No approved therapy; life-threatening condition   |
| **Development Stage & Timelines**       | Discovery / Pre-IND                        | Phase 0 / enabling                    | Phase 1                                | Phase 2                                    | Phase 3 positive or NDA filed                      |
| **Capital Efficiency to PoC**           | >$500 M                                    | $200-500 M                            | $100-200 M                            | $20-100 M                                  | <$20 M or mostly sunk by partner                  |
| **Peak-Sales Potential**                | < $0.2 B                                   | $0.2-0.5 B                            | $0.5-1 B                              | $1-2 B                                     | > $2 B                                           |
| **IP Strength & Duration**              | Expired / none                              | < 2030 single patent                  | 2030-2035                              | 2035-2040 multiple families               | ≥ 2040, broad CMC + method + formulation          |
| **Probability of Technical Success**    | < 5 %                                      | 5-15 %                                | 15-30 %                                | 30-50 %                                   | > 50 %                                           |
| **Competitive Landscape**               | Numerous late-stage peers                  | Several mid-stage peers              | Some early peers                      | Few direct competitors                     | White-space / unique pathway                      |
| **Transactability**                     | Owner unwilling / price unrealistic        | Hard-to-engage                       | Negotiable but stiff                  | Constructive negotiations                 | Owner actively seeking partnership                |
| **Regulatory Path Complexity**          | Unclear endpoints / no precedent           | Novel but mappable                   | Standard but arduous                  | Accelerated pathway possible              | Established surrogate endpoints / RMAT, etc.      |
| **Strategic Fit**                       | Outside strategy                            | Peripheral                           | Adjacent                               | Strong fit                                | Core to strategy / synergy with current assets    |

## Decision Framework:
- PURSUE: Asset warrants further investigation (scores ≥5 in key areas OR high unmet need with reasonable development stage)
- DON'T PURSUE: Multiple critical weaknesses or clear deal-breakers

## Handling Limited Information:
When limited public information is available:
1. Score based on available data and reasonable inferences
2. For unknown criteria, assign neutral scores (5-6) unless red flags exist
3. Note information gaps in rationales but don't penalize excessively
4. Consider that limited disclosure may indicate:
   - Proprietary/confidential development
   - Early but promising asset
   - Opportunity for competitive advantage through early engagement

IMPORTANT: Search for information about the asset, company, indication, and therapeutic area. Base assessments on available data while recognizing that direct company engagement may reveal additional value.

Return your response as JSON in this exact format:
{{
  "pursue": true/false,
  "fail_reasons": ["reason1", "reason2"],
  "info_confidence": 85,
  "rationale": "Brief explanation of decision (1-2 sentences)",
  "scoring": {{
    "novelty_differentiation_score": 7,
    "unmet_medical_need_score": 8,
    "development_stage_score": 6,
    "capital_efficiency_score": 5,
    "peak_sales_potential_score": 8,
    "ip_strength_duration_score": 7,
    "probability_technical_success_score": 6,
    "competitive_landscape_score": 7,
    "transactability_score": 4,
    "regulatory_path_complexity_score": 8,
    "strategic_fit_score": 6
  }},
  "scoring_rationale": {{
    "novelty_differentiation_rationale": "Brief sentence explaining the score",
    "unmet_medical_need_rationale": "Brief sentence explaining the score",
    "development_stage_rationale": "Brief sentence explaining the score",
    "capital_efficiency_rationale": "Brief sentence explaining the score",
    "peak_sales_potential_rationale": "Brief sentence explaining the score",
    "ip_strength_duration_rationale": "Brief sentence explaining the score",
    "probability_technical_success_rationale": "Brief sentence explaining the score",
    "competitive_landscape_rationale": "Brief sentence explaining the score",
    "transactability_rationale": "Brief sentence explaining the score",
    "regulatory_path_complexity_rationale": "Brief sentence explaining the score",
    "strategic_fit_rationale": "Brief sentence explaining the score"
  }}
}}

Common fail reasons: "pre_clinical_only", "market_too_small", "overcrowded_space", "weak_differentiation", "regulatory_barriers", "poor_strategic_fit"

Note: Recommend "pursue" for assets addressing high unmet needs, novel mechanisms, or strategic areas even with limited public data.\"\"\"
    
    def _build_screening_prompt_messages(self, asset_name: str, company_name: str, indication: str, unmet_need_score: str, is_repurposing: bool, mechanism_of_action: Optional[str] = None) -> List[Dict[str, Any]]:
        """Build messages for the asset screening prompt with caching support."""
        repurposing_context = " (repurposing opportunity)" if is_repurposing else " (primary indication)"
        moa_context = f"\nMechanism of Action: {mechanism_of_action}" if mechanism_of_action else ""
        
        # Create cacheable system message with framework
        system_content = []
        if self.enable_prompt_caching:
            system_content = [
                {
                    "type": "text",
                    "text": self._get_screening_framework(),
                    "cache_control": {"type": "ephemeral"}  # This will be cached
                }
            ]
        else:
            system_content = [
                {
                    "type": "text", 
                    "text": self._get_screening_framework()
                }
            ]
        
        # Variable user content (not cached)
        user_content = f\"\"\"Asset: {asset_name}
Company: {company_name}
Indication: {indication}{repurposing_context}
Unmet Need Score: {unmet_need_score}{moa_context}

Please analyze this asset-indication pair according to the scoring framework and return your assessment as JSON.\"\"\"

        return [
            {
                "role": "system",
                "content": system_content
            },
            {
                "role": "user", 
                "content": user_content
            }
        ]
    
    def _build_screening_prompt(self, asset_name: str, company_name: str, indication: str, unmet_need_score: str, is_repurposing: bool, mechanism_of_action: Optional[str] = None) -> str:
        """Build the asset screening prompt (legacy format without caching)."""
        repurposing_context = " (repurposing opportunity)" if is_repurposing else " (primary indication)"
        moa_context = f"\nMechanism of Action: {mechanism_of_action}" if mechanism_of_action else ""
        
        framework = self._get_screening_framework()
        
        return f\"\"\"{framework}

Asset: {asset_name}
Company: {company_name}
Indication: {indication}{repurposing_context}
Unmet Need Score: {unmet_need_score}{moa_context}

Please analyze this asset-indication pair according to the scoring framework and return your assessment as JSON.\"\"\"

    async def _call_llm_with_retry(self, model: str, prompt: str = None, use_web_search: bool = False, messages: List[Dict[str, Any]] = None) -> str:
        """Call LLM (via direct HTTP, LiteLLM gateway, or direct Anthropic API) with retry logic."""
        # Check if we should use direct Anthropic API
        if self.use_direct_anthropic and self.anthropic_client and "claude" in model.lower():
            if messages:
                return await self._call_anthropic_direct_messages(model, messages, use_web_search)
            else:
                return await self._call_anthropic_direct(model, prompt, use_web_search)
        else:
            # For OpenAI models, use LiteLLM gateway with proper web search support
            if messages:
                return await self._call_litellm_gateway_messages(model, messages, use_web_search)
            else:
                return await self._call_litellm_gateway(model, prompt, use_web_search)
    
    async def _call_anthropic_direct(self, model: str, prompt: str, use_web_search: bool = False) -> str:
        """Call Anthropic API directly with web search support."""
        last_error = None
        retry_delay = 1.0  # Initial retry delay in seconds
        
        for attempt in range(self.max_retries + 1):
            try:
                # Add JSON format instruction to prompt
                formatted_prompt = f"{prompt}\n\nIMPORTANT: Respond with valid JSON only. Do not include any text before or after the JSON."
                
                # Web search will be handled via tools parameter below
                
                # Add thinking instruction if enabled
                if self.claude_extended_thinking and "claude" in model.lower():
                    formatted_prompt = f"<thinking>\nLet me carefully analyze this request and think through the best approach to provide an accurate, well-researched response.\n</thinking>\n\n{formatted_prompt}"
                
                # Prepare kwargs for Anthropic call
                anthropic_kwargs = {
                    "model": model,
                    "max_tokens": 4000,
                    "temperature": 0.1,
                    "messages": [{"role": "user", "content": formatted_prompt}]
                }
                
                
                # Add web search tool if requested
                if use_web_search:
                    anthropic_kwargs["tools"] = [{
                        "type": "web_search_20250305",
                        "name": "web_search",
                        "max_uses": 5
                    }]
                # Add extended thinking if enabled and supported
                # Note: Extended thinking is often enabled by prompt design rather than headers
                # if self.claude_extended_thinking and "claude" in model.lower():
                #     anthropic_kwargs["extra_headers"] = {"anthropic-beta": "thinking-2025-01-10"}
                
                # Call Anthropic directly
                response = await self.anthropic_client.messages.create(**anthropic_kwargs)
                
                # Extract the response content (handle tool use and thinking mode)
                output_text = ""
                
                # Handle responses with tool use (web search) and thinking mode
                for content_block in response.content:
                    if hasattr(content_block, 'type'):
                        if content_block.type == "text":
                            output_text = content_block.text
                        elif content_block.type == "tool_use":
                            # Skip tool use blocks, we want the final text response
                            continue
                
                # Fallback if no text found
                if not output_text:
                    # Try to get text from any content block that has text attribute
                    for content_block in response.content:
                        if hasattr(content_block, 'text'):
                            output_text = content_block.text
                            break
                
                # Log the raw response for debugging
                self.logger.debug(f"Raw Anthropic response: {output_text[:200]}...")
                
                # Handle empty or None response
                if not output_text or output_text.strip() == "":
                    raise LLMError("Empty response from Anthropic")
                
                return self._clean_json_response(output_text)
                
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    # Longer delays for web search operations due to rate limiting
                    delay = retry_delay * 2 if use_web_search else retry_delay
                    self.logger.warning(f"Anthropic call attempt {attempt + 1} failed: {str(e)}. Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                    retry_delay *= 1.5  # Exponential backoff
                else:
                    self.logger.error(f"Anthropic call failed after {self.max_retries + 1} attempts")
        
        raise LLMError(f"Anthropic call failed after retries: {str(last_error)}")
    
    async def _call_gateway_direct_http(self, model: str, prompt: str, use_web_search: bool = False) -> str:
        """Call the gateway directly via HTTP, bypassing LiteLLM."""
        last_error = None
        retry_delay = 1.0
        
        for attempt in range(self.max_retries + 1):
            try:
                # Add JSON format instruction to prompt
                formatted_prompt = f"{prompt}\n\nIMPORTANT: Respond with valid JSON only. Do not include any text before or after the JSON."
                
                # Prepare the request payload
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": formatted_prompt}],
                    "temperature": 1.0 if model.startswith("o") else 0.1,
                }
                
                # Add o3 effort level if this is an o3 model
                if model.startswith("o") and self.o3_effort in ["low", "medium", "high"]:
                    payload["reasoning_effort"] = self.o3_effort
                
                # Handle web search if requested
                if use_web_search:
                    # For all models via this gateway, add web search instruction to prompt
                    # The gateway appears to handle web search via prompt instructions rather than parameters
                    web_search_note = "\n\nIMPORTANT: Use your web search capabilities to find current, up-to-date information about this topic. Search for recent clinical trials, company announcements, scientific publications, and regulatory filings."
                    payload["messages"][0]["content"] = formatted_prompt + web_search_note
                
                # Prepare headers
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
                }
                
                # Make the HTTP request
                url = f"{self.gateway_url}chat/completions"
                async with httpx.AsyncClient(timeout=120.0) as client:
                    response = await client.post(url, json=payload, headers=headers)
                    
                    if response.status_code == 200:
                        result = response.json()
                        output_text = result["choices"][0]["message"]["content"]
                        
                        # Log the raw response for debugging
                        self.logger.debug(f"Raw gateway response: {output_text[:200]}...")
                        
                        # Handle empty or None response
                        if not output_text or output_text.strip() == "":
                            raise LLMError("Empty response from gateway")
                        
                        return self._clean_json_response(output_text)
                    else:
                        raise LLMError(f"Gateway returned HTTP {response.status_code}: {response.text}")
                
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    self.logger.warning(f"Gateway HTTP call attempt {attempt + 1} failed: {str(e)}. Retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 1.5  # Exponential backoff
                else:
                    self.logger.error(f"Gateway HTTP call failed after {self.max_retries + 1} attempts")
        
        raise LLMError(f"Gateway HTTP call failed after retries: {str(last_error)}")
    
    async def _call_litellm_gateway(self, model: str, prompt: str, use_web_search: bool = False) -> str:
        """Call LiteLLM gateway with retry logic for handling rate limits and transient errors."""
        last_error = None
        retry_delay = 1.0
        
        for attempt in range(self.max_retries + 1):
            try:
                # Add JSON format instruction to prompt
                formatted_prompt = f"{prompt}\n\nIMPORTANT: Respond with valid JSON only. Do not include any text before or after the JSON."
                
                # Prepare messages for LiteLLM
                messages = [{"role": "user", "content": formatted_prompt}]
                
                # Prepare kwargs for LiteLLM completion call
                # O-series models only support temperature=1
                temperature = 1.0 if model.startswith("o") else 0.1
                kwargs = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                }
                
                # Handle web search if requested
                if use_web_search:
                    # Check if model supports web search
                    if model in ["o3", "gpt-4.1"] or "gpt" in model.lower():
                        # Use LiteLLM web search for OpenAI models via search_responses API
                        kwargs["response_format"] = {"type": "json_object"}
                        # Add web search instruction to the prompt itself
                        web_search_instruction = "\n\nIMPORTANT: Search the web for current, up-to-date information about this topic. Use multiple search queries to find information from clinical trials, company announcements, scientific publications, and regulatory filings."
                        messages[0]["content"] = formatted_prompt + web_search_instruction
                        kwargs["messages"] = messages
                    else:
                        # For non-OpenAI models, add web search instruction to prompt
                        web_search_note = "\n\nIMPORTANT: Use your web search capabilities to find current, up-to-date information about this topic. Search for recent clinical trials, company announcements, scientific publications, and regulatory filings."
                        messages[0]["content"] = formatted_prompt + web_search_note
                        kwargs["messages"] = messages
                
                # Call LiteLLM completion with explicit api_base
                kwargs["api_base"] = self.gateway_url
                response = await litellm.acompletion(**kwargs, timeout=120)
                
                # Extract the response content
                output_text = response.choices[0].message.content
                
                # Log the raw response for debugging
                self.logger.debug(f"Raw LiteLLM response: {output_text[:200]}...")
                
                # Handle empty or None response
                if not output_text or output_text.strip() == "":
                    raise LLMError("Empty response from LiteLLM")
                
                return self._clean_json_response(output_text)
                
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    self.logger.warning(f"LiteLLM call attempt {attempt + 1} failed: {str(e)}. Retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 1.5  # Exponential backoff
                else:
                    self.logger.error(f"LiteLLM call failed after {self.max_retries + 1} attempts")
        
        raise LLMError(f"LiteLLM call failed after retries: {str(last_error)}")
    
    async def _call_litellm_gateway_messages(self, model: str, messages: List[Dict[str, Any]], use_web_search: bool = False) -> str:
        """Call LiteLLM gateway with messages format for web search support."""
        last_error = None
        retry_delay = 1.0
        
        for attempt in range(self.max_retries + 1):
            try:
                # Prepare kwargs for LiteLLM completion call
                temperature = 1.0 if model.startswith("o") else 0.1
                kwargs = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                }
                
                # Handle web search if requested
                if use_web_search:
                    # Check if model supports web search
                    if model in ["o3", "gpt-4.1"] or "gpt" in model.lower():
                        # Use LiteLLM web search for OpenAI models
                        kwargs["response_format"] = {"type": "json_object"}
                        # Add web search instruction to the last user message
                        web_search_instruction = "\n\nIMPORTANT: Search the web for current, up-to-date information about this topic. Use multiple search queries to find information from clinical trials, company announcements, scientific publications, and regulatory filings."
                        if messages and messages[-1]["role"] == "user":
                            messages[-1]["content"] += web_search_instruction
                        kwargs["messages"] = messages
                    else:
                        # For non-OpenAI models, add web search instruction
                        web_search_note = "\n\nIMPORTANT: Use your web search capabilities to find current, up-to-date information about this topic."
                        if messages and messages[-1]["role"] == "user":
                            messages[-1]["content"] += web_search_note
                        kwargs["messages"] = messages
                
                # Add JSON format instruction to last user message
                if messages and messages[-1]["role"] == "user":
                    messages[-1]["content"] += "\n\nIMPORTANT: Respond with valid JSON only. Do not include any text before or after the JSON."
                
                # Call LiteLLM completion with explicit api_base
                kwargs["api_base"] = self.gateway_url
                response = await litellm.acompletion(**kwargs, timeout=120)
                
                # Extract the response content
                output_text = response.choices[0].message.content
                
                # Log the raw response for debugging
                if use_web_search:
                    self.logger.info(f"LiteLLM web search response: {output_text[:500]}...")
                else:
                    self.logger.debug(f"Raw LiteLLM response: {output_text[:200]}...")
                
                # Handle empty or None response
                if not output_text or output_text.strip() == "":
                    raise LLMError("Empty response from LiteLLM")
                
                return self._clean_json_response(output_text)
                
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    # Longer delays for web search operations due to rate limiting
                    delay = retry_delay * 2 if use_web_search else retry_delay
                    self.logger.warning(f"LiteLLM messages call attempt {attempt + 1} failed: {str(e)}. Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                    retry_delay *= 1.5  # Exponential backoff
                else:
                    self.logger.error(f"LiteLLM messages call failed after {self.max_retries + 1} attempts")
        
        raise LLMError(f"LiteLLM messages call failed after retries: {str(last_error)}")
    
    async def _call_anthropic_direct_messages(self, model: str, messages: List[Dict[str, Any]], use_web_search: bool = False) -> str:
        """Call Anthropic API directly with messages format (supports caching)."""
        last_error = None
        retry_delay = 1.0
        
        for attempt in range(self.max_retries + 1):
            try:
                # Add thinking instruction if enabled
                if self.claude_extended_thinking and "claude" in model.lower():
                    if messages and messages[-1]["role"] == "user":
                        thinking_prompt = "<thinking>\nLet me carefully analyze this request and think through the best approach to provide an accurate, well-researched response.\n</thinking>\n\n"
                        messages[-1]["content"] = thinking_prompt + messages[-1]["content"]
                
                # Add JSON format instruction to last user message
                if messages and messages[-1]["role"] == "user":
                    messages[-1]["content"] += "\n\nIMPORTANT: Respond with valid JSON only. Do not include any text before or after the JSON."
                
                # Extract system messages and prepare kwargs for Anthropic messages call
                system_messages = []
                user_messages = []
                
                for message in messages:
                    if message["role"] == "system":
                        system_messages.append(message["content"])
                    else:
                        user_messages.append(message)
                
                anthropic_kwargs = {
                    "model": model,
                    "max_tokens": 4000,
                    "temperature": 0.1,
                    "messages": user_messages
                }
                
                # Add system content as separate parameter if present
                if system_messages:
                    # Combine all system content
                    if len(system_messages) == 1 and isinstance(system_messages[0], list):
                        anthropic_kwargs["system"] = system_messages[0]
                    elif len(system_messages) == 1 and isinstance(system_messages[0], str):
                        anthropic_kwargs["system"] = system_messages[0]
                    else:
                        # Multiple system messages - combine them
                        combined_system = []
                        for sys_msg in system_messages:
                            if isinstance(sys_msg, list):
                                combined_system.extend(sys_msg)
                            else:
                                combined_system.append({"type": "text", "text": str(sys_msg)})
                        anthropic_kwargs["system"] = combined_system
                
                # Add web search tool if requested
                if use_web_search:
                    anthropic_kwargs["tools"] = [{
                        "type": "web_search_20250305",
                        "name": "web_search",
                        "max_uses": 5
                    }]
                
                # Add extended thinking if enabled and supported
                # Note: Extended thinking is often enabled by prompt design rather than headers
                # if self.claude_extended_thinking and "claude" in model.lower():
                #     anthropic_kwargs["extra_headers"] = {"anthropic-beta": "thinking-2025-01-10"}
                
                # Call Anthropic directly with messages
                response = await self.anthropic_client.messages.create(**anthropic_kwargs)
                
                # Extract the response content (handle tool use and thinking mode)
                output_text = ""
                
                # Handle responses with tool use (web search) and thinking mode
                for content_block in response.content:
                    if hasattr(content_block, 'type'):
                        if content_block.type == "text":
                            output_text = content_block.text
                        elif content_block.type == "tool_use":
                            # Skip tool use blocks, we want the final text response
                            continue
                
                # Fallback if no text found
                if not output_text:
                    # Try to get text from any content block that has text attribute
                    for content_block in response.content:
                        if hasattr(content_block, 'text'):
                            output_text = content_block.text
                            break
                
                # Log the raw response for debugging
                self.logger.debug(f"Raw Anthropic response: {output_text[:200]}...")
                
                # Handle empty or None response
                if not output_text or output_text.strip() == "":
                    raise LLMError("Empty response from Anthropic")
                
                return self._clean_json_response(output_text)
                
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    # Longer delays for web search operations due to rate limiting
                    delay = retry_delay * 2 if use_web_search else retry_delay
                    self.logger.warning(f"Anthropic messages call attempt {attempt + 1} failed: {str(e)}. Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                    retry_delay *= 1.5  # Exponential backoff
                else:
                    self.logger.error(f"Anthropic messages call failed after {self.max_retries + 1} attempts")
        
        raise LLMError(f"Anthropic messages call failed after retries: {str(last_error)}")
    
    async def _call_gateway_direct_http_messages(self, model: str, messages: List[Dict[str, Any]], use_web_search: bool = False) -> str:
        """Call the gateway directly via HTTP with messages format (supports caching)."""
        last_error = None
        retry_delay = 1.0
        
        for attempt in range(self.max_retries + 1):
            try:
                # Add JSON format instruction to last user message
                if messages and messages[-1]["role"] == "user":
                    messages[-1]["content"] += "\n\nIMPORTANT: Respond with valid JSON only. Do not include any text before or after the JSON."
                
                # Prepare the request payload
                payload = {
                    "model": model,
                    "messages": messages,
                    "temperature": 1.0 if model.startswith("o") else 0.1,
                }
                
                # Add o3 effort level if this is an o3 model
                if model.startswith("o") and self.o3_effort in ["low", "medium", "high"]:
                    payload["reasoning_effort"] = self.o3_effort
                
                # Handle web search if requested
                if use_web_search:
                    # For all models via this gateway, add web search instruction to prompt
                    web_search_note = "\n\nIMPORTANT: Use your web search capabilities to find current, up-to-date information about this topic. Search for recent clinical trials, company announcements, scientific publications, and regulatory filings."
                    if messages and messages[-1]["role"] == "user":
                        messages[-1]["content"] += web_search_note
                
                # Prepare headers
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
                }
                
                # Make the HTTP request
                url = f"{self.gateway_url}chat/completions"
                async with httpx.AsyncClient(timeout=120.0) as client:
                    response = await client.post(url, json=payload, headers=headers)
                    
                    if response.status_code == 200:
                        result = response.json()
                        output_text = result["choices"][0]["message"]["content"]
                        
                        # Log the raw response for debugging
                        self.logger.debug(f"Raw gateway response: {output_text[:200]}...")
                        
                        # Handle empty or None response
                        if not output_text or output_text.strip() == "":
                            raise LLMError("Empty response from gateway")
                        
                        return self._clean_json_response(output_text)
                    else:
                        raise LLMError(f"Gateway returned HTTP {response.status_code}: {response.text}")
                
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    self.logger.warning(f"Gateway HTTP messages call attempt {attempt + 1} failed: {str(e)}. Retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 1.5  # Exponential backoff
                else:
                    self.logger.error(f"Gateway HTTP messages call failed after {self.max_retries + 1} attempts")
        
        raise LLMError(f"Gateway HTTP messages call failed after retries: {str(last_error)}")
    
    def _clean_json_response(self, output_text: str) -> str:
        """Clean and extract JSON from response text."""
        # Try to clean up the response and extract JSON
        output_text = output_text.strip()
        
        # Look for JSON content between code blocks or clean text
        if "```json" in output_text:
            start = output_text.find("```json") + 7
            end = output_text.find("```", start)
            if end != -1:
                output_text = output_text[start:end].strip()
        elif "```" in output_text:
            start = output_text.find("```") + 3
            end = output_text.find("```", start)
            if end != -1:
                output_text = output_text[start:end].strip()
        
        # Look for JSON objects in the text
        if not output_text.startswith('{'):
            # Try to find the first { and last }
            start = output_text.find('{')
            end = output_text.rfind('}')
            if start != -1 and end != -1 and end > start:
                output_text = output_text[start:end+1]
        
        return output_text

    # ================== BATCH PROCESSING METHODS ==================
    
    async def prompt_a_repurposing_batch(
        self, 
        assets: List[Dict[str, Any]],
        prefer_anthropic_batch: bool = False
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Batch version of prompt_a_repurposing for processing multiple assets efficiently.
        
        Args:
            assets: List of dicts with keys: asset_name, company_name, mechanism_of_action, primary_indication
            prefer_anthropic_batch: Whether to prefer Anthropic batch API when possible
            
        Returns:
            Dict mapping asset keys to their repurposing indications
        """
        if not assets:
            return {}
            
        self.logger.info(f"Processing batch repurposing for {len(assets)} assets")
        
        # Create batch requests
        batch_requests = []
        asset_keys = []
        
        for i, asset in enumerate(assets):
            asset_name = asset["asset_name"]
            company_name = asset["company_name"]
            mechanism_of_action = asset.get("mechanism_of_action")
            primary_indication = asset.get("primary_indication")
            
            asset_key = f"{asset_name}_{company_name}"
            asset_keys.append(asset_key)
            
            prompt = self._build_repurposing_prompt(asset_name, company_name, mechanism_of_action, primary_indication)
            
            batch_request = BatchRequest(
                custom_id=f"repurposing_{i:04d}_{asset_key}",
                model=self.models["prompt_a"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=1.0 if self.models["prompt_a"].startswith("o") else 0.0,
                metadata={"asset_name": asset_name, "company_name": company_name, "type": "repurposing"}
            )
            batch_requests.append(batch_request)
        
        # Process batch
        try:
            batch_responses = await self.batch_manager.process_batch(
                batch_requests, 
                prefer_anthropic_batch=prefer_anthropic_batch
            )
            
            # Parse results
            results = {}
            for i, response in enumerate(batch_responses):
                asset_key = asset_keys[i] if i < len(asset_keys) else f"unknown_{i}"
                
                if response.success and response.content:
                    try:
                        result = json.loads(self._clean_json_response(response.content))
                        indications = result.get("repurposing_indications", [])
                        
                        # Validate and format the response
                        formatted_indications = []
                        for indication in indications[:5]:  # Limit to 5 as per requirements
                            if isinstance(indication, dict) and "indication" in indication and "plausibility" in indication:
                                formatted_indications.append({
                                    "indication": indication["indication"],
                                    "plausibility": indication["plausibility"],
                                    "source": "repurposing"
                                })
                        
                        results[asset_key] = formatted_indications
                        self.logger.debug(f"Generated {len(formatted_indications)} repurposing indications for {asset_key}")
                        
                    except (json.JSONDecodeError, KeyError) as e:
                        self.logger.error(f"Failed to parse repurposing response for {asset_key}: {e}")
                        results[asset_key] = []
                else:
                    self.logger.error(f"Repurposing failed for {asset_key}: {response.error}")
                    results[asset_key] = []
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch repurposing failed: {e}")
            raise LLMError(f"Batch repurposing failed: {str(e)}")

    async def prompt_mechanism_of_action_lookup_batch(
        self, 
        assets: List[Dict[str, str]],
        prefer_anthropic_batch: bool = False
    ) -> Dict[str, str]:
        """
        Batch version of mechanism of action lookup for multiple assets.
        
        Args:
            assets: List of dicts with keys: asset_name, company_name
            prefer_anthropic_batch: Whether to prefer Anthropic batch API when possible
            
        Returns:
            Dict mapping asset keys to their mechanisms of action
        """
        if not assets:
            return {}
            
        self.logger.info(f"Processing batch MOA lookup for {len(assets)} assets")
        
        # Create batch requests
        batch_requests = []
        asset_keys = []
        
        for i, asset in enumerate(assets):
            asset_name = asset["asset_name"]
            company_name = asset["company_name"]
            
            asset_key = f"{asset_name}_{company_name}"
            asset_keys.append(asset_key)
            
            prompt = self._build_mechanism_of_action_prompt(asset_name, company_name)
            
            batch_request = BatchRequest(
                custom_id=f"moa_{i:04d}_{asset_key}",
                model=self.models["moa_lookup"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.1,
                metadata={"asset_name": asset_name, "company_name": company_name, "type": "moa_lookup"}
            )
            batch_requests.append(batch_request)
        
        # Process batch
        try:
            batch_responses = await self.batch_manager.process_batch(
                batch_requests, 
                prefer_anthropic_batch=prefer_anthropic_batch,
                max_wait_time=1800  # 30 minutes for web search tasks
            )
            
            # Parse results
            results = {}
            for i, response in enumerate(batch_responses):
                asset_key = asset_keys[i] if i < len(asset_keys) else f"unknown_{i}"
                
                if response.success and response.content:
                    try:
                        result = json.loads(self._clean_json_response(response.content))
                        
                        if "mechanism_of_action" not in result:
                            self.logger.error(f"Missing mechanism_of_action field for {asset_key}")
                            results[asset_key] = "Unknown"
                            continue
                        
                        mechanism_of_action = result["mechanism_of_action"].strip()
                        if not mechanism_of_action or mechanism_of_action.lower() in ['unknown', 'not found', 'unclear']:
                            self.logger.warning(f"Could not determine mechanism of action for {asset_key}")
                            results[asset_key] = "Unknown"
                        else:
                            results[asset_key] = mechanism_of_action
                            self.logger.debug(f"Found mechanism of action for {asset_key}: {mechanism_of_action}")
                        
                    except (json.JSONDecodeError, KeyError) as e:
                        self.logger.error(f"Failed to parse MOA response for {asset_key}: {e}")
                        results[asset_key] = "Unknown"
                else:
                    self.logger.error(f"MOA lookup failed for {asset_key}: {response.error}")
                    results[asset_key] = "Unknown"
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch MOA lookup failed: {e}")
            raise LLMError(f"Batch MOA lookup failed: {str(e)}")

    async def prompt_primary_indication_lookup_batch(
        self, 
        assets: List[Dict[str, str]],
        prefer_anthropic_batch: bool = False
    ) -> Dict[str, str]:
        """
        Batch version of primary indication lookup for multiple assets.
        
        Args:
            assets: List of dicts with keys: asset_name, company_name
            prefer_anthropic_batch: Whether to prefer Anthropic batch API when possible
            
        Returns:
            Dict mapping asset keys to their primary indications
        """
        if not assets:
            return {}
            
        self.logger.info(f"Processing batch primary indication lookup for {len(assets)} assets")
        
        # Create batch requests
        batch_requests = []
        asset_keys = []
        
        for i, asset in enumerate(assets):
            asset_name = asset["asset_name"]
            company_name = asset["company_name"]
            
            asset_key = f"{asset_name}_{company_name}"
            asset_keys.append(asset_key)
            
            prompt = self._build_primary_indication_prompt(asset_name, company_name)
            
            batch_request = BatchRequest(
                custom_id=f"primary_indication_{i:04d}_{asset_key}",
                model=self.models["primary_indication_lookup"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.1,
                metadata={"asset_name": asset_name, "company_name": company_name, "type": "primary_indication_lookup"}
            )
            batch_requests.append(batch_request)
        
        # Process batch
        try:
            batch_responses = await self.batch_manager.process_batch(
                batch_requests, 
                prefer_anthropic_batch=prefer_anthropic_batch,
                max_wait_time=1800  # 30 minutes for web search tasks
            )
            
            # Parse results
            results = {}
            for i, response in enumerate(batch_responses):
                asset_key = asset_keys[i] if i < len(asset_keys) else f"unknown_{i}"
                
                if response.success and response.content:
                    try:
                        result = json.loads(self._clean_json_response(response.content))
                        
                        if "primary_indication" not in result:
                            self.logger.error(f"Missing primary_indication field for {asset_key}")
                            results[asset_key] = "Unknown"
                            continue
                        
                        primary_indication = result["primary_indication"].strip()
                        if not primary_indication or primary_indication.lower() in ['unknown', 'not found', 'unclear']:
                            self.logger.warning(f"Could not determine primary indication for {asset_key}")
                            results[asset_key] = "Unknown"
                        else:
                            results[asset_key] = primary_indication
                            self.logger.debug(f"Found primary indication for {asset_key}: {primary_indication}")
                        
                    except (json.JSONDecodeError, KeyError) as e:
                        self.logger.error(f"Failed to parse primary indication response for {asset_key}: {e}")
                        results[asset_key] = "Unknown"
                else:
                    self.logger.error(f"Primary indication lookup failed for {asset_key}: {response.error}")
                    results[asset_key] = "Unknown"
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch primary indication lookup failed: {e}")
            raise LLMError(f"Batch primary indication lookup failed: {str(e)}")

    async def prompt_c_asset_screen_batch(
        self, 
        screening_requests: List[Dict[str, Any]],
        prefer_anthropic_batch: bool = True  # Default to True for screening since it's Claude
    ) -> Dict[str, Dict[str, Any]]:
        """
        Batch version of prompt_c_asset_screen for processing multiple asset-indication pairs.
        
        Args:
            screening_requests: List of dicts with keys: asset_name, company_name, indication, 
                              unmet_need_score, is_repurposing, mechanism_of_action
            prefer_anthropic_batch: Whether to prefer Anthropic batch API when possible
            
        Returns:
            Dict mapping request keys to their screening results
        """
        if not screening_requests:
            return {}
            
        self.logger.info(f"Processing batch asset screening for {len(screening_requests)} requests")
        
        # Create batch requests
        batch_requests = []
        request_keys = []
        
        for i, req in enumerate(screening_requests):
            asset_name = req["asset_name"]
            company_name = req["company_name"]
            indication = req["indication"]
            unmet_need_score = req["unmet_need_score"]
            is_repurposing = req.get("is_repurposing", False)
            mechanism_of_action = req.get("mechanism_of_action")
            
            request_key = f"{asset_name}_{company_name}_{indication}"
            request_keys.append(request_key)
            
            # Use messages format to support prompt caching
            model = self.models["prompt_c"]
            supports_caching = (self.enable_prompt_caching and 
                              supports_prompt_caching(f"anthropic/{model}") and 
                              "claude" in model.lower())
            
            if supports_caching:
                messages = self._build_screening_prompt_messages(
                    asset_name, company_name, indication, unmet_need_score, is_repurposing, mechanism_of_action
                )
                batch_request = BatchRequest(
                    custom_id=f"screening_{i:04d}_{request_key}",
                    model=model,
                    messages=messages,
                    max_tokens=4000,
                    temperature=0.1,
                    metadata={
                        "asset_name": asset_name, 
                        "company_name": company_name, 
                        "indication": indication, 
                        "type": "asset_screening"
                    }
                )
            else:
                prompt = self._build_screening_prompt(
                    asset_name, company_name, indication, unmet_need_score, is_repurposing, mechanism_of_action
                )
                batch_request = BatchRequest(
                    custom_id=f"screening_{i:04d}_{request_key}",
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=4000,
                    temperature=0.1,
                    metadata={
                        "asset_name": asset_name, 
                        "company_name": company_name, 
                        "indication": indication, 
                        "type": "asset_screening"
                    }
                )
            
            batch_requests.append(batch_request)
        
        # Process batch
        try:
            batch_responses = await self.batch_manager.process_batch(
                batch_requests, 
                prefer_anthropic_batch=prefer_anthropic_batch,
                max_wait_time=3600  # 1 hour for complex screening tasks
            )
            
            # Parse results
            results = {}
            for i, response in enumerate(batch_responses):
                request_key = request_keys[i] if i < len(request_keys) else f"unknown_{i}"
                
                if response.success and response.content:
                    try:
                        result = json.loads(self._clean_json_response(response.content))
                        
                        # Handle insufficient_data response
                        if result.get("insufficient_data", False):
                            standardized_result = {
                                "pursue": False,
                                "fail_reasons": ["insufficient_data"],
                                "info_confidence": int(result.get("info_confidence", 10)),
                                "rationale": result.get("reason", "Insufficient data available"),
                                "scoring": {},
                                "scoring_rationale": {}
                            }
                            
                            # Add None values for all scoring fields
                            scoring_fields = [
                                "novelty_differentiation_score", "unmet_medical_need_score", "development_stage_score",
                                "capital_efficiency_score", "peak_sales_potential_score", "ip_strength_duration_score",
                                "probability_technical_success_score", "competitive_landscape_score", "transactability_score",
                                "regulatory_path_complexity_score", "strategic_fit_score"
                            ]
                            rationale_fields = [
                                "novelty_differentiation_rationale", "unmet_medical_need_rationale", "development_stage_rationale",
                                "capital_efficiency_rationale", "peak_sales_potential_rationale", "ip_strength_duration_rationale",
                                "probability_technical_success_rationale", "competitive_landscape_rationale", "transactability_rationale",
                                "regulatory_path_complexity_rationale", "strategic_fit_rationale"
                            ]
                            
                            for field in scoring_fields:
                                standardized_result["scoring"][field] = None
                            for field in rationale_fields:
                                standardized_result["scoring_rationale"][field] = None
                            
                            results[request_key] = standardized_result
                            self.logger.info(f"Insufficient data response for {request_key}")
                            continue
                        
                        # Validate standard response fields
                        required_fields = ["pursue", "fail_reasons", "info_confidence", "rationale", "scoring", "scoring_rationale"]
                        missing_fields = [field for field in required_fields if field not in result]
                        if missing_fields:
                            self.logger.error(f"Missing required fields for {request_key}: {missing_fields}")
                            results[request_key] = self._create_error_response("Missing required fields")
                            continue
                        
                        # Validate and clean scoring fields
                        required_scoring_fields = [
                            "novelty_differentiation_score", "unmet_medical_need_score", "development_stage_score",
                            "capital_efficiency_score", "peak_sales_potential_score", "ip_strength_duration_score",
                            "probability_technical_success_score", "competitive_landscape_score", "transactability_score",
                            "regulatory_path_complexity_score", "strategic_fit_score"
                        ]
                        
                        for field in required_scoring_fields:
                            if field not in result["scoring"]:
                                self.logger.error(f"Missing scoring field {field} for {request_key}")
                                results[request_key] = self._create_error_response(f"Missing scoring field: {field}")
                                break
                            try:
                                score = int(result["scoring"][field])
                                result["scoring"][field] = max(0, min(10, score))
                            except (ValueError, TypeError):
                                self.logger.error(f"Invalid score value for {field} in {request_key}")
                                results[request_key] = self._create_error_response(f"Invalid score: {field}")
                                break
                        else:
                            # All scoring fields validated successfully
                            result["pursue"] = bool(result["pursue"])
                            result["fail_reasons"] = result["fail_reasons"] if isinstance(result["fail_reasons"], list) else []
                            result["info_confidence"] = int(result["info_confidence"])
                            
                            results[request_key] = result
                            self.logger.debug(f"Screening result for {request_key}: pursue={result['pursue']}")
                        
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        self.logger.error(f"Failed to parse screening response for {request_key}: {e}")
                        results[request_key] = self._create_error_response(f"Parse error: {str(e)}")
                else:
                    self.logger.error(f"Asset screening failed for {request_key}: {response.error}")
                    results[request_key] = self._create_error_response(f"API error: {response.error}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch asset screening failed: {e}")
            raise LLMError(f"Batch asset screening failed: {str(e)}")

    def _create_error_response(self, error_msg: str) -> Dict[str, Any]:
        """Create a standardized error response for failed screening requests."""
        return {
            "pursue": False,
            "fail_reasons": ["processing_error"],
            "info_confidence": 0,
            "rationale": f"Processing failed: {error_msg}",
            "scoring": {field: None for field in [
                "novelty_differentiation_score", "unmet_medical_need_score", "development_stage_score",
                "capital_efficiency_score", "peak_sales_potential_score", "ip_strength_duration_score",
                "probability_technical_success_score", "competitive_landscape_score", "transactability_score",
                "regulatory_path_complexity_score", "strategic_fit_score"
            ]},
            "scoring_rationale": {field: None for field in [
                "novelty_differentiation_rationale", "unmet_medical_need_rationale", "development_stage_rationale",
                "capital_efficiency_rationale", "peak_sales_potential_rationale", "ip_strength_duration_rationale",
                "probability_technical_success_rationale", "competitive_landscape_rationale", "transactability_rationale",
                "regulatory_path_complexity_rationale", "strategic_fit_rationale"
            ]}
        }

    async def close_batch_manager(self):
        """Cleanly close the batch manager and any background tasks."""
        if self.batch_manager:
            await self.batch_manager.close()
            self.logger.info("Batch manager closed.") 