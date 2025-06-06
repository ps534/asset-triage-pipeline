"""
Batch processing manager for LLM requests supporting both LiteLLM and Anthropic batching.

This module provides:
1. LiteLLM multi-model batching (immediate response to multiple models)
2. Anthropic Message Batches API (asynchronous batch processing) 
3. Intelligent batch grouping and optimization
4. Fallback to individual requests when batching fails
"""

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import anthropic
import httpx
import litellm
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

logger = logging.getLogger(__name__)


class BatchType(Enum):
    """Supported batch processing types."""
    LITELLM_MULTI_MODEL = "litellm_multi_model"
    ANTHROPIC_MESSAGE_BATCH = "anthropic_message_batch"
    INDIVIDUAL_FALLBACK = "individual_fallback"


class BatchStatus(Enum):
    """Batch processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class BatchRequest:
    """Individual request within a batch."""
    custom_id: str
    model: str
    messages: List[Dict[str, Any]]
    system: Optional[str] = None
    max_tokens: int = 2000
    temperature: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchResponse:
    """Response for an individual request within a batch."""
    custom_id: str
    success: bool
    content: Optional[str] = None
    error: Optional[str] = None
    model_used: Optional[str] = None
    tokens_used: Optional[int] = None
    processing_time: Optional[float] = None


@dataclass
class BatchJob:
    """Represents a batch processing job."""
    batch_id: str
    batch_type: BatchType
    requests: List[BatchRequest]
    status: BatchStatus = BatchStatus.PENDING
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    responses: List[BatchResponse] = field(default_factory=list)
    anthropic_batch_id: Optional[str] = None
    error: Optional[str] = None


class BatchManager:
    """
    Manages batch processing for LLM requests with support for multiple batching strategies.
    
    Features:
    - LiteLLM multi-model batching for immediate responses
    - Anthropic Message Batches API for cost-effective async processing
    - Intelligent batch grouping by model and request type
    - Automatic fallback to individual requests
    - Retry logic and error handling
    """
    
    def __init__(
        self,
        gateway_url: str,
        anthropic_api_key: str,
        max_batch_size: int = 100,
        max_concurrent_batches: int = 5,
        anthropic_batch_enabled: bool = True,
        litellm_multi_model_enabled: bool = True,
    ):
        self.gateway_url = gateway_url
        self.anthropic_api_key = anthropic_api_key
        self.max_batch_size = max_batch_size
        self.max_concurrent_batches = max_concurrent_batches
        self.anthropic_batch_enabled = anthropic_batch_enabled
        self.litellm_multi_model_enabled = litellm_multi_model_enabled
        
        # Initialize clients with shorter timeouts
        self.anthropic_client = anthropic.AsyncAnthropic(api_key=anthropic_api_key, http_client=httpx.AsyncClient(verify=False, timeout=60.0))
        self.httpx_client = httpx.AsyncClient(timeout=60.0, verify=False)
        
        # Configure LiteLLM
        litellm.api_base = gateway_url
        litellm.drop_params = True
        
        # Disable SSL verification for LiteLLM async calls
        import ssl
        litellm.ssl_verify = False
        # Set global SSL context to not verify certificates
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        litellm.ssl_context = ssl_context
        
        # Track active batches
        self.active_batches: Dict[str, BatchJob] = {}
        self.batch_semaphore = asyncio.Semaphore(max_concurrent_batches)
        
        # Rate limiting for individual requests
        self.request_semaphore = asyncio.Semaphore(1)  # Process one request at a time to avoid rate limits
        
        # Supported models for different batch types
        self.anthropic_models = {
            "claude-opus-4-20250514",
            "claude-sonnet-4-20250514", 
            "claude-3-7-sonnet-20250219",
            "claude-3-5-sonnet-20240620",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-haiku-20240307",
            "claude-3-opus-20240229"
        }
        
        logger.info(f"BatchManager initialized with gateway: {gateway_url}")
        logger.info(f"Anthropic batching: {anthropic_batch_enabled}, LiteLLM multi-model: {litellm_multi_model_enabled}")

    async def process_batch(
        self,
        requests: List[BatchRequest],
        prefer_anthropic_batch: bool = False,
        max_wait_time: int = 3600  # 1 hour default
    ) -> List[BatchResponse]:
        """
        Process a batch of requests using the most appropriate batching strategy.
        
        Args:
            requests: List of BatchRequest objects to process
            prefer_anthropic_batch: Whether to prefer Anthropic batch API when possible
            max_wait_time: Maximum time to wait for async batch processing (seconds)
            
        Returns:
            List of BatchResponse objects with results
        """
        if not requests:
            return []
            
        batch_id = f"batch_{uuid.uuid4().hex[:8]}"
        logger.info(f"Processing batch {batch_id} with {len(requests)} requests")
        
        # Group requests by optimal batching strategy
        batch_groups = self._group_requests_for_batching(requests, prefer_anthropic_batch)
        
        all_responses = []
        
        for batch_type, group_requests in batch_groups.items():
            if not group_requests:
                continue
                
            logger.info(f"Processing {len(group_requests)} requests using {batch_type.value}")
            
            try:
                if batch_type == BatchType.ANTHROPIC_MESSAGE_BATCH:
                    responses = await self._process_anthropic_batch(group_requests, max_wait_time)
                elif batch_type == BatchType.LITELLM_MULTI_MODEL:
                    responses = await self._process_litellm_multi_model_batch(group_requests)
                else:  # INDIVIDUAL_FALLBACK
                    responses = await self._process_individual_requests(group_requests)
                    
                all_responses.extend(responses)
                
            except Exception as e:
                logger.error(f"Batch processing failed for {batch_type.value}: {e}")
                # Fallback to individual processing
                logger.info(f"Falling back to individual processing for {len(group_requests)} requests")
                fallback_responses = await self._process_individual_requests(group_requests)
                all_responses.extend(fallback_responses)
        
        # Sort responses by original request order
        request_order = {req.custom_id: i for i, req in enumerate(requests)}
        all_responses.sort(key=lambda r: request_order.get(r.custom_id, float('inf')))
        
        logger.info(f"Batch {batch_id} completed with {len(all_responses)} responses")
        return all_responses

    def _group_requests_for_batching(
        self,
        requests: List[BatchRequest],
        prefer_anthropic_batch: bool
    ) -> Dict[BatchType, List[BatchRequest]]:
        """
        Group requests by optimal batching strategy based on models and preferences.
        """
        groups = {
            BatchType.ANTHROPIC_MESSAGE_BATCH: [],
            BatchType.LITELLM_MULTI_MODEL: [],
            BatchType.INDIVIDUAL_FALLBACK: []
        }
        
        # Analyze request characteristics
        anthropic_requests = []
        other_requests = []
        
        for request in requests:
            if request.model in self.anthropic_models:
                anthropic_requests.append(request)
            else:
                other_requests.append(request)
        
        # Decision logic for batching strategy
        if (prefer_anthropic_batch and 
            self.anthropic_batch_enabled and 
            len(anthropic_requests) >= 5):  # Minimum batch size for efficiency
            groups[BatchType.ANTHROPIC_MESSAGE_BATCH] = anthropic_requests
            groups[BatchType.INDIVIDUAL_FALLBACK] = other_requests
            
        elif (self.litellm_multi_model_enabled and 
              len(other_requests) > 0):
            # Use LiteLLM multi-model for mixed model batches
            unique_models = list(set(req.model for req in other_requests))
            if len(unique_models) > 1 and len(other_requests) <= 20:  # Good for multi-model
                groups[BatchType.LITELLM_MULTI_MODEL] = other_requests
                groups[BatchType.INDIVIDUAL_FALLBACK] = anthropic_requests
            else:
                groups[BatchType.INDIVIDUAL_FALLBACK] = requests
        else:
            # Default to individual processing
            groups[BatchType.INDIVIDUAL_FALLBACK] = requests
            
        return groups

    async def _process_anthropic_batch(
        self,
        requests: List[BatchRequest],
        max_wait_time: int
    ) -> List[BatchResponse]:
        """Process requests using Anthropic Message Batches API."""
        
        # Convert to Anthropic batch format
        anthropic_requests = []
        for req in requests:
            messages = [{"role": msg["role"], "content": msg["content"]} for msg in req.messages]
            
            params = MessageCreateParamsNonStreaming(
                model=req.model,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                messages=messages
            )
            
            if req.system:
                # Anthropic batch API requires system to be a list, not a string
                if isinstance(req.system, str):
                    params["system"] = [{"type": "text", "text": req.system}]
                else:
                    params["system"] = req.system
                
            anthropic_requests.append(Request(
                custom_id=req.custom_id,
                params=params
            ))
        
        logger.info(f"Creating Anthropic batch with {len(anthropic_requests)} requests")
        
        # Create batch
        message_batch = await self.anthropic_client.messages.batches.create(
            requests=anthropic_requests
        )
        
        batch_id = message_batch.id
        logger.info(f"Created Anthropic batch {batch_id}, status: {message_batch.processing_status}")
        
        # Poll for completion
        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            batch_status = await self.anthropic_client.messages.batches.retrieve(batch_id)
            
            if batch_status.processing_status == "ended":
                logger.info(f"Anthropic batch {batch_id} completed")
                break
            elif batch_status.processing_status in ["failed", "canceled", "expired"]:
                raise Exception(f"Anthropic batch failed with status: {batch_status.processing_status}")
                
            await asyncio.sleep(30)  # Poll every 30 seconds
        else:
            raise Exception(f"Anthropic batch {batch_id} timed out after {max_wait_time} seconds")
        
        # Retrieve results
        responses = []
        async for result in self.anthropic_client.messages.batches.results(batch_id):
            if result.result.type == "succeeded":
                content = result.result.message.content[0].text if result.result.message.content else ""
                tokens_used = (result.result.message.usage.input_tokens + 
                             result.result.message.usage.output_tokens)
                
                responses.append(BatchResponse(
                    custom_id=result.custom_id,
                    success=True,
                    content=content,
                    model_used=result.result.message.model,
                    tokens_used=tokens_used
                ))
            else:
                error_msg = getattr(result.result, 'error', {}).get('message', 'Unknown error')
                responses.append(BatchResponse(
                    custom_id=result.custom_id,
                    success=False,
                    error=error_msg
                ))
        
        return responses

    async def _process_litellm_multi_model_batch(
        self,
        requests: List[BatchRequest]
    ) -> List[BatchResponse]:
        """Process requests using LiteLLM multi-model batching."""
        
        if not requests:
            return []
            
        # Group by unique prompts for multi-model calls
        prompt_groups = {}
        for req in requests:
            prompt_key = json.dumps({
                "messages": req.messages,
                "system": req.system,
                "max_tokens": req.max_tokens,
                "temperature": req.temperature
            }, sort_keys=True)
            
            if prompt_key not in prompt_groups:
                prompt_groups[prompt_key] = {
                    "models": [],
                    "custom_ids": [],
                    "base_request": req
                }
            prompt_groups[prompt_key]["models"].append(req.model)
            prompt_groups[prompt_key]["custom_ids"].append(req.custom_id)
        
        responses = []
        
        for prompt_key, group in prompt_groups.items():
            models_str = ",".join(group["models"])
            base_req = group["base_request"]
            
            try:
                # Use LiteLLM multi-model call
                litellm_response = await litellm.acompletion(
                    model=models_str,
                    messages=base_req.messages,
                    max_tokens=base_req.max_tokens,
                    temperature=base_req.temperature,
                    api_base=self.gateway_url
                )
                
                # litellm returns a list when multiple models are specified
                if isinstance(litellm_response, list):
                    for i, response in enumerate(litellm_response):
                        if i < len(group["custom_ids"]):
                            content = response.choices[0].message.content if response.choices else ""
                            tokens_used = getattr(response.usage, 'total_tokens', None)
                            
                            responses.append(BatchResponse(
                                custom_id=group["custom_ids"][i],
                                success=True,
                                content=content,
                                model_used=response.model,
                                tokens_used=tokens_used
                            ))
                else:
                    # Single response - assign to first custom_id
                    content = litellm_response.choices[0].message.content if litellm_response.choices else ""
                    tokens_used = getattr(litellm_response.usage, 'total_tokens', None)
                    
                    responses.append(BatchResponse(
                        custom_id=group["custom_ids"][0],
                        success=True,
                        content=content,
                        model_used=litellm_response.model,
                        tokens_used=tokens_used
                    ))
                    
            except Exception as e:
                logger.error(f"LiteLLM multi-model call failed: {e}")
                # Add error responses for all requests in this group
                for custom_id in group["custom_ids"]:
                    responses.append(BatchResponse(
                        custom_id=custom_id,
                        success=False,
                        error=str(e)
                    ))
        
        return responses

    async def _process_individual_requests(
        self,
        requests: List[BatchRequest]
    ) -> List[BatchResponse]:
        """Process requests individually as fallback."""
        
        async def process_single_request(request: BatchRequest) -> BatchResponse:
            try:
                # Add timeout to prevent hanging requests
                timeout_seconds = 45
                
                # Use appropriate client based on model
                if request.model in self.anthropic_models:
                    response = await asyncio.wait_for(
                        self.anthropic_client.messages.create(
                            model=request.model,
                            max_tokens=request.max_tokens,
                            temperature=request.temperature,
                            messages=request.messages,
                            system=request.system
                        ),
                        timeout=timeout_seconds
                    )
                    
                    content = response.content[0].text if response.content else ""
                    tokens_used = response.usage.input_tokens + response.usage.output_tokens
                    
                    return BatchResponse(
                        custom_id=request.custom_id,
                        success=True,
                        content=content,
                        model_used=response.model,
                        tokens_used=tokens_used
                    )
                else:
                    # Use LiteLLM for other models
                    response = await asyncio.wait_for(
                        litellm.acompletion(
                            model=request.model,
                            messages=request.messages,
                            max_tokens=request.max_tokens,
                            temperature=request.temperature,
                            api_base=self.gateway_url
                        ),
                        timeout=timeout_seconds
                    )
                    
                    content = response.choices[0].message.content if response.choices else ""
                    tokens_used = getattr(response.usage, 'total_tokens', None)
                    
                    return BatchResponse(
                        custom_id=request.custom_id,
                        success=True,
                        content=content,
                        model_used=response.model,
                        tokens_used=tokens_used
                    )
                    
            except Exception as e:
                logger.error(f"Individual request {request.custom_id} failed: {e}")
                return BatchResponse(
                    custom_id=request.custom_id,
                    success=False,
                    error=str(e)
                )
        
        # Process with concurrency limit using the class semaphore
        async def process_with_semaphore(request):
            async with self.request_semaphore:  # Use existing rate limiting
                result = await process_single_request(request)
                # Small delay to be gentle on the gateway
                await asyncio.sleep(0.5)
                return result
        
        responses = await asyncio.gather(*[
            process_with_semaphore(req) for req in requests
        ])
        
        return responses

    async def create_batch_requests(
        self,
        prompts_data: List[Dict[str, Any]],
        model: str,
        custom_id_prefix: str = "req"
    ) -> List[BatchRequest]:
        """
        Helper method to create BatchRequest objects from prompt data.
        
        Args:
            prompts_data: List of dicts with 'messages', optional 'system', etc.
            model: Model to use for all requests
            custom_id_prefix: Prefix for generating custom IDs
            
        Returns:
            List of BatchRequest objects
        """
        requests = []
        
        for i, data in enumerate(prompts_data):
            # Fix custom_id format: no spaces, alphanumeric + underscore/dash only
            safe_id = f"{custom_id_prefix}_{i:04d}_{int(time.time())}"
            safe_id = safe_id.replace(" ", "_").replace("(", "").replace(")", "")[:64]
            
            request = BatchRequest(
                custom_id=safe_id,
                model=model,
                messages=data["messages"],
                system=data.get("system"),
                max_tokens=data.get("max_tokens", 2000),
                temperature=data.get("temperature", 0.0),
                metadata=data.get("metadata", {})
            )
            
            requests.append(request)
        
        return requests

    async def close(self):
        """Clean up resources."""
        await self.httpx_client.aclose()
        await self.anthropic_client.close()