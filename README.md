# Asset-Indication Filtering Pipeline

An AI-assisted pipeline for triaging clinical-stage assets to identify high-potential opportunities for business development.

## Overview

This pipeline processes spreadsheets containing asset/company information through four AI-powered stages:

1. **Timeline Research**: Generate comprehensive development timeline using web search
2. **Prompt A**: Generate up to 5 plausible repurposing indications per asset
3. **Prompt B**: Look up unmet need scores using fuzzy matching 
4. **Prompt C**: Screen each asset-indication pair for pursue/not-pursue decisions

The goal is to achieve a ~10% keep-rate while maintaining <1% false-negative errors.

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env and add your OPENAI_API_KEY
   # Optionally set LITELLM_PROXY_URL if using a proxy server
   ```

3. **Prepare unmet need lookup table** (optional):
   - Use the provided `data/unmet_need_lookup.csv` or
   - Create your own CSV with columns: `indication`, `unmet_need_score`

## Usage

### Basic Usage
```bash
python filter_assets.py input_file.xlsx
```

### Dry Run (Preview Only)
```bash
python filter_assets.py input_file.xlsx --dry-run
```

### With Custom Options
```bash
python filter_assets.py input_file.xlsx \
    --unmet-need-csv custom_lookup.csv \
    --max-assets 10 \
    --verbose
```

### Input File Requirements

Your input file (Excel or CSV) must contain columns:
- `Asset Name`: Name of the clinical asset
- `Company Name`: Company developing the asset

Example:
```csv
Asset Name,Company Name
Aducanumab,Biogen
Keytruda,Merck
```

## Output

The pipeline generates a new "Filtered" worksheet (or Excel file for CSV inputs) with columns:

- **Asset Name**: Original asset name
- **Company**: Company name
- **Development Timeline**: Comprehensive development history and milestones
- **Indication**: Primary or repurposed indication
- **Pursue**: Yes/No/Error recommendation
- **Fail Reasons**: Why asset was rejected (if applicable)
- **Degree of Unmet Need**: High/Medium/Low based on lookup table
- **Repurposing**: Yes if from repurposing prompt
- **Rationale**: Brief explanation of decision
- **Error**: Error details if processing failed

The output includes conditional formatting:
- 🟢 Green: Pursue = Yes
- 🔴 Red: Pursue = No  
- 🟡 Yellow: Pursue = Error

## Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `--dry-run` | Preview mode, no file modification | False |
| `--verbose` | Enable detailed logging | False |
| `--unmet-need-csv` | Path to custom unmet need lookup | `data/unmet_need_lookup.csv` |
| `--max-assets` | Limit processing (for testing) | None |

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `OPENAI_API_KEY` | OpenAI API key for LLM calls | Yes | - |
| `LITELLM_PROXY_URL` | LiteLLM proxy server URL | No | - |
| `LITELLM_LOG` | LiteLLM logging level | No | - |
| `TIMELINE_MODEL` | Model for development timeline research | No | `claude-sonnet-4-20250514` |
| `PROMPT_A_MODEL` | Model for repurposing generation | No | `gpt-4o-mini` |
| `PROMPT_C_MODEL` | Model for asset screening | No | `gpt-4o` |

## Performance

- **Rate Limiting**: ≤15 concurrent LLM calls
- **Expected Throughput**: ~7000 assets in ≤4 hours
- **Target Metrics**: 8-12% pursue rate, <2% error rate

### Model Selection Strategy

The pipeline uses different models optimized for each task:

- **Timeline Research**: `claude-sonnet-4-20250514` by default
  - Comprehensive web search for development history and milestones
  - Claude's web search capabilities provide detailed timeline research
  
- **Prompt A (Repurposing)**: `gpt-4o-mini` by default
  - Generates creative repurposing ideas based on known mechanisms and timeline
  - Smaller model is cost-effective for this creative task
  
- **Prompt C (Asset Screening)**: `gpt-4o` by default  
  - Complex business development reasoning with timeline context
  - Requires stronger reasoning capabilities for accurate screening

This approach balances **cost efficiency** with **quality** - using a smaller model for simpler tasks while ensuring the critical screening step uses the most capable model.

## Architecture

```
src/
├── parsers/
│   └── input_parser.py          # Excel/CSV parsing & validation
├── llm/
│   └── prompt_manager.py        # Three LLM prompts with retry logic
├── data/
│   └── unmet_need_lookup.py     # Fuzzy matching for indications
├── pipeline/
│   └── pipeline_orchestrator.py # Async orchestration & rate limiting
└── output/
    └── spreadsheet_generator.py # Excel output with formatting
```

## Error Handling

The pipeline includes comprehensive error handling:

- **Input validation**: Checks for required columns and file format
- **Environment validation**: Ensures API keys are set
- **LLM retries**: Automatic retry on rate limits/timeouts
- **Graceful degradation**: Continues processing if some assets fail
- **Detailed logging**: Use `--verbose` for troubleshooting

## Development

### Running Tests
```bash
# Create a small test file
echo "Asset Name,Company Name\nTest Drug,Test Co" > test_input.csv

# Test with dry run
python filter_assets.py test_input.csv --dry-run --max-assets 1
```

### Customizing Prompts

Edit the prompt templates in `src/llm/prompt_manager.py`:
- `_build_repurposing_prompt()`: Modify repurposing criteria
- `_build_screening_prompt()`: Adjust screening criteria

### Adding New Unmet Need Data

Update `data/unmet_need_lookup.csv` with your custom indication scores (1-10 scale).

## Troubleshooting

**"Missing required environment variables"**
- Ensure `OPENAI_API_KEY` is set in your `.env` file

**"Missing required columns"** 
- Check that your input file has `Asset Name` and `Company Name` columns

**High error rates (>5%)**
- Check API connectivity and rate limits
- Verify asset names are valid/recognizable

**Low pursue rates (<5%)**
- Review screening criteria in prompts
- Check if unmet need scores are appropriate

## License

This project is for internal business development use.