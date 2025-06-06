## Relevant Files

- filter_assets.py - Main CLI script entry point
- src/parsers/input_parser.py - Handles parsing and validation
- src/llm/prompt_manager.py - Manages the three LLM prompts
- src/data/unmet_need_lookup.py - Handles fuzzy matching and lookup
- src/pipeline/pipeline_orchestrator.py - Orchestrates the three-stage pipeline
- src/output/spreadsheet_generator.py - Generates the filtered output
- data/unmet_need_lookup.csv - Sample unmet need scores lookup table
- requirements.txt - Project dependencies
- .env.example - Environment variable template
- README.md - Complete setup and usage documentation

## Tasks

- [x] 1.0 Set up project structure and dependencies
- [x] 2.0 Implement input parsing and validation system
- [x] 3.0 Build the three-prompt LLM pipeline
  - [x] 3.1 Implement Prompt A - Repurposing Enumerator
  - [x] 3.2 Implement Prompt B - Unmet Need Lookup with fuzzy matching
  - [x] 3.3 Implement Prompt C - Asset Screening with web search
  - [x] 3.4 Build pipeline orchestration with rate limiting and error handling
- [x] 4.0 Create spreadsheet output generation system
- [x] 5.0 Develop CLI interface and comprehensive error handling