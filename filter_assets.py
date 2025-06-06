#!/usr/bin/env python3
"""
Asset-Indication Filtering Pipeline
Main CLI entry point for processing asset lists through AI-assisted filtering.
"""

import click
import sys
import logging
import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
import time

from src.parsers.input_parser import InputParser, InputParserError
from src.pipeline.pipeline_orchestrator import PipelineOrchestrator
from src.output.spreadsheet_generator import SpreadsheetGenerator, SpreadsheetGeneratorError
from src.data.unmet_need_lookup import UnmetNeedLookupError


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def validate_environment():
    """Validate required environment variables and dependencies."""
    required_env_vars = ['OPENAI_API_KEY']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        click.echo(f"Error: Missing required environment variables: {', '.join(missing_vars)}", err=True)
        click.echo("Please set these in your .env file or environment.", err=True)
        return False
    
    # Show proxy configuration if set
    proxy_url = os.getenv("LITELLM_PROXY_URL")
    if proxy_url:
        click.echo(f"Using LiteLLM proxy: {proxy_url}")
    
    return True


@click.command()
@click.argument('input_file', type=click.Path(exists=True, path_type=Path))
@click.option('--dry-run', is_flag=True, help='Output CSV preview to stdout instead of modifying file')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--unmet-need-csv', type=click.Path(exists=True, path_type=Path), 
              help='Path to unmet need lookup CSV file')
@click.option('--max-assets', type=int, help='Limit processing to first N assets (for testing)')
def main(input_file: Path, dry_run: bool, verbose: bool, unmet_need_csv: Path = None, max_assets: int = None):
    """
    Process asset list through AI-assisted filtering pipeline.
    
    INPUT_FILE: Path to Excel (.xlsx) or CSV file containing Asset Name and Company Name columns.
    
    The pipeline will:
    1. Parse and validate the input file
    2. Generate repurposing indications for each asset (Prompt A)
    3. Look up unmet need scores for all indications (Prompt B)
    4. Screen each asset-indication pair for pursue/not-pursue decision (Prompt C)
    5. Generate filtered output with conditional formatting
    
    Required: OPENAI_API_KEY environment variable must be set.
    """
    setup_logging(verbose)
    load_dotenv()
    
    logger = logging.getLogger(__name__)
    
    # Validate environment
    if not validate_environment():
        sys.exit(1)
    
    # Set default unmet need CSV if not provided
    if unmet_need_csv is None:
        default_csv = Path(__file__).parent / "data" / "unmet_need_lookup.csv"
        if default_csv.exists():
            unmet_need_csv = default_csv
            logger.info(f"Using default unmet need lookup: {unmet_need_csv}")
        else:
            logger.warning("No unmet need lookup CSV provided and default not found")
    
    logger.info(f"Starting asset filtering pipeline for {input_file}")
    start_time = time.time()
    
    try:
        # Step 1: Parse and validate input file
        click.echo("📋 Parsing input file...")
        parser = InputParser()
        df = parser.parse_file(input_file)
        
        # Apply max_assets limit if specified
        if max_assets and max_assets < len(df):
            df = df.head(max_assets)
            click.echo(f"⚠️  Limited to first {max_assets} assets for testing")
        
        summary = parser.get_summary(df)
        logger.info(f"Parsed input: {summary}")
        
        click.echo(f"✅ Parsed {summary['total_rows']} assets from {summary['unique_companies']} companies")
        
        if dry_run:
            click.echo("\n=== DRY RUN MODE ===")
            click.echo(f"Input file: {input_file}")
            click.echo(f"Total rows: {summary['total_rows']}")
            click.echo(f"Unique assets: {summary['unique_assets']}")
            click.echo(f"Unique companies: {summary['unique_companies']}")
            click.echo("\nFirst 5 rows:")
            click.echo(df.head().to_string())
            
            # Show what would be processed
            click.echo(f"\n🔄 Would process through 3-stage LLM pipeline:")
            click.echo(f"   • Prompt A: Generate repurposing indications")
            click.echo(f"   • Prompt B: Look up unmet need scores")
            click.echo(f"   • Prompt C: Screen for pursue/not-pursue decisions")
            click.echo(f"   • Generate filtered Excel output with conditional formatting")
            return
        
        # Step 2: Initialize pipeline orchestrator
        click.echo("🤖 Initializing AI pipeline...")
        try:
            orchestrator = PipelineOrchestrator(
                unmet_need_csv_path=str(unmet_need_csv) if unmet_need_csv else None
            )
        except UnmetNeedLookupError as e:
            click.echo(f"⚠️  Warning: Unmet need lookup failed: {e}")
            click.echo("Continuing without unmet need scoring...")
            orchestrator = PipelineOrchestrator()
        
        # Step 3: Process through pipeline
        click.echo(f"🔄 Processing {len(df)} assets through 3-stage pipeline...")
        click.echo("   This may take several minutes...")
        
        async def run_pipeline():
            return await orchestrator.process_assets(df)
        
        # Run the async pipeline
        results = asyncio.run(run_pipeline())
        
        # Step 4: Generate output
        click.echo("📊 Generating filtered output...")
        generator = SpreadsheetGenerator()
        generator.generate_filtered_sheet(input_file, results)
        
        # Step 5: Show summary statistics
        stats = generator.get_summary_stats(results)
        processing_time = time.time() - start_time
        
        click.echo("\n✅ Pipeline completed successfully!")
        click.echo(f"⏱️  Processing time: {processing_time:.1f} seconds")
        click.echo(f"📈 Results summary:")
        click.echo(f"   • Total asset-indication pairs: {stats.get('total_asset_indication_pairs', 0)}")
        click.echo(f"   • Primary indications: {stats.get('primary_indications', 0)}")
        click.echo(f"   • Repurposing indications: {stats.get('repurposing_indications', 0)}")
        click.echo(f"   • Pursue recommendations: {stats.get('pursue_count', 0)} ({stats.get('pursue_rate_percent', 0)}%)")
        click.echo(f"   • Don't pursue: {stats.get('dont_pursue_count', 0)}")
        click.echo(f"   • Errors: {stats.get('error_count', 0)} ({stats.get('error_rate_percent', 0)}%)")
        
        if input_file.suffix.lower() == '.csv':
            output_file = input_file.parent / f"{input_file.stem}_filtered.xlsx"
            click.echo(f"📄 Output saved to: {output_file}")
        else:
            click.echo(f"📄 Added 'Filtered' sheet to: {input_file}")
        
        # Success metrics check
        if stats.get('pursue_rate_percent', 0) > 15:
            click.echo("⚠️  Warning: Pursue rate >15% - consider reviewing screening criteria")
        if stats.get('error_rate_percent', 0) > 5:
            click.echo("⚠️  Warning: Error rate >5% - check API connectivity and asset data quality")
            
    except InputParserError as e:
        logger.error(f"Input parsing failed: {e}")
        click.echo(f"❌ Input parsing error: {e}", err=True)
        click.echo("💡 Check that your file has 'Asset Name' and 'Company Name' columns", err=True)
        sys.exit(1)
    except SpreadsheetGeneratorError as e:
        logger.error(f"Output generation failed: {e}")
        click.echo(f"❌ Output generation error: {e}", err=True)
        sys.exit(1)
    except KeyboardInterrupt:
        click.echo("\n⚠️  Pipeline interrupted by user", err=True)
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        click.echo(f"❌ Unexpected error: {e}", err=True)
        click.echo("💡 Run with --verbose for detailed error information", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()