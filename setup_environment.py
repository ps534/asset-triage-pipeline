#!/usr/bin/env python3
"""
Setup script to help configure environment variables for the asset screening pipeline.
"""

import os
from pathlib import Path

def setup_environment():
    """Interactive setup for environment variables."""
    
    print("=== Asset Screening Pipeline Setup ===")
    print()
    print("This script will help you set up the required environment variables.")
    print("You can also create a .env file or set these variables in your shell.")
    print()
    
    # Check current configuration
    current_vars = {
        "LITELLM_GATEWAY_URL": os.getenv("LITELLM_GATEWAY_URL"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
        "USE_DIRECT_ANTHROPIC": os.getenv("USE_DIRECT_ANTHROPIC"),
        "PROMPT_A_MODEL": os.getenv("PROMPT_A_MODEL"),
        "PROMPT_C_MODEL": os.getenv("PROMPT_C_MODEL"),
    }
    
    print("Current Configuration:")
    for key, value in current_vars.items():
        status = "✓ Set" if value else "❌ Not set"
        masked_value = "***" if value and "KEY" in key else value
        print(f"  {key}: {status} {f'({masked_value})' if masked_value else ''}")
    
    print()
    
    # Generate .env file content
    env_content = f"""# LiteLLM Gateway Configuration
LITELLM_GATEWAY_URL=https://llmgateway.experiment.trialspark.com/

# Model Configuration
PROMPT_A_MODEL=o3
PROMPT_C_MODEL=claude-sonnet-4-20250514
MOA_LOOKUP_MODEL=claude-sonnet-4-20250514
PRIMARY_INDICATION_MODEL=claude-sonnet-4-20250514

# API Access Configuration
USE_DIRECT_ANTHROPIC=true
ANTHROPIC_API_KEY=your-anthropic-key-here
OPENAI_API_KEY=your-openai-key-here

# Web Search Configuration
WEB_SEARCH_CONTEXT_SIZE=medium
"""
    
    # Write .env file
    env_file = Path(".env")
    if env_file.exists():
        print("⚠ .env file already exists. Backing up to .env.backup")
        env_file.rename(".env.backup")
    
    with open(env_file, "w") as f:
        f.write(env_content)
    
    print(f"✓ Created .env file at {env_file.absolute()}")
    print()
    print("Next steps:")
    print("1. Edit the .env file and add your actual API keys:")
    print("   - Replace 'your-openai-key-here' with your OpenAI API key")
    print("   - Replace 'your-anthropic-key-here' with your Anthropic API key")
    print()
    print("2. Install dependencies:")
    print("   pip install -r requirements.txt")
    print()
    print("3. Test the connection:")
    print("   python test_litellm_connection.py")
    print()
    print("4. Test the full pipeline:")
    print("   python test_pipeline_integration.py")
    print()
    
    # Show example of loading .env
    print("To load the .env file in Python:")
    print("""
from dotenv import load_dotenv
load_dotenv()  # This loads the .env file
""")

if __name__ == "__main__":
    setup_environment()