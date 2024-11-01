import click
import uvicorn
import os
import sys
from pathlib import Path

@click.group()
def cli():
    """DeepSeek Vision-Language Model CLI"""
    pass

@cli.command()
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=8000, help='Port to bind to')
@click.option('--model-path', envvar='DEEPSEEK_MODEL_PATH', help='Path to model checkpoint')
def api(host: str, port: int, model_path: str):
    """Start the DeepSeek-VL API server"""
    if not model_path:
        click.echo("Error: DEEPSEEK_MODEL_PATH environment variable or --model-path must be set")
        sys.exit(1)
        
    # Add the current directory to Python path to find api.py
    module_dir = Path(__file__).parent
    sys.path.insert(0, str(module_dir))
    
    # Set environment variable for the API
    os.environ['DEEPSEEK_MODEL_PATH'] = model_path
    
    click.echo(f"Starting API server on {host}:{port}")
    click.echo(f"Using model from: {model_path}")
    
    uvicorn.run("api:app", host=host, port=port, reload=False)

if __name__ == '__main__':
    cli() 