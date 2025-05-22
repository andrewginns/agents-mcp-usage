import json
import os
import subprocess
import tempfile
import argparse
import asyncio
import sys
import time
from typing import Optional

from loguru import logger
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

# Configure loguru
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="DEBUG",
)
logger.add(
    "mermaid_validator.log",
    rotation="10 MB",
    retention="1 week",
    level="DEBUG",
)

# Add a file logger with more details for debugging
logger.add(
    "mermaid_raw_input.log",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}",
    level="DEBUG",
    filter=lambda record: "raw_input" in record["extra"],
    rotation="10 MB",
)

# Patch mcp.run to capture raw input
original_run = FastMCP.run


def patched_run(self, transport: str = "stdio", *args, **kwargs):
    logger.info(f"Starting MCP server with transport: {transport}")

    # Patch stdio handling if needed
    if transport == "stdio":
        # Store the original stdin.read
        original_stdin_read = sys.stdin.read
        original_stdin_readline = sys.stdin.readline

        def patched_read(n=-1):
            data = original_stdin_read(n)
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            logger.bind(raw_input=True).debug(
                f"[STDIN READ][{timestamp}] Length: {len(data)}, Data: {data}"
            )
            return data

        def patched_readline(size=-1):
            data = original_stdin_readline(size)
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            logger.bind(raw_input=True).debug(
                f"[STDIN READLINE][{timestamp}] Line: {data}"
            )
            return data

        # Patch stdin.read
        sys.stdin.read = patched_read
        sys.stdin.readline = patched_readline

    # Call the original run method
    return original_run(self, transport, *args, **kwargs)


# Apply the patch
FastMCP.run = patched_run

mcp = FastMCP("mermaid-validator")


class MermaidValidationResult(BaseModel):
    """Result of mermaid diagram validation."""

    is_valid: bool = Field(description="Whether the mermaid diagram is valid")
    error_message: Optional[str] = Field(
        None, description="Error message if the diagram is invalid"
    )


@mcp.tool()
async def validate_mermaid_diagram(diagram_text: str) -> MermaidValidationResult:
    """Validate a mermaid diagram.

    Uses mermaid-cli to validate an input string as a mermaid diagram.
    Expects input to be mermaid syntax only, no wrapping code blocks or ```mermaid tags.

    Args:
        diagram_text: The mermaid diagram text to validate

    Returns:
        A MermaidValidationResult object containing validation results
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    logger.info(f"MCP tool called: validate_mermaid_diagram at {timestamp}")
    logger.debug(f"Input diagram text (first 100 chars): {diagram_text[:100]}...")
    # Log the full diagram text to the raw input log
    logger.bind(raw_input=True).debug(
        f"[TOOL CALL][{timestamp}] Full diagram text: {diagram_text}"
    )

    temp_file_path = None
    puppeteer_config_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".mmd", mode="w", delete=False
        ) as temp_file:
            temp_file.write(diagram_text)
            temp_file_path = temp_file.name
            logger.debug(f"Created temporary file: {temp_file_path}")

        puppeteer_config = {"args": ["--no-sandbox", "--disable-setuid-sandbox"]}
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False
        ) as config_file:
            json.dump(puppeteer_config, config_file)
            puppeteer_config_path = config_file.name
            logger.debug(f"Created puppeteer config file: {puppeteer_config_path}")

        # Run mermaid-cli to validate input string as mermaid diagram
        logger.debug("Running mermaid-cli validation...")
        result = subprocess.run(
            [
                "npx",
                "-y",
                "@mermaid-js/mermaid-cli",
                "-i",
                temp_file_path,
                "--puppeteerConfigFile",
                puppeteer_config_path,
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            logger.info("Mermaid diagram validation successful")
            return MermaidValidationResult(is_valid=True, error_message=None)
        else:
            logger.warning(f"Mermaid diagram validation failed: {result.stderr}")
            logger.bind(raw_input=True).debug(
                f"[VALIDATION ERROR][{timestamp}] {result.stderr}"
            )
            return MermaidValidationResult(
                is_valid=False,
                error_message=f"Mermaid diagram is invalid: {result.stderr}",
            )
    except Exception as e:
        logger.error(f"Error validating mermaid diagram: {str(e)}")
        logger.bind(raw_input=True).debug(f"[EXCEPTION][{timestamp}] {str(e)}")
        return MermaidValidationResult(
            is_valid=False,
            error_message=f"Error validating mermaid diagram: {str(e)}",
        )
    finally:
        for file_path in [temp_file_path, puppeteer_config_path]:
            if file_path and os.path.exists(file_path):
                try:
                    os.unlink(file_path)
                    logger.debug(f"Deleted temporary file: {file_path}")
                except Exception as e:
                    logger.error(f"Error deleting temporary file {file_path}: {e}")


@mcp.resource("example://mermaid-diagram")
def get_example_mermaid_diagram():
    """Provides an example mermaid diagram for the client.

    Returns:
        Dict containing an example mermaid diagram
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    logger.info(f"Resource called: example://mermaid-diagram at {timestamp}")
    logger.bind(raw_input=True).debug(
        f"[RESOURCE CALL][{timestamp}] example://mermaid-diagram"
    )
    return """
    ```mermaid
    graph TD
        A[Start] --> B{Is it valid?}
        B -->|Yes| C[Output valid result]
        B -->|No| D[Output error message]
        C --> E[End]
        D --> E
    ```
    """


@mcp.prompt("validate-string-as-mermaid")
def validate_string_as_mermaid(diagram_text: str) -> MermaidValidationResult:
    """Validate a string as a mermaid diagram.

    Uses mermaid-cli to validate an input string as a mermaid diagram.
    Expects input to be mermaid syntax only, no wrapping code blocks or ```mermaid tags.
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    logger.info(f"Prompt called: validate-string-as-mermaid at {timestamp}")
    logger.debug(f"Original input (first 100 chars): {diagram_text[:100]}...")
    logger.bind(raw_input=True).debug(
        f"[PROMPT CALL][{timestamp}] Original full input: {diagram_text}"
    )

    # Strip whitespace, remove backticks and ```mermaid markers
    input_str = diagram_text.strip()

    # Remove ```mermaid and ``` markers
    if input_str.startswith("```mermaid"):
        input_str = input_str[len("```mermaid") :].strip()
        logger.debug("Removed ```mermaid prefix")
    if input_str.endswith("```"):
        input_str = input_str[:-3].strip()
        logger.debug("Removed ``` suffix")

    # Remove any remaining backticks
    input_str = input_str.replace("`", "")

    logger.debug(
        f"Cleaned input for validation (first 100 chars): {input_str[:100]}..."
    )
    logger.bind(raw_input=True).debug(
        f"[PROMPT CALL][{timestamp}] Cleaned full input: {input_str}"
    )

    # Validate the cleaned diagram
    return validate_mermaid_diagram(input_str)


if __name__ == "__main__":
    logger.info("Starting mermaid-validator MCP server")
    parser = argparse.ArgumentParser(description="Mermaid diagram validator")
    parser.add_argument(
        "--debug", action="store_true", help="Run validation on a placeholder diagram"
    )
    args = parser.parse_args()

    if args.debug:
        # Example diagram for debugging
        logger.info("Running in debug mode with example diagram")
        debug_diagram = get_example_mermaid_diagram()
        # Run the validation function
        result = asyncio.run(validate_string_as_mermaid(debug_diagram))
        logger.info(f"Debug validation result: {result}")
        print(f"Debug validation result: {result}")
    else:
        transport = os.getenv("MCP_TRANSPORT", "stdio")
        mcp.run(transport=transport)
