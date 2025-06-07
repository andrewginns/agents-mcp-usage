import os
import tempfile
import subprocess
import base64
import json
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

mcp = FastMCP("mermaid-validator")


class MermaidValidationResult(BaseModel):
    """Result of mermaid diagram validation."""

    is_valid: bool = Field(description="Whether the mermaid diagram is valid")
    error_message: Optional[str] = Field(
        None, description="Error message if the diagram is invalid"
    )
    diagram_image: Optional[str] = Field(
        None, description="Base64-encoded PNG image of the rendered diagram if valid"
    )


@mcp.tool()
async def validate_mermaid_diagram(diagram_text: str) -> MermaidValidationResult:
    """
    Validate a mermaid diagram and render it as a PNG image if valid.

    Uses mermaid-cli to validate and render the diagram. Requires mermaid-cli
    to be installed globally via npm: npm install -g @mermaid-js/mermaid-cli

    Args:
        diagram_text: The mermaid diagram text to validate

    Returns:
        A MermaidValidationResult object containing validation results
    """
    temp_file_path = None
    output_file_name = None
    puppeteer_config_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".mmd", mode="w", delete=False
        ) as temp_file:
            temp_file.write(diagram_text)
            temp_file_path = temp_file.name

        output_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        output_file.close()
        output_file_name = output_file.name

        puppeteer_config = {"args": ["--no-sandbox", "--disable-setuid-sandbox"]}
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False
        ) as config_file:
            json.dump(puppeteer_config, config_file)
            puppeteer_config_path = config_file.name

        result = subprocess.run(
            [
                "npx",
                "-y",
                "@mermaid-js/mermaid-cli",
                "-i",
                temp_file_path,
                "-o",
                output_file_name,
                "--puppeteerConfigFile",
                puppeteer_config_path,
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            with open(output_file_name, "rb") as f:
                diagram_image = base64.b64encode(f.read()).decode("utf-8")

            return MermaidValidationResult(
                is_valid=True, error_message=None, diagram_image=diagram_image
            )
        else:
            return MermaidValidationResult(
                is_valid=False,
                error_message=f"Mermaid diagram is invalid: {result.stderr}",
                diagram_image=None,
            )
    except Exception as e:
        return MermaidValidationResult(
            is_valid=False,
            error_message=f"Error validating mermaid diagram: {str(e)}",
            diagram_image=None,
        )
    finally:
        for file_path in [temp_file_path, output_file_name, puppeteer_config_path]:
            if file_path and os.path.exists(file_path):
                try:
                    os.unlink(file_path)
                except OSError:
                    # Ignore file deletion errors during cleanup
                    pass


@mcp.resource("example://mermaid-diagram")
def get_example_mermaid_diagram():
    """
    Provides an example mermaid diagram for the client.

    Returns:
        Dict containing an example mermaid diagram
    """
    return {
        "diagram": """
        graph TD
            A[Start] --> B{Is it valid?}
            B -->|Yes| C[Output valid result]
            B -->|No| D[Output error message]
            C --> E[End]
            D --> E
        """
    }


if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "stdio")

    mcp.run(transport=transport)
