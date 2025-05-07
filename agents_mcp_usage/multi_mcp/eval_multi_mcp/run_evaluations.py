#!/usr/bin/env python
import asyncio
import json
import os
from datetime import datetime
from typing import Dict, List

import logfire
import pandas as pd
from dotenv import load_dotenv

from multi_mcp.eval_multi_mcp.evals_pydantic_mcp import (
    EvalResult,
    InputPrompt,
    evaluate_result,
    main,
)
from multi_mcp.mermaid_diagrams import invalid_mermaid_diagram, valid_mermaid_diagram

# Load environment variables
load_dotenv()

# Configure logging
logfire.configure(
    send_to_logfire="if-token-present", service_name="pydantic-multi-mcp-evals"
)


# Define test cases
TEST_CASES = [
    {
        "name": "fix_mermaid_add_time",
        "query": f"Add the current time and fix the mermaid diagram syntax using the validator: {invalid_mermaid_diagram}. Return only the fixed mermaid diagram between backticks.",
        "description": "Basic test case to fix mermaid diagram and add time",
    },
    {
        "name": "mermaid_only",
        "query": f"Fix this mermaid diagram syntax using the validator: {invalid_mermaid_diagram}. Return only the fixed mermaid diagram between backticks.",
        "description": "Tests if the agent uses only the mermaid validator MCP",
    },
    {
        "name": "time_only",
        "query": "What's the current time? Use the time tool to get it.",
        "description": "Tests if the agent uses only the time MCP tool",
    },
    {
        "name": "complex_task",
        "query": f"""Create a mermaid diagram showing the flow of this application. 
                  Include the current time somewhere in the diagram. 
                  Start with this diagram as a base and improve it: {invalid_mermaid_diagram}
                  Make sure to validate your diagram with the mermaid validator.""",
        "description": "Complex task requiring both MCPs and creativity",
    },
]


async def run_single_evaluation(test_case: Dict) -> Dict:
    """
    Run a single evaluation test case and return the results

    Args:
        test_case: Dictionary containing the test case details

    Returns:
        Dictionary with test results
    """
    try:
        query = InputPrompt(question=test_case["query"])
        start_time = datetime.now()

        # Run the agent
        result = await main(query, request_limit=10)

        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds()

        # Evaluate the result
        eval_result = await evaluate_result(result)

        # Add metadata to results
        return {
            "test_case": test_case["name"],
            "description": test_case["description"],
            "execution_time": execution_time,
            "result": result,
            **eval_result.dict(),  # Unpack the evaluation results
        }
    except Exception as e:
        logfire.error(f"Error running test case {test_case['name']}: {e}")
        return {
            "test_case": test_case["name"],
            "description": test_case["description"],
            "execution_time": -1,
            "result": f"ERROR: {str(e)}",
            "used_both_mcps": False,
            "mermaid_diagram_valid": False,
            "diagram_similarity_score": 0.0,
            "overall_score": 0.0,
            "error": str(e),
        }


async def run_all_evaluations() -> List[Dict]:
    """
    Run all evaluation test cases and return the results

    Returns:
        List of dictionaries with test results
    """
    results = []
    for test_case in TEST_CASES:
        logfire.info(f"Running test case: {test_case['name']}")
        result = await run_single_evaluation(test_case)
        results.append(result)
        logfire.info(
            f"Completed test case: {test_case['name']} with score: {result['overall_score']:.2f}"
        )
    return results


def save_results(results: List[Dict], output_dir: str = "evaluation_results"):
    """
    Save evaluation results to files

    Args:
        results: List of dictionaries with test results
        output_dir: Directory to save results in
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save as JSON
    json_path = os.path.join(output_dir, f"eval_results_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save as CSV
    csv_path = os.path.join(output_dir, f"eval_results_{timestamp}.csv")
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)

    # Generate summary
    summary = {
        "timestamp": timestamp,
        "num_test_cases": len(results),
        "avg_overall_score": sum(r["overall_score"] for r in results) / len(results),
        "avg_execution_time": sum(
            r["execution_time"] for r in results if r["execution_time"] > 0
        )
        / len(results),
        "success_rate": sum(1 for r in results if r.get("error") is None)
        / len(results),
        "scores_by_test": {r["test_case"]: r["overall_score"] for r in results},
    }

    summary_path = os.path.join(output_dir, f"eval_summary_{timestamp}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logfire.info(f"Results saved to {output_dir}")
    return json_path, csv_path, summary_path


async def main_eval():
    """Main evaluation function"""
    logfire.info("Starting evaluations")
    results = await run_all_evaluations()
    json_path, csv_path, summary_path = save_results(results)

    # Print summary to console
    with open(summary_path, "r") as f:
        summary = json.load(f)

    print("\n===== EVALUATION SUMMARY =====")
    print(f"Test cases run: {summary['num_test_cases']}")
    print(f"Average score: {summary['avg_overall_score']:.2f}")
    print(f"Success rate: {summary['success_rate']:.2%}")
    print(f"Average execution time: {summary['avg_execution_time']:.2f} seconds")
    print("\nScores by test case:")
    for test, score in summary["scores_by_test"].items():
        print(f"  - {test}: {score:.2f}")
    print(f"\nDetailed results saved to:\n{json_path}\n{csv_path}")

    return results


if __name__ == "__main__":
    asyncio.run(main_eval())
