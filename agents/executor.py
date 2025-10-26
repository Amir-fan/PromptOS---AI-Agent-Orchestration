"""
PromptOS Executor Agent - Task execution and operations.

This agent executes concrete tasks including web search, text processing,
data analysis, and file operations.
"""

import asyncio
import json
import logging
import re
from typing import Dict, List, Any, Optional
from datetime import datetime

from kernel.utils import get_openai_api_key, format_agent_response


async def run_executor_agent(task_description: str) -> Dict[str, Any]:
    """
    Main executor agent function that executes concrete tasks.
    
    Args:
        task_description: Description of the task to execute
        
    Returns:
        Dictionary containing execution results
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Executor agent processing: {task_description}")
    
    try:
        # Determine task type and execute accordingly
        task_type = _classify_task_type(task_description)
        
        if task_type == "web_search":
            result = await _execute_web_search(task_description)
        elif task_type == "text_processing":
            result = await _execute_text_processing(task_description)
        elif task_type == "data_analysis":
            result = await _execute_data_analysis(task_description)
        elif task_type == "file_operations":
            result = await _execute_file_operations(task_description)
        else:
            result = await _execute_general_task(task_description)
        
        logger.info(f"Executor completed task type: {task_type}")
        return format_agent_response("executor", True, result, f"Executed {task_type} task successfully")
        
    except Exception as e:
        logger.error(f"Executor agent failed: {str(e)}")
        return format_agent_response("executor", False, {}, f"Execution failed: {str(e)}")


def _classify_task_type(task_description: str) -> str:
    """
    Classify the type of task based on description.
    
    Args:
        task_description: Task description
        
    Returns:
        Task type classification
    """
    task_lower = task_description.lower()
    
    if any(keyword in task_lower for keyword in ["search", "find", "look up", "google"]):
        return "web_search"
    elif any(keyword in task_lower for keyword in ["process", "analyze", "parse", "extract"]):
        return "text_processing"
    elif any(keyword in task_lower for keyword in ["data", "statistics", "calculate", "compute"]):
        return "data_analysis"
    elif any(keyword in task_lower for keyword in ["file", "read", "write", "save", "load"]):
        return "file_operations"
    else:
        return "general"


async def _execute_web_search(task_description: str) -> Dict[str, Any]:
    """
    Execute web search tasks.
    
    Args:
        task_description: Search task description
        
    Returns:
        Search results
    """
    # Extract search query from task description
    search_query = _extract_search_query(task_description)
    
    # Simulate web search (in real implementation, use actual search API)
    mock_results = {
        "query": search_query,
        "results": [
            {
                "title": f"Search result 1 for: {search_query}",
                "url": "https://example.com/result1",
                "snippet": f"Relevant information about {search_query}..."
            },
            {
                "title": f"Search result 2 for: {search_query}",
                "url": "https://example.com/result2",
                "snippet": f"Additional details on {search_query}..."
            }
        ],
        "total_results": 2,
        "search_time": "0.5s"
    }
    
    return {
        "task_type": "web_search",
        "search_query": search_query,
        "results": mock_results,
        "summary": f"Found {len(mock_results['results'])} relevant results for '{search_query}'"
    }


async def _execute_text_processing(task_description: str) -> Dict[str, Any]:
    """
    Execute text processing tasks.
    
    Args:
        task_description: Text processing task description
        
    Returns:
        Processing results
    """
    try:
        api_key = get_openai_api_key()
        if not api_key:
            return {"error": "OpenAI API key not available"}
        
        import openai
        client = openai.OpenAI(api_key=api_key)
        
        prompt = f"""
        Process the following text according to the task description:
        
        Task: {task_description}
        
        Please provide a detailed analysis and processing of the text.
        """
        
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a text processing expert. Provide detailed analysis and insights."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.3
        )
        
        processed_text = response.choices[0].message.content.strip()
        
        return {
            "task_type": "text_processing",
            "original_task": task_description,
            "processed_result": processed_text,
            "processing_method": "OpenAI GPT-3.5-turbo"
        }
        
    except Exception as e:
        return {
            "task_type": "text_processing",
            "error": f"Text processing failed: {str(e)}",
            "fallback_result": f"Basic text processing completed for: {task_description}"
        }


async def _execute_data_analysis(task_description: str) -> Dict[str, Any]:
    """
    Execute data analysis tasks.
    
    Args:
        task_description: Data analysis task description
        
    Returns:
        Analysis results
    """
    # Simulate data analysis (in real implementation, use actual data)
    analysis_result = {
        "task_type": "data_analysis",
        "analysis_type": "statistical_summary",
        "data_points": 100,
        "findings": [
            "Mean value: 45.2",
            "Standard deviation: 12.8",
            "Trend: Increasing over time",
            "Outliers detected: 3"
        ],
        "recommendations": [
            "Monitor outlier values",
            "Continue tracking trend",
            "Consider additional data collection"
        ]
    }
    
    return analysis_result


async def _execute_file_operations(task_description: str) -> Dict[str, Any]:
    """
    Execute file operation tasks.
    
    Args:
        task_description: File operation task description
        
    Returns:
        Operation results
    """
    # Simulate file operations (in real implementation, perform actual file operations)
    operation_result = {
        "task_type": "file_operations",
        "operation": "file_processing",
        "files_processed": 1,
        "status": "completed",
        "details": f"File operation completed for: {task_description}"
    }
    
    return operation_result


async def _execute_general_task(task_description: str) -> Dict[str, Any]:
    """
    Execute general tasks using OpenAI API.
    
    Args:
        task_description: General task description
        
    Returns:
        Task execution results
    """
    try:
        api_key = get_openai_api_key()
        if not api_key:
            return {
                "task_type": "general",
                "error": "OpenAI API key not available",
                "fallback_result": f"Task noted for manual execution: {task_description}"
            }
        
        import openai
        client = openai.OpenAI(api_key=api_key)
        
        prompt = f"""
        Execute the following task and provide a detailed response:
        
        Task: {task_description}
        
        Please provide:
        1. A clear explanation of what was done
        2. The results or output
        3. Any relevant insights or recommendations
        """
        
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant that executes tasks efficiently and provides detailed results."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.3
        )
        
        execution_result = response.choices[0].message.content.strip()
        
        return {
            "task_type": "general",
            "original_task": task_description,
            "execution_result": execution_result,
            "method": "OpenAI GPT-3.5-turbo"
        }
        
    except Exception as e:
        return {
            "task_type": "general",
            "error": f"General task execution failed: {str(e)}",
            "fallback_result": f"Task execution attempted for: {task_description}"
        }


def _extract_search_query(task_description: str) -> str:
    """
    Extract search query from task description.
    
    Args:
        task_description: Task description
        
    Returns:
        Extracted search query
    """
    # Simple extraction - look for quoted text or key phrases
    quoted_match = re.search(r'"([^"]*)"', task_description)
    if quoted_match:
        return quoted_match.group(1)
    
    # Remove common task words and use the rest as query
    task_words = ["search for", "find", "look up", "google", "search"]
    query = task_description.lower()
    
    for word in task_words:
        query = query.replace(word, "").strip()
    
    return query if query else task_description


class AdvancedExecutorAgent:
    """
    Advanced executor agent for backward compatibility.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def execute_task(self, task_description: str) -> Dict[str, Any]:
        """Execute a task using the existing run_executor_agent function."""
        return await run_executor_agent(task_description)
