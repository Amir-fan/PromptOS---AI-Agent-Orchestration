"""
PromptOS Reporter Agent - Reasoning trace compilation and reporting.

This agent compiles reasoning traces from all agents and generates
transparent, comprehensive reports of the orchestration process.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from kernel.utils import get_openai_api_key, format_agent_response


async def run_reporter_agent(main_task: str, final_data: Dict[str, Any], agent_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Main reporter agent function that compiles final reports.
    
    Args:
        main_task: The main task description
        final_data: Dictionary containing final task data
        agent_responses: List of agent responses
        
    Returns:
        Dictionary containing the final report
    """
    logger = logging.getLogger(__name__)
    logger.info("Reporter agent compiling final report")
    
    try:
        # Use the provided parameters directly
        
        # Generate reasoning trace
        reasoning_trace = await _generate_reasoning_trace(agent_responses)
        
        # Generate executive summary
        executive_summary = await _generate_executive_summary(main_task, final_data, agent_responses)
        
        # Generate detailed report
        detailed_report = await _generate_detailed_report(main_task, final_data, agent_responses)
        
        # Generate transparency log
        transparency_log = await _generate_transparency_log(agent_responses)
        
        # Compile final report
        final_report = {
            "task": main_task,
            "executive_summary": executive_summary,
            "reasoning_trace": reasoning_trace,
            "detailed_report": detailed_report,
            "transparency_log": transparency_log,
            "metadata": {
                "report_generated_at": datetime.now().isoformat(),
                "total_agents_involved": len(agent_responses),
                "report_version": "1.0"
            }
        }
        
        logger.info("Reporter completed final report compilation")
        return format_agent_response("reporter", True, final_report, "Final report compiled successfully")
        
    except Exception as e:
        logger.error(f"Reporter agent failed: {str(e)}")
        return format_agent_response("reporter", False, {}, f"Reporting failed: {str(e)}")


async def _generate_reasoning_trace(agent_responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generate a step-by-step reasoning trace from agent responses.
    
    Args:
        agent_responses: List of agent response data
        
    Returns:
        List of reasoning trace steps
    """
    trace = []
    
    for i, response in enumerate(agent_responses):
        step = {
            "step_number": i + 1,
            "agent_id": response.get("agent_id", "unknown"),
            "timestamp": response.get("timestamp", datetime.now().isoformat()),
            "action": _describe_agent_action(response),
            "input": _extract_input_summary(response),
            "output": _extract_output_summary(response),
            "reasoning": response.get("reasoning", ""),
            "success": response.get("success", False)
        }
        trace.append(step)
    
    return trace


def _describe_agent_action(response: Dict[str, Any]) -> str:
    """
    Generate a human-readable description of what the agent did.
    
    Args:
        response: Agent response data
        
    Returns:
        Action description
    """
    agent_id = response.get("agent_id", "unknown")
    success = response.get("success", False)
    
    action_descriptions = {
        "planner": "Analyzed task and created execution plan",
        "executor": "Executed assigned subtasks",
        "critic": "Evaluated output quality and coherence",
        "ethics": "Performed safety and bias checks",
        "reporter": "Compiled final results and reasoning trace"
    }
    
    base_action = action_descriptions.get(agent_id, f"Processed task using {agent_id} agent")
    
    if not success:
        base_action += " (with errors)"
    
    return base_action


def _extract_input_summary(response: Dict[str, Any]) -> str:
    """
    Extract a summary of the agent's input.
    
    Args:
        response: Agent response data
        
    Returns:
        Input summary
    """
    result = response.get("result", {})
    
    if isinstance(result, dict):
        if "original_task" in result:
            return result["original_task"]
        elif "subtasks" in result:
            return f"Task decomposition: {len(result['subtasks'])} subtasks"
        elif "main_task" in result:
            return result["main_task"]
        else:
            return "Structured input data"
    else:
        return str(result)[:100] + "..." if len(str(result)) > 100 else str(result)


def _extract_output_summary(response: Dict[str, Any]) -> str:
    """
    Extract a summary of the agent's output.
    
    Args:
        response: Agent response data
        
    Returns:
        Output summary
    """
    result = response.get("result", {})
    
    if isinstance(result, dict):
        if "summary" in result:
            return result["summary"]
        elif "execution_result" in result:
            return result["execution_result"][:200] + "..." if len(result["execution_result"]) > 200 else result["execution_result"]
        elif "subtasks" in result:
            return f"Generated {len(result['subtasks'])} subtasks"
        else:
            return f"Generated {len(result)} data fields"
    else:
        return str(result)[:200] + "..." if len(str(result)) > 200 else str(result)


async def _generate_executive_summary(main_task: str, final_data: Dict[str, Any], 
                                    agent_responses: List[Dict[str, Any]]) -> str:
    """
    Generate an executive summary of the entire process.
    
    Args:
        main_task: Original main task
        final_data: Final processed data
        agent_responses: All agent responses
        
    Returns:
        Executive summary text
    """
    try:
        api_key = get_openai_api_key()
        if api_key:
            # Use OpenAI for sophisticated summary generation
            return await _generate_ai_summary(main_task, final_data, agent_responses, api_key)
    except Exception as e:
        logging.warning(f"AI summary generation failed, using fallback: {str(e)}")
    
    # Fallback to rule-based summary
    return _generate_fallback_summary(main_task, final_data, agent_responses)


async def _generate_ai_summary(main_task: str, final_data: Dict[str, Any], 
                              agent_responses: List[Dict[str, Any]], api_key: str) -> str:
    """
    Generate executive summary using OpenAI API.
    
    Args:
        main_task: Original main task
        final_data: Final processed data
        agent_responses: All agent responses
        api_key: OpenAI API key
        
    Returns:
        AI-generated executive summary
    """
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        
        # Prepare data for summary
        summary_data = {
            "main_task": main_task,
            "agent_count": len(agent_responses),
            "successful_agents": sum(1 for r in agent_responses if r.get("success", False)),
            "final_data_keys": list(final_data.keys()) if isinstance(final_data, dict) else []
        }
        
        prompt = f"""
        Generate a concise executive summary for an AI orchestration system that processed the following task:
        
        Task: {main_task}
        Agents Involved: {summary_data['agent_count']}
        Successful Agents: {summary_data['successful_agents']}
        Final Data: {summary_data['final_data_keys']}
        
        The summary should:
        1. Briefly describe what was accomplished
        2. Highlight key findings or results
        3. Note any issues or limitations
        4. Be professional and concise (2-3 paragraphs max)
        """
        
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert at creating executive summaries for AI system outputs. Be concise and professional."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.3
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        logging.error(f"AI summary generation failed: {str(e)}")
        return _generate_fallback_summary(main_task, final_data, agent_responses)


def _generate_fallback_summary(main_task: str, final_data: Dict[str, Any], 
                              agent_responses: List[Dict[str, Any]]) -> str:
    """
    Generate fallback executive summary using rule-based approach.
    
    Args:
        main_task: Original main task
        final_data: Final processed data
        agent_responses: All agent responses
        
    Returns:
        Fallback executive summary
    """
    successful_agents = sum(1 for r in agent_responses if r.get("success", False))
    total_agents = len(agent_responses)
    
    summary = f"""
    Executive Summary
    
    Task: {main_task}
    
    The PromptOS orchestration system successfully processed the requested task using {total_agents} specialized AI agents. 
    {successful_agents} out of {total_agents} agents completed their assigned tasks successfully.
    
    The system employed a multi-agent approach with specialized roles including planning, execution, quality evaluation, 
    and safety checking to ensure comprehensive and reliable task completion.
    """
    
    if successful_agents < total_agents:
        summary += f"\n\nNote: {total_agents - successful_agents} agents encountered issues during execution, but the overall process completed."
    
    return summary.strip()


async def _generate_detailed_report(main_task: str, final_data: Dict[str, Any], 
                                  agent_responses: List[Dict[str, Any]]) -> str:
    """
    Generate a detailed technical report.
    
    Args:
        main_task: Original main task
        final_data: Final processed data
        agent_responses: All agent responses
        
    Returns:
        Detailed report text
    """
    report = f"""
    Detailed Technical Report
    
    Task: {main_task}
    Processing Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    Agent Performance Summary:
    """
    
    for response in agent_responses:
        agent_id = response.get("agent_id", "unknown")
        success = response.get("success", False)
        reasoning = response.get("reasoning", "")
        
        report += f"""
    - {agent_id.upper()}: {'✓ Success' if success else '✗ Failed'}
      Reasoning: {reasoning[:100]}{'...' if len(reasoning) > 100 else ''}
        """
    
    report += f"""
    
    Final Data Structure:
    {json.dumps(final_data, indent=2) if isinstance(final_data, dict) else str(final_data)}
    
    System Metadata:
    - Total Agents: {len(agent_responses)}
    - Successful Executions: {sum(1 for r in agent_responses if r.get('success', False))}
    - Processing Method: Multi-agent orchestration with quality assurance
    """
    
    return report.strip()


async def _generate_transparency_log(agent_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate a transparency log for audit and debugging purposes.
    
    Args:
        agent_responses: All agent responses
        
    Returns:
        Transparency log data
    """
    log = {
        "system_info": {
            "log_generated_at": datetime.now().isoformat(),
            "total_interactions": len(agent_responses),
            "system_version": "PromptOS 0.1.0"
        },
        "agent_interactions": [],
        "performance_metrics": {
            "success_rate": 0,
            "average_processing_time": 0,
            "error_count": 0
        }
    }
    
    successful_count = 0
    error_count = 0
    
    for response in agent_responses:
        interaction = {
            "agent_id": response.get("agent_id", "unknown"),
            "timestamp": response.get("timestamp", datetime.now().isoformat()),
            "success": response.get("success", False),
            "input_hash": hash(str(response.get("result", {}))),
            "output_hash": hash(str(response.get("reasoning", ""))),
            "processing_metadata": {
                "data_size": len(str(response.get("result", {}))),
                "reasoning_length": len(response.get("reasoning", ""))
            }
        }
        log["agent_interactions"].append(interaction)
        
        if response.get("success", False):
            successful_count += 1
        else:
            error_count += 1
    
    # Calculate metrics
    log["performance_metrics"]["success_rate"] = successful_count / len(agent_responses) if agent_responses else 0
    log["performance_metrics"]["error_count"] = error_count
    
    return log


class AdvancedReporterAgent:
    """
    Advanced reporter agent for backward compatibility.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def generate_report(self, main_task: str, final_data: Dict[str, Any], 
                            agent_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate report using the existing run_reporter_agent function."""
        return await run_reporter_agent(main_task, final_data, agent_responses)
    
    async def compile_results(self, main_task: str, final_data: Dict[str, Any], 
                            agent_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compile results - alias for generate_report for backward compatibility."""
        return await self.generate_report(main_task, final_data, agent_responses)
