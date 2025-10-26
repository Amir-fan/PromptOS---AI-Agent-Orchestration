"""
PromptOS Critic Agent - Quality evaluation and assessment.

This agent evaluates the quality, coherence, and completeness of outputs
from other agents and provides improvement suggestions.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from kernel.utils import get_openai_api_key, format_agent_response


async def run_critic_agent(agent_outputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main critic agent function that evaluates agent outputs.
    
    Args:
        agent_outputs: Dictionary containing outputs from all agents
        
    Returns:
        Dictionary containing evaluation results and suggestions
    """
    logger = logging.getLogger(__name__)
    logger.info("Critic agent evaluating agent outputs")
    
    try:
        # Extract main task and results
        main_task = agent_outputs.get("main_task", "")
        results = agent_outputs.get("results", [])
        
        # Evaluate each result
        evaluations = []
        overall_score = 0
        
        for result in results:
            evaluation = await _evaluate_single_result(result)
            evaluations.append(evaluation)
            overall_score += evaluation.get("score", 0)
        
        # Calculate average score
        avg_score = overall_score / len(evaluations) if evaluations else 0
        
        # Generate overall assessment
        overall_assessment = await _generate_overall_assessment(evaluations, main_task)
        
        # Create improvement suggestions
        suggestions = await _generate_improvement_suggestions(evaluations, avg_score)
        
        critic_result = {
            "overall_score": avg_score,
            "individual_evaluations": evaluations,
            "overall_assessment": overall_assessment,
            "improvement_suggestions": suggestions,
            "evaluation_timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Critic completed evaluation with overall score: {avg_score:.2f}")
        return format_agent_response("critic", True, critic_result, f"Evaluated {len(evaluations)} results with score {avg_score:.2f}")
        
    except Exception as e:
        logger.error(f"Critic agent failed: {str(e)}")
        return format_agent_response("critic", False, {}, f"Criticism failed: {str(e)}")


async def _evaluate_single_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate a single agent result.
    
    Args:
        result: Single agent result to evaluate
        
    Returns:
        Evaluation details for the result
    """
    try:
        # Extract key information
        agent_id = result.get("agent_id", "unknown")
        success = result.get("success", False)
        result_data = result.get("result", {})
        reasoning = result.get("reasoning", "")
        
        # Basic evaluation criteria
        score = 0
        max_score = 100
        issues = []
        strengths = []
        
        # Check success status
        if success:
            score += 30
            strengths.append("Task completed successfully")
        else:
            score += 0
            issues.append("Task failed to complete")
        
        # Check result completeness
        if result_data and isinstance(result_data, dict):
            if len(result_data) > 0:
                score += 20
                strengths.append("Result contains data")
            else:
                score += 5
                issues.append("Result data is empty")
        else:
            issues.append("Result data is missing or invalid")
        
        # Check reasoning quality
        if reasoning and len(reasoning.strip()) > 10:
            score += 25
            strengths.append("Good reasoning provided")
        else:
            score += 5
            issues.append("Insufficient reasoning provided")
        
        # Check for errors
        if "error" in str(result_data).lower():
            score -= 20
            issues.append("Result contains errors")
        
        # Use OpenAI for detailed evaluation if available
        detailed_evaluation = await _get_detailed_evaluation(result)
        if detailed_evaluation:
            score += detailed_evaluation.get("additional_score", 0)
            issues.extend(detailed_evaluation.get("additional_issues", []))
            strengths.extend(detailed_evaluation.get("additional_strengths", []))
        
        # Ensure score is within bounds
        score = max(0, min(100, score))
        
        return {
            "agent_id": agent_id,
            "score": score,
            "max_score": max_score,
            "issues": issues,
            "strengths": strengths,
            "detailed_evaluation": detailed_evaluation,
            "evaluation_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Error evaluating single result: {str(e)}")
        return {
            "agent_id": result.get("agent_id", "unknown"),
            "score": 0,
            "max_score": 100,
            "issues": [f"Evaluation error: {str(e)}"],
            "strengths": [],
            "detailed_evaluation": None,
            "evaluation_timestamp": datetime.now().isoformat()
        }


async def _get_detailed_evaluation(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Get detailed evaluation using OpenAI API.
    
    Args:
        result: Result to evaluate
        
    Returns:
        Detailed evaluation or None if API unavailable
    """
    try:
        api_key = get_openai_api_key()
        if not api_key:
            return None
        
        import openai
        client = openai.OpenAI(api_key=api_key)
        
        prompt = f"""
        Evaluate the following AI agent result for quality, completeness, and usefulness:
        
        Agent ID: {result.get('agent_id', 'unknown')}
        Success: {result.get('success', False)}
        Result: {json.dumps(result.get('result', {}), indent=2)}
        Reasoning: {result.get('reasoning', '')}
        
        Provide evaluation in JSON format:
        {{
            "quality_score": 0-100,
            "completeness_score": 0-100,
            "usefulness_score": 0-100,
            "issues": ["list of specific issues"],
            "strengths": ["list of strengths"],
            "recommendations": ["list of improvement recommendations"]
        }}
        """
        
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert evaluator of AI agent outputs. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.3
        )
        
        content = response.choices[0].message.content.strip()
        
        # Parse JSON response
        try:
            evaluation = json.loads(content)
            
            # Calculate additional score based on detailed evaluation
            additional_score = (
                evaluation.get("quality_score", 0) * 0.4 +
                evaluation.get("completeness_score", 0) * 0.3 +
                evaluation.get("usefulness_score", 0) * 0.3
            ) / 10  # Scale to 0-10 additional points
            
            return {
                "additional_score": additional_score,
                "additional_issues": evaluation.get("issues", []),
                "additional_strengths": evaluation.get("strengths", []),
                "recommendations": evaluation.get("recommendations", []),
                "detailed_scores": {
                    "quality": evaluation.get("quality_score", 0),
                    "completeness": evaluation.get("completeness_score", 0),
                    "usefulness": evaluation.get("usefulness_score", 0)
                }
            }
            
        except (json.JSONDecodeError, KeyError):
            return None
            
    except Exception as e:
        logging.error(f"Detailed evaluation failed: {str(e)}")
        return None


async def _generate_overall_assessment(evaluations: List[Dict[str, Any]], main_task: str) -> str:
    """
    Generate overall assessment of all evaluations.
    
    Args:
        evaluations: List of individual evaluations
        main_task: Original main task
        
    Returns:
        Overall assessment text
    """
    if not evaluations:
        return "No evaluations available for assessment."
    
    total_score = sum(eval_data.get("score", 0) for eval_data in evaluations)
    avg_score = total_score / len(evaluations)
    
    all_issues = []
    all_strengths = []
    
    for eval_data in evaluations:
        all_issues.extend(eval_data.get("issues", []))
        all_strengths.extend(eval_data.get("strengths", []))
    
    assessment = f"""
    Overall Assessment for Task: {main_task}
    
    Average Score: {avg_score:.2f}/100
    Total Evaluations: {len(evaluations)}
    
    Key Strengths:
    {chr(10).join(f"- {strength}" for strength in all_strengths[:5])}
    
    Main Issues:
    {chr(10).join(f"- {issue}" for issue in all_issues[:5])}
    
    Overall Quality: {'Excellent' if avg_score >= 80 else 'Good' if avg_score >= 60 else 'Needs Improvement' if avg_score >= 40 else 'Poor'}
    """
    
    return assessment.strip()


async def _generate_improvement_suggestions(evaluations: List[Dict[str, Any]], avg_score: float) -> List[str]:
    """
    Generate improvement suggestions based on evaluations.
    
    Args:
        evaluations: List of individual evaluations
        avg_score: Average score across all evaluations
        
    Returns:
        List of improvement suggestions
    """
    suggestions = []
    
    # Score-based suggestions
    if avg_score < 40:
        suggestions.append("Consider redesigning the task decomposition approach")
        suggestions.append("Review agent capabilities and assignments")
    elif avg_score < 60:
        suggestions.append("Improve result validation and error handling")
        suggestions.append("Enhance reasoning quality in agent responses")
    elif avg_score < 80:
        suggestions.append("Fine-tune agent parameters for better performance")
        suggestions.append("Add more detailed logging and monitoring")
    else:
        suggestions.append("Maintain current performance levels")
        suggestions.append("Consider optimization for efficiency")
    
    # Issue-based suggestions
    all_issues = []
    for eval_data in evaluations:
        all_issues.extend(eval_data.get("issues", []))
    
    if any("error" in issue.lower() for issue in all_issues):
        suggestions.append("Implement better error handling and recovery mechanisms")
    
    if any("reasoning" in issue.lower() for issue in all_issues):
        suggestions.append("Improve reasoning quality and transparency")
    
    if any("empty" in issue.lower() or "missing" in issue.lower() for issue in all_issues):
        suggestions.append("Enhance data validation and completeness checks")
    
    return suggestions[:5]  # Limit to top 5 suggestions


class AdvancedCriticAgent:
    """
    Advanced critic agent for backward compatibility.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def evaluate_output(self, output: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Evaluate output using the existing run_critic_agent function."""
        return await run_critic_agent(output, context or {})
