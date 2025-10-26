"""
PromptOS Ethics Agent - Safety and bias checking.

This agent performs safety checks, bias detection, and privacy validation
on outputs to ensure ethical AI operation.
"""

import asyncio
import json
import logging
import re
from typing import Dict, List, Any, Optional
from datetime import datetime

from kernel.utils import get_openai_api_key, format_agent_response


async def run_ethics_agent(ethics_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main ethics agent function that performs safety and bias checks.
    
    Args:
        ethics_data: Dictionary containing task and evaluation data
        
    Returns:
        Dictionary containing ethics assessment results
    """
    logger = logging.getLogger(__name__)
    logger.info("Ethics agent performing safety and bias checks")
    
    try:
        # Extract data for analysis
        task = ethics_data.get("task", "")
        evaluation = ethics_data.get("evaluation", {})
        
        # Perform various ethics checks
        safety_check = await _perform_safety_check(task, evaluation)
        bias_check = await _perform_bias_check(task, evaluation)
        privacy_check = await _perform_privacy_check(task, evaluation)
        fairness_check = await _perform_fairness_check(task, evaluation)
        
        # Calculate overall ethics score
        ethics_score = _calculate_ethics_score(safety_check, bias_check, privacy_check, fairness_check)
        
        # Generate recommendations
        recommendations = await _generate_ethics_recommendations(safety_check, bias_check, privacy_check, fairness_check)
        
        # Determine overall assessment
        overall_assessment = _determine_overall_assessment(ethics_score, safety_check, bias_check, privacy_check, fairness_check)
        
        ethics_result = {
            "ethics_score": ethics_score,
            "safety_check": safety_check,
            "bias_check": bias_check,
            "privacy_check": privacy_check,
            "fairness_check": fairness_check,
            "overall_assessment": overall_assessment,
            "recommendations": recommendations,
            "assessment_timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Ethics assessment completed with score: {ethics_score}")
        return format_agent_response("ethics", True, ethics_result, f"Ethics assessment completed with score {ethics_score}")
        
    except Exception as e:
        logger.error(f"Ethics agent failed: {str(e)}")
        return format_agent_response("ethics", False, {}, f"Ethics check failed: {str(e)}")


async def _perform_safety_check(task: str, evaluation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform safety checks on the task and evaluation.
    
    Args:
        task: Original task description
        evaluation: Evaluation data
        
    Returns:
        Safety check results
    """
    safety_issues = []
    safety_score = 100  # Start with perfect score
    
    # Check for potentially harmful content
    harmful_keywords = [
        "harm", "hurt", "dangerous", "illegal", "malicious", "attack",
        "violence", "discrimination", "hate", "threat", "abuse"
    ]
    
    task_lower = task.lower()
    for keyword in harmful_keywords:
        if keyword in task_lower:
            safety_issues.append(f"Potentially harmful keyword detected: {keyword}")
            safety_score -= 20
    
    # Check evaluation data for safety concerns
    evaluation_str = json.dumps(evaluation, default=str).lower()
    for keyword in harmful_keywords:
        if keyword in evaluation_str:
            safety_issues.append(f"Safety concern in evaluation data: {keyword}")
            safety_score -= 15
    
    # Use AI for advanced safety analysis if available
    ai_safety_check = await _perform_ai_safety_check(task, evaluation)
    if ai_safety_check:
        safety_issues.extend(ai_safety_check.get("issues", []))
        safety_score -= ai_safety_check.get("penalty", 0)
    
    # Ensure score doesn't go below 0
    safety_score = max(0, safety_score)
    
    return {
        "score": safety_score,
        "issues": safety_issues,
        "status": "safe" if safety_score >= 80 else "caution" if safety_score >= 60 else "unsafe",
        "ai_analysis": ai_safety_check
    }


async def _perform_ai_safety_check(task: str, evaluation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Perform AI-powered safety check using OpenAI API.
    
    Args:
        task: Task description
        evaluation: Evaluation data
        
    Returns:
        AI safety analysis or None if unavailable
    """
    try:
        api_key = get_openai_api_key()
        if not api_key:
            return None
        
        import openai
        client = openai.OpenAI(api_key=api_key)
        
        prompt = f"""
        Analyze the following task and evaluation data for safety concerns:
        
        Task: {task}
        Evaluation: {json.dumps(evaluation, default=str, indent=2)}
        
        Check for:
        1. Potential harm to individuals or groups
        2. Illegal or unethical activities
        3. Privacy violations
        4. Bias or discrimination
        5. Misinformation or harmful content
        
        Respond in JSON format:
        {{
            "safety_issues": ["list of specific safety concerns"],
            "risk_level": "low/medium/high",
            "penalty": 0-50,
            "recommendations": ["list of safety recommendations"]
        }}
        """
        
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI safety expert. Analyze content for potential harm and respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.1
        )
        
        content = response.choices[0].message.content.strip()
        
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return None
            
    except Exception as e:
        logging.error(f"AI safety check failed: {str(e)}")
        return None


async def _perform_bias_check(task: str, evaluation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform bias detection on the task and evaluation.
    
    Args:
        task: Task description
        evaluation: Evaluation data
        
    Returns:
        Bias check results
    """
    bias_issues = []
    bias_score = 100
    
    # Check for demographic bias indicators
    demographic_terms = [
        "race", "gender", "age", "religion", "ethnicity", "nationality",
        "sexual orientation", "disability", "socioeconomic"
    ]
    
    task_lower = task.lower()
    for term in demographic_terms:
        if term in task_lower:
            # Check if used in potentially biased context
            context_patterns = [
                r"all\s+\w*\s*" + term,
                r"no\s+\w*\s*" + term,
                r"only\s+\w*\s*" + term
            ]
            
            for pattern in context_patterns:
                if re.search(pattern, task_lower):
                    bias_issues.append(f"Potential demographic bias detected: {term}")
                    bias_score -= 25
    
    # Check evaluation data for bias indicators
    evaluation_str = json.dumps(evaluation, default=str).lower()
    bias_keywords = ["unfair", "discriminat", "prejudice", "stereotype"]
    
    for keyword in bias_keywords:
        if keyword in evaluation_str:
            bias_issues.append(f"Bias indicator in evaluation: {keyword}")
            bias_score -= 15
    
    # Use AI for advanced bias detection
    ai_bias_check = await _perform_ai_bias_check(task, evaluation)
    if ai_bias_check:
        bias_issues.extend(ai_bias_check.get("issues", []))
        bias_score -= ai_bias_check.get("penalty", 0)
    
    bias_score = max(0, bias_score)
    
    return {
        "score": bias_score,
        "issues": bias_issues,
        "status": "unbiased" if bias_score >= 80 else "minor_bias" if bias_score >= 60 else "biased",
        "ai_analysis": ai_bias_check
    }


async def _perform_ai_bias_check(task: str, evaluation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Perform AI-powered bias detection.
    
    Args:
        task: Task description
        evaluation: Evaluation data
        
    Returns:
        AI bias analysis or None if unavailable
    """
    try:
        api_key = get_openai_api_key()
        if not api_key:
            return None
        
        import openai
        client = openai.OpenAI(api_key=api_key)
        
        prompt = f"""
        Analyze the following content for potential bias:
        
        Task: {task}
        Evaluation: {json.dumps(evaluation, default=str, indent=2)}
        
        Look for:
        1. Demographic bias (race, gender, age, etc.)
        2. Confirmation bias
        3. Selection bias
        4. Cultural bias
        5. Language bias
        
        Respond in JSON format:
        {{
            "bias_issues": ["list of specific bias concerns"],
            "bias_types": ["list of detected bias types"],
            "penalty": 0-50,
            "recommendations": ["list of bias mitigation recommendations"]
        }}
        """
        
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a bias detection expert. Analyze content for various types of bias and respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.1
        )
        
        content = response.choices[0].message.content.strip()
        
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return None
            
    except Exception as e:
        logging.error(f"AI bias check failed: {str(e)}")
        return None


async def _perform_privacy_check(task: str, evaluation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform privacy validation checks.
    
    Args:
        task: Task description
        evaluation: Evaluation data
        
    Returns:
        Privacy check results
    """
    privacy_issues = []
    privacy_score = 100
    
    # Check for personal information patterns
    personal_info_patterns = [
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
        r'\b\d{4}-\d{4}-\d{4}-\d{4}\b',  # Credit card
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
        r'\b\d{3}-\d{3}-\d{4}\b',  # Phone
    ]
    
    task_text = task + " " + json.dumps(evaluation, default=str)
    
    for pattern in personal_info_patterns:
        if re.search(pattern, task_text):
            privacy_issues.append("Potential personal information detected")
            privacy_score -= 30
    
    # Check for sensitive keywords
    sensitive_keywords = [
        "password", "secret", "confidential", "private", "personal",
        "ssn", "social security", "credit card", "bank account"
    ]
    
    task_lower = task.lower()
    for keyword in sensitive_keywords:
        if keyword in task_lower:
            privacy_issues.append(f"Sensitive keyword detected: {keyword}")
            privacy_score -= 20
    
    privacy_score = max(0, privacy_score)
    
    return {
        "score": privacy_score,
        "issues": privacy_issues,
        "status": "privacy_safe" if privacy_score >= 80 else "privacy_concern" if privacy_score >= 60 else "privacy_risk",
        "recommendations": [
            "Review data handling practices",
            "Implement data anonymization",
            "Ensure compliance with privacy regulations"
        ] if privacy_score < 80 else []
    }


async def _perform_fairness_check(task: str, evaluation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform fairness assessment.
    
    Args:
        task: Task description
        evaluation: Evaluation data
        
    Returns:
        Fairness check results
    """
    fairness_issues = []
    fairness_score = 100
    
    # Check for fairness indicators in task
    unfair_patterns = [
        r"only\s+\w+",  # "only men", "only women"
        r"no\s+\w+",    # "no minorities"
        r"exclude\s+\w+",  # "exclude certain groups"
    ]
    
    task_lower = task.lower()
    for pattern in unfair_patterns:
        if re.search(pattern, task_lower):
            fairness_issues.append("Potential unfair exclusion detected")
            fairness_score -= 25
    
    # Check evaluation data for fairness
    evaluation_str = json.dumps(evaluation, default=str).lower()
    if "unfair" in evaluation_str or "discriminat" in evaluation_str:
        fairness_issues.append("Unfair treatment indicated in evaluation")
        fairness_score -= 30
    
    fairness_score = max(0, fairness_score)
    
    return {
        "score": fairness_score,
        "issues": fairness_issues,
        "status": "fair" if fairness_score >= 80 else "unfair_concerns" if fairness_score >= 60 else "unfair",
        "recommendations": [
            "Review inclusion criteria",
            "Ensure equal treatment",
            "Implement fairness monitoring"
        ] if fairness_score < 80 else []
    }


def _calculate_ethics_score(safety_check: Dict[str, Any], bias_check: Dict[str, Any], 
                           privacy_check: Dict[str, Any], fairness_check: Dict[str, Any]) -> float:
    """
    Calculate overall ethics score from individual checks.
    
    Args:
        safety_check: Safety check results
        bias_check: Bias check results
        privacy_check: Privacy check results
        fairness_check: Fairness check results
        
    Returns:
        Overall ethics score (0-100)
    """
    weights = {
        "safety": 0.4,
        "bias": 0.3,
        "privacy": 0.2,
        "fairness": 0.1
    }
    
    weighted_score = (
        safety_check.get("score", 0) * weights["safety"] +
        bias_check.get("score", 0) * weights["bias"] +
        privacy_check.get("score", 0) * weights["privacy"] +
        fairness_check.get("score", 0) * weights["fairness"]
    )
    
    return round(weighted_score, 2)


async def _generate_ethics_recommendations(safety_check: Dict[str, Any], bias_check: Dict[str, Any], 
                                         privacy_check: Dict[str, Any], fairness_check: Dict[str, Any]) -> List[str]:
    """
    Generate ethics recommendations based on all checks.
    
    Args:
        safety_check: Safety check results
        bias_check: Bias check results
        privacy_check: Privacy check results
        fairness_check: Fairness check results
        
    Returns:
        List of recommendations
    """
    recommendations = []
    
    # Safety recommendations
    if safety_check.get("score", 0) < 80:
        recommendations.extend([
            "Implement additional safety checks",
            "Review content for potential harm",
            "Add human oversight for sensitive tasks"
        ])
    
    # Bias recommendations
    if bias_check.get("score", 0) < 80:
        recommendations.extend([
            "Implement bias detection tools",
            "Diversify training data",
            "Add bias mitigation strategies"
        ])
    
    # Privacy recommendations
    if privacy_check.get("score", 0) < 80:
        recommendations.extend([
            "Implement data anonymization",
            "Review data handling practices",
            "Ensure privacy compliance"
        ])
    
    # Fairness recommendations
    if fairness_check.get("score", 0) < 80:
        recommendations.extend([
            "Review inclusion criteria",
            "Implement fairness monitoring",
            "Ensure equal treatment"
        ])
    
    # General recommendations
    if not recommendations:
        recommendations.append("Maintain current ethical standards")
    
    return recommendations[:10]  # Limit to top 10 recommendations


def _determine_overall_assessment(ethics_score: float, safety_check: Dict[str, Any], 
                                bias_check: Dict[str, Any], privacy_check: Dict[str, Any], 
                                fairness_check: Dict[str, Any]) -> str:
    """
    Determine overall ethics assessment.
    
    Args:
        ethics_score: Overall ethics score
        safety_check: Safety check results
        bias_check: Bias check results
        privacy_check: Privacy check results
        fairness_check: Fairness check results
        
    Returns:
        Overall assessment text
    """
    if ethics_score >= 90:
        return "Excellent ethical compliance with no significant issues detected."
    elif ethics_score >= 80:
        return "Good ethical compliance with minor issues that should be monitored."
    elif ethics_score >= 60:
        return "Moderate ethical compliance with several issues that need attention."
    elif ethics_score >= 40:
        return "Poor ethical compliance with significant issues requiring immediate attention."
    else:
        return "Critical ethical compliance issues detected. Immediate review and remediation required."


class AdvancedEthicsAgent:
    """
    Advanced ethics agent for backward compatibility.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def evaluate_ethics(self, task: str, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate ethics using the existing run_ethics_agent function."""
        return await run_ethics_agent(task, evaluation)
