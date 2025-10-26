"""
PromptOS Advanced Planner Agent - OpenAI-level task decomposition.

This agent provides sophisticated task planning with multi-step reasoning,
dependency analysis, and optimization capabilities.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import networkx as nx

from config import config
from core.memory import AdvancedMemoryManager
from core.monitoring import TraceCollector


@dataclass
class Subtask:
    """Enhanced subtask representation."""
    id: str
    description: str
    agent_id: str
    priority: int = 1
    estimated_duration: int = 60
    dependencies: List[str] = None
    parallel_execution: bool = True
    retry_count: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ExecutionPlan:
    """Comprehensive execution plan."""
    plan_id: str
    main_task: str
    subtasks: List[Subtask]
    execution_graph: nx.DiGraph
    estimated_total_duration: int
    parallel_groups: List[List[str]]
    optimization_suggestions: List[str]
    confidence_score: float
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class AdvancedPlannerAgent:
    """
    Advanced planner agent with OpenAI-level capabilities.
    
    Features:
    - Multi-step reasoning
    - Dependency analysis
    - Parallel execution optimization
    - Context-aware planning
    - Learning from past executions
    """
    
    def __init__(self, memory_manager: AdvancedMemoryManager = None):
        """Initialize the advanced planner agent."""
        self.logger = logging.getLogger(__name__)
        self.memory_manager = memory_manager or AdvancedMemoryManager()
        self.trace_collector = TraceCollector()
        
        # Planning capabilities
        self.agent_capabilities = {
            'planner': ['task_decomposition', 'dependency_analysis', 'optimization'],
            'executor': ['web_search', 'text_processing', 'data_analysis', 'file_operations'],
            'critic': ['quality_assessment', 'coherence_check', 'improvement_suggestions'],
            'reporter': ['trace_compilation', 'report_generation', 'transparency_logging'],
            'ethics': ['safety_check', 'bias_detection', 'privacy_validation']
        }
        
        # Planning patterns
        self.planning_patterns = {
            'research': ['gather_info', 'analyze_data', 'synthesize_findings', 'generate_report'],
            'analysis': ['define_scope', 'collect_data', 'process_data', 'draw_conclusions'],
            'creation': ['plan_structure', 'gather_resources', 'create_content', 'review_output'],
            'optimization': ['assess_current', 'identify_issues', 'propose_solutions', 'implement_changes']
        }
    
    async def create_execution_plan(self, task_description: str) -> Dict[str, Any]:
        """
        Create a comprehensive execution plan for a task.
        
        Args:
            task_description: Task to plan
            
        Returns:
            Execution plan dictionary
        """
        span_id = self.trace_collector.start_trace("create_execution_plan")
        
        try:
            # Analyze task
            task_analysis = await self._analyze_task(task_description)
            
            # Get relevant context
            context = await self.memory_manager.get_context(task_description)
            
            # Generate subtasks
            subtasks = await self._generate_subtasks(task_description, task_analysis, context)
            
            # Analyze dependencies
            execution_graph = await self._analyze_dependencies(subtasks)
            
            # Optimize for parallel execution
            parallel_groups = await self._optimize_parallel_execution(execution_graph)
            
            # Calculate estimates
            estimated_duration = await self._calculate_estimated_duration(subtasks, parallel_groups)
            
            # Generate optimization suggestions
            suggestions = await self._generate_optimization_suggestions(task_analysis, subtasks)
            
            # Calculate confidence score
            confidence = await self._calculate_confidence_score(task_analysis, subtasks, context)
            
            # Create execution plan
            plan = ExecutionPlan(
                plan_id=f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                main_task=task_description,
                subtasks=subtasks,
                execution_graph=execution_graph,
                estimated_total_duration=estimated_duration,
                parallel_groups=parallel_groups,
                optimization_suggestions=suggestions,
                confidence_score=confidence
            )
            
            # Store plan in memory
            await self.memory_manager.store_memory(
                content=json.dumps(asdict(plan), default=str),
                node_type='execution_plan',
                metadata={'task': task_description, 'confidence': confidence}
            )
            
            self.trace_collector.end_trace(span_id, "ok")
            
            return asdict(plan)
            
        except Exception as e:
            self.trace_collector.end_trace(span_id, "error")
            self.logger.error(f"Failed to create execution plan: {e}")
            raise
    
    async def _analyze_task(self, task_description: str) -> Dict[str, Any]:
        """Analyze task characteristics and requirements."""
        analysis = {
            'complexity': 'medium',
            'domain': 'general',
            'estimated_effort': 'medium',
            'requires_external_data': False,
            'requires_creativity': False,
            'requires_analysis': False,
            'output_format': 'text',
            'keywords': [],
            'intent': 'unknown'
        }
        
        task_lower = task_description.lower()
        
        # Analyze complexity
        complexity_indicators = {
            'simple': ['summarize', 'list', 'find', 'search'],
            'medium': ['analyze', 'compare', 'evaluate', 'explain'],
            'complex': ['research', 'develop', 'create', 'design', 'implement']
        }
        
        for complexity, indicators in complexity_indicators.items():
            if any(indicator in task_lower for indicator in indicators):
                analysis['complexity'] = complexity
                break
        
        # Analyze domain
        domain_indicators = {
            'technical': ['code', 'programming', 'software', 'api', 'database'],
            'business': ['strategy', 'marketing', 'sales', 'finance', 'management'],
            'research': ['study', 'investigate', 'research', 'analysis', 'data'],
            'creative': ['write', 'create', 'design', 'art', 'content']
        }
        
        for domain, indicators in domain_indicators.items():
            if any(indicator in task_lower for indicator in indicators):
                analysis['domain'] = domain
                break
        
        # Analyze requirements
        if any(word in task_lower for word in ['search', 'find', 'gather', 'collect']):
            analysis['requires_external_data'] = True
        
        if any(word in task_lower for word in ['create', 'write', 'design', 'develop']):
            analysis['requires_creativity'] = True
        
        if any(word in task_lower for word in ['analyze', 'evaluate', 'assess', 'examine']):
            analysis['requires_analysis'] = True
        
        # Extract keywords
        analysis['keywords'] = [word for word in task_lower.split() if len(word) > 3]
        
        # Determine intent
        if '?' in task_description:
            analysis['intent'] = 'question'
        elif any(word in task_lower for word in ['create', 'make', 'build', 'generate']):
            analysis['intent'] = 'creation'
        elif any(word in task_lower for word in ['analyze', 'evaluate', 'assess']):
            analysis['intent'] = 'analysis'
        else:
            analysis['intent'] = 'general'
        
        return analysis
    
    async def _generate_subtasks(self, task_description: str, task_analysis: Dict[str, Any], 
                               context: Dict[str, Any]) -> List[Subtask]:
        """Generate subtasks based on task analysis."""
        subtasks = []
        
        # Get planning pattern
        pattern = self._get_planning_pattern(task_analysis)
        
        # Generate subtasks based on pattern
        for i, step in enumerate(pattern):
            subtask = await self._create_subtask(
                task_description, step, i, task_analysis, context
            )
            subtasks.append(subtask)
        
        # Add domain-specific subtasks
        domain_subtasks = await self._add_domain_specific_subtasks(
            task_description, task_analysis, context
        )
        subtasks.extend(domain_subtasks)
        
        # Add quality assurance subtasks
        qa_subtasks = await self._add_quality_assurance_subtasks(
            task_description, task_analysis
        )
        subtasks.extend(qa_subtasks)
        
        return subtasks
    
    def _get_planning_pattern(self, task_analysis: Dict[str, Any]) -> List[str]:
        """Get appropriate planning pattern for task."""
        intent = task_analysis.get('intent', 'general')
        complexity = task_analysis.get('complexity', 'medium')
        
        if intent == 'research':
            return self.planning_patterns['research']
        elif intent == 'analysis':
            return self.planning_patterns['analysis']
        elif intent == 'creation':
            return self.planning_patterns['creation']
        elif intent == 'optimization':
            return self.planning_patterns['optimization']
        else:
            # Default pattern based on complexity
            if complexity == 'simple':
                return ['understand_task', 'execute_task', 'verify_result']
            elif complexity == 'complex':
                return ['plan_approach', 'gather_resources', 'execute_phases', 'integrate_results', 'validate_output']
            else:
                return ['analyze_requirements', 'execute_task', 'review_output']
    
    async def _create_subtask(self, task_description: str, step: str, index: int,
                            task_analysis: Dict[str, Any], context: Dict[str, Any]) -> Subtask:
        """Create a specific subtask."""
        # Determine appropriate agent
        agent_id = self._select_agent_for_step(step, task_analysis)
        
        # Create subtask description
        description = await self._generate_subtask_description(
            task_description, step, task_analysis, context
        )
        
        # Estimate duration
        estimated_duration = self._estimate_duration(step, task_analysis)
        
        # Determine priority
        priority = self._calculate_priority(step, index, task_analysis)
        
        return Subtask(
            id=f"subtask_{index}_{step}",
            description=description,
            agent_id=agent_id,
            priority=priority,
            estimated_duration=estimated_duration,
            metadata={
                'step': step,
                'index': index,
                'complexity': task_analysis.get('complexity', 'medium')
            }
        )
    
    def _select_agent_for_step(self, step: str, task_analysis: Dict[str, Any]) -> str:
        """Select appropriate agent for a step."""
        step_lower = step.lower()
        
        # Mapping based on step characteristics
        if any(word in step_lower for word in ['gather', 'search', 'find', 'collect']):
            return 'executor'
        elif any(word in step_lower for word in ['analyze', 'process', 'evaluate']):
            return 'executor'
        elif any(word in step_lower for word in ['review', 'check', 'validate', 'assess']):
            return 'critic'
        elif any(word in step_lower for word in ['report', 'summarize', 'compile']):
            return 'reporter'
        elif any(word in step_lower for word in ['safety', 'ethics', 'bias']):
            return 'ethics'
        else:
            return 'executor'  # Default
    
    async def _generate_subtask_description(self, task_description: str, step: str,
                                          task_analysis: Dict[str, Any], 
                                          context: Dict[str, Any]) -> str:
        """Generate detailed subtask description."""
        # Use context to enhance description
        context_info = ""
        if context.get('relevant_memories'):
            context_info = f"Consider previous similar tasks: {len(context['relevant_memories'])} relevant memories found."
        
        # Generate step-specific description
        descriptions = {
            'gather_info': f"Research and gather information about: {task_description}",
            'analyze_data': f"Analyze the collected data for: {task_description}",
            'synthesize_findings': f"Synthesize findings into coherent insights for: {task_description}",
            'generate_report': f"Generate a comprehensive report for: {task_description}",
            'understand_task': f"Analyze and understand the requirements: {task_description}",
            'execute_task': f"Execute the main task: {task_description}",
            'verify_result': f"Verify and validate the result for: {task_description}",
            'plan_approach': f"Develop a detailed approach for: {task_description}",
            'gather_resources': f"Identify and gather necessary resources for: {task_description}",
            'execute_phases': f"Execute the planned phases for: {task_description}",
            'integrate_results': f"Integrate all results for: {task_description}",
            'validate_output': f"Validate the final output for: {task_description}"
        }
        
        base_description = descriptions.get(step, f"Execute step '{step}' for: {task_description}")
        
        if context_info:
            base_description += f" {context_info}"
        
        return base_description
    
    def _estimate_duration(self, step: str, task_analysis: Dict[str, Any]) -> int:
        """Estimate duration for a step in seconds."""
        complexity = task_analysis.get('complexity', 'medium')
        
        base_durations = {
            'gather_info': 120,
            'analyze_data': 180,
            'synthesize_findings': 90,
            'generate_report': 120,
            'understand_task': 30,
            'execute_task': 300,
            'verify_result': 60,
            'plan_approach': 60,
            'gather_resources': 90,
            'execute_phases': 600,
            'integrate_results': 120,
            'validate_output': 90
        }
        
        base_duration = base_durations.get(step, 120)
        
        # Adjust for complexity
        complexity_multipliers = {
            'simple': 0.5,
            'medium': 1.0,
            'complex': 2.0
        }
        
        multiplier = complexity_multipliers.get(complexity, 1.0)
        
        return int(base_duration * multiplier)
    
    def _calculate_priority(self, step: str, index: int, task_analysis: Dict[str, Any]) -> int:
        """Calculate priority for a subtask (1-5, higher is more important)."""
        # Base priority on step order and importance
        if index == 0:  # First step
            return 5
        elif 'verify' in step.lower() or 'validate' in step.lower():
            return 4
        elif 'analyze' in step.lower() or 'synthesize' in step.lower():
            return 3
        else:
            return 2
    
    async def _add_domain_specific_subtasks(self, task_description: str, 
                                          task_analysis: Dict[str, Any],
                                          context: Dict[str, Any]) -> List[Subtask]:
        """Add domain-specific subtasks."""
        subtasks = []
        domain = task_analysis.get('domain', 'general')
        
        if domain == 'technical':
            # Add technical validation
            subtasks.append(Subtask(
                id="tech_validation",
                description=f"Validate technical requirements for: {task_description}",
                agent_id="critic",
                priority=3,
                estimated_duration=60
            ))
        
        elif domain == 'research':
            # Add literature review
            subtasks.append(Subtask(
                id="literature_review",
                description=f"Conduct literature review for: {task_description}",
                agent_id="executor",
                priority=4,
                estimated_duration=300
            ))
        
        return subtasks
    
    async def _add_quality_assurance_subtasks(self, task_description: str,
                                            task_analysis: Dict[str, Any]) -> List[Subtask]:
        """Add quality assurance subtasks."""
        subtasks = []
        
        # Always add quality check
        subtasks.append(Subtask(
            id="quality_check",
            description=f"Perform quality assessment for: {task_description}",
            agent_id="critic",
            priority=4,
            estimated_duration=90
        ))
        
        # Add ethics check for sensitive tasks
        if any(word in task_description.lower() for word in ['personal', 'private', 'sensitive', 'data']):
            subtasks.append(Subtask(
                id="ethics_check",
                description=f"Perform ethics and safety check for: {task_description}",
                agent_id="ethics",
                priority=5,
                estimated_duration=60
            ))
        
        return subtasks
    
    async def _analyze_dependencies(self, subtasks: List[Subtask]) -> nx.DiGraph:
        """Analyze dependencies between subtasks."""
        graph = nx.DiGraph()
        
        # Add all subtasks as nodes
        for subtask in subtasks:
            graph.add_node(subtask.id, **asdict(subtask))
        
        # Analyze dependencies based on step order and content
        for i, subtask in enumerate(subtasks):
            # Earlier subtasks are dependencies
            for j in range(i):
                prev_subtask = subtasks[j]
                
                # Check if current subtask depends on previous
                if self._has_dependency(subtask, prev_subtask):
                    graph.add_edge(prev_subtask.id, subtask.id)
                    subtask.dependencies.append(prev_subtask.id)
        
        return graph
    
    def _has_dependency(self, subtask: Subtask, prev_subtask: Subtask) -> bool:
        """Check if subtask depends on previous subtask."""
        # Simple dependency rules
        step = subtask.metadata.get('step', '')
        prev_step = prev_subtask.metadata.get('step', '')
        
        # Sequential dependencies
        sequential_pairs = [
            ('gather_info', 'analyze_data'),
            ('analyze_data', 'synthesize_findings'),
            ('synthesize_findings', 'generate_report'),
            ('plan_approach', 'gather_resources'),
            ('gather_resources', 'execute_phases'),
            ('execute_phases', 'integrate_results')
        ]
        
        for before, after in sequential_pairs:
            if prev_step == before and step == after:
                return True
        
        # Quality assurance depends on execution
        if step in ['quality_check', 'verify_result', 'validate_output']:
            return True
        
        return False
    
    async def _optimize_parallel_execution(self, graph: nx.DiGraph) -> List[List[str]]:
        """Optimize subtasks for parallel execution."""
        # Use topological sort to find execution levels
        try:
            topo_order = list(nx.topological_sort(graph))
        except nx.NetworkXError:
            # Graph has cycles, use simple ordering
            topo_order = list(graph.nodes())
        
        # Group nodes by level (nodes that can run in parallel)
        levels = []
        remaining = set(topo_order)
        
        while remaining:
            # Find nodes with no dependencies in remaining set
            current_level = []
            for node in list(remaining):
                predecessors = set(graph.predecessors(node))
                if not predecessors.intersection(remaining):
                    current_level.append(node)
                    remaining.remove(node)
            
            if current_level:
                levels.append(current_level)
            else:
                # Break cycles by adding remaining nodes
                levels.append(list(remaining))
                break
        
        return levels
    
    async def _calculate_estimated_duration(self, subtasks: List[Subtask], 
                                          parallel_groups: List[List[str]]) -> int:
        """Calculate estimated total duration considering parallel execution."""
        total_duration = 0
        
        for group in parallel_groups:
            # Duration of parallel group is the max duration in the group
            group_duration = 0
            for subtask_id in group:
                subtask = next(s for s in subtasks if s.id == subtask_id)
                group_duration = max(group_duration, subtask.estimated_duration)
            
            total_duration += group_duration
        
        return total_duration
    
    async def _generate_optimization_suggestions(self, task_analysis: Dict[str, Any],
                                               subtasks: List[Subtask]) -> List[str]:
        """Generate optimization suggestions for the plan."""
        suggestions = []
        
        # Check for potential optimizations
        if len(subtasks) > 10:
            suggestions.append("Consider breaking down into smaller, more focused subtasks")
        
        if any(s.estimated_duration > 600 for s in subtasks):
            suggestions.append("Some subtasks may be too large - consider splitting them")
        
        if task_analysis.get('requires_external_data'):
            suggestions.append("Consider caching external data to improve performance")
        
        if task_analysis.get('complexity') == 'complex':
            suggestions.append("For complex tasks, consider adding intermediate validation steps")
        
        return suggestions
    
    async def _calculate_confidence_score(self, task_analysis: Dict[str, Any],
                                        subtasks: List[Subtask], 
                                        context: Dict[str, Any]) -> float:
        """Calculate confidence score for the execution plan."""
        base_confidence = 0.7
        
        # Adjust based on task complexity
        complexity = task_analysis.get('complexity', 'medium')
        if complexity == 'simple':
            base_confidence += 0.2
        elif complexity == 'complex':
            base_confidence -= 0.1
        
        # Adjust based on available context
        if context.get('relevant_memories'):
            base_confidence += 0.1
        
        # Adjust based on subtask count
        if len(subtasks) < 3:
            base_confidence -= 0.1
        elif len(subtasks) > 8:
            base_confidence -= 0.05
        
        return max(0.0, min(1.0, base_confidence))


async def run_planner_agent(task_description: str) -> Dict[str, Any]:
    """
    Main planner agent function for backward compatibility.
    
    Args:
        task_description: The task to plan
        
    Returns:
        Dict containing the execution plan
    """
    planner = AdvancedPlannerAgent()
    return await planner.create_execution_plan(task_description)