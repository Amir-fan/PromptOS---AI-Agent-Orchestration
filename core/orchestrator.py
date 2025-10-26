"""
PromptOS Core Orchestrator - Advanced agent coordination system.

This module implements OpenAI-level orchestration capabilities with
advanced features like parallel execution, learning, and monitoring.
"""

import asyncio
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging

from config import config
from core.memory import AdvancedMemoryManager
from core.monitoring import MetricsCollector, TraceCollector
from core.learning import AgentLearner
from core.security import SecurityManager
from agents.registry import AgentRegistry


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Task:
    """Enhanced task representation."""
    id: str
    description: str
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    assigned_agents: List[str] = None
    dependencies: List[str] = None
    timeout: int = 300
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.assigned_agents is None:
            self.assigned_agents = []
        if self.dependencies is None:
            self.dependencies = []
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AgentResponse:
    """Enhanced agent response with metrics."""
    agent_id: str
    task_id: str
    success: bool
    result: Dict[str, Any]
    reasoning: str
    confidence: float = 0.0
    execution_time: float = 0.0
    tokens_used: int = 0
    cost: float = 0.0
    timestamp: datetime = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}


class AdvancedOrchestrator:
    """
    Advanced orchestrator with OpenAI-level capabilities.
    
    Features:
    - Parallel task execution
    - Agent learning and optimization
    - Advanced monitoring and tracing
    - Circuit breakers and retry logic
    - Security and compliance
    - Performance optimization
    """
    
    def __init__(self):
        """Initialize the advanced orchestrator."""
        self.logger = logging.getLogger(__name__)
        self.memory_manager = AdvancedMemoryManager()
        self.metrics_collector = MetricsCollector()
        self.trace_collector = TraceCollector()
        self.agent_learner = AgentLearner()
        self.security_manager = SecurityManager()
        self.agent_registry = AgentRegistry()
        
        # Task management
        self.active_tasks: Dict[str, Task] = {}
        self.task_queue = asyncio.PriorityQueue()
        self.completed_tasks: List[Task] = []
        
        # Agent management
        self.agent_responses: List[AgentResponse] = []
        self.agent_performance: Dict[str, Dict[str, Any]] = {}
        
        # Circuit breakers
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        
        # Initialize system
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize the orchestrator system."""
        self.logger.info("Initializing Advanced Orchestrator...")
        
        # Start background tasks
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're in an async context, create tasks
                asyncio.create_task(self._task_processor())
                asyncio.create_task(self._metrics_collector())
                asyncio.create_task(self._cleanup_old_tasks())
            else:
                # If no event loop is running, we'll start tasks later
                self._background_tasks_started = False
        except RuntimeError:
            # No event loop running, we'll start tasks later
            self._background_tasks_started = False
        
        self.logger.info("Advanced Orchestrator initialized successfully")
    
    async def _ensure_background_tasks_started(self):
        """Ensure background tasks are started."""
        if not hasattr(self, '_background_tasks_started') or not self._background_tasks_started:
            asyncio.create_task(self._task_processor())
            asyncio.create_task(self._metrics_collector())
            asyncio.create_task(self._cleanup_old_tasks())
            self._background_tasks_started = True
            self.logger.info("Background tasks started")
    
    async def execute_task(self, description: str, priority: TaskPriority = TaskPriority.NORMAL,
                          timeout: int = 300, metadata: Dict[str, Any] = None) -> str:
        """
        Execute a task with advanced orchestration.
        
        Args:
            description: Task description
            priority: Task priority
            timeout: Task timeout in seconds
            metadata: Additional task metadata
            
        Returns:
            Task ID
        """
        # Ensure background tasks are running
        await self._ensure_background_tasks_started()
        
        task_id = str(uuid.uuid4())
        
        # Create task
        task = Task(
            id=task_id,
            description=description,
            priority=priority,
            timeout=timeout,
            metadata=metadata or {}
        )
        
        # Security check
        if not await self.security_manager.validate_task(task):
            raise ValueError("Task failed security validation")
        
        # Add to queue
        await self.task_queue.put((priority.value, time.time(), task))
        self.active_tasks[task_id] = task
        
        # Log task creation
        await self.trace_collector.trace_task_created(task.id, task.description)
        await self.metrics_collector.increment_counter("tasks_created")
        
        return task_id
    
    async def _task_processor(self):
        """Background task processor."""
        while True:
            try:
                # Get next task from queue
                priority, timestamp, task = await self.task_queue.get()
                
                # Process task
                await self._process_task(task)
                
            except Exception as e:
                self.logger.error(f"Task processor error: {e}")
                await asyncio.sleep(1)
    
    async def _process_task(self, task: Task):
        """Process a single task."""
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        
        try:
            # Plan task execution
            execution_plan = await self._create_execution_plan(task)
            
            # Execute plan
            result = await self._execute_plan(task, execution_plan)
            
            # Complete task
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = result
            
            # Move to completed tasks
            self.completed_tasks.append(task)
            del self.active_tasks[task.id]
            
            # Update metrics
            execution_time = (task.completed_at - task.started_at).total_seconds()
            await self.metrics_collector.record_histogram("task_execution_time", execution_time)
            await self.metrics_collector.increment_counter("tasks_completed")
            
            # Learn from execution
            await self.agent_learner.learn_from_execution(task, result)
            
        except Exception as e:
            # Handle task failure
            await self._handle_task_failure(task, e)
    
    async def _create_execution_plan(self, task: Task) -> Dict[str, Any]:
        """Create an execution plan for the task."""
        # Use planner agent to create plan
        from agents.planner import AdvancedPlannerAgent
        
        planner = AdvancedPlannerAgent()
        plan = await planner.create_execution_plan(task.description)
        
        # Optimize plan based on learning
        optimized_plan = await self.agent_learner.optimize_plan(plan)
        
        return optimized_plan
    
    async def _execute_plan(self, task: Task, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the task plan."""
        subtasks = plan.get("subtasks", [])
        results = []
        
        # Execute subtasks in parallel where possible
        if plan.get("parallel_execution", True):
            # Group subtasks by dependency level
            dependency_groups = self._group_by_dependencies(subtasks)
            
            for group in dependency_groups:
                # Execute group in parallel
                group_results = await asyncio.gather(
                    *[self._execute_subtask(task, subtask) for subtask in group],
                    return_exceptions=True
                )
                results.extend(group_results)
        else:
            # Execute sequentially
            for subtask in subtasks:
                result = await self._execute_subtask(task, subtask)
                results.append(result)
        
        # Compile final result
        return await self._compile_results(task, results)
    
    async def _execute_subtask(self, task: Task, subtask: Dict[str, Any]) -> AgentResponse:
        """Execute a single subtask."""
        agent_id = subtask.get("agent_id", "executor")
        subtask_description = subtask.get("description", "")
        
        # Check circuit breaker
        if not await self._check_circuit_breaker(agent_id):
            raise Exception(f"Circuit breaker open for agent {agent_id}")
        
        # Execute with monitoring
        start_time = time.time()
        
        try:
            # Get agent
            agent = self.agent_registry.get_agent(agent_id)
            
            # Execute
            result = await agent.execute(subtask_description, task.metadata)
            
            # Record success
            execution_time = time.time() - start_time
            response = AgentResponse(
                agent_id=agent_id,
                task_id=task.id,
                success=True,
                result=result,
                reasoning=result.get("reasoning", ""),
                confidence=result.get("confidence", 0.0),
                execution_time=execution_time,
                tokens_used=result.get("tokens_used", 0),
                cost=result.get("cost", 0.0)
            )
            
            # Update performance metrics
            await self._update_agent_performance(agent_id, response)
            
            # Reset circuit breaker on success
            await self._reset_circuit_breaker(agent_id)
            
            return response
            
        except Exception as e:
            # Record failure
            execution_time = time.time() - start_time
            response = AgentResponse(
                agent_id=agent_id,
                task_id=task.id,
                success=False,
                result={"error": str(e)},
                reasoning="Task execution failed",
                execution_time=execution_time
            )
            
            # Update circuit breaker
            await self._record_circuit_breaker_failure(agent_id)
            
            return response
    
    async def _compile_results(self, task: Task, results: List[AgentResponse]) -> Dict[str, Any]:
        """Compile results from all agents."""
        # Use reporter agent to compile results
        from agents.reporter import AdvancedReporterAgent
        
        reporter = AdvancedReporterAgent()
        
        # Convert AgentResponse objects to dict format
        agent_responses = []
        for result in results:
            if hasattr(result, 'agent_id'):  # Check if it's an AgentResponse object
                agent_responses.append({
                    "agent_id": result.agent_id,
                    "success": result.success,
                    "result": result.result,
                    "reasoning": result.reasoning,
                    "timestamp": result.timestamp.isoformat() if result.timestamp else None
                })
            else:
                # Handle Exception objects or other types
                agent_responses.append({
                    "agent_id": "unknown",
                    "success": False,
                    "result": {"error": str(result)},
                    "reasoning": "Execution failed",
                    "timestamp": datetime.now().isoformat()
                })
        
        compiled_result = await reporter.compile_results(
            task.description, 
            {"task_id": task.id, "status": "completed"}, 
            agent_responses
        )
        
        # Store in memory
        await self.memory_manager.store_task_result(task.id, compiled_result)
        
        return compiled_result
    
    def _group_by_dependencies(self, subtasks: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group subtasks by dependency level for parallel execution."""
        # Simple implementation - can be enhanced with topological sorting
        groups = []
        remaining = subtasks.copy()
        
        while remaining:
            # Find subtasks with no dependencies
            current_group = []
            for subtask in remaining[:]:
                if not subtask.get("dependencies"):
                    current_group.append(subtask)
                    remaining.remove(subtask)
            
            if not current_group:
                # Circular dependency or error
                groups.append(remaining)
                break
            
            groups.append(current_group)
        
        return groups
    
    async def _check_circuit_breaker(self, agent_id: str) -> bool:
        """Check if circuit breaker allows execution."""
        if agent_id not in self.circuit_breakers:
            return True
        
        breaker = self.circuit_breakers[agent_id]
        
        if breaker["state"] == "OPEN":
            # Check if timeout has passed
            if time.time() - breaker["last_failure"] > breaker["timeout"]:
                breaker["state"] = "HALF_OPEN"
                return True
            return False
        
        return True
    
    async def _record_circuit_breaker_failure(self, agent_id: str):
        """Record a circuit breaker failure."""
        if agent_id not in self.circuit_breakers:
            self.circuit_breakers[agent_id] = {
                "failures": 0,
                "state": "CLOSED",
                "threshold": 5,
                "timeout": 60,
                "last_failure": 0
            }
        
        breaker = self.circuit_breakers[agent_id]
        breaker["failures"] += 1
        breaker["last_failure"] = time.time()
        
        if breaker["failures"] >= breaker["threshold"]:
            breaker["state"] = "OPEN"
            self.logger.warning(f"Circuit breaker opened for agent {agent_id}")
    
    async def _reset_circuit_breaker(self, agent_id: str):
        """Reset circuit breaker on success."""
        if agent_id in self.circuit_breakers:
            self.circuit_breakers[agent_id]["failures"] = 0
            self.circuit_breakers[agent_id]["state"] = "CLOSED"
    
    async def _update_agent_performance(self, agent_id: str, response: AgentResponse):
        """Update agent performance metrics."""
        if agent_id not in self.agent_performance:
            self.agent_performance[agent_id] = {
                "total_executions": 0,
                "successful_executions": 0,
                "total_time": 0.0,
                "average_confidence": 0.0,
                "total_tokens": 0,
                "total_cost": 0.0
            }
        
        perf = self.agent_performance[agent_id]
        perf["total_executions"] += 1
        
        if response.success:
            perf["successful_executions"] += 1
        
        perf["total_time"] += response.execution_time
        perf["total_tokens"] += response.tokens_used
        perf["total_cost"] += response.cost
        
        # Update averages
        perf["average_confidence"] = (
            (perf["average_confidence"] * (perf["total_executions"] - 1) + response.confidence) 
            / perf["total_executions"]
        )
    
    async def _handle_task_failure(self, task: Task, error: Exception):
        """Handle task failure with retry logic."""
        task.retry_count += 1
        task.error = str(error)
        
        if task.retry_count < task.max_retries:
            # Retry task
            self.logger.warning(f"Retrying task {task.id} (attempt {task.retry_count})")
            await self.task_queue.put((task.priority.value, time.time(), task))
        else:
            # Mark as failed
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()
            self.completed_tasks.append(task)
            
            if task.id in self.active_tasks:
                del self.active_tasks[task.id]
            
            await self.metrics_collector.increment_counter("tasks_failed")
            self.logger.error(f"Task {task.id} failed after {task.max_retries} retries: {error}")
    
    async def _metrics_collector(self):
        """Background metrics collection."""
        while True:
            try:
                # Collect system metrics
                metrics = {
                    "active_tasks": len(self.active_tasks),
                    "completed_tasks": len(self.completed_tasks),
                    "queue_size": self.task_queue.qsize(),
                    "agent_responses": len(self.agent_responses)
                }
                
                # Send to monitoring system
                await self.metrics_collector.record_metrics(metrics)
                
                await asyncio.sleep(10)  # Collect every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(10)
    
    async def _cleanup_old_tasks(self):
        """Clean up old completed tasks."""
        while True:
            try:
                # Remove tasks older than 24 hours
                cutoff_time = datetime.now() - timedelta(hours=24)
                
                self.completed_tasks = [
                    task for task in self.completed_tasks
                    if task.completed_at and task.completed_at > cutoff_time
                ]
                
                await asyncio.sleep(3600)  # Cleanup every hour
                
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")
                await asyncio.sleep(3600)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "queue_size": self.task_queue.qsize(),
            "agent_responses": len(self.agent_responses),
            "registered_agents": len(self.agent_registry.agents),
            "memory_nodes": self.memory_manager.graph.number_of_nodes(),
            "agent_performance": self.agent_performance,
            "circuit_breakers": {
                agent_id: breaker["state"] 
                for agent_id, breaker in self.circuit_breakers.items()
            },
            "uptime": time.time() - getattr(self, "_start_time", time.time()),
            "status": "operational"
        }
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task."""
        # Check active tasks
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            return asdict(task)
        
        # Check completed tasks
        for task in self.completed_tasks:
            if task.id == task_id:
                return asdict(task)
        
        return None
