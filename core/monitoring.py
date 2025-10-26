"""
PromptOS Monitoring System - Advanced observability and metrics.

This module provides OpenAI-level monitoring capabilities including
metrics collection, distributed tracing, and real-time dashboards.
"""

import asyncio
import time
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import logging
from pathlib import Path

from config import config


@dataclass
class Metric:
    """Metric data structure."""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Trace:
    """Distributed trace data structure."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    operation_name: str = ""
    start_time: datetime = None
    end_time: Optional[datetime] = None
    duration: float = 0.0
    tags: Dict[str, str] = None
    logs: List[Dict[str, Any]] = None
    status: str = "ok"  # ok, error, timeout
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.start_time is None:
            self.start_time = datetime.now()
        if self.tags is None:
            self.tags = {}
        if self.logs is None:
            self.logs = []


class MetricsCollector:
    """
    Advanced metrics collector with real-time aggregation.
    
    Features:
    - Counter, gauge, histogram metrics
    - Real-time aggregation
    - Custom metrics
    - Performance tracking
    """
    
    def __init__(self):
        """Initialize metrics collector."""
        self.logger = logging.getLogger(__name__)
        
        # Metric storage
        self.counters = defaultdict(float)
        self.gauges = defaultdict(float)
        self.histograms = defaultdict(list)
        self.custom_metrics = defaultdict(list)
        
        # Aggregation settings
        self.aggregation_window = 60  # seconds
        self.max_histogram_samples = 1000
        
        # Background tasks
        self._aggregation_task = None
        self._start_aggregation()
    
    def _start_aggregation(self):
        """Start background aggregation task."""
        if self._aggregation_task is None or self._aggregation_task.done():
            self._aggregation_task = asyncio.create_task(self._aggregate_metrics())
    
    async def increment_counter(self, name: str, value: float = 1.0, tags: Dict[str, str] = None):
        """Increment a counter metric."""
        key = self._create_key(name, tags)
        self.counters[key] += value
        
        # Store custom metric
        metric = Metric(
            name=name,
            value=value,
            timestamp=datetime.now(),
            tags=tags or {}
        )
        self.custom_metrics[name].append(metric)
    
    async def set_gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Set a gauge metric."""
        key = self._create_key(name, tags)
        self.gauges[key] = value
        
        # Store custom metric
        metric = Metric(
            name=name,
            value=value,
            timestamp=datetime.now(),
            tags=tags or {}
        )
        self.custom_metrics[name].append(metric)
    
    async def record_histogram(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a histogram metric."""
        key = self._create_key(name, tags)
        
        # Add to histogram
        self.histograms[key].append({
            'value': value,
            'timestamp': datetime.now()
        })
        
        # Limit samples
        if len(self.histograms[key]) > self.max_histogram_samples:
            self.histograms[key] = self.histograms[key][-self.max_histogram_samples:]
        
        # Store custom metric
        metric = Metric(
            name=name,
            value=value,
            timestamp=datetime.now(),
            tags=tags or {}
        )
        self.custom_metrics[name].append(metric)
    
    async def record_metrics(self, metrics: Dict[str, Any]):
        """Record multiple metrics at once."""
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                await self.set_gauge(name, value)
            elif isinstance(value, dict) and 'value' in value:
                tags = value.get('tags', {})
                await self.set_gauge(name, value['value'], tags)
    
    def _create_key(self, name: str, tags: Dict[str, str] = None) -> str:
        """Create a unique key for a metric."""
        if not tags:
            return name
        
        tag_str = ','.join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}[{tag_str}]"
    
    async def _aggregate_metrics(self):
        """Background task to aggregate metrics."""
        while True:
            try:
                await asyncio.sleep(self.aggregation_window)
                
                # Clean old metrics
                cutoff_time = datetime.now() - timedelta(hours=1)
                
                for name, metrics in self.custom_metrics.items():
                    self.custom_metrics[name] = [
                        m for m in metrics 
                        if m.timestamp > cutoff_time
                    ]
                
                # Log aggregated metrics
                await self._log_aggregated_metrics()
                
            except Exception as e:
                self.logger.error(f"Metrics aggregation error: {e}")
                await asyncio.sleep(10)
    
    async def _log_aggregated_metrics(self):
        """Log aggregated metrics."""
        aggregated = {
            'counters': dict(self.counters),
            'gauges': dict(self.gauges),
            'histogram_stats': {}
        }
        
        # Calculate histogram statistics
        for key, samples in self.histograms.items():
            if samples:
                values = [s['value'] for s in samples]
                aggregated['histogram_stats'][key] = {
                    'count': len(values),
                    'min': min(values),
                    'max': max(values),
                    'mean': sum(values) / len(values),
                    'p50': self._percentile(values, 50),
                    'p95': self._percentile(values, 95),
                    'p99': self._percentile(values, 99)
                }
        
        self.logger.info(f"Aggregated metrics: {json.dumps(aggregated, indent=2)}")
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get current metrics summary."""
        return {
            'counters': dict(self.counters),
            'gauges': dict(self.gauges),
            'histogram_count': len(self.histograms),
            'custom_metrics_count': len(self.custom_metrics),
            'timestamp': datetime.now().isoformat()
        }


class TraceCollector:
    """
    Distributed tracing collector for request flows.
    
    Features:
    - Request tracing
    - Span correlation
    - Performance analysis
    - Error tracking
    """
    
    def __init__(self):
        """Initialize trace collector."""
        self.logger = logging.getLogger(__name__)
        
        # Trace storage
        self.active_traces = {}
        self.completed_traces = []
        self.trace_metrics = defaultdict(list)
        
        # Settings
        self.max_traces = 1000
        self.trace_retention_hours = 24
    
    def start_trace(self, operation_name: str, trace_id: str = None, 
                   parent_span_id: str = None) -> str:
        """Start a new trace."""
        if trace_id is None:
            trace_id = f"trace_{int(time.time() * 1000000)}"
        
        span_id = f"span_{int(time.time() * 1000000)}"
        
        trace = Trace(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name
        )
        
        self.active_traces[span_id] = trace
        
        return span_id
    
    def end_trace(self, span_id: str, status: str = "ok"):
        """End a trace."""
        if span_id in self.active_traces:
            trace = self.active_traces[span_id]
            trace.end_time = datetime.now()
            trace.duration = (trace.end_time - trace.start_time).total_seconds()
            trace.status = status
            
            # Move to completed traces
            self.completed_traces.append(trace)
            del self.active_traces[span_id]
            
            # Record metrics
            self.trace_metrics[trace.operation_name].append({
                'duration': trace.duration,
                'status': status,
                'timestamp': trace.start_time
            })
            
            # Cleanup old traces
            self._cleanup_old_traces()
    
    def add_trace_log(self, span_id: str, message: str, level: str = "info", 
                     metadata: Dict[str, Any] = None):
        """Add a log entry to a trace."""
        if span_id in self.active_traces:
            log_entry = {
                'message': message,
                'level': level,
                'timestamp': datetime.now().isoformat(),
                'metadata': metadata or {}
            }
            self.active_traces[span_id].logs.append(log_entry)
    
    def add_trace_tag(self, span_id: str, key: str, value: str):
        """Add a tag to a trace."""
        if span_id in self.active_traces:
            self.active_traces[span_id].tags[key] = value
    
    def _cleanup_old_traces(self):
        """Clean up old traces."""
        cutoff_time = datetime.now() - timedelta(hours=self.trace_retention_hours)
        
        self.completed_traces = [
            trace for trace in self.completed_traces
            if trace.start_time > cutoff_time
        ]
        
        # Limit total traces
        if len(self.completed_traces) > self.max_traces:
            self.completed_traces = self.completed_traces[-self.max_traces:]
    
    def get_trace_summary(self) -> Dict[str, Any]:
        """Get trace summary statistics."""
        if not self.completed_traces:
            return {'total_traces': 0}
        
        durations = [trace.duration for trace in self.completed_traces]
        statuses = [trace.status for trace in self.completed_traces]
        
        return {
            'total_traces': len(self.completed_traces),
            'active_traces': len(self.active_traces),
            'average_duration': sum(durations) / len(durations),
            'min_duration': min(durations),
            'max_duration': max(durations),
            'error_rate': statuses.count('error') / len(statuses),
            'operations': {
                op: len(traces) 
                for op, traces in self.trace_metrics.items()
            }
        }
    
    def get_traces_by_operation(self, operation_name: str) -> List[Dict[str, Any]]:
        """Get traces for a specific operation."""
        traces = [
            trace for trace in self.completed_traces
            if trace.operation_name == operation_name
        ]
        
        return [asdict(trace) for trace in traces]
    
    async def trace_task_created(self, task_id: str, task_description: str):
        """Trace task creation."""
        trace = Trace(
            trace_id=str(uuid.uuid4()),
            span_id=str(uuid.uuid4()),
            operation_name="task_created",
            start_time=datetime.now(),
            metadata={
                "task_id": task_id,
                "task_description": task_description[:100] + "..." if len(task_description) > 100 else task_description
            }
        )
        self.active_traces[task_id] = trace
        self.logger.debug(f"Started trace for task {task_id}")
    
    async def trace_task_completed(self, task_id: str, status: str = "success", result: Any = None):
        """Trace task completion."""
        if task_id in self.active_traces:
            trace = self.active_traces[task_id]
            trace.end_time = datetime.now()
            trace.duration = (trace.end_time - trace.start_time).total_seconds()
            trace.status = status
            if result:
                trace.metadata["result_summary"] = str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
            
            self.completed_traces.append(trace)
            del self.active_traces[task_id]
            self.logger.debug(f"Completed trace for task {task_id}")
    
    async def trace_agent_execution(self, task_id: str, agent_name: str, status: str = "success"):
        """Trace agent execution."""
        if task_id in self.active_traces:
            trace = self.active_traces[task_id]
            if "agent_executions" not in trace.metadata:
                trace.metadata["agent_executions"] = []
            trace.metadata["agent_executions"].append({
                "agent": agent_name,
                "status": status,
                "timestamp": datetime.now().isoformat()
            })


class HealthChecker:
    """
    System health checker with dependency monitoring.
    
    Features:
    - Service health checks
    - Dependency monitoring
    - Alert generation
    - Performance thresholds
    """
    
    def __init__(self, metrics_collector: MetricsCollector):
        """Initialize health checker."""
        self.metrics_collector = metrics_collector
        self.logger = logging.getLogger(__name__)
        
        # Health checks
        self.health_checks = {}
        self.health_status = {}
        
        # Thresholds
        self.thresholds = {
            'response_time_ms': 1000,
            'error_rate': 0.05,
            'memory_usage_percent': 80,
            'cpu_usage_percent': 80
        }
    
    def register_health_check(self, name: str, check_func: Callable[[], bool]):
        """Register a health check function."""
        self.health_checks[name] = check_func
    
    async def check_health(self) -> Dict[str, Any]:
        """Perform all health checks."""
        health_status = {
            'overall_status': 'healthy',
            'checks': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Run registered health checks
        for name, check_func in self.health_checks.items():
            try:
                result = await check_func() if asyncio.iscoroutinefunction(check_func) else check_func()
                health_status['checks'][name] = {
                    'status': 'healthy' if result else 'unhealthy',
                    'result': result
                }
            except Exception as e:
                health_status['checks'][name] = {
                    'status': 'error',
                    'error': str(e)
                }
                health_status['overall_status'] = 'degraded'
        
        # Check performance thresholds
        await self._check_performance_thresholds(health_status)
        
        # Update overall status
        if any(check['status'] == 'unhealthy' for check in health_status['checks'].values()):
            health_status['overall_status'] = 'unhealthy'
        
        self.health_status = health_status
        return health_status
    
    async def _check_performance_thresholds(self, health_status: Dict[str, Any]):
        """Check performance thresholds."""
        metrics = self.metrics_collector.get_metrics_summary()
        
        # Check response time
        if 'response_time_ms' in metrics['gauges']:
            response_time = metrics['gauges']['response_time_ms']
            if response_time > self.thresholds['response_time_ms']:
                health_status['checks']['response_time'] = {
                    'status': 'degraded',
                    'message': f'Response time {response_time}ms exceeds threshold'
                }
        
        # Check error rate
        if 'error_rate' in metrics['gauges']:
            error_rate = metrics['gauges']['error_rate']
            if error_rate > self.thresholds['error_rate']:
                health_status['checks']['error_rate'] = {
                    'status': 'unhealthy',
                    'message': f'Error rate {error_rate} exceeds threshold'
                }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        return self.health_status
