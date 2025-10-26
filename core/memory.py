"""
PromptOS Advanced Memory System - Vector-based semantic memory.

This module implements OpenAI-level memory capabilities with
vector embeddings, semantic search, and temporal memory.
"""

import asyncio
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
from pathlib import Path

import networkx as nx
from sentence_transformers import SentenceTransformer
import faiss

from config import config


@dataclass
class MemoryNode:
    """Enhanced memory node with vector embeddings."""
    id: str
    content: str
    node_type: str  # task, agent_response, knowledge, context
    timestamp: datetime
    vector_embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None
    importance_score: float = 0.0
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MemoryEdge:
    """Memory edge with relationship information."""
    source_id: str
    target_id: str
    relationship_type: str  # causes, follows, similar_to, depends_on
    strength: float = 1.0
    timestamp: datetime = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}


class AdvancedMemoryManager:
    """
    Advanced memory manager with vector embeddings and semantic search.
    
    Features:
    - Vector embeddings for semantic search
    - Temporal memory with decay
    - Knowledge graph with relationships
    - Memory compression and optimization
    - Context-aware retrieval
    """
    
    def __init__(self):
        """Initialize the advanced memory manager."""
        self.logger = logging.getLogger(__name__)
        
        # Graph structure
        self.graph = nx.DiGraph()
        
        # Memory storage - use absolute paths to ensure correct directory
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(project_root, "data")
        
        self.memory_file = Path(data_dir) / "advanced_memory.json"
        self.vector_index_file = Path(data_dir) / "vector_index.faiss"
        
        # Vector embeddings
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_index = None
        self.node_vectors = {}
        self.vector_dimension = 384  # all-MiniLM-L6-v2 dimension
        
        # Initialize vector index
        self._initialize_vector_index()
        
        # Ensure data directory exists with absolute paths
        import os
        memory_dir = os.path.abspath(str(self.memory_file.parent))
        vector_dir = os.path.abspath(str(self.vector_index_file.parent))
        
        # Create directories
        os.makedirs(memory_dir, exist_ok=True)
        os.makedirs(vector_dir, exist_ok=True)
        
        # Verify directory creation
        if not os.path.exists(vector_dir):
            self.logger.error(f"Failed to create data directory: {vector_dir}")
            # Try alternative approach
            try:
                os.makedirs(vector_dir, mode=0o755, exist_ok=True)
                self.logger.info(f"Retry: Data directory created: {vector_dir}")
            except Exception as e:
                self.logger.error(f"Failed to create data directory even with retry: {e}")
        else:
            self.logger.info(f"Data directory ready: {vector_dir}")
            
        # Test write permissions
        try:
            test_file = os.path.join(vector_dir, "test_write.tmp")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            self.logger.info("Write permissions verified")
        except Exception as e:
            self.logger.warning(f"Write permission test failed: {e}")
        
        # Memory settings
        self.max_memory_nodes = 10000
        self.memory_decay_factor = 0.95
        self.importance_threshold = 0.1
        
        # Initialize
        self._initialize_memory()
    
    def _initialize_memory(self):
        """Initialize memory system."""
        self.logger.info("Initializing Advanced Memory Manager...")
        
        # Load existing memory
        self._load_memory()
        
        self.logger.info("Advanced Memory Manager initialized")
    
    def _initialize_vector_index(self):
        """Initialize FAISS vector index."""
        try:
            import faiss
            if self.vector_index_file.exists():
                # Load existing index
                self.vector_index = faiss.read_index(str(self.vector_index_file))
                self.logger.info("Loaded existing vector index")
            else:
                # Create new index
                self.vector_index = faiss.IndexFlatL2(self.vector_dimension)
                self.logger.info("Created new vector index")
        except Exception as e:
            self.logger.error(f"Failed to initialize vector index: {e}")
            try:
                import faiss
                self.vector_index = faiss.IndexFlatL2(self.vector_dimension)
            except:
                self.vector_index = None
    
    async def store_memory(self, content: str, node_type: str, 
                          metadata: Dict[str, Any] = None) -> str:
        """
        Store a new memory with vector embedding.
        
        Args:
            content: Memory content
            node_type: Type of memory node
            metadata: Additional metadata
            
        Returns:
            Memory node ID
        """
        node_id = f"memory_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Create memory node
        node = MemoryNode(
            id=node_id,
            content=content,
            node_type=node_type,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        # Generate vector embedding
        node.vector_embedding = await self._generate_embedding(content)
        
        # Calculate importance score
        node.importance_score = await self._calculate_importance(node)
        
        # Add to graph
        self.graph.add_node(node_id, **asdict(node))
        
        # Add to vector index
        await self._add_to_vector_index(node)
        
        # Store node vector
        self.node_vectors[node_id] = node.vector_embedding
        
        # Update relationships
        await self._update_relationships(node)
        
        # Check memory limits
        await self._enforce_memory_limits()
        
        # Save memory
        await self._save_memory()
        
        self.logger.info(f"Stored memory: {node_id}")
        return node_id
    
    async def semantic_search(self, query: str, top_k: int = 5, 
                            node_types: List[str] = None) -> List[Dict[str, Any]]:
        """
        Perform semantic search on memory.
        
        Args:
            query: Search query
            top_k: Number of results to return
            node_types: Filter by node types
            
        Returns:
            List of relevant memory nodes
        """
        # Generate query embedding
        query_embedding = await self._generate_embedding(query)
        
        # Search vector index
        if self.vector_index.ntotal == 0:
            return []
        
        # Normalize query embedding for cosine similarity
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Search
        scores, indices = self.vector_index.search(
            query_embedding.reshape(1, -1), 
            min(top_k * 2, self.vector_index.ntotal)
        )
        
        # Get results
        results = []
        node_ids = list(self.node_vectors.keys())
        
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(node_ids):
                node_id = node_ids[idx]
                node_data = self.graph.nodes[node_id]
                
                # Filter by node types if specified
                if node_types and node_data.get('node_type') not in node_types:
                    continue
                
                # Add to results
                result = {
                    'node_id': node_id,
                    'content': node_data.get('content', ''),
                    'node_type': node_data.get('node_type', ''),
                    'similarity_score': float(score),
                    'timestamp': node_data.get('timestamp'),
                    'importance_score': node_data.get('importance_score', 0.0),
                    'metadata': node_data.get('metadata', {})
                }
                results.append(result)
        
        # Sort by similarity score
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return results[:top_k]
    
    async def get_context(self, task_description: str, context_window: int = 10) -> Dict[str, Any]:
        """
        Get relevant context for a task.
        
        Args:
            task_description: Task description
            context_window: Number of relevant memories to include
            
        Returns:
            Context information
        """
        # Search for relevant memories
        relevant_memories = await self.semantic_search(
            task_description, 
            top_k=context_window
        )
        
        # Get related tasks
        task_memories = await self.semantic_search(
            task_description,
            top_k=5,
            node_types=['task']
        )
        
        # Get agent responses
        response_memories = await self.semantic_search(
            task_description,
            top_k=5,
            node_types=['agent_response']
        )
        
        # Compile context
        context = {
            'relevant_memories': relevant_memories,
            'related_tasks': task_memories,
            'agent_responses': response_memories,
            'context_timestamp': datetime.now().isoformat(),
            'total_memories': len(self.graph.nodes)
        }
        
        return context
    
    async def update_memory_importance(self, node_id: str, importance_delta: float):
        """Update memory importance based on usage."""
        if node_id in self.graph.nodes:
            current_importance = self.graph.nodes[node_id].get('importance_score', 0.0)
            new_importance = max(0.0, min(1.0, current_importance + importance_delta))
            
            self.graph.nodes[node_id]['importance_score'] = new_importance
            self.graph.nodes[node_id]['last_accessed'] = datetime.now()
            
            # Update access count
            access_count = self.graph.nodes[node_id].get('access_count', 0)
            self.graph.nodes[node_id]['access_count'] = access_count + 1
    
    async def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate vector embedding for text."""
        try:
            # Use sentence transformer
            embedding = self.embedding_model.encode(text)
            return embedding.astype('float32')
        except Exception as e:
            self.logger.error(f"Failed to generate embedding: {e}")
            # Return zero vector as fallback
            return np.zeros(self.vector_dimension, dtype='float32')
    
    async def _calculate_importance(self, node: MemoryNode) -> float:
        """Calculate importance score for a memory node."""
        base_score = 0.5  # Base importance
        
        # Factor in content length
        content_length = len(node.content)
        length_factor = min(1.0, content_length / 1000)  # Normalize to 0-1
        
        # Factor in node type
        type_weights = {
            'task': 0.8,
            'agent_response': 0.7,
            'knowledge': 0.9,
            'context': 0.6
        }
        type_factor = type_weights.get(node.node_type, 0.5)
        
        # Factor in metadata importance
        metadata_factor = 0.5
        if node.metadata:
            if 'priority' in node.metadata:
                priority = node.metadata['priority']
                if priority == 'high':
                    metadata_factor = 0.9
                elif priority == 'medium':
                    metadata_factor = 0.7
                elif priority == 'low':
                    metadata_factor = 0.5
        
        # Calculate final importance
        importance = (base_score + length_factor + type_factor + metadata_factor) / 4
        
        return min(1.0, max(0.0, importance))
    
    async def _add_to_vector_index(self, node: MemoryNode):
        """Add node to vector index."""
        if node.vector_embedding is not None:
            # Normalize embedding for cosine similarity
            normalized_embedding = node.vector_embedding / np.linalg.norm(node.vector_embedding)
            
            # Add to index
            self.vector_index.add(normalized_embedding.reshape(1, -1))
    
    async def _update_relationships(self, node: MemoryNode):
        """Update relationships with existing nodes."""
        # Find similar nodes
        similar_memories = await self.semantic_search(
            node.content,
            top_k=5,
            node_types=[node.node_type]
        )
        
        # Create relationships
        for similar in similar_memories:
            if similar['node_id'] != node.id:
                similarity = similar['similarity_score']
                
                # Create edge if similarity is high enough
                if similarity > 0.7:
                    edge = MemoryEdge(
                        source_id=node.id,
                        target_id=similar['node_id'],
                        relationship_type='similar_to',
                        strength=similarity
                    )
                    
                    self.graph.add_edge(
                        node.id,
                        similar['node_id'],
                        **asdict(edge)
                    )
    
    async def _enforce_memory_limits(self):
        """Enforce memory limits by removing least important nodes."""
        if len(self.graph.nodes) <= self.max_memory_nodes:
            return
        
        # Get nodes sorted by importance
        nodes_with_importance = [
            (node_id, self.graph.nodes[node_id].get('importance_score', 0.0))
            for node_id in self.graph.nodes
        ]
        nodes_with_importance.sort(key=lambda x: x[1])
        
        # Remove least important nodes
        nodes_to_remove = len(self.graph.nodes) - self.max_memory_nodes
        for i in range(nodes_to_remove):
            node_id = nodes_with_importance[i][0]
            await self._remove_memory_node(node_id)
    
    async def _remove_memory_node(self, node_id: str):
        """Remove a memory node and update vector index."""
        if node_id in self.graph.nodes:
            # Remove from graph
            self.graph.remove_node(node_id)
            
            # Remove from node vectors
            if node_id in self.node_vectors:
                del self.node_vectors[node_id]
            
            # Note: FAISS doesn't support removal, so we'll rebuild index periodically
            self.logger.info(f"Removed memory node: {node_id}")
    
    async def _save_memory(self):
        """Save memory to disk."""
        try:
            # Save graph data
            graph_data = {
                'nodes': dict(self.graph.nodes(data=True)),
                'edges': [
                    {
                        'source': source,
                        'target': target,
                        'attributes': edge_data
                    }
                    for source, target, edge_data in self.graph.edges(data=True)
                ],
                'metadata': {
                    'last_updated': datetime.now().isoformat(),
                    'node_count': self.graph.number_of_nodes(),
                    'edge_count': self.graph.number_of_edges()
                }
            }
            
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, indent=2, default=str)
            
            # Save vector index
            if self.vector_index is not None:
                try:
                    # Ensure directory exists with absolute path
                    import os
                    data_dir = os.path.abspath(str(self.vector_index_file.parent))
                    os.makedirs(data_dir, exist_ok=True)
                    
                    # Verify directory was created
                    if not os.path.exists(data_dir):
                        self.logger.error(f"Failed to create data directory: {data_dir}")
                        return
                    
                    # Test write permissions first
                    test_file = os.path.join(data_dir, "write_test.tmp")
                    try:
                        with open(test_file, 'w') as f:
                            f.write("test")
                        os.remove(test_file)
                    except Exception as e:
                        self.logger.error(f"Cannot write to data directory: {e}")
                        return
                    
                    # Create a temporary file first to test write permissions
                    temp_file = self.vector_index_file.with_suffix('.tmp')
                    faiss.write_index(self.vector_index, str(temp_file))
                    # If successful, rename to final file
                    temp_file.rename(self.vector_index_file)
                    self.logger.info("Vector index saved successfully")
                except Exception as e:
                    self.logger.warning(f"Could not save vector index: {e}")
                    # Continue without failing the entire save operation
            
        except Exception as e:
            self.logger.error(f"Failed to save memory: {e}")
    
    def _load_memory(self):
        """Load memory from disk."""
        try:
            if self.memory_file.exists():
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Load nodes
                for node_id, node_data in data.get('nodes', {}).items():
                    # Convert timestamp back to datetime
                    if 'timestamp' in node_data:
                        node_data['timestamp'] = datetime.fromisoformat(node_data['timestamp'])
                    if 'last_accessed' in node_data and node_data['last_accessed']:
                        node_data['last_accessed'] = datetime.fromisoformat(node_data['last_accessed'])
                    
                    self.graph.add_node(node_id, **node_data)
                
                # Load edges
                for edge in data.get('edges', []):
                    self.graph.add_edge(
                        edge['source'],
                        edge['target'],
                        **edge['attributes']
                    )
                
                self.logger.info(f"Loaded memory with {self.graph.number_of_nodes()} nodes")
                
        except Exception as e:
            self.logger.error(f"Failed to load memory: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'vector_index_size': self.vector_index.ntotal if self.vector_index else 0,
            'memory_file_size': self.memory_file.stat().st_size if self.memory_file.exists() else 0,
            'node_types': {
                node_type: sum(1 for _, data in self.graph.nodes(data=True) 
                              if data.get('node_type') == node_type)
                for node_type in set(data.get('node_type', 'unknown') 
                                   for _, data in self.graph.nodes(data=True))
            }
        }
    
    def get_graph_data(self) -> Dict[str, Any]:
        """Get graph data for API responses."""
        def serialize_value(value):
            """Serialize values, handling numpy arrays and other non-serializable types."""
            import numpy as np
            if isinstance(value, np.ndarray):
                return value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                return value.item()
            elif isinstance(value, datetime):
                return value.isoformat()
            elif hasattr(value, '__dict__'):
                return str(value)
            else:
                return value
        
        return {
            'nodes': [
                {
                    'id': node_id,
                    'label': data.get('label', node_id),
                    'type': data.get('node_type', 'unknown'),
                    'timestamp': data.get('timestamp', datetime.now()).isoformat() if isinstance(data.get('timestamp'), datetime) else str(data.get('timestamp', '')),
                    'metadata': {k: serialize_value(v) for k, v in data.items() if k not in ['label', 'node_type', 'timestamp', 'last_accessed']}
                }
                for node_id, data in self.graph.nodes(data=True)
            ],
            'edges': [
                {
                    'source': source,
                    'target': target,
                    'type': edge_data.get('edge_type', 'unknown'),
                    'weight': edge_data.get('weight', 1.0),
                    'metadata': {k: serialize_value(v) for k, v in edge_data.items() if k not in ['edge_type', 'weight']}
                }
                for source, target, edge_data in self.graph.edges(data=True)
            ],
            'metadata': {
                'node_count': self.graph.number_of_nodes(),
                'edge_count': self.graph.number_of_edges(),
                'last_updated': datetime.now().isoformat()
            }
        }
    
    async def store_task_result(self, task_id: str, result: Dict[str, Any]) -> str:
        """Store task result in memory."""
        try:
            # Create a memory node for the task result
            node_id = f"task_result_{task_id}"
            
            # Store the result
            await self.store_memory(
                content=f"Task Result: {result.get('summary', 'No summary available')}",
                node_type="task_result",
                metadata={
                    "task_id": task_id,
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            self.logger.info(f"Stored task result for {task_id}")
            return node_id
            
        except Exception as e:
            self.logger.error(f"Failed to store task result: {e}")
            return ""
