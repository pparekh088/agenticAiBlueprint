# Standard RAG Implementation Guide for Enterprise Teams

## Quick Start RAG Template

This guide provides production-ready RAG implementation patterns for enterprise financial institution teams.

## Table of Contents
1. [RAG Pipeline Implementation](#rag-pipeline-implementation)
2. [Vector Database Setup](#vector-database-setup)
3. [Document Processing](#document-processing)
4. [Query Enhancement](#query-enhancement)
5. [Evaluation Framework](#evaluation-framework)
6. [Production Deployment](#production-deployment)

---

## RAG Pipeline Implementation

### Core RAG Service

```python
# rag_service.py
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from datetime import datetime
import hashlib
import json

@dataclass
class RAGConfig:
    """Configuration for RAG pipeline"""
    embedding_model: str = "text-embedding-3-large"
    llm_model: str = "gpt-4"
    chunk_size: int = 512
    chunk_overlap: int = 128
    top_k: int = 10
    similarity_threshold: float = 0.75
    max_context_tokens: int = 4000
    temperature: float = 0.7
    enable_reranking: bool = True
    enable_caching: bool = True
    cache_ttl: int = 3600

class EnterpriseRAGPipeline:
    """Production-ready RAG pipeline for financial institutions"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.vector_store = None
        self.embedder = None
        self.llm_client = None
        self.reranker = None
        self.cache = {}
        self.metrics = {
            "queries_processed": 0,
            "avg_latency": 0,
            "cache_hits": 0,
            "errors": 0
        }
        
    def initialize(self):
        """Initialize all components"""
        self._setup_vector_store()
        self._setup_embedder()
        self._setup_llm()
        self._setup_reranker()
        
    def _setup_vector_store(self):
        """Initialize vector database connection"""
        # Implementation for your chosen vector DB
        pass
        
    def _setup_embedder(self):
        """Initialize embedding model"""
        # Implementation for embedding service
        pass
        
    def _setup_llm(self):
        """Initialize LLM client"""
        # Implementation for LLM service
        pass
        
    def _setup_reranker(self):
        """Initialize reranking model"""
        # Implementation for reranking service
        pass
    
    def process_query(
        self, 
        query: str, 
        user_context: Optional[Dict[str, Any]] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a user query through the RAG pipeline
        
        Args:
            query: User's question
            user_context: Additional context about the user
            filters: Metadata filters for retrieval
            
        Returns:
            Response dictionary with answer and metadata
        """
        start_time = datetime.now()
        
        try:
            # Check cache
            cache_key = self._generate_cache_key(query, filters)
            if self.config.enable_caching and cache_key in self.cache:
                self.metrics["cache_hits"] += 1
                return self.cache[cache_key]
            
            # Enhance query
            enhanced_query = self._enhance_query(query, user_context)
            
            # Retrieve relevant documents
            retrieved_docs = self._retrieve_documents(
                enhanced_query, 
                filters, 
                self.config.top_k
            )
            
            # Rerank if enabled
            if self.config.enable_reranking:
                retrieved_docs = self._rerank_documents(
                    query, 
                    retrieved_docs
                )
            
            # Build context
            context = self._build_context(retrieved_docs)
            
            # Generate response
            response = self._generate_response(
                query, 
                context, 
                user_context
            )
            
            # Prepare result
            result = {
                "answer": response["text"],
                "sources": [doc["metadata"] for doc in retrieved_docs[:5]],
                "confidence": response.get("confidence", 0.0),
                "latency_ms": (datetime.now() - start_time).total_seconds() * 1000,
                "timestamp": datetime.now().isoformat()
            }
            
            # Update cache
            if self.config.enable_caching:
                self.cache[cache_key] = result
            
            # Update metrics
            self._update_metrics(result["latency_ms"])
            
            return result
            
        except Exception as e:
            self.metrics["errors"] += 1
            return {
                "error": str(e),
                "answer": "I apologize, but I encountered an error processing your request.",
                "timestamp": datetime.now().isoformat()
            }
    
    def _generate_cache_key(self, query: str, filters: Optional[Dict]) -> str:
        """Generate cache key for query"""
        key_data = f"{query}_{json.dumps(filters or {}, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _enhance_query(self, query: str, context: Optional[Dict]) -> str:
        """Enhance query with context and expansion"""
        enhanced = query
        
        # Add temporal context
        if context and "date_range" in context:
            enhanced += f" (considering data from {context['date_range']})"
        
        # Add user role context
        if context and "user_role" in context:
            enhanced += f" (from perspective of {context['user_role']})"
        
        # Query expansion could be added here
        
        return enhanced
    
    def _retrieve_documents(
        self, 
        query: str, 
        filters: Optional[Dict],
        top_k: int
    ) -> List[Dict]:
        """Retrieve relevant documents from vector store"""
        # Embed query
        query_embedding = self.embedder.embed(query)
        
        # Search vector store
        results = self.vector_store.search(
            embedding=query_embedding,
            filters=filters,
            top_k=top_k * 2,  # Retrieve more for filtering
            threshold=self.config.similarity_threshold
        )
        
        # Post-process results
        processed_results = []
        for result in results:
            if self._validate_document(result):
                processed_results.append({
                    "content": result["content"],
                    "metadata": result["metadata"],
                    "score": result["score"]
                })
        
        return processed_results[:top_k]
    
    def _validate_document(self, doc: Dict) -> bool:
        """Validate document for compliance and quality"""
        # Check for required metadata
        if "source" not in doc.get("metadata", {}):
            return False
        
        # Check content quality
        if len(doc.get("content", "")) < 50:
            return False
        
        # Additional validation rules
        
        return True
    
    def _rerank_documents(self, query: str, docs: List[Dict]) -> List[Dict]:
        """Rerank documents for better relevance"""
        if not self.reranker:
            return docs
        
        # Prepare pairs for reranking
        pairs = [(query, doc["content"]) for doc in docs]
        
        # Get reranking scores
        scores = self.reranker.score(pairs)
        
        # Sort by new scores
        for i, doc in enumerate(docs):
            doc["rerank_score"] = scores[i]
        
        return sorted(docs, key=lambda x: x["rerank_score"], reverse=True)
    
    def _build_context(self, docs: List[Dict]) -> str:
        """Build context from retrieved documents"""
        context_parts = []
        total_tokens = 0
        
        for i, doc in enumerate(docs):
            # Estimate tokens (rough approximation)
            doc_tokens = len(doc["content"]) // 4
            
            if total_tokens + doc_tokens > self.config.max_context_tokens:
                break
            
            # Format document
            source = doc["metadata"].get("source", "Unknown")
            date = doc["metadata"].get("date", "N/A")
            
            context_parts.append(
                f"[Source {i+1}: {source} | Date: {date}]\n{doc['content']}\n"
            )
            
            total_tokens += doc_tokens
        
        return "\n---\n".join(context_parts)
    
    def _generate_response(
        self, 
        query: str, 
        context: str,
        user_context: Optional[Dict]
    ) -> Dict[str, Any]:
        """Generate response using LLM"""
        
        # Build system prompt
        system_prompt = self._build_system_prompt(user_context)
        
        # Build user prompt
        user_prompt = f"""Based on the following context, please answer the question.
        
Context:
{context}

Question: {query}

Please provide a comprehensive answer based solely on the provided context. If the context doesn't contain enough information, please state that clearly."""
        
        # Call LLM
        response = self.llm_client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self.config.temperature,
            max_tokens=1000
        )
        
        return {
            "text": response["text"],
            "confidence": self._calculate_confidence(response, context)
        }
    
    def _build_system_prompt(self, user_context: Optional[Dict]) -> str:
        """Build system prompt based on context"""
        base_prompt = """You are an AI assistant for a financial institution. 
You provide accurate, compliant, and helpful responses based on official documentation and policies.
Always maintain professional tone and ensure regulatory compliance."""
        
        if user_context:
            if "department" in user_context:
                base_prompt += f"\nUser is from {user_context['department']} department."
            if "clearance_level" in user_context:
                base_prompt += f"\nUser has {user_context['clearance_level']} clearance."
        
        return base_prompt
    
    def _calculate_confidence(self, response: Dict, context: str) -> float:
        """Calculate confidence score for response"""
        # Simple heuristic - can be replaced with more sophisticated method
        confidence = 0.5
        
        # Check if response references the context
        if any(phrase in response["text"].lower() for phrase in 
               ["according to", "based on", "the document states"]):
            confidence += 0.2
        
        # Check context quality
        if len(context) > 1000:
            confidence += 0.2
        
        # Check for uncertainty markers
        if any(phrase in response["text"].lower() for phrase in 
               ["i'm not sure", "unclear", "no information"]):
            confidence -= 0.3
        
        return max(0.0, min(1.0, confidence))
    
    def _update_metrics(self, latency: float):
        """Update performance metrics"""
        self.metrics["queries_processed"] += 1
        
        # Update rolling average latency
        n = self.metrics["queries_processed"]
        self.metrics["avg_latency"] = (
            (self.metrics["avg_latency"] * (n - 1) + latency) / n
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return {
            **self.metrics,
            "cache_hit_rate": (
                self.metrics["cache_hits"] / 
                max(1, self.metrics["queries_processed"])
            ),
            "error_rate": (
                self.metrics["errors"] / 
                max(1, self.metrics["queries_processed"])
            )
        }
```

---

## Vector Database Setup

### Qdrant Configuration

```python
# vector_store.py
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue
)
import uuid
from typing import List, Dict, Any, Optional

class EnterpriseVectorStore:
    """Enterprise-grade vector store implementation"""
    
    def __init__(self, 
                 host: str = "localhost",
                 port: int = 6333,
                 collection_name: str = "enterprise_docs",
                 vector_size: int = 1536):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Ensure collection exists with proper configuration"""
        collections = self.client.get_collections().collections
        
        if not any(c.name == self.collection_name for c in collections):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                ),
                # Optimizations for large-scale deployments
                optimizers_config={
                    "indexing_threshold": 20000,
                    "memmap_threshold": 50000,
                },
                # Enable sharding for scalability
                shard_number=6,
                replication_factor=2
            )
    
    def insert_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Insert documents into vector store"""
        points = []
        ids = []
        
        for doc in documents:
            doc_id = str(uuid.uuid4())
            ids.append(doc_id)
            
            point = PointStruct(
                id=doc_id,
                vector=doc["embedding"],
                payload={
                    "content": doc["content"],
                    "metadata": doc.get("metadata", {}),
                    "timestamp": doc.get("timestamp", datetime.now().isoformat()),
                    "source": doc.get("source", "unknown"),
                    "classification": doc.get("classification", "public")
                }
            )
            points.append(point)
        
        # Batch insert for efficiency
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True
        )
        
        return ids
    
    def search(self,
               embedding: List[float],
               filters: Optional[Dict[str, Any]] = None,
               top_k: int = 10,
               threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        
        # Build filter conditions
        filter_conditions = []
        if filters:
            for key, value in filters.items():
                filter_conditions.append(
                    FieldCondition(
                        key=f"metadata.{key}",
                        match=MatchValue(value=value)
                    )
                )
        
        # Perform search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=embedding,
            query_filter=Filter(must=filter_conditions) if filter_conditions else None,
            limit=top_k,
            score_threshold=threshold
        )
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "id": result.id,
                "score": result.score,
                "content": result.payload.get("content", ""),
                "metadata": result.payload.get("metadata", {}),
                "source": result.payload.get("source", "unknown")
            })
        
        return formatted_results
    
    def delete_documents(self, ids: List[str]):
        """Delete documents by IDs"""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=ids
        )
    
    def update_metadata(self, doc_id: str, metadata: Dict[str, Any]):
        """Update document metadata"""
        self.client.set_payload(
            collection_name=self.collection_name,
            payload={"metadata": metadata},
            points=[doc_id]
        )
```

---

## Document Processing

### Advanced Document Chunking

```python
# document_processor.py
import re
from typing import List, Dict, Any, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
import PyPDF2
import docx
import pandas as pd
from bs4 import BeautifulSoup

class EnterpriseDocumentProcessor:
    """Document processing for various enterprise formats"""
    
    def __init__(self, 
                 chunk_size: int = 512,
                 chunk_overlap: int = 128,
                 preserve_tables: bool = True):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.preserve_tables = preserve_tables
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
    
    def process_document(self, 
                        file_path: str,
                        metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Process document into chunks with metadata"""
        
        # Detect file type
        file_type = self._detect_file_type(file_path)
        
        # Extract content based on type
        if file_type == "pdf":
            content = self._process_pdf(file_path)
        elif file_type == "docx":
            content = self._process_docx(file_path)
        elif file_type == "xlsx":
            content = self._process_excel(file_path)
        elif file_type == "html":
            content = self._process_html(file_path)
        else:
            content = self._process_text(file_path)
        
        # Clean and normalize content
        content = self._clean_content(content)
        
        # Smart chunking
        chunks = self._smart_chunk(content)
        
        # Add metadata to chunks
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                **(metadata or {}),
                "chunk_index": i,
                "total_chunks": len(chunks),
                "file_path": file_path,
                "file_type": file_type
            }
            
            processed_chunks.append({
                "content": chunk,
                "metadata": chunk_metadata
            })
        
        return processed_chunks
    
    def _detect_file_type(self, file_path: str) -> str:
        """Detect file type from extension"""
        extension = file_path.lower().split('.')[-1]
        return extension
    
    def _process_pdf(self, file_path: str) -> str:
        """Extract text from PDF"""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def _process_docx(self, file_path: str) -> str:
        """Extract text from DOCX"""
        doc = docx.Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        
        # Extract tables if needed
        if self.preserve_tables:
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join(cell.text for cell in row.cells)
                    text += row_text + "\n"
        
        return text
    
    def _process_excel(self, file_path: str) -> str:
        """Extract text from Excel"""
        text = ""
        xls = pd.ExcelFile(file_path)
        
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name)
            # Convert DataFrame to text
            text += f"\n=== Sheet: {sheet_name} ===\n"
            text += df.to_string() + "\n"
        
        return text
    
    def _process_html(self, file_path: str) -> str:
        """Extract text from HTML"""
        with open(file_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file.read(), 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text
    
    def _process_text(self, file_path: str) -> str:
        """Process plain text file"""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def _clean_content(self, content: str) -> str:
        """Clean and normalize content"""
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove special characters that might cause issues
        content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', content)
        
        # Normalize quotes
        content = content.replace('"', '"').replace('"', '"')
        content = content.replace(''', "'").replace(''', "'")
        
        return content.strip()
    
    def _smart_chunk(self, content: str) -> List[str]:
        """Smart chunking that preserves context"""
        
        # First, try to identify natural sections
        sections = self._identify_sections(content)
        
        chunks = []
        for section in sections:
            if len(section) <= self.chunk_size:
                chunks.append(section)
            else:
                # Split large sections
                section_chunks = self.text_splitter.split_text(section)
                chunks.extend(section_chunks)
        
        return chunks
    
    def _identify_sections(self, content: str) -> List[str]:
        """Identify natural sections in content"""
        # Look for common section markers
        section_patterns = [
            r'\n#{1,6}\s+.*?\n',  # Markdown headers
            r'\n[A-Z][A-Z\s]+\n',  # All caps headers
            r'\n\d+\.\s+.*?\n',    # Numbered sections
            r'\n[A-Z]\.\s+.*?\n',  # Letter sections
        ]
        
        # Find all section positions
        positions = []
        for pattern in section_patterns:
            for match in re.finditer(pattern, content):
                positions.append(match.start())
        
        if not positions:
            return [content]
        
        # Sort positions
        positions = sorted(set(positions))
        positions.append(len(content))
        
        # Extract sections
        sections = []
        start = 0
        for pos in positions:
            section = content[start:pos].strip()
            if section:
                sections.append(section)
            start = pos
        
        return sections
```

---

## Query Enhancement

### Advanced Query Processing

```python
# query_enhancer.py
from typing import List, Dict, Any, Optional
import spacy
from sentence_transformers import SentenceTransformer

class QueryEnhancer:
    """Enhance queries for better retrieval"""
    
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.domain_synonyms = self._load_domain_synonyms()
        self.query_templates = self._load_query_templates()
    
    def enhance_query(self, 
                     query: str,
                     context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Enhance query with multiple strategies"""
        
        # Parse query
        doc = self.nlp(query)
        
        # Extract entities
        entities = self._extract_entities(doc)
        
        # Identify query type
        query_type = self._classify_query(query)
        
        # Expand with synonyms
        expanded_query = self._expand_with_synonyms(query)
        
        # Generate sub-queries
        sub_queries = self._generate_sub_queries(query, query_type)
        
        # Add temporal context
        if context and "time_range" in context:
            temporal_query = self._add_temporal_context(query, context["time_range"])
        else:
            temporal_query = query
        
        return {
            "original": query,
            "expanded": expanded_query,
            "temporal": temporal_query,
            "sub_queries": sub_queries,
            "entities": entities,
            "query_type": query_type
        }
    
    def _extract_entities(self, doc) -> List[Dict[str, str]]:
        """Extract named entities from query"""
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })
        return entities
    
    def _classify_query(self, query: str) -> str:
        """Classify query type"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["what", "define", "explain"]):
            return "definition"
        elif any(word in query_lower for word in ["how", "process", "procedure"]):
            return "procedural"
        elif any(word in query_lower for word in ["why", "reason", "cause"]):
            return "causal"
        elif any(word in query_lower for word in ["when", "date", "time"]):
            return "temporal"
        elif any(word in query_lower for word in ["compare", "difference", "versus"]):
            return "comparison"
        else:
            return "general"
    
    def _expand_with_synonyms(self, query: str) -> str:
        """Expand query with domain-specific synonyms"""
        expanded = query
        
        for term, synonyms in self.domain_synonyms.items():
            if term.lower() in query.lower():
                # Add synonyms in parentheses
                synonym_str = " OR ".join(synonyms)
                expanded += f" ({synonym_str})"
        
        return expanded
    
    def _generate_sub_queries(self, query: str, query_type: str) -> List[str]:
        """Generate sub-queries based on query type"""
        sub_queries = [query]  # Always include original
        
        if query_type == "procedural":
            sub_queries.extend([
                f"steps for {query}",
                f"process of {query}",
                f"how to {query}"
            ])
        elif query_type == "definition":
            sub_queries.extend([
                f"definition of {query}",
                f"what is {query}",
                f"meaning of {query}"
            ])
        elif query_type == "comparison":
            sub_queries.extend([
                f"differences in {query}",
                f"comparison of {query}",
                f"pros and cons {query}"
            ])
        
        return list(set(sub_queries))  # Remove duplicates
    
    def _add_temporal_context(self, query: str, time_range: str) -> str:
        """Add temporal context to query"""
        return f"{query} (within {time_range})"
    
    def _load_domain_synonyms(self) -> Dict[str, List[str]]:
        """Load financial domain synonyms"""
        return {
            "aml": ["anti-money laundering", "money laundering prevention"],
            "kyc": ["know your customer", "customer identification"],
            "roi": ["return on investment", "investment return"],
            "apr": ["annual percentage rate", "yearly interest rate"],
            "etf": ["exchange traded fund", "traded fund"],
            "ipo": ["initial public offering", "going public"],
            "m&a": ["mergers and acquisitions", "merger", "acquisition"],
            "p&l": ["profit and loss", "income statement"],
            "nav": ["net asset value", "fund value"],
            "eps": ["earnings per share", "share earnings"]
        }
    
    def _load_query_templates(self) -> Dict[str, List[str]]:
        """Load query templates for different intents"""
        return {
            "compliance": [
                "What are the compliance requirements for {topic}?",
                "How do we ensure compliance with {topic}?",
                "What are the regulatory guidelines for {topic}?"
            ],
            "risk": [
                "What are the risks associated with {topic}?",
                "How do we assess risk for {topic}?",
                "What is the risk profile of {topic}?"
            ],
            "process": [
                "What is the process for {topic}?",
                "How do we handle {topic}?",
                "What are the steps for {topic}?"
            ]
        }
```

---

## Evaluation Framework

### RAG Evaluation Metrics

```python
# evaluation.py
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score
import json
from datetime import datetime

class RAGEvaluator:
    """Comprehensive RAG evaluation framework"""
    
    def __init__(self):
        self.metrics_history = []
        self.thresholds = {
            "faithfulness": 0.95,
            "relevance": 0.90,
            "coherence": 0.85,
            "latency_p95": 2000,  # ms
            "precision": 0.85,
            "recall": 0.90
        }
    
    def evaluate_retrieval(self, 
                          queries: List[str],
                          retrieved_docs: List[List[Dict]],
                          ground_truth: List[List[str]]) -> Dict[str, float]:
        """Evaluate retrieval quality"""
        
        precisions = []
        recalls = []
        mrrs = []
        
        for query, retrieved, truth in zip(queries, retrieved_docs, ground_truth):
            # Calculate precision and recall
            retrieved_ids = [doc["id"] for doc in retrieved]
            
            precision = self._calculate_precision(retrieved_ids, truth)
            recall = self._calculate_recall(retrieved_ids, truth)
            mrr = self._calculate_mrr(retrieved_ids, truth)
            
            precisions.append(precision)
            recalls.append(recall)
            mrrs.append(mrr)
        
        return {
            "precision_mean": np.mean(precisions),
            "precision_std": np.std(precisions),
            "recall_mean": np.mean(recalls),
            "recall_std": np.std(recalls),
            "mrr_mean": np.mean(mrrs),
            "mrr_std": np.std(mrrs),
            "f1_score": 2 * (np.mean(precisions) * np.mean(recalls)) / 
                       (np.mean(precisions) + np.mean(recalls))
        }
    
    def evaluate_generation(self,
                          questions: List[str],
                          generated_answers: List[str],
                          contexts: List[str],
                          ground_truth_answers: Optional[List[str]] = None) -> Dict[str, float]:
        """Evaluate generation quality"""
        
        faithfulness_scores = []
        relevance_scores = []
        coherence_scores = []
        
        for q, a, c in zip(questions, generated_answers, contexts):
            faithfulness = self._evaluate_faithfulness(a, c)
            relevance = self._evaluate_relevance(a, q)
            coherence = self._evaluate_coherence(a)
            
            faithfulness_scores.append(faithfulness)
            relevance_scores.append(relevance)
            coherence_scores.append(coherence)
        
        results = {
            "faithfulness_mean": np.mean(faithfulness_scores),
            "faithfulness_std": np.std(faithfulness_scores),
            "relevance_mean": np.mean(relevance_scores),
            "relevance_std": np.std(relevance_scores),
            "coherence_mean": np.mean(coherence_scores),
            "coherence_std": np.std(coherence_scores)
        }
        
        # Add comparison with ground truth if available
        if ground_truth_answers:
            similarity_scores = []
            for generated, truth in zip(generated_answers, ground_truth_answers):
                similarity = self._calculate_similarity(generated, truth)
                similarity_scores.append(similarity)
            
            results["similarity_to_truth_mean"] = np.mean(similarity_scores)
            results["similarity_to_truth_std"] = np.std(similarity_scores)
        
        return results
    
    def evaluate_end_to_end(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Comprehensive end-to-end evaluation"""
        
        retrieval_metrics = []
        generation_metrics = []
        latency_metrics = []
        
        for test_case in test_cases:
            # Run RAG pipeline
            start_time = datetime.now()
            result = test_case["rag_pipeline"].process_query(test_case["query"])
            latency = (datetime.now() - start_time).total_seconds() * 1000
            
            # Evaluate retrieval
            if "expected_docs" in test_case:
                retrieval_score = self._evaluate_retrieval_single(
                    result["sources"],
                    test_case["expected_docs"]
                )
                retrieval_metrics.append(retrieval_score)
            
            # Evaluate generation
            generation_score = self._evaluate_generation_single(
                test_case["query"],
                result["answer"],
                test_case.get("expected_answer")
            )
            generation_metrics.append(generation_score)
            
            # Track latency
            latency_metrics.append(latency)
        
        # Aggregate results
        results = {
            "timestamp": datetime.now().isoformat(),
            "num_test_cases": len(test_cases),
            "retrieval": {
                "mean_score": np.mean(retrieval_metrics) if retrieval_metrics else None,
                "std_score": np.std(retrieval_metrics) if retrieval_metrics else None
            },
            "generation": {
                "mean_score": np.mean(generation_metrics),
                "std_score": np.std(generation_metrics)
            },
            "latency": {
                "mean": np.mean(latency_metrics),
                "p50": np.percentile(latency_metrics, 50),
                "p95": np.percentile(latency_metrics, 95),
                "p99": np.percentile(latency_metrics, 99)
            },
            "pass_rate": self._calculate_pass_rate(
                retrieval_metrics, 
                generation_metrics, 
                latency_metrics
            )
        }
        
        # Store metrics history
        self.metrics_history.append(results)
        
        return results
    
    def _calculate_precision(self, retrieved: List[str], relevant: List[str]) -> float:
        """Calculate precision at k"""
        if not retrieved:
            return 0.0
        
        relevant_set = set(relevant)
        retrieved_set = set(retrieved)
        
        return len(retrieved_set & relevant_set) / len(retrieved_set)
    
    def _calculate_recall(self, retrieved: List[str], relevant: List[str]) -> float:
        """Calculate recall at k"""
        if not relevant:
            return 0.0
        
        relevant_set = set(relevant)
        retrieved_set = set(retrieved)
        
        return len(retrieved_set & relevant_set) / len(relevant_set)
    
    def _calculate_mrr(self, retrieved: List[str], relevant: List[str]) -> float:
        """Calculate Mean Reciprocal Rank"""
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant:
                return 1.0 / (i + 1)
        return 0.0
    
    def _evaluate_faithfulness(self, answer: str, context: str) -> float:
        """Evaluate if answer is faithful to context"""
        # Simplified implementation - in production, use LLM-based evaluation
        # or more sophisticated NLI models
        
        # Check if key phrases from answer appear in context
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())
        
        overlap = len(answer_words & context_words) / len(answer_words)
        
        return min(1.0, overlap)
    
    def _evaluate_relevance(self, answer: str, question: str) -> float:
        """Evaluate if answer is relevant to question"""
        # Simplified implementation - use embedding similarity in production
        
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        
        overlap = len(question_words & answer_words) / len(question_words)
        
        return min(1.0, overlap * 1.5)  # Boost score slightly
    
    def _evaluate_coherence(self, answer: str) -> float:
        """Evaluate answer coherence"""
        # Check for basic coherence indicators
        
        score = 1.0
        
        # Check for incomplete sentences
        if not answer.endswith(('.', '!', '?')):
            score -= 0.1
        
        # Check for reasonable length
        word_count = len(answer.split())
        if word_count < 10:
            score -= 0.2
        elif word_count > 500:
            score -= 0.1
        
        # Check for repetition
        sentences = answer.split('.')
        if len(sentences) != len(set(sentences)):
            score -= 0.2
        
        return max(0.0, score)
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between texts"""
        # Simplified - use embedding similarity in production
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union
    
    def _evaluate_retrieval_single(self, 
                                  retrieved: List[Dict],
                                  expected: List[str]) -> float:
        """Evaluate single retrieval result"""
        retrieved_ids = [doc.get("id", "") for doc in retrieved]
        return self._calculate_precision(retrieved_ids, expected)
    
    def _evaluate_generation_single(self,
                                  question: str,
                                  answer: str,
                                  expected: Optional[str] = None) -> float:
        """Evaluate single generation result"""
        score = self._evaluate_coherence(answer)
        
        if expected:
            similarity = self._calculate_similarity(answer, expected)
            score = (score + similarity) / 2
        
        return score
    
    def _calculate_pass_rate(self,
                            retrieval_metrics: List[float],
                            generation_metrics: List[float],
                            latency_metrics: List[float]) -> float:
        """Calculate overall pass rate based on thresholds"""
        
        passed = 0
        total = len(generation_metrics)
        
        for i in range(total):
            if (generation_metrics[i] >= self.thresholds["coherence"] and
                latency_metrics[i] <= self.thresholds["latency_p95"]):
                passed += 1
        
        return passed / total if total > 0 else 0.0
    
    def generate_report(self) -> str:
        """Generate evaluation report"""
        if not self.metrics_history:
            return "No evaluation data available"
        
        latest = self.metrics_history[-1]
        
        report = f"""
RAG Evaluation Report
Generated: {latest['timestamp']}
===================================

Test Cases Evaluated: {latest['num_test_cases']}

Retrieval Performance:
- Mean Score: {latest['retrieval']['mean_score']:.3f} if latest['retrieval']['mean_score'] else 'N/A'}
- Std Dev: {latest['retrieval']['std_score']:.3f} if latest['retrieval']['std_score'] else 'N/A'}

Generation Performance:
- Mean Score: {latest['generation']['mean_score']:.3f}
- Std Dev: {latest['generation']['std_score']:.3f}

Latency Metrics:
- Mean: {latest['latency']['mean']:.1f}ms
- P50: {latest['latency']['p50']:.1f}ms
- P95: {latest['latency']['p95']:.1f}ms
- P99: {latest['latency']['p99']:.1f}ms

Overall Pass Rate: {latest['pass_rate']:.1%}

Threshold Compliance:
- Latency P95 < {self.thresholds['latency_p95']}ms: {'✓' if latest['latency']['p95'] < self.thresholds['latency_p95'] else '✗'}
- Generation Score > {self.thresholds['coherence']}: {'✓' if latest['generation']['mean_score'] > self.thresholds['coherence'] else '✗'}
"""
        
        return report
```

---

## Production Deployment

### Docker Configuration

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spacy model
RUN python -m spacy download en_core_web_sm

# Copy application
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV RAG_CONFIG_PATH=/app/config/rag_config.yaml

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Kubernetes Deployment

```yaml
# rag-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-service
  namespace: ai-platform
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-service
  template:
    metadata:
      labels:
        app: rag-service
    spec:
      containers:
      - name: rag-service
        image: enterprise-rag:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        env:
        - name: VECTOR_DB_HOST
          value: "qdrant-service"
        - name: VECTOR_DB_PORT
          value: "6333"
        - name: LLM_API_KEY
          valueFrom:
            secretKeyRef:
              name: llm-credentials
              key: api-key
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: rag-service
  namespace: ai-platform
spec:
  selector:
    app: rag-service
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rag-service-hpa
  namespace: ai-platform
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-service
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Monitoring Configuration

```yaml
# prometheus-config.yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'rag-service'
    static_configs:
      - targets: ['rag-service:8000']
    metrics_path: '/metrics'

  - job_name: 'vector-db'
    static_configs:
      - targets: ['qdrant-service:6333']
    metrics_path: '/metrics'

# Alerting rules
rule_files:
  - 'alerts.yaml'

---
# alerts.yaml
groups:
  - name: rag_alerts
    interval: 30s
    rules:
      - alert: HighLatency
        expr: rag_request_duration_seconds{quantile="0.95"} > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High RAG latency detected"
          description: "95th percentile latency is above 2 seconds"
      
      - alert: HighErrorRate
        expr: rate(rag_errors_total[5m]) > 0.01
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate in RAG service"
          description: "Error rate is above 1%"
      
      - alert: LowCacheHitRate
        expr: rag_cache_hit_rate < 0.3
        for: 10m
        labels:
          severity: info
        annotations:
          summary: "Low cache hit rate"
          description: "Cache hit rate is below 30%"
```

This comprehensive RAG implementation guide provides production-ready code and configurations for deploying a robust RAG system in an enterprise financial institution environment.