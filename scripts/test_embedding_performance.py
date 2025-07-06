#!/usr/bin/env python3
"""
Comprehensive performance test for embedding pipeline with 50 companies from corpus.json.
Tests GPU performance, memory usage, and embedding quality.

Usage:
    python scripts/test_embedding_performance.py

This script will:
1. Load 50 companies from corpus.json
2. Test embedding service with GPU optimization
3. Test vector storage performance
4. Generate comprehensive performance report
"""

import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.components.embedding_service import EmbeddingService
from src.components.vector_storage import VectorStorage


class PerformanceMonitor:
    """Monitor system performance during embedding operations."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.start_memory = None
        self.end_memory = None
        self.peak_memory = None
        self.gpu_memory_allocated = None
        self.gpu_memory_cached = None
        
    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.start_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
        self.peak_memory = self.start_memory
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.gpu_memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            self.gpu_memory_cached = torch.cuda.memory_reserved() / 1024 / 1024  # MB
            
    def update_peak_memory(self):
        """Update peak memory usage."""
        current_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
        self.peak_memory = max(self.peak_memory, current_memory)
        
    def stop_monitoring(self):
        """Stop performance monitoring and calculate metrics."""
        self.end_time = time.time()
        self.end_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
        
        # Update peak memory one final time
        self.update_peak_memory()
        
    def get_metrics(self) -> dict[str, Any]:
        """Get performance metrics."""
        duration = self.end_time - self.start_time if self.end_time else 0
        memory_used = self.end_memory - self.start_memory if self.end_memory else 0
        peak_memory_used = self.peak_memory - self.start_memory if self.peak_memory else 0
        
        metrics = {
            "duration_seconds": duration,
            "memory_used_mb": memory_used,
            "peak_memory_used_mb": peak_memory_used,
            "start_memory_mb": self.start_memory,
            "end_memory_mb": self.end_memory,
            "peak_memory_mb": self.peak_memory,
        }
        
        if torch.cuda.is_available():
            metrics.update({
                "gpu_memory_allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
                "gpu_memory_cached_mb": torch.cuda.memory_reserved() / 1024 / 1024,
                "gpu_peak_memory_allocated_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
                "gpu_peak_memory_cached_mb": torch.cuda.max_memory_reserved() / 1024 / 1024,
                "gpu_available": True,
                "gpu_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            })
        else:
            metrics.update({
                "gpu_available": False,
                "gpu_device_name": None,
            })
            
        return metrics


class EmbeddingPerformanceTest:
    """Comprehensive performance test for embedding pipeline."""
    
    def __init__(self, num_companies: int = 50):
        self.num_companies = num_companies
        self.logger = logging.getLogger(__name__)
        self.monitor = PerformanceMonitor()
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        
    def load_corpus_data(self) -> list[dict[str, Any]]:
        """Load and prepare corpus data for testing."""
        corpus_path = Path("data/corpus.json")
        
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
            
        with open(corpus_path, encoding="utf-8") as f:
            corpus_data = json.load(f)
            
        # Select first N companies for testing
        selected_companies = corpus_data[:self.num_companies]
        
        self.logger.info(f"Loaded {len(selected_companies)} companies from corpus")
        return selected_companies
        
    def prepare_documents(self, companies: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Prepare documents for embedding processing."""
        documents = []
        
        for idx, company in enumerate(companies):
            # Split text into chunks (simple by paragraphs)
            text_chunks = company["text"].split("\n\n")
            
            for chunk_idx, chunk in enumerate(text_chunks):
                if chunk.strip():  # Skip empty chunks
                    doc = {
                        "doc_id": f"company_{idx}",
                        "chunk_index": chunk_idx,
                        "text": chunk.strip(),
                        "company_name": company["title"],
                        "entities": [],  # Would be populated by NER
                        "relations": [],  # Would be populated by RE
                        "source_file": "corpus.json",
                    }
                    documents.append(doc)
                    
        self.logger.info(f"Created {len(documents)} document chunks from {len(companies)} companies")
        return documents
        
    def test_embedding_service(self, documents: list[dict[str, Any]]) -> tuple[bool, dict[str, Any], list[dict[str, Any]]]:
        """Test embedding service performance."""
        self.logger.info("Testing Embedding Service Performance")
        
        # Initialize embedding service
        embedding_service = EmbeddingService(
            model_name="Qwen/Qwen3-Embedding-4B",
            device="cuda" if torch.cuda.is_available() else "cpu",
            embedding_dim=2560,
            batch_size=32,
        )
        
        # Start monitoring
        self.monitor.start_monitoring()
        
        try:
            # Load model
            model_load_start = time.time()
            model_loaded = embedding_service.load_model()
            model_load_time = time.time() - model_load_start
            
            if not model_loaded:
                return False, {"error": "Failed to load embedding model"}, []
                
            self.logger.info(f"Model loaded in {model_load_time:.2f} seconds")
            
            # Update memory after model loading
            self.monitor.update_peak_memory()
            
            # Process documents
            processing_start = time.time()
            processed_docs = embedding_service.process_documents(documents)
            processing_time = time.time() - processing_start
            
            if not processed_docs:
                return False, {"error": "Failed to process documents"}, []
                
            # Stop monitoring
            self.monitor.stop_monitoring()
            
            # Calculate metrics
            metrics = self.monitor.get_metrics()
            metrics.update({
                "model_load_time_seconds": model_load_time,
                "processing_time_seconds": processing_time,
                "total_documents": len(documents),
                "processed_documents": len(processed_docs),
                "documents_per_second": len(processed_docs) / processing_time if processing_time > 0 else 0,
                "success": True,
            })
            
            self.logger.info(f"Processed {len(processed_docs)} documents in {processing_time:.2f} seconds")
            self.logger.info(f"Processing rate: {metrics['documents_per_second']:.2f} documents/second")
            
            return True, metrics, processed_docs
            
        except Exception as e:
            self.monitor.stop_monitoring()
            error_metrics = self.monitor.get_metrics()
            error_metrics.update({
                "error": str(e),
                "traceback": traceback.format_exc(),
                "success": False,
            })
            self.logger.error(f"Embedding service test failed: {e}")
            return False, error_metrics, []
            
    def test_vector_storage(self, processed_docs: list[dict[str, Any]]) -> tuple[bool, dict[str, Any]]:
        """Test vector storage performance."""
        self.logger.info("Testing Vector Storage Performance")
        
        # Initialize vector storage
        vector_storage = VectorStorage(
            db_path=Path("./output/test_vector_store1"),
            table_name="performance_test",
            embedding_dim=2560,
            batch_size=100,
        )
        
        storage_start = time.time()
        
        try:
            # Connect to database
            vector_storage.connect()
            
            # Verify connection was successful
            if vector_storage.db is None:
                raise ValueError("Failed to establish database connection")
            
            # Create table with initial data
            if processed_docs:
                vector_storage.create_table([processed_docs[0]])
                
                # Add all documents
                vector_storage.add_documents(processed_docs)
                
            # Get table info
            table_info = vector_storage.get_table_info()
            
            # Test search performance
            search_start = time.time()
            if processed_docs:
                # Use first document's embedding as query
                query_vector = processed_docs[0]["vector"]
                search_results = vector_storage.search(query_vector, top_k=10)
                search_time = time.time() - search_start
                
                # Test company filtering
                filter_start = time.time()
                company_name = processed_docs[0]["company_name"]
                filtered_results = vector_storage.search(
                    query_vector, top_k=5, filter_company=company_name
                )
                filter_time = time.time() - filter_start
            else:
                search_results = []
                filtered_results = []
                search_time = 0
                filter_time = 0
                
            storage_time = time.time() - storage_start
            
            # Clean up
            vector_storage.close()
            
            metrics = {
                "storage_time_seconds": storage_time,
                "search_time_seconds": search_time,
                "filter_time_seconds": filter_time,
                "total_documents_stored": len(processed_docs),
                "table_info": table_info,
                "search_results_count": len(search_results),
                "filtered_results_count": len(filtered_results),
                "success": True,
            }
            
            self.logger.info(f"Stored {len(processed_docs)} documents in {storage_time:.2f} seconds")
            self.logger.info(f"Search completed in {search_time:.4f} seconds")
            self.logger.info(f"Filtered search completed in {filter_time:.4f} seconds")
            
            return True, metrics
            
        except Exception as e:
            vector_storage.close()
            error_metrics = {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "success": False,
            }
            self.logger.error(f"Vector storage test failed: {e}")
            return False, error_metrics
            
    def analyze_embedding_quality(self, processed_docs: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze embedding quality metrics."""
        if not processed_docs:
            return {"error": "No processed documents to analyze"}
            
        embeddings = np.array([doc["vector"] for doc in processed_docs])
        
        # Calculate quality metrics
        metrics = {
            "embedding_shape": embeddings.shape,
            "embedding_dtype": str(embeddings.dtype),
            "mean_norm": float(np.mean(np.linalg.norm(embeddings, axis=1))),
            "std_norm": float(np.std(np.linalg.norm(embeddings, axis=1))),
            "min_norm": float(np.min(np.linalg.norm(embeddings, axis=1))),
            "max_norm": float(np.max(np.linalg.norm(embeddings, axis=1))),
        }
        
        # Calculate pairwise similarities (sample)
        if len(embeddings) > 1:
            # Calculate similarities between first 10 embeddings
            sample_size = min(10, len(embeddings))
            sample_embeddings = embeddings[:sample_size]
            
            # Normalize embeddings
            normalized = sample_embeddings / np.linalg.norm(sample_embeddings, axis=1, keepdims=True)
            
            # Calculate cosine similarities
            similarities = np.dot(normalized, normalized.T)
            
            # Get upper triangle (excluding diagonal)
            upper_triangle = similarities[np.triu_indices_from(similarities, k=1)]
            
            metrics.update({
                "sample_similarity_stats": {
                    "mean": float(np.mean(upper_triangle)),
                    "std": float(np.std(upper_triangle)),
                    "min": float(np.min(upper_triangle)),
                    "max": float(np.max(upper_triangle)),
                    "median": float(np.median(upper_triangle)),
                }
            })
            
        return metrics
        
    def test_batch_size_performance(self, documents: list[dict[str, Any]]) -> dict[str, Any]:
        """Test performance with different batch sizes."""
        batch_sizes = [8, 16, 32, 64]
        batch_results = {}
        
        self.logger.info("Testing batch size performance")
        
        # Use subset of documents for batch testing
        test_docs = documents[:100] if len(documents) > 100 else documents
        
        for batch_size in batch_sizes:
            if batch_size > len(test_docs):
                continue
                
            self.logger.info(f"Testing batch size: {batch_size}")
            
            try:
                # Initialize service with specific batch size
                service = EmbeddingService(
                    model_name="Qwen/Qwen3-Embedding-4B",
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    batch_size=batch_size,
                )
                
                # Load model
                if not service.load_model():
                    continue
                
                # Time the processing
                start_time = time.time()
                processed = service.process_documents(test_docs)
                end_time = time.time()
                
                processing_time = end_time - start_time
                docs_per_second = len(processed) / processing_time if processing_time > 0 else 0
                
                batch_results[batch_size] = {
                    "processing_time": processing_time,
                    "documents_processed": len(processed),
                    "docs_per_second": docs_per_second,
                    "success": len(processed) > 0,
                }
                
                self.logger.info(f"Batch size {batch_size}: {docs_per_second:.2f} docs/sec")
                
            except Exception as e:
                batch_results[batch_size] = {
                    "error": str(e),
                    "success": False,
                }
                self.logger.error(f"Batch size {batch_size} failed: {e}")
                
        return batch_results
        
    def run_comprehensive_test(self) -> dict[str, Any]:
        """Run comprehensive performance test."""
        self.logger.info(f"Starting comprehensive performance test with {self.num_companies} companies")
        
        test_results = {
            "test_config": {
                "num_companies": self.num_companies,
                "gpu_available": torch.cuda.is_available(),
                "gpu_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "pytorch_version": torch.__version__,
                "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "cpu_count": psutil.cpu_count(),
                "total_memory_mb": psutil.virtual_memory().total / 1024 / 1024,
            },
            "data_loading": {},
            "embedding_service": {},
            "vector_storage": {},
            "embedding_quality": {},
            "batch_performance": {},
            "overall_success": False,
        }
        
        try:
            # Load corpus data
            load_start = time.time()
            companies = self.load_corpus_data()
            documents = self.prepare_documents(companies)
            load_time = time.time() - load_start
            
            test_results["data_loading"] = {
                "load_time_seconds": load_time,
                "companies_loaded": len(companies),
                "documents_created": len(documents),
                "avg_doc_length": np.mean([len(doc["text"]) for doc in documents]),
                "success": True,
            }
            
            # Test embedding service
            embed_success, embed_metrics, processed_docs = self.test_embedding_service(documents)
            test_results["embedding_service"] = embed_metrics
            
            if embed_success and processed_docs:
                # Test vector storage
                storage_success, storage_metrics = self.test_vector_storage(processed_docs)
                test_results["vector_storage"] = storage_metrics
                
                # Analyze embedding quality
                quality_metrics = self.analyze_embedding_quality(processed_docs)
                test_results["embedding_quality"] = quality_metrics
                
                # Test batch size performance
                batch_metrics = self.test_batch_size_performance(documents)
                test_results["batch_performance"] = batch_metrics
                
                test_results["overall_success"] = storage_success
            else:
                test_results["overall_success"] = False
                
        except Exception as e:
            test_results["error"] = str(e)
            test_results["traceback"] = traceback.format_exc()
            self.logger.error(f"Comprehensive test failed: {e}")
            
        return test_results
        
    def generate_performance_report(self, results: dict[str, Any]) -> str:
        """Generate a detailed performance report."""
        report = []
        report.append("=" * 80)
        report.append("EMBEDDING PIPELINE PERFORMANCE TEST REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Test configuration
        config = results.get("test_config", {})
        report.append("TEST CONFIGURATION:")
        report.append(f"  • Number of companies: {config.get('num_companies', 'N/A')}")
        report.append(f"  • GPU available: {config.get('gpu_available', 'N/A')}")
        report.append(f"  • GPU device: {config.get('gpu_device', 'N/A')}")
        report.append(f"  • PyTorch version: {config.get('pytorch_version', 'N/A')}")
        report.append(f"  • CPU count: {config.get('cpu_count', 'N/A')}")
        report.append(f"  • Total memory: {config.get('total_memory_mb', 0):.0f} MB")
        report.append(f"  • Test timestamp: {config.get('test_timestamp', 'N/A')}")
        report.append("")
        
        # Data loading results
        data_loading = results.get("data_loading", {})
        report.append("DATA LOADING RESULTS:")
        report.append(f"  • Load time: {data_loading.get('load_time_seconds', 0):.2f} seconds")
        report.append(f"  • Companies loaded: {data_loading.get('companies_loaded', 0)}")
        report.append(f"  • Documents created: {data_loading.get('documents_created', 0)}")
        report.append(f"  • Average document length: {data_loading.get('avg_doc_length', 0):.0f} chars")
        report.append(f"  • Success: {data_loading.get('success', False)}")
        report.append("")
        
        # Embedding service results
        embedding = results.get("embedding_service", {})
        report.append("EMBEDDING SERVICE RESULTS:")
        report.append(f"  • Success: {embedding.get('success', False)}")
        report.append(f"  • Model load time: {embedding.get('model_load_time_seconds', 0):.2f} seconds")
        report.append(f"  • Processing time: {embedding.get('processing_time_seconds', 0):.2f} seconds")
        report.append(f"  • Total duration: {embedding.get('duration_seconds', 0):.2f} seconds")
        report.append(f"  • Documents processed: {embedding.get('processed_documents', 0)}")
        report.append(f"  • Processing rate: {embedding.get('documents_per_second', 0):.2f} docs/sec")
        report.append("")
        
        # Memory usage
        report.append("MEMORY USAGE:")
        report.append(f"  • Memory used: {embedding.get('memory_used_mb', 0):.1f} MB")
        report.append(f"  • Peak memory used: {embedding.get('peak_memory_used_mb', 0):.1f} MB")
        if embedding.get('gpu_available', False):
            report.append(f"  • GPU memory allocated: {embedding.get('gpu_memory_allocated_mb', 0):.1f} MB")
            report.append(f"  • GPU peak memory: {embedding.get('gpu_peak_memory_allocated_mb', 0):.1f} MB")
        report.append("")
        
        # Vector storage results
        storage = results.get("vector_storage", {})
        report.append("VECTOR STORAGE RESULTS:")
        report.append(f"  • Success: {storage.get('success', False)}")
        report.append(f"  • Storage time: {storage.get('storage_time_seconds', 0):.2f} seconds")
        report.append(f"  • Search time: {storage.get('search_time_seconds', 0):.4f} seconds")
        report.append(f"  • Filter time: {storage.get('filter_time_seconds', 0):.4f} seconds")
        report.append(f"  • Documents stored: {storage.get('total_documents_stored', 0)}")
        report.append(f"  • Search results: {storage.get('search_results_count', 0)}")
        report.append(f"  • Filtered results: {storage.get('filtered_results_count', 0)}")
        report.append("")
        
        # Batch performance
        batch_perf = results.get("batch_performance", {})
        if batch_perf:
            report.append("BATCH SIZE PERFORMANCE:")
            for batch_size, metrics in batch_perf.items():
                if metrics.get("success", False):
                    report.append(f"  • Batch size {batch_size}: {metrics.get('docs_per_second', 0):.2f} docs/sec")
            report.append("")
        
        # Embedding quality
        quality = results.get("embedding_quality", {})
        if "embedding_shape" in quality:
            report.append("EMBEDDING QUALITY ANALYSIS:")
            report.append(f"  • Embedding shape: {quality.get('embedding_shape', 'N/A')}")
            report.append(f"  • Mean norm: {quality.get('mean_norm', 0):.4f}")
            report.append(f"  • Std norm: {quality.get('std_norm', 0):.4f}")
            report.append(f"  • Min norm: {quality.get('min_norm', 0):.4f}")
            report.append(f"  • Max norm: {quality.get('max_norm', 0):.4f}")
            
            sim_stats = quality.get("sample_similarity_stats", {})
            if sim_stats:
                report.append("  • Sample similarity stats:")
                report.append(f"    - Mean: {sim_stats.get('mean', 0):.4f}")
                report.append(f"    - Std: {sim_stats.get('std', 0):.4f}")
                report.append(f"    - Min: {sim_stats.get('min', 0):.4f}")
                report.append(f"    - Max: {sim_stats.get('max', 0):.4f}")
                report.append(f"    - Median: {sim_stats.get('median', 0):.4f}")
        report.append("")
        
        # Overall status
        report.append("OVERALL TEST STATUS:")
        report.append(f"  • Success: {results.get('overall_success', False)}")
        if "error" in results:
            report.append(f"  • Error: {results['error']}")
        report.append("")
        
        # Performance summary
        if results.get("overall_success", False):
            report.append("PERFORMANCE SUMMARY:")
            embed_rate = embedding.get('documents_per_second', 0)
            storage_time = storage.get('storage_time_seconds', 0)
            search_time = storage.get('search_time_seconds', 0)
            total_docs = embedding.get('processed_documents', 0)
            
            report.append(f"  • Overall throughput: {embed_rate:.2f} documents/second")
            report.append(f"  • Storage efficiency: {total_docs/storage_time:.2f} docs/sec stored" if storage_time > 0 else "  • Storage efficiency: N/A")
            report.append(f"  • Search latency: {search_time*1000:.2f} ms")
            report.append(f"  • Memory efficiency: {embedding.get('peak_memory_used_mb', 0)/total_docs:.2f} MB/doc" if total_docs > 0 else "  • Memory efficiency: N/A")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)


def main():
    """Main function to run the performance test."""
    print("🚀 Starting Embedding Performance Test")
    print(f"📊 Testing with 50 companies from corpus.json")
    print(f"🔧 GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🎮 GPU Device: {torch.cuda.get_device_name(0)}")
    print()
    
    # Create performance test instance
    performance_test = EmbeddingPerformanceTest(num_companies=50)
    
    # Run comprehensive test
    results = performance_test.run_comprehensive_test()
    
    # Generate and print report
    report = performance_test.generate_performance_report(results)
    print(report)
    
    # Save results to file
    results_path = Path("output/embedding_performance_results1.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📁 Detailed results saved to: {results_path}")
    
    # Performance insights
    if results.get("overall_success", False):
        embed_rate = results.get("embedding_service", {}).get("documents_per_second", 0)
        peak_gpu_mem = results.get("embedding_service", {}).get("gpu_peak_memory_allocated_mb", 0)
        
        print("\n🎯 PERFORMANCE INSIGHTS:")
        print(f"   • Processing Rate: {embed_rate:.1f} documents/second")
        if peak_gpu_mem > 0:
            print(f"   • Peak GPU Memory: {peak_gpu_mem:.0f} MB")
        print(f"   • Vector Storage: Ready for semantic search")
        print(f"   ✅ System is ready for production workloads!")
    else:
        print("\n❌ Test failed. Check logs and results for details.")
    
    # Return exit code based on success
    return 0 if results.get("overall_success", False) else 1


if __name__ == "__main__":
    exit(main())