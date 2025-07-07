#!/usr/bin/env python3
"""
基于工作正常的10公司版本的大规模数据集处理器
===========================================
复制test_10_companies_optimized_full.py的成功架构，扩展到5341公司
保持相同的NER/RE处理逻辑，确保数据质量
"""

import json
import logging
import sys
import time
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Any
import gc
import psutil
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import Settings
from src.components.knowledge_graph_constructor import KnowledgeGraphConstructor
from src.components.embedding_service import EmbeddingService
from src.components.vector_storage import VectorStorage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('full_dataset_pipeline_fixed.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FixedFullDatasetProcessor(KnowledgeGraphConstructor):
    """
    基于成功的10公司版本的大规模处理器
    使用相同的NER/RE处理逻辑，确保数据质量一致性
    """
    
    def __init__(
        self,
        max_workers: int = 20,
        batch_size: int = 100,
        enable_embeddings: bool = True,
        embedding_batch_size: int = 8,
        checkpoint_interval: int = 5,
        memory_limit_gb: float = 32.0
    ):
        """
        初始化处理器，使用与10公司测试相同的架构
        """
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.enable_embeddings = enable_embeddings
        self.embedding_batch_size = embedding_batch_size
        self.checkpoint_interval = checkpoint_interval
        self.memory_limit_gb = memory_limit_gb
        
        # 统计信息
        self.stats = {
            'total_documents': 0,
            'processed_documents': 0,
            'total_entities': 0,
            'total_relations': 0,
            'processing_time': 0,
            'start_time': None,
            'batches_completed': 0,
            'errors': []
        }
        
        # 输出路径
        self.output_dir = Path("output/full_dataset")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 检查点文件
        self.checkpoint_file = self.output_dir / "processing_checkpoint.json"
        self.progress_file = self.output_dir / "processing_progress.json"
        
        # 嵌入模型加载状态
        self.embedding_model_loaded = False
        
        logger.info(f"🚀 FixedFullDatasetProcessor initialized:")
        logger.info(f"   • Max workers: {max_workers}")
        logger.info(f"   • Batch size: {batch_size}")
        logger.info(f"   • Embeddings: {'✅' if enable_embeddings else '❌'}")
        logger.info(f"   • Checkpoint interval: {checkpoint_interval}")
        logger.info(f"   • Memory limit: {memory_limit_gb} GB")
    
    def initialize_components(self) -> bool:
        """初始化所有组件 - 基于10公司版本的成功模式"""
        try:
            logger.info("🔄 Initializing components...")
            
            # 初始化嵌入服务
            if self.enable_embeddings:
                try:
                    self.embedding_service = EmbeddingService(
                        batch_size=self.embedding_batch_size
                    )
                    logger.info("✅ Embedding service initialized")
                    
                    # 立即加载嵌入模型（只加载一次）
                    logger.info("🔄 Loading embedding model (one-time initialization)...")
                    model_load_start = time.time()
                    if self.embedding_service.load_model():
                        self.embedding_model_loaded = True
                        model_load_time = time.time() - model_load_start
                        logger.info(f"✅ Embedding model loaded successfully in {model_load_time:.2f}s")
                    else:
                        logger.error("❌ Failed to load embedding model")
                        self.enable_embeddings = False
                        
                except Exception as e:
                    logger.error(f"❌ Failed to initialize embedding service: {e}")
                    self.enable_embeddings = False
                
                # 初始化向量存储
                if self.enable_embeddings:
                    try:
                        self.vector_storage = VectorStorage(
                            db_path=self.output_dir / "vector_store"
                        )
                        self.vector_storage.connect()
                        logger.info("✅ Vector storage initialized")
                    except Exception as e:
                        logger.error(f"❌ Failed to initialize vector storage: {e}")
                        self.enable_embeddings = False
            
            # 初始化知识图谱构造器（使用父类初始化）
            super().__init__(
                embedding_service=self.embedding_service if self.enable_embeddings else None,
                vector_storage=self.vector_storage if self.enable_embeddings else None
            )
            logger.info("✅ Knowledge graph constructor initialized")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Component initialization failed: {e}")
            return False
    
    def process_full_dataset(self, corpus_path: Path) -> bool:
        """处理完整数据集 - 使用批次化的10公司处理逻辑"""
        try:
            logger.info("🚀 FIXED FULL DATASET PROCESSING STARTED")
            logger.info("=" * 60)
            
            # 加载数据
            logger.info(f"📂 Loading corpus from {corpus_path}")
            with open(corpus_path, 'r', encoding='utf-8') as f:
                corpus_data = json.load(f)
            
            self.stats['total_documents'] = len(corpus_data)
            self.stats['start_time'] = time.time()
            
            logger.info(f"📊 Total documents: {self.stats['total_documents']}")
            
            # 计算批次信息
            total_batches = (self.stats['total_documents'] + self.batch_size - 1) // self.batch_size
            
            logger.info(f"📋 Processing in {total_batches} batches of {self.batch_size} docs each")
            
            # 初始化组件
            if not self.initialize_components():
                logger.error("❌ Failed to initialize components")
                return False
            
            # 批次处理
            for batch_idx in range(total_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, self.stats['total_documents'])
                
                # 准备批次文档（使用与10公司测试相同的格式）
                batch_docs = []
                for i in range(start_idx, end_idx):
                    doc = corpus_data[i]
                    formatted_doc = {
                        'id': f"company_{i:04d}",
                        'title': doc.get('title', f'Company {i}'),
                        'text': doc.get('text', ''),
                        'idx': doc.get('idx', i)
                    }
                    batch_docs.append(formatted_doc)
                
                logger.info(f"\n📦 Processing batch {batch_idx + 1}/{total_batches}: {len(batch_docs)} documents")
                batch_start = time.time()
                
                # 使用与10公司测试完全相同的处理逻辑
                batch_results, batch_graph = self.process_batch_like_10_companies(batch_docs)
                
                batch_time = time.time() - batch_start
                
                # 更新统计信息
                self.stats['processed_documents'] += len(batch_docs)
                self.stats['batches_completed'] += 1
                self.stats['processing_time'] += batch_time
                
                logger.info(f"✅ Batch {batch_idx + 1} completed in {batch_time:.2f}s")
                logger.info(f"📊 Progress: {self.stats['processed_documents']}/{self.stats['total_documents']} "
                          f"({self.stats['processed_documents']/self.stats['total_documents']*100:.1f}%)")
                
                # 保存进度
                self.save_progress(batch_results, batch_idx)
                
                # 检查点保存
                if (batch_idx + 1) % self.checkpoint_interval == 0:
                    self.save_checkpoint(batch_idx, self.stats['processed_documents'])
                
                # 内存清理
                self.cleanup_memory()
            
            # 最终统计
            total_time = time.time() - self.stats['start_time']
            
            logger.info("\n" + "=" * 60)
            logger.info("🎉 FIXED FULL DATASET PROCESSING COMPLETED!")
            logger.info("=" * 60)
            logger.info(f"⏱️  Total time: {total_time/3600:.2f} hours")
            logger.info(f"📊 Processing speed: {self.stats['total_documents']/(total_time/3600):.0f} companies/hour")
            logger.info(f"📁 Results saved in: {self.output_dir}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Fixed full dataset processing failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def process_batch_like_10_companies(self, batch_docs: List[Dict]) -> Tuple[Dict, Any]:
        """
        使用KnowledgeGraphConstructor的标准处理逻辑
        这确保了NER/RE结果的一致性
        """
        logger.info(f"🚀 Processing {len(batch_docs)} documents using standard logic")
        
        # 使用父类的process_documents方法，这是标准的处理流程
        results, graph = super().process_documents(batch_docs)
        
        return results, graph
    
    
    def save_progress(self, batch_results: Dict, batch_idx: int):
        """保存进度信息"""
        progress_data = {
            'current_batch': batch_idx + 1,
            'total_batches': (self.stats['total_documents'] + self.batch_size - 1) // self.batch_size,
            'processed_documents': self.stats['processed_documents'],
            'total_documents': self.stats['total_documents'],
            'processing_time': self.stats['processing_time'],
            'timestamp': datetime.now().isoformat(),
            'memory_usage': self.check_memory_usage()
        }
        
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)
    
    def save_checkpoint(self, batch_idx: int, processed_docs: int):
        """保存检查点"""
        checkpoint_data = {
            'batch_idx': batch_idx,
            'processed_docs': processed_docs,
            'timestamp': datetime.now().isoformat(),
            'stats': self.stats
        }
        
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 Checkpoint saved: batch {batch_idx}, processed {processed_docs}")
    
    def check_memory_usage(self) -> Dict[str, float]:
        """检查内存使用情况"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_gb = memory_info.rss / (1024 ** 3)
        
        return {
            'memory_gb': memory_gb,
            'memory_percent': process.memory_percent(),
            'available_gb': psutil.virtual_memory().available / (1024 ** 3)
        }
    
    def cleanup_memory(self):
        """清理内存"""
        gc.collect()


def main():
    """主函数"""
    try:
        # 配置 - 基于10公司测试的成功参数
        config = {
            'max_workers': 20,              # 与10公司测试类似的并行度
            'batch_size': 50,               # 较小批次确保稳定性
            'enable_embeddings': True,
            'embedding_batch_size': 4,      # 保守的嵌入批次
            'checkpoint_interval': 2,       # 频繁检查点
            'memory_limit_gb': 32.0
        }
        
        # 加载数据
        project_root = Path(__file__).parent.parent
        corpus_path = project_root / "data" / "corpus.json"
        
        logger.info("📂 Loading documents...")
        if not corpus_path.exists():
            logger.error(f"❌ Corpus file not found: {corpus_path}")
            return 1
        
        # 创建修复版处理器
        processor = FixedFullDatasetProcessor(**config)
        
        # 运行处理
        success = processor.process_full_dataset(corpus_path)
        
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"❌ Main execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)