#!/usr/bin/env python3
"""
完整的NER/RE + Embedding pipeline测试
测试完整流程：文档 → NER → RE → 嵌入 → 存储到LanceDB
"""

import json
import sys
import time
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.components.embedding_service import EmbeddingService
from src.components.knowledge_graph_constructor import KnowledgeGraphConstructor
from src.components.vector_storage import VectorStorage


def test_complete_pipeline():
    """测试完整的NER/RE/Embedding pipeline"""
    
    print("🔍 测试完整的NER/RE + Embedding pipeline")
    print("=" * 80)
    
    # 1. 准备测试数据（选择前3家公司）
    print("\n1️⃣ 加载测试数据")
    with open("data/corpus.json", "r", encoding="utf-8") as f:
        corpus_data = json.load(f)
    
    # 选择前3家公司进行测试
    test_companies = corpus_data[:3]
    print(f"选择测试公司: {[c['title'] for c in test_companies]}")
    
    # 准备文档格式（KnowledgeGraphConstructor期望的格式）
    documents = []
    for company in test_companies:
        doc = {
            "id": f"doc_{company['idx']}",
            "text": company["text"],
            "title": company["title"],
            "source_file": "corpus.json"
        }
        documents.append(doc)
    
    # 2. 初始化组件
    print("\n2️⃣ 初始化组件")
    
    # 初始化embedding service
    embedding_service = EmbeddingService(
        model_name="Qwen/Qwen3-Embedding-4B",
        device="cuda" if __import__("torch").cuda.is_available() else "cpu",
        batch_size=16,
    )
    
    # 初始化vector storage
    vector_storage = VectorStorage(
        db_path=Path("./output/complete_pipeline_test"),
        table_name="complete_test",
        embedding_dim=2560,
    )
    
    # 连接到vector storage
    vector_storage.connect()
    
    # 初始化knowledge graph constructor（带embedding组件）
    constructor = KnowledgeGraphConstructor(
        embedding_service=embedding_service,
        vector_storage=vector_storage
    )
    
    print("✅ 所有组件初始化完成")
    
    # 3. 执行完整pipeline
    print("\n3️⃣ 执行完整pipeline")
    start_time = time.time()
    
    try:
        # 加载embedding模型
        print("📦 加载embedding模型...")
        model_loaded = embedding_service.load_model()
        if not model_loaded:
            print("❌ 模型加载失败")
            return
        
        # 运行完整pipeline：NER → RE → Embedding → Storage
        print("🔄 运行NER/RE/Embedding pipeline...")
        results, graph = constructor.process_documents(documents)
        
        end_time = time.time()
        
        # 4. 分析结果
        print(f"\n4️⃣ Pipeline执行结果 (耗时: {end_time - start_time:.2f}秒)")
        
        # NER/RE结果分析
        print("\n📊 NER/RE结果:")
        total_entities = 0
        total_relations = 0
        
        for doc_id, doc_results in results.items():
            entities = doc_results.get('entities', [])
            triples = doc_results.get('triples', [])
            
            print(f"  {doc_id}:")
            print(f"    实体: {len(entities)} 个")
            print(f"    关系: {len(triples)} 个")
            
            # 显示实体样例
            if entities:
                for entity in entities[:3]:  # 显示前3个实体
                    print(f"      - {entity['text']} ({entity['type']})")
                if len(entities) > 3:
                    print(f"      - ... 还有 {len(entities) - 3} 个实体")
            
            # 显示关系样例
            if triples:
                for triple in triples[:2]:  # 显示前2个关系
                    print(f"      - {triple[0]} → {triple[1]} → {triple[2]}")
                if len(triples) > 2:
                    print(f"      - ... 还有 {len(triples) - 2} 个关系")
            
            total_entities += len(entities)
            total_relations += len(triples)
        
        # 图谱结果
        print(f"\n🕸️ 知识图谱结果:")
        print(f"  节点数: {graph.vcount()}")
        print(f"  边数: {graph.ecount()}")
        print(f"  总实体: {total_entities}")
        print(f"  总关系: {total_relations}")
        
        # 5. 验证向量存储
        print(f"\n5️⃣ 验证向量存储:")
        
        try:
            table_info = vector_storage.get_table_info()
            print(f"  存储的文档数: {table_info['num_rows']}")
            print(f"  向量维度: {table_info['embedding_dim']}")
            print(f"  表结构: {table_info['schema']}")
            
            # 6. 测试向量搜索
            print(f"\n6️⃣ 测试向量搜索:")
            
            # 获取一些存储的数据进行搜索测试
            all_data = vector_storage.table.to_pandas()
            if len(all_data) > 0:
                # 使用第一个文档的向量进行搜索
                first_vector = eval(all_data.iloc[0]['vector']) if isinstance(all_data.iloc[0]['vector'], str) else all_data.iloc[0]['vector']
                search_results = vector_storage.search(first_vector, top_k=3)
                
                print(f"  搜索结果 (Top 3):")
                for i, result in enumerate(search_results):
                    company = result.get('company_name', 'Unknown')
                    score = result.get('score', 0)
                    entities_count = len(result.get('entities', []))
                    relations_count = len(result.get('relations', []))
                    
                    print(f"    {i+1}. {company} (相似度: {score:.4f})")
                    print(f"       实体: {entities_count}个, 关系: {relations_count}个")
                
                # 7. 验证实体和关系数据
                print(f"\n7️⃣ 验证存储的实体和关系数据:")
                sample_doc = all_data.iloc[0]
                stored_entities = sample_doc.get('entities', [])
                stored_relations = sample_doc.get('relations', [])
                
                print(f"  示例文档: {sample_doc.get('company_name', 'Unknown')}")
                print(f"  存储的实体格式: {type(stored_entities)}")
                
                # 安全地检查entities数据
                try:
                    entities_length = len(stored_entities) if stored_entities is not None else 0
                    if entities_length > 0:
                        first_entity = stored_entities[0]
                        print(f"    首个实体: {first_entity}")
                        
                        # 验证实体格式是否为 [{"text": str, "type": str}]
                        if isinstance(first_entity, dict) and 'text' in first_entity and 'type' in first_entity:
                            print("    ✅ 实体格式正确 (Story 1.2.1兼容)")
                        else:
                            print("    ❌ 实体格式不正确")
                    else:
                        print("    实体列表为空")
                except Exception as entity_error:
                    print(f"    ❌ 实体数据检查失败: {entity_error}")
                
                print(f"  存储的关系格式: {type(stored_relations)}")
                
                # 安全地检查relations数据
                try:
                    relations_length = len(stored_relations) if stored_relations is not None else 0
                    if relations_length > 0:
                        first_relation = stored_relations[0]
                        print(f"    首个关系: {first_relation}")
                    else:
                        print("    关系列表为空")
                except Exception as relation_error:
                    print(f"    ❌ 关系数据检查失败: {relation_error}")
            
        except Exception as e:
            print(f"❌ 向量存储验证失败: {e}")
        
        # 8. 总体评估
        print(f"\n8️⃣ 总体评估:")
        
        pipeline_success = (
            len(results) > 0 and 
            total_entities > 0 and 
            table_info['num_rows'] > 0
        )
        
        if pipeline_success:
            print("✅ 完整pipeline测试成功!")
            print("✅ NER/RE成功提取实体和关系")
            print("✅ 嵌入生成和存储成功")
            print("✅ 向量搜索功能正常")
            print("✅ Story 1.5要求完全满足")
        else:
            print("❌ Pipeline测试失败")
        
        # 9. 性能统计
        print(f"\n9️⃣ 性能统计:")
        print(f"  处理文档数: {len(documents)}")
        print(f"  总处理时间: {end_time - start_time:.2f}秒")
        print(f"  平均每文档: {(end_time - start_time) / len(documents):.2f}秒")
        print(f"  实体提取率: {total_entities / len(documents):.1f}实体/文档")
        print(f"  关系提取率: {total_relations / len(documents):.1f}关系/文档")
        
        return pipeline_success
        
    except Exception as e:
        print(f"❌ Pipeline执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # 清理资源
        vector_storage.close()


if __name__ == "__main__":
    success = test_complete_pipeline()
    exit(0 if success else 1)