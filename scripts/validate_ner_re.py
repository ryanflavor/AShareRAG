#!/usr/bin/env python3
"""
验证NER和RE功能的实际效果
使用真实的DeepSeek API和corpus.json数据测试一个公司的命名实体识别和关系抽取
"""

import json
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.adapters import LLMAdapter
from src.components.knowledge_graph_constructor import KnowledgeGraphConstructor
from src.components.data_ingestor import DataIngestor
from config.settings import Settings


def main():
    """主函数：验证NER和RE功能"""

    print("🔍 开始验证NER和RE功能...")
    print("=" * 60)

    try:
        # 1. 加载配置
        print("📋 1. 加载配置...")
        settings = Settings()
        print(
            f"   - DeepSeek API配置: {'✅ 已配置' if settings.deepseek_api_key.startswith('sk-') else '❌ 未配置'}"
        )
        print(f"   - 模型: {settings.deepseek_model}")

        # 2. 初始化组件
        print("\n🔧 2. 初始化组件...")
        llm_adapter = LLMAdapter(enable_cache=True, high_throughput=True)
        kg_constructor = KnowledgeGraphConstructor()
        data_ingestor = DataIngestor()
        print("   - LLM Adapter: ✅")
        print("   - Knowledge Graph Constructor: ✅")
        print("   - Data Ingestor: ✅")

        # 3. 加载数据
        print("\n📂 3. 加载corpus.json数据...")
        documents = data_ingestor.load_corpus("data/corpus.json")
        print(f"   - 成功加载 {len(documents)} 个公司文档")

        # 4. 选择测试公司（取第一个：GQY视讯）
        test_company = documents[0]
        print(f"\n🏢 4. 测试公司: {test_company.title}")
        print(f"   - 文档长度: {len(test_company.text)} 字符")

        # 显示公司文本的前500字符
        preview_text = (
            test_company.text[:500] + "..."
            if len(test_company.text) > 500
            else test_company.text
        )
        print(f"   - 文档预览: {preview_text}")

        # 5. 测试NER功能
        print(f"\n🏷️  5. 测试命名实体识别（NER）...")
        print("   发送请求到DeepSeek API...")

        entities = llm_adapter.extract_entities(test_company.text, include_types=True)

        print(f"   - 识别到 {len(entities)} 个命名实体:")
        for i, entity in enumerate(entities[:10], 1):  # 只显示前10个
            print(f"     {i}. {entity['text']} ({entity['type']})")
        if len(entities) > 10:
            print(f"     ... 还有 {len(entities) - 10} 个实体")

        # 6. 测试RE功能
        print(f"\n🔗 6. 测试关系抽取（RE）...")
        print("   发送请求到DeepSeek API...")

        relations = llm_adapter.extract_relations(test_company.text, entities)

        print(f"   - 抽取到 {len(relations)} 个关系三元组:")
        for i, triple in enumerate(relations[:10], 1):  # 只显示前10个
            subject, predicate, obj = triple
            print(f"     {i}. ({subject}, {predicate}, {obj})")
        if len(relations) > 10:
            print(f"     ... 还有 {len(relations) - 10} 个关系")

        # 7. 测试知识图谱构建
        print(f"\n🕸️  7. 测试知识图谱构建...")
        # 转换Document对象为字典格式
        doc_dict = {
            "id": test_company.idx,
            "text": test_company.text,
            "title": test_company.title,
        }
        results, graph = kg_constructor.process_documents([doc_dict])

        print(f"   - 图谱统计:")
        print(f"     * 顶点数量: {graph.vcount()}")
        print(f"     * 边数量: {graph.ecount()}")

        # 显示一些顶点信息
        if graph.vcount() > 0:
            print(f"   - 顶点示例 (前5个):")
            for i in range(min(5, graph.vcount())):
                vertex = graph.vs[i]
                name = vertex["name"]
                entity_type = vertex.attributes().get("entity_type", "未知")
                print(f"     {i + 1}. {name} ({entity_type})")

        # 显示一些边信息
        if graph.ecount() > 0:
            print(f"   - 关系示例 (前5个):")
            for i in range(min(5, graph.ecount())):
                edge = graph.es[i]
                source_name = graph.vs[edge.source]["name"]
                target_name = graph.vs[edge.target]["name"]
                relation = edge["relation"]
                print(f"     {i + 1}. {source_name} --[{relation}]--> {target_name}")

        # 8. 验证结果质量
        print(f"\n✅ 8. 结果质量评估:")

        # 检查是否识别到公司名称
        company_names = ["GQY视讯", "GQY", "视讯"]
        found_company = any(
            any(name.lower() in entity["text"].lower() for name in company_names)
            for entity in entities
        )
        print(f"   - 公司名称识别: {'✅ 成功' if found_company else '❌ 失败'}")

        # 检查是否识别到关键业务词汇
        business_keywords = ["显示", "拼接", "LED", "系统集成", "智慧城市"]
        found_business = any(
            any(keyword in entity["text"] for keyword in business_keywords)
            for entity in entities
        )
        print(f"   - 业务关键词识别: {'✅ 成功' if found_business else '❌ 失败'}")

        # 检查关系抽取质量
        valid_relations = sum(
            1
            for triple in relations
            if len(triple) == 3 and all(len(str(x).strip()) > 0 for x in triple)
        )
        relation_quality = valid_relations / len(relations) if relations else 0
        print(
            f"   - 关系三元组质量: {relation_quality:.1%} ({valid_relations}/{len(relations)} 个有效)"
        )

        # 检查是否包含有意义的关系
        meaningful_relations = []
        for triple in relations:
            subject, predicate, obj = triple
            # 检查是否包含公司相关的关系
            if any(name in subject or name in obj for name in company_names):
                meaningful_relations.append(triple)

        print(f"   - 公司相关关系: {len(meaningful_relations)} 个")
        if meaningful_relations:
            print("     示例:")
            for triple in meaningful_relations[:3]:
                subject, predicate, obj = triple
                print(f"       * ({subject}, {predicate}, {obj})")

        print(f"\n🎉 验证完成！")
        print(f"NER成功识别 {len(entities)} 个实体，RE成功抽取 {len(relations)} 个关系")
        print(f"知识图谱包含 {graph.vcount()} 个顶点和 {graph.ecount()} 条边")

        return True

    except Exception as e:
        print(f"\n❌ 验证过程中出现错误: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
