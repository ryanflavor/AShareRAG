#!/usr/bin/env python3
"""
Analyze the generated knowledge graph and vector storage outputs.
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

import igraph as ig
import lancedb
from src.components.vector_storage import VectorStorage

def analyze_graph(graph_path: Path):
    """Analyze the GraphML file."""
    print("🔍 ANALYZING KNOWLEDGE GRAPH")
    print("="*50)
    
    # Load graph
    graph = ig.read(str(graph_path))
    
    print(f"📊 Basic Statistics:")
    print(f"   • Vertices: {graph.vcount()}")
    print(f"   • Edges: {graph.ecount()}")
    print(f"   • Directed: {graph.is_directed()}")
    print(f"   • Density: {graph.density():.4f}")
    
    # Entity type analysis
    entity_types = {}
    for vertex in graph.vs:
        entity_type = vertex["entity_type"] if "entity_type" in vertex.attributes() else "UNKNOWN"
        entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
    
    print(f"\n🏷️  Entity Type Distribution:")
    for entity_type, count in sorted(entity_types.items(), key=lambda x: x[1], reverse=True):
        print(f"   • {entity_type}: {count}")
    
    # Top entities by degree
    degrees = graph.degree()
    vertex_degrees = [(i, d) for i, d in enumerate(degrees)]
    vertex_degrees.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n🎯 Top Entities by Connectivity:")
    for i, (vertex_id, degree) in enumerate(vertex_degrees[:10]):
        vertex = graph.vs[vertex_id]
        name = vertex["name"] if "name" in vertex.attributes() else "Unknown"
        entity_type = vertex["entity_type"] if "entity_type" in vertex.attributes() else "UNKNOWN"
        print(f"   {i+1:2d}. {name[:30]:<30} | {entity_type:<15} | Degree: {degree}")
    
    # Relation analysis
    relations = {}
    for edge in graph.es:
        relation = edge["relation"] if "relation" in edge.attributes() else "UNKNOWN"
        relations[relation] = relations.get(relation, 0) + 1
    
    print(f"\n🔗 Relation Type Distribution (Top 10):")
    for relation, count in sorted(relations.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"   • {relation}: {count}")
    
    # Connected components
    components = graph.components()
    component_sizes = [len(c) for c in components]
    component_sizes.sort(reverse=True)
    
    print(f"\n🌐 Connected Components:")
    print(f"   • Total components: {len(components)}")
    print(f"   • Largest component: {component_sizes[0] if component_sizes else 0}")
    print(f"   • Component sizes: {component_sizes[:5]}")
    
    # Company validation
    companies = []
    for vertex in graph.vs:
        if vertex["entity_type"] == "COMPANY" if "entity_type" in vertex.attributes() else False:
            companies.append(vertex["name"] if "name" in vertex.attributes() else "Unknown")
    
    print(f"\n🏢 Companies in Graph:")
    for i, company in enumerate(sorted(companies), 1):
        print(f"   {i:2d}. {company}")
    
    return {
        "vertices": graph.vcount(),
        "edges": graph.ecount(),
        "companies": len(companies),
        "entity_types": len(entity_types),
        "relations": len(relations),
        "components": len(components)
    }

def analyze_vector_storage(db_path: Path):
    """Analyze the vector storage."""
    print("\n🎯 ANALYZING VECTOR STORAGE")
    print("="*50)
    
    try:
        # Connect to database
        db = lancedb.connect(str(db_path))
        
        # List tables
        tables = db.table_names()
        print(f"📋 Available tables: {tables}")
        
        if "ashare_documents" in tables:
            table = db.open_table("ashare_documents")
            
            # Get basic info
            count = table.count_rows()
            print(f"📊 Document count: {count}")
            
            # Sample a few records
            if count > 0:
                sample = table.head(3).to_pandas()
                print(f"\n📝 Sample records:")
                for i, row in sample.iterrows():
                    print(f"   {i+1}. ID: {row.get('id', 'N/A')}")
                    print(f"      Company: {row.get('company_name', 'N/A')}")
                    print(f"      Relations: {row.get('relations_count', 0)}")
                    print(f"      Vector dim: {len(row.get('vector', [])) if 'vector' in row else 'N/A'}")
                    print()
            
            return {"document_count": count, "vector_dim": len(sample.iloc[0]['vector']) if count > 0 and 'vector' in sample.columns else 0}
        else:
            print("⚠️  No ashare_documents table found")
            return {"document_count": 0, "vector_dim": 0}
            
    except Exception as e:
        print(f"❌ Error analyzing vector storage: {e}")
        return {"document_count": 0, "vector_dim": 0}

def analyze_consistency(graph_stats, vector_stats):
    """Analyze consistency between graph and vector storage."""
    print("\n🔍 CONSISTENCY ANALYSIS")
    print("="*50)
    
    # Expected 10 companies
    expected_companies = 10
    expected_docs = 10
    
    print(f"📊 Data Consistency:")
    print(f"   • Expected companies: {expected_companies}")
    print(f"   • Found companies in graph: {graph_stats['companies']}")
    print(f"   • Expected documents: {expected_docs}")
    print(f"   • Found documents in vectors: {vector_stats['document_count']}")
    
    # Validation
    issues = []
    
    if graph_stats['companies'] != expected_companies:
        issues.append(f"Company count mismatch: expected {expected_companies}, found {graph_stats['companies']}")
    
    if vector_stats['document_count'] != expected_docs:
        issues.append(f"Document count mismatch: expected {expected_docs}, found {vector_stats['document_count']}")
    
    if vector_stats['vector_dim'] not in [2560, 0]:  # Qwen3 embedding dimension
        issues.append(f"Unexpected vector dimension: {vector_stats['vector_dim']}")
    
    if issues:
        print(f"\n⚠️  Issues found:")
        for issue in issues:
            print(f"   • {issue}")
    else:
        print(f"\n✅ All consistency checks passed!")
    
    # Data quality assessment
    print(f"\n📈 Data Quality Assessment:")
    
    # Graph richness
    avg_degree = graph_stats['edges'] * 2 / graph_stats['vertices'] if graph_stats['vertices'] > 0 else 0
    print(f"   • Average degree: {avg_degree:.2f}")
    
    if avg_degree > 3.0:
        print(f"   • Connectivity: 🟢 Excellent (>3.0)")
    elif avg_degree > 2.0:
        print(f"   • Connectivity: 🟡 Good (>2.0)")
    else:
        print(f"   • Connectivity: 🔴 Sparse (<2.0)")
    
    # Entity diversity
    entity_ratio = graph_stats['entity_types'] / graph_stats['companies'] if graph_stats['companies'] > 0 else 0
    print(f"   • Entity type diversity: {entity_ratio:.1f} types per company")
    
    if entity_ratio > 8:
        print(f"   • Diversity: 🟢 Rich (>8 types/company)")
    elif entity_ratio > 5:
        print(f"   • Diversity: 🟡 Moderate (>5 types/company)")  
    else:
        print(f"   • Diversity: 🔴 Limited (<5 types/company)")

def main():
    """Main analysis function."""
    project_root = Path(__file__).parent.parent
    graph_path = project_root / "output" / "graph" / "10_companies_full_optimized.graphml"
    vector_path = project_root / "output" / "vector_store"
    
    print("🔬 COMPREHENSIVE OUTPUT ANALYSIS")
    print("="*60)
    print(f"📂 Project: {project_root}")
    print(f"📄 Graph: {graph_path}")
    print(f"🎯 Vectors: {vector_path}")
    print()
    
    if not graph_path.exists():
        print(f"❌ Graph file not found: {graph_path}")
        return 1
    
    if not vector_path.exists():
        print(f"❌ Vector storage not found: {vector_path}")
        return 1
    
    # Analyze components
    graph_stats = analyze_graph(graph_path)
    vector_stats = analyze_vector_storage(vector_path)
    
    # Consistency analysis
    analyze_consistency(graph_stats, vector_stats)
    
    print(f"\n🎉 Analysis completed successfully!")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)