#!/usr/bin/env python3
"""
Debug vector storage to understand why there are 11 documents instead of 10.
"""

import sys
from pathlib import Path
import lancedb

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

def debug_vector_storage():
    """Debug the vector storage contents."""
    project_root = Path(__file__).parent.parent
    vector_path = project_root / "output" / "vector_store"
    
    print("🔍 DEBUGGING VECTOR STORAGE")
    print("="*50)
    
    try:
        # Connect to database
        db = lancedb.connect(str(vector_path))
        
        # Open table
        table = db.open_table("ashare_documents")
        
        # Get all records
        all_data = table.to_pandas()
        
        print(f"📊 Total records: {len(all_data)}")
        print(f"📋 Columns: {list(all_data.columns)}")
        
        # Analyze document IDs
        print(f"\n📄 Document ID Analysis:")
        doc_ids = all_data['doc_id'].value_counts()
        print(doc_ids)
        
        # Check for duplicates
        print(f"\n🔍 Duplicate Analysis:")
        unique_combinations = all_data[['doc_id', 'chunk_index']].drop_duplicates()
        print(f"   • Unique (doc_id, chunk_index) combinations: {len(unique_combinations)}")
        print(f"   • Total records: {len(all_data)}")
        
        if len(unique_combinations) != len(all_data):
            print("   ⚠️  Found duplicate records!")
            
            # Show duplicates
            duplicated = all_data[all_data.duplicated(['doc_id', 'chunk_index'], keep=False)]
            print(f"   • Duplicated records: {len(duplicated)}")
            
            print(f"\n🔍 Duplicate details:")
            for i, row in duplicated.iterrows():
                print(f"   {i}: doc_id={row['doc_id']}, company={row['company_name']}, chunk={row['chunk_index']}")
        
        # Show all document details
        print(f"\n📋 All Documents:")
        for i, row in all_data.iterrows():
            print(f"   {i+1:2d}. ID: {row['doc_id']:<12} | Company: {row['company_name']:<15} | Chunk: {row['chunk_index']} | Relations: {row['relations_count']}")
        
        # Check company coverage
        companies = all_data['company_name'].unique()
        print(f"\n🏢 Companies in vector storage: {len(companies)}")
        for i, company in enumerate(sorted(companies), 1):
            count = len(all_data[all_data['company_name'] == company])
            print(f"   {i:2d}. {company:<15} - {count} records")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    debug_vector_storage()