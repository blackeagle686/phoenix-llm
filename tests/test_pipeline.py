import sys
import os
# Add current directory to path
sys.path.append(os.getcwd())

from data.pipeline import DataPipeline
from data.loader import TextDataLoader

def test_pipeline():
    print("Initializing DataPipeline (wikitext)...")
    # Use wikitext-2-raw-v1 for testing
    pipeline = DataPipeline("wikitext", "wikitext-2-raw-v1", split="test")
    
    print("Initializing TextDataLoader...")
    loader = TextDataLoader(pipeline, chunk_size=512)
    
    print("Streaming first 5 batches...")
    batch_count = 0
    for batch in loader.get_batches(batch_size=2):
        print(f"\n--- Batch {batch_count} ---")
        for i, text in enumerate(batch):
            print(f"Chunk {i} (Length: {len(text)})")
            print(f"Sample: {text[:100]}...")
        
        batch_count += 1
        if batch_count >= 5:
            break
            
    print("\nData Pipeline Verification SUCCESSFUL!")

if __name__ == "__main__":
    test_pipeline()
