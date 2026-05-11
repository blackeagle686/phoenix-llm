from datasets import load_dataset
from .filters import clean_html, normalize_text, quality_filter, exact_deduplicate
from typing import Iterator, Set

class DataPipeline:
    def __init__(self, dataset_name: str, config_name: str = None, split: str = 'train'):
        self.dataset_name = dataset_name
        self.config_name = config_name
        self.split = split
        self.seen_hashes: Set[str] = set()

    def stream(self) -> Iterator[str]:
        """Streams, cleans, and filters the dataset."""
        print(f"Streaming dataset: {self.dataset_name} (split: {self.split})...")
        
        # Stream the dataset from HuggingFace
        dataset = load_dataset(self.dataset_name, self.config_name, split=self.split, streaming=True)
        
        for item in dataset:
            # Most HF datasets have a 'text' column, but some use 'content' or 'body'
            text = item.get('text') or item.get('content') or item.get('body')
            if not text:
                continue
                
            # 1. Clean
            text = clean_html(text)
            text = normalize_text(text)
            
            # 2. Quality Filter
            if not quality_filter(text):
                continue
                
            # 3. Deduplicate
            if not exact_deduplicate(text, self.seen_hashes):
                continue
                
            yield text
