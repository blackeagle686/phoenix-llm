from .pipeline import DataPipeline
from typing import Iterator

class TextDataLoader:
    def __init__(self, pipeline: DataPipeline, chunk_size: int = 1024):
        self.pipeline = pipeline
        self.chunk_size = chunk_size

    def __iter__(self) -> Iterator[str]:
        """Yields cleaned text in chunks of approximately chunk_size."""
        buffer = ""
        for text in self.pipeline.stream():
            buffer += text + " "
            
            while len(buffer) >= self.chunk_size:
                # Yield a chunk
                chunk = buffer[:self.chunk_size]
                buffer = buffer[self.chunk_size:]
                yield chunk
                
    def get_batches(self, batch_size: int):
        """Yields batches of text chunks."""
        iterator = iter(self)
        while True:
            batch = []
            try:
                for _ in range(batch_size):
                    batch.append(next(iterator))
                yield batch
            except StopIteration:
                if batch:
                    yield batch
                break
