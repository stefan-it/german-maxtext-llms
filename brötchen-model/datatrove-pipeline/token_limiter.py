from datatrove.data import Document
from datatrove.pipeline.base import PipelineStep
from transformers import AutoTokenizer
from typing import Generator

class TokenLimiter(PipelineStep):
    def __init__(self, tokenizer_name_or_path: str, max_tokens: int):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        self.max_tokens = max_tokens
        self.tokens_seen = 0
    
    def run(self, data: Generator[Document, None, None], rank: int = 0, world_size: int = 1):
        for doc in data:
            if self.tokens_seen >= self.max_tokens:
                break

            tokens = self.tokenizer.encode(doc.text)
            remaining = self.max_tokens - self.tokens_seen

            doc.metadata['subtokens'] = len(tokens)

            if len(tokens) > remaining:
                self.tokens_seen = self.max_tokens
                yield doc
                break
            else:
                self.tokens_seen += len(tokens)
                yield doc