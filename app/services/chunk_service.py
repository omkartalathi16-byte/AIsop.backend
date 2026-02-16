import nltk
import logging

logger = logging.getLogger(__name__)


class ChunkService:
    def __init__(self, sentences_per_chunk: int = 5):
        self.sentences_per_chunk = sentences_per_chunk
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt')
            nltk.download('punkt_tab')

    def chunk_text(self, text: str) -> list[str]:
        sentences = nltk.sent_tokenize(text)
        chunks = []
        for i in range(0, len(sentences), self.sentences_per_chunk):
            chunk = " ".join(sentences[i:i + self.sentences_per_chunk])
            chunks.append(chunk)
        return chunks
