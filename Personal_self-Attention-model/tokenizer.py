import json
import re

SPECIAL_TOKENS = {
    "PAD": "<PAD>",
    "UNK": "<UNK>",
    "CLS": "<CLS>",
    "SEP": "<SEP>",
    "MASK": "<MASK>"
}

class Tokenizer:
    def __init__(self, vocab_json=None):
        if vocab_json:
            with open(vocab_json, 'r') as f:
                self.vocab = json.load(f)
        else:
            print("No vocabulary file provided. Initializing empty vocabulary. Provide texts to generate vocabulary.")
            self.vocab = {}

    def get_vocab_size(self):
        return len(self.vocab)
    
    def _tokenize2(self, text):
        '''
            splits text into words and punctuation
            eg: "Hello, world!" -> ["hello", ",", "world", "!"]
        '''
        text = text.lower().strip()
        tokens = re.findall(r"\w+|[^\w\s]", text)
        return tokens
    
    def _get_likely_tokens(self, texts, max_size):
        """
            Build WordPiece-like vocab from training corpus.
            This is a simplified version:
            1. Start with all characters.
            2. Iteratively merge most frequent pairs until vocab_size.
        """
        vocab = set()

        corpus_tokens = []
        for text in texts:
            for word in self._tokenize2(text):
                word = word + "</w>"  # End of word token
                corpus_tokens.append(list(word))
                vocab.update(list(word))


    def _tokenize(self, text):
        text = text.lower().strip()
        return text.split()

    
    
    def generate_vocab(self, texts):
        unique_tokens = set()
        
        for special_token in SPECIAL_TOKENS.values():
            unique_tokens.add(special_token)

        for text in texts:
            tokens = self._tokenize(text)
            unique_tokens.update(tokens)
        
        self.vocab = {token: idx for idx, token in enumerate(sorted(unique_tokens))}

    def save_vocab(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.vocab, f)

    def get_ids(self, text, pad_length=None):
        tokens = self._tokenize(text)
        tokens = [token if token in self.vocab else SPECIAL_TOKENS["UNK"] for token in tokens]

        if pad_length is None:
            pad_length = len(tokens) + 2
        
        if len(tokens) > pad_length - 2:
            tokens = tokens[:pad_length - 2]
        
        ids = (
            [self.vocab[SPECIAL_TOKENS["CLS"]]] +
            [self.vocab[token] for token in tokens] +
            [self.vocab[SPECIAL_TOKENS["PAD"]]] * (pad_length - len(tokens) - 2) +
            [self.vocab[SPECIAL_TOKENS["SEP"]]]
        )
        return ids
    

    def convert_ids_to_tokens(self, ids):
        inv_vocab = {v: k for k, v in self.vocab.items()}
        return [inv_vocab[id] for id in ids if id in inv_vocab]