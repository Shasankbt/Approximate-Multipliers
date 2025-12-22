from transformers import BertTokenizer

class Tokenizer:
    def __init__(self, model_name='bert-base-uncased'):
        """
        Initialize tokenizer with pretrained BERT tokenizer from Hugging Face
        Args:
            model_name: HuggingFace model name (default: 'bert-base-uncased')
        """
        self.tokenizer_ = BertTokenizer.from_pretrained(model_name)
        self.model_name_ = model_name
        
        # Set up special token IDs as class members
        self.pad_id = self.tokenizer_.pad_token_id
        self.mask_id = self.tokenizer_.mask_token_id
        self.cls_id = self.tokenizer_.cls_token_id
        self.sep_id = self.tokenizer_.sep_token_id
        self.unk_id = self.tokenizer_.unk_token_id
        
        # Get the vocabulary
        self.vocab = self.tokenizer_.vocab
    
    def get_vocab_size(self):
        """Returns the size of the vocabulary"""
        return len(self.vocab)
    
    def get_ids(self, text, pad_length=None):
        """
        Convert text to token IDs with CLS, SEP, padding and truncation
        """
        if pad_length is not None:
            # Use built-in padding and truncation
            ids = self.tokenizer_.encode(
                text,
                add_special_tokens=True,
                max_length=pad_length,
                padding='max_length',
                truncation=True
            )
        else:
            # No padding/truncation
            ids = self.tokenizer_.encode(text, add_special_tokens=True)
    
        return ids
    
    def convert_ids_to_tokens(self, ids):
        return self.tokenizer_.convert_ids_to_tokens(ids)
    
    def decode(self, ids, skip_special_tokens=True):
        return self.tokenizer_.decode(ids, skip_special_tokens=skip_special_tokens)