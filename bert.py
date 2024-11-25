from transformers import BertTokenizer

def tokenize_texts(texts, max_length=512):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return tokenizer(list(texts), padding=True, truncation=True, max_length=max_length, return_tensors="pt")
