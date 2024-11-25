import torch  # Add this import
from transformers import BertTokenizer

def predict(model, tokenizer, news_text):
    # Tokenize and prepare the input
    inputs = tokenizer(news_text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Determine the device (GPU or CPU)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Perform the prediction
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1).item()

    return "Significant Rise" if prediction == 1 else "Significant Drop"
