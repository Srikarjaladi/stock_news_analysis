from transformers import BertForSequenceClassification, AdamW, get_scheduler
import torch
from torch.utils.data import DataLoader


def train_model(train_dataset, val_dataset, epochs=4, batch_size=16):
    # Model, optimizer, scheduler setup
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # Training loop
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = {key: val.to(device) for key, val in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {key: val.to(device) for key, val in batch.items()}
                outputs = model(**batch)
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct += (predictions == batch['labels']).sum().item()
                total += batch['labels'].size(0)
        print(f'Epoch {epoch}, Validation Accuracy: {correct / total}')

    return model
