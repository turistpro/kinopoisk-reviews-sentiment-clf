import torch
from torch import nn
from torch.utils.data import TensorDataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

def tensorDataset(tokenizer, dataset, max_length = 128):
    tokens = tokenizer.batch_encode_plus(
        dataset['text'],
        max_length = max_length,
        padding = 'max_length',
        truncation = True
    )

    seq = torch.tensor(tokens['input_ids'])
    mask = torch.tensor(tokens['attention_mask'])
    y = torch.tensor(dataset['labels'])

    # Dataset обучающей выборки
    return TensorDataset(seq, mask, y)

def train(model, data_loader, optimizer, scheduler, device):
    model.train()
    for step, batch in tqdm(enumerate(data_loader), total = len(data_loader)):
        optimizer.zero_grad()
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions)