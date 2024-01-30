from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

from model import tokenizer, BERTClassifier
from functions import train, evaluate, tensorDataset

print('start...')
# Set up parameters
num_classes = 2
max_length = 128
batch_size = 16
num_epochs = 4
learning_rate = 2e-5
deviceId = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available else "cpu")
print("device =", deviceId)
device = torch.device(deviceId)

print('load dataset...')
#
dataset = load_dataset("zloelias/kinopoisk-reviews")


print('embeddings...')
# Dataset обучающей выборки
train_data = tensorDataset(tokenizer, dataset['train'])
# Dataset тестовой выборки
test_data = tensorDataset(tokenizer, dataset['test'])

print('train...')
model = BERTClassifier(num_classes).to(device)

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(test_data, batch_size=batch_size)


optimizer = AdamW(model.parameters(), lr=learning_rate)

total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train(model, train_dataloader, optimizer, scheduler, device)
        accuracy, report = evaluate(model, val_dataloader, device)
        print(f"Validation Accuracy: {accuracy:.4f}")
        print(report)

print('save...')
torch.save(model.state_dict(), "bert_classifier.pth")