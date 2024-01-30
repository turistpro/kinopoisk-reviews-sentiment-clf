from torch import nn
from transformers import BertModel, BertTokenizer

tokenizer = BertTokenizer.from_pretrained("ai-forever/ruBert-base", do_lower_case=False)
baseModel = BertModel.from_pretrained("ai-forever/ruBert-base")

for param in baseModel.parameters():
    param.requires_grad = False

class BERTClassifier(nn.Module):
    
    def __init__(self, num_classes):
            super(BERTClassifier, self).__init__()
            self.bert = baseModel
            self.fc1 = nn.Linear(self.bert.config.hidden_size, 512)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.1)
            self.fc2 = nn.Linear(512, num_classes)
            self.softmax = nn.LogSoftmax(dim = 1)

    def forward(self, input_ids, attention_mask):
            outputs  = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            x = self.fc1(outputs.pooler_output)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            x = self.softmax(x)
            return x