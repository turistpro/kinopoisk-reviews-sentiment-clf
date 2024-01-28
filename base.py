from transformers import BertForMaskedLM, BertTokenizer

tokenizer = BertTokenizer.from_pretrained("ai-forever/ruBert-base", do_lower_case=False)
baseModel = BertForMaskedLM.from_pretrained("ai-forever/ruBert-base")

for param in baseModel.parameters():
    param.requires_grad = False