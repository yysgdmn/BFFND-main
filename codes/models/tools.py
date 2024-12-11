from transformers import BertTokenizer, BertModel

def pretrain_bert_token():
    print('da taloader中加载token')
    tokenizer = BertTokenizer.from_pretrained("roberta_wwm")
    return tokenizer


def pretrain_bert_models():
    # tokenizer = BertTokenizer.from_pretrained("pretrain/roberta_wwm")
    print('model中加载bert')
    # model = BertModel.from_pretrained("E:\daima\FakeSV-main\codes\\roberta_wwm").cuda()
    model = BertModel.from_pretrained("roberta_wwm").cuda()
    for param in model.parameters(): 
        param.requires_grad = False
    return model