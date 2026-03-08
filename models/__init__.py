import torch

def calculate(inputs, attn_score):
    attn_score = attn_score[0]
    scores, indexs = torch.sort(attn_score,descending=True)
    top_3_token = []
    for i in range(indexs.shape[0]):
        ids = inputs[i,indexs[i,:3]]    
        top_3_token.append(ids.cpu().numpy().tolist())
    return top_3_token


