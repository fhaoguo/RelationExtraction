import torch
from torch import nn

from .base_model import BagRE


class BagAverage(BagRE):
    """
    Average policy for bag-level relation extraction.
    """

    def __init__(self, sentence_encoder, num_class, rel2id):
        """
        Args:
            sentence_encoder: encoder for sentences
            num_class: number of classes
            id2rel: dictionary of id -> relation name mapping
        """
        super().__init__()
        self.sentence_encoder = sentence_encoder
        self.num_class = num_class
        self.fc = nn.Linear(self.sentence_encoder.hidden_size, num_class)
        self.softmax = nn.Softmax(-1)
        self.rel2id = rel2id
        self.id2rel = {}
        self.drop = nn.Dropout()
        for rel, id in rel2id.items():
            self.id2rel[id] = rel

    def infer(self, bag):
        """
        Args:
            bag: bag of sentences with the same entity pair
                [{
                  'text' or 'token': ..., 
                  'h': {'pos': [start, end], ...}, 
                  't': {'pos': [start, end], ...}
                }]
        Return:
            (relation, score)
        """
        pass

        """
        tokens = []
        pos1s = []
        pos2s = []
        masks = []
        for item in bag:
            if 'text' in item:
                token, pos1, pos2, mask = self.tokenizer(item['text'], 
                    item['h']['pos'], item['t']['pos'], is_token=False, padding=True)
            else:
                token, pos1, pos2, mask = self.tokenizer(item['token'], 
                    item['h']['pos'], item['t']['pos'], is_token=True, padding=True)
            tokens.append(token)
            pos1s.append(pos1)
            pos2s.append(pos2)
            masks.append(mask)
        tokens = torch.cat(tokens, 0) # (n, L)
        pos1s = torch.cat(pos1s, 0)
        pos2s = torch.cat(pos2s, 0)
        masks = torch.cat(masks, 0) 
        scope = torch.tensor([[0, len(bag)]]).long() # (1, 2)
        bag_logits = self.forward(None, scope, tokens, pos1s, pos2s, masks, train=False).squeeze(0) # (N) after softmax
        score, pred = bag_logits.max()
        score = score.item()
        pred = pred.item()
        rel = self.id2rel[pred]
        return (rel, score)
        """
    
    def forward(self, label, scope, token, pos1, pos2, mask=None, train=True, bag_size=None):
        """
        Args:
            label: (B), label of the bag
            scope: (B), scope for each bag
            token: (nsum, L), index of tokens
            pos1: (nsum, L), relative position to head entity
            pos2: (nsum, L), relative position to tail entity
            mask: (nsum, L), used for piece-wise CNN
        Return:
            logits, (B, N)
        """
        if bag_size > 0:
            token = token.view(-1, token.size(-1))
            pos1 = pos1.view(-1, pos1.size(-1))
            pos2 = pos2.view(-1, pos2.size(-1))
            if mask is not None:
                mask = mask.view(-1, mask.size(-1))
        else:
            begin, end = scope[0][0], scope[-1][1]
            token = token[:, begin:end, :].view(-1, token.size(-1))
            pos1 = pos1[:, begin:end, :].view(-1, pos1.size(-1))
            pos2 = pos2[:, begin:end, :].view(-1, pos2.size(-1))
            if mask is not None:
                mask = mask[:, begin:end, :].view(-1, mask.size(-1))
            scope = torch.sub(scope, torch.zeros_like(scope).fill_(begin))

        if train or bag_size > 0:
            if mask is not None:
                rep = self.sentence_encoder(token, pos1, pos2, mask) # (nsum, H) 
            else:
                rep = self.sentence_encoder(token, pos1, pos2) # (nsum, H) 
        else:
            rep = []
            bs = 256
            total_bs = len(token) // bs + (1 if len(token) % bs != 0 else 0)
            for b in range(total_bs):
                with torch.no_grad():
                    left = bs * b
                    right = min(bs * (b + 1), len(token))
                    if mask is not None:        
                        rep.append(self.sentence_encoder(token[left:right], pos1[left:right], pos2[left:right], mask[left:right]).detach()) # (nsum, H) 
                    else:
                        rep.append(self.sentence_encoder(token[left:right], pos1[left:right], pos2[left:right]).detach()) # (nsum, H) 
            rep = torch.cat(rep, 0)

        # Average
        bag_rep = []
        if bag_size is None or bag_size == 0:
            for i in range(len(scope)):
                bag_rep.append(rep[scope[i][0]:scope[i][1]].mean(0))
            bag_rep = torch.stack(bag_rep, 0) # (B, H)
        else:
            batch_size = len(scope)
            rep = rep.view(batch_size, bag_size, -1) # (B, bag, H)
            bag_rep = rep.mean(1) # (B, H)
        bag_rep = self.drop(bag_rep)
        bag_logits = self.fc(bag_rep) # (B, N)

        if not train:
            bag_logits = torch.softmax(bag_logits, -1) 

        return bag_logits

