import torch
from torch import nn

from .base_model import BagRE


class BagAttention(BagRE):
    """
    Instance attention for bag-level relation extraction.
    """

    def __init__(self, sentence_encoder, num_class, rel2id, use_diag=True):
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
        if use_diag:
            self.use_diag = True
            self.diag = nn.Parameter(torch.ones(self.sentence_encoder.hidden_size))
        else:
            self.use_diag = False

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
        self.eval()
        tokens = []
        pos1s = []
        pos2s = []
        masks = []
        for item in bag:
            token, pos1, pos2, mask = self.sentence_encoder.tokenize(item)
            tokens.append(token)
            pos1s.append(pos1)
            pos2s.append(pos2)
            masks.append(mask)
        tokens = torch.cat(tokens, 0).unsqueeze(0) # (n, L)
        pos1s = torch.cat(pos1s, 0).unsqueeze(0)
        pos2s = torch.cat(pos2s, 0).unsqueeze(0)
        masks = torch.cat(masks, 0).unsqueeze(0)
        scope = torch.tensor([[0, len(bag)]]).long() # (1, 2)
        bag_logits = self.forward(None, scope, tokens, pos1s, pos2s, masks, train=False).squeeze(0) # (N) after softmax
        score, pred = bag_logits.max(0)
        score = score.item()
        pred = pred.item()
        rel = self.id2rel[pred]
        return (rel, score)
    
    def forward(self, label, scope, token, pos1, pos2, mask=None, train=True, bag_size=0):
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

        Dirty hack:
            When the encoder is BERT, the input is actually token, att_mask, pos1, pos2, but
            since the arguments are then fed into BERT encoder with the original order,
            the encoder can actually work out correclty.
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

        # Attention
        if train:
            if mask is not None:
                rep = self.sentence_encoder(token, pos1, pos2, mask) # (nsum, H) 
            else:
                rep = self.sentence_encoder(token, pos1, pos2) # (nsum, H) 

            if bag_size == 0:
                bag_rep = []
                query = torch.zeros((rep.size(0))).long()
                if torch.cuda.is_available():
                    query = query.cuda()
                for i in range(len(scope)):
                    query[scope[i][0]:scope[i][1]] = label[i]
                att_mat = self.fc.weight[query] # (nsum, H)
                if self.use_diag:
                    att_mat = att_mat * self.diag.unsqueeze(0)
                att_score = (rep * att_mat).sum(-1) # (nsum)

                for i in range(len(scope)):
                    bag_mat = rep[scope[i][0]:scope[i][1]] # (n, H)
                    softmax_att_score = self.softmax(att_score[scope[i][0]:scope[i][1]]) # (n)
                    bag_rep.append((softmax_att_score.unsqueeze(-1) * bag_mat).sum(0)) # (n, 1) * (n, H) -> (n, H) -> (H)
                bag_rep = torch.stack(bag_rep, 0) # (B, H)
            else:
                batch_size = label.size(0)
                query = label.unsqueeze(1) # (B, 1)
                att_mat = self.fc.weight[query] # (B, 1, H)
                if self.use_diag:
                    att_mat = att_mat * self.diag.unsqueeze(0)
                rep = rep.view(batch_size, bag_size, -1)
                att_score = (rep * att_mat).sum(-1) # (B, bag)
                softmax_att_score = self.softmax(att_score) # (B, bag)
                bag_rep = (softmax_att_score.unsqueeze(-1) * rep).sum(1) # (B, bag, 1) * (B, bag, H) -> (B, bag, H) -> (B, H)
            bag_rep = self.drop(bag_rep)
            bag_logits = self.fc(bag_rep) # (B, N)
        else:

            if bag_size == 0:
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

                bag_logits = []
                att_mat = self.fc.weight.transpose(0, 1)
                if self.use_diag:
                    att_mat = att_mat * self.diag.unsqueeze(1)
                att_score = torch.matmul(rep, att_mat) # (nsum, H) * (H, N) -> (nsum, N)
                for i in range(len(scope)):
                    bag_mat = rep[scope[i][0]:scope[i][1]] # (n, H)
                    softmax_att_score = self.softmax(att_score[scope[i][0]:scope[i][1]].transpose(0, 1)) # (N, (softmax)n) 
                    rep_for_each_rel = torch.matmul(softmax_att_score, bag_mat) # (N, n) * (n, H) -> (N, H)
                    logit_for_each_rel = self.softmax(self.fc(rep_for_each_rel)) # ((each rel)N, (logit)N)
                    logit_for_each_rel = logit_for_each_rel.diag() # (N)
                    bag_logits.append(logit_for_each_rel)
                bag_logits = torch.stack(bag_logits, 0) # after **softmax**
            else:
                if mask is not None:
                    rep = self.sentence_encoder(token, pos1, pos2, mask) # (nsum, H) 
                else:
                    rep = self.sentence_encoder(token, pos1, pos2) # (nsum, H) 

                batch_size = rep.size(0) // bag_size
                att_mat = self.fc.weight.transpose(0, 1)
                if self.use_diag:
                    att_mat = att_mat * self.diag.unsqueeze(1) 
                att_score = torch.matmul(rep, att_mat) # (nsum, H) * (H, N) -> (nsum, N)
                att_score = att_score.view(batch_size, bag_size, -1) # (B, bag, N)
                rep = rep.view(batch_size, bag_size, -1) # (B, bag, H)
                softmax_att_score = self.softmax(att_score.transpose(1, 2)) # (B, N, (softmax)bag)
                rep_for_each_rel = torch.matmul(softmax_att_score, rep) # (B, N, bag) * (B, bag, H) -> (B, N, H)
                bag_logits = self.softmax(self.fc(rep_for_each_rel)).diagonal(dim1=1, dim2=2) # (B, (each rel)N)
        return bag_logits

