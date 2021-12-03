import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def batch_pairwise_squared_distance(x, squared=False):
    bsz, max_len, dim = x.size()
    x_sq = (x**2).sum(dim=2)
    prod = torch.bmm(x, x.transpose(1,2))
    dist = (x_sq.unsqueeze(1) + x_sq.unsqueeze(2) - 2*prod).clamp(min=1e-12)
    if squared == True:
        dist = torch.sqrt(dist).clone()
    #dist[dist!=dist] = 0
    dist[:, range(max_len), range(max_len)] = 0
    return dist

def central_squared_distance(x, squared=False):
    bsz, max_len, w_size, dim = x.size()
    center = int((w_size-1)/2)
    x_sq = (x**2).sum(dim=3)
    prod = torch.bmm(x.view(-1, w_size, dim), x.view(-1, w_size, dim).transpose(1,2))
    prod = prod.view(bsz, max_len, w_size, w_size)[:,:,center,:]
    dist = (x_sq + x_sq[:,:,center].unsqueeze(2) - 2*prod).clamp(min=1e-12)
    if squared == True:
        dist = torch.sqrt(dist)   
    dist[:, range(max_len), center] = 0
    return dist

def window_index(w_size, bsz, length):

    w_size_2 = (w_size - 1)/2
    idx = torch.arange(0, length).unsqueeze(0).unsqueeze(2).repeat(bsz, 1, w_size)
    idx_range = torch.range(-w_size_2, w_size_2).expand_as(idx)
    idx = idx + idx_range
    idx = torch.clamp(idx, 0, length-1)
    idx_base = torch.arange(0, bsz).view(-1,1,1)*length
    idx = (idx + idx_base)
    return idx


class DistillKL(nn.Module):
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T
    def forward(self, y_s, y_t,mode="classification"):
        if mode == "regression":
            loss = F.mse_loss((y_s/self.T).view(-1), (y_t/self.T).view(-1))
        else:
            p_s = F.log_softmax(y_s/self.T, dim=-1)
            p_t = F.softmax(y_t/self.T, dim=-1)
            loss = -torch.sum(p_t * p_s, dim=-1).mean()
        return loss

class PKD_loss(nn.Module):
    ##Only CLS Token
    ##Input Dimension : Batchsize, n_layer, hidden_Size
    def __init__(self, p, normalize=False):
        super(PKD_loss, self).__init__()
        self.p = p
        self.normalize=normalize

    def forward(self, teacher_patience, student_patience):
        if self.normalize:
            if len(teacher_patience.size()) == 4:
                teacher_patience = F.normalize(teacher_patience, p=self.p, dim=3)
                student_patience = F.normalize(student_patience, p=self.p, dim=3)
            elif len(teacher_patience.size()) == 3:
                teacher_patience = F.normalize(teacher_patience, p=self.p, dim=2)
                student_patience = F.normalize(student_patience, p=self.p, dim=2)                

        return F.mse_loss(teacher_patience.float(), student_patience.float())

class WR_Dist(nn.Module):
    def __init__(self):
        super(WR_Dist, self).__init__()
    
    def forward(self, t_embed, s_embed, attention_mask, distance, lossfunc, normalize=False, squard=False):
        bsz, layer_num, max_len, dim = t_embed.size()
        _, _, _, sdim = s_embed.size()
        t_embed = t_embed.view(-1, max_len, dim)
        s_embed = s_embed.view(-1, max_len, sdim)
        mask = self.make_mask(attention_mask, layer_num)
        mask = mask.view(-1, max_len, max_len)
        with torch.no_grad():
            if distance == "cos":
                t_norm = F.normalize(t_embed, p=2, dim=2)
                t_d = torch.bmm(t_norm, t_norm.transpose(1,2))
                t_d = t_d * mask
                diagonal = (torch.ones(max_len, max_len) - torch.eye(max_len, max_len)).to(t_embed.device) 
                t_d = t_d.masked_fill(diagonal == 0, -np.inf)
                t_d = t_d.masked_fill(mask == 0, -np.inf)
                t_d = F.softmax(t_d, dim=-1)
                t_d = t_d * mask
            elif distance=="l2":
                t_d = batch_pairwise_squared_distance(t_embed, squared=False)
                if normalize:
                    t_d = t_d * mask
                    nonzero = torch.sum((t_d.view(bsz*layer_num, -1) > 0), dim=-1)
                    mean_td = t_d.view(bsz*layer_num, -1).sum(dim=-1) / nonzero
                    t_d = t_d / mean_td.unsqueeze(1).unsqueeze(2)
                else:
                    t_d = t_d * mask
        if distance == "cos":
            s_norm = F.normalize(s_embed, p=2, dim=2)
            s_d = torch.bmm(s_norm, s_norm.transpose(1,2))
            s_d = s_d * mask
            s_d = s_d.masked_fill(diagonal == 0, -np.inf)
            s_d = s_d.masked_fill(mask == 0, -np.inf)
            s_d = F.log_softmax(s_d, dim=-1)
            s_d = s_d * mask

        elif distance=="l2":
            s_d = batch_pairwise_squared_distance(s_embed, squared=False)
            if normalize:
                s_d = s_d * mask
                nonzero = torch.sum((s_d.view(bsz*layer_num, -1) > 0), dim=-1)
                mean_sd = s_d.view(bsz*layer_num, -1).sum(dim=-1) / nonzero
                s_d = s_d / mean_sd.unsqueeze(1).unsqueeze(2)
            else:
                s_d = s_d * mask

        if lossfunc == "kldiv":
            return F.kl_div(s_d, t_d, reduction="sum") / mask.sum().item()
        elif lossfunc == "l1loss":
            return F.l1_loss(s_d, t_d, reduction='sum') / mask.sum().item()
        elif lossfunc == "l2loss":
            return F.mse_loss(s_d, t_d, reduction='sum') / mask.sum().item()
        elif lossfunc =='smoothl1':
            return F.smooth_l1_loss(s_d, t_d, reduction='sum') / mask.sum().item()

    def make_mask(self, attention_mask, layers):
        mask = attention_mask.unsqueeze(2) * attention_mask.unsqueeze(1)
        return mask.unsqueeze(1).repeat(1,layers,1,1).float()

##Update WRDIST with window

class WR_Angle(nn.Module):
    def __init__(self):
        super(WR_Angle, self).__init__()
    def forward(self, t_embed, s_embed, attention_mask, lossfunc):
        bsz, layer_num, max_len, dim = t_embed.size()
        bsz, layer_num, max_len, sdim = s_embed.size()
        t_embed = t_embed.view(-1, max_len, dim)
        s_embed = s_embed.view(-1, max_len, sdim)

        mask  = self.make_mask(attention_mask, layer_num)
        mask = mask.view(-1, max_len, max_len, max_len)
        with torch.no_grad():
            #1441
            t_sub = (t_embed.unsqueeze(1) - t_embed.unsqueeze(2))   #1873
            t_sub = F.normalize(t_sub, p=2, dim=3).view(-1,max_len,dim) #2305
            t_angle = torch.bmm(t_sub, t_sub.transpose(1,2)).view(-1, max_len, max_len, max_len)
            t_angle = t_angle * mask

        s_sub = (s_embed.unsqueeze(1) - s_embed.unsqueeze(2))   #2737
        s_sub = F.normalize(s_sub, p=2, dim=3).view(-1, max_len, sdim)   #3169
        s_angle = torch.bmm(s_sub, s_sub.transpose(1,2)).view(-1, max_len, max_len, max_len)
        s_angle = s_angle * mask

        if lossfunc == "l1loss":
            return F.l1_loss(s_angle, t_angle, reduction='sum') / mask.sum().item()
        elif lossfunc == "l2loss":
            return F.mse_loss(s_angle, t_angle, reduction='sum') / mask.sum().item()
        elif lossfunc == "smoothl1":
            return F.smooth_l1_loss(s_angle, t_angle, reduction='sum') / mask.sum().item()

    def make_mask(self, attention_mask, layers):
        mask = attention_mask.unsqueeze(2).unsqueeze(3) * attention_mask.unsqueeze(1).unsqueeze(3) * attention_mask.unsqueeze(1).unsqueeze(2)
        return mask.unsqueeze(1).repeat(1,layers,1,1,1).float()

class WR_Angle_window(nn.Module):
    def __init__(self):
        super(WR_Angle_window, self).__init__()
    def forward(self, t_embed, s_embed, attention_mask, lossfunc, window=5):
        assert (window % 2) == 1
        bsz, layer_num, max_len, dim = t_embed.size()
        bsz, layer_num, max_len, sdim = s_embed.size()
        t_embed = t_embed.view(-1, max_len, dim)
        s_embed = s_embed.view(-1, max_len, sdim)
        new_bsz = bsz * layer_num
        idx = window_index(window, new_bsz, max_len)
        #idx = idx.long().unsqueeze(1).repeat(1, layer_num,1,1).view(-1, max_len, window)
        idx = idx.long()
        t_round_emb = t_embed.view(new_bsz*max_len, -1)[idx, :]
        s_round_emb = s_embed.view(new_bsz*max_len, -1)[idx, :]
        mask = self.make_mask(attention_mask, layer_num, window)
        mask = mask.view(-1, max_len, window, window)

        with torch.no_grad():
            t_sub = (t_embed.unsqueeze(2) - t_round_emb)
            # bsz, len, window, window, dim
            t_sub = F.normalize(t_sub, p=2, dim=3).view(-1, window, dim)
            t_angle = torch.bmm(t_sub, t_sub.transpose(1,2)).view(new_bsz, max_len, window, window)
            t_angle = t_angle * mask
        s_sub = (s_embed.unsqueeze(2) - s_round_emb)   #2737
        s_sub = F.normalize(s_sub, p=2, dim=3).view(-1, window, sdim)   #3169
        s_angle = torch.bmm(s_sub, s_sub.transpose(1,2)).view(new_bsz, max_len, window, window)
        s_angle = s_angle * mask

        if lossfunc == "l1loss":
            return F.l1_loss(s_angle, t_angle, reduction='sum') / mask.sum().item()
        elif lossfunc == "l2loss":
            return F.mse_loss(s_angle, t_angle, reduction='sum') / mask.sum().item()
        elif lossfunc == "smoothl1":
            return F.smooth_l1_loss(s_angle, t_angle, reduction='sum') / mask.sum().item()

    def make_mask(self, attention_mask, layers, window):
        mask = attention_mask.unsqueeze(2).unsqueeze(3).repeat(1,1,window,window)
        return mask.unsqueeze(1).repeat(1,layers,1,1,1).float()

class LTR_Dist(nn.Module):
    def __init__(self):
        super(LTR_Dist, self).__init__()
    
    def forward(self, t_embed, s_embed, attention_mask, distance, lossfunc, normalize=False, squard=False):
        bsz, layer_num, max_len, dim = t_embed.size()
        bsz, layer_num, max_len, sdim = s_embed.size()
        t_embed = t_embed.transpose(1,2).reshape(-1, layer_num, dim)
        s_embed = s_embed.transpose(1,2).reshape(-1, layer_num, sdim)

        mask = self.make_mask(attention_mask, layer_num).view(-1, layer_num, layer_num)
        mask = mask.view(-1, layer_num, layer_num)

        with torch.no_grad():
            if distance == "cos":
                t_norm = F.normalize(t_embed, p=2, dim=2)
                t_d = torch.bmm(t_norm, t_norm.transpose(1,2))
                t_d = t_d * mask
                diagonal = (torch.ones(layer_num, layer_num) - torch.eye(layer_num, layer_num)).to(t_embed.device)
                t_d = t_d.masked_fill(diagonal == 0, -np.inf)
                t_d = t_d.masked_fill(mask == 0, -np.inf)
                #t_d = t_d.masked_fill(t_d == 1.0, -np.inf)
                t_d = F.softmax(t_d, dim=-1)
                t_d = t_d * mask

            elif distance == "l2":
                t_d = batch_pairwise_squared_distance(t_embed, squared=False)
                if normalize:
                    t_d = t_d * mask
                    nonzero = torch.sum((t_d.view(bsz*max_len, -1) > 0), dim=-1)
                    nonzero[nonzero==0] = 1
                    mean_td = t_d.view(bsz*max_len, -1).sum(dim=-1) / nonzero
                    mean_td[mean_td==0] = 1
                    t_d = t_d / mean_td.unsqueeze(1).unsqueeze(2)
                else:
                    t_d = t_d * mask
        if distance == "cos":
            s_norm = F.normalize(s_embed, p=2, dim=2)
            s_d = torch.bmm(s_norm, s_norm.transpose(1,2))
            s_d = s_d * mask
            s_d = s_d.masked_fill(diagonal == 0, -np.inf)
            s_d = s_d.masked_fill(mask == 0, -np.inf)
            #s_d = s_d.masked_fill(s_d == 1.0, -np.inf)
            s_d = F.log_softmax(s_d, dim=-1)
            s_d = s_d * mask

        elif distance == "l2":
            s_d = batch_pairwise_squared_distance(s_embed, squared=False)
            if normalize:
                s_d = s_d * mask
                nonzero = torch.sum((s_d.view(bsz*max_len, -1) > 0), dim=-1)
                nonzero[nonzero==0] = 1
                mean_sd = s_d.view(bsz*max_len, -1).sum(dim=-1) / nonzero
                mean_sd[mean_sd==0] = 1
                s_d = s_d / mean_sd.unsqueeze(1).unsqueeze(2)
            else:
                s_d = s_d * mask

        if lossfunc == "kldiv":
            return F.kl_div(s_d, t_d, reduction="sum") / mask.sum().item()
        elif lossfunc == "l1loss":
            return F.l1_loss(s_d, t_d, reduction='sum') / mask.sum().item()
        elif lossfunc == "l2loss":
            return F.mse_loss(s_d, t_d, reduction='sum') / mask.sum().item()
        elif lossfunc =='smoothl1':
            return F.smooth_l1_loss(s_d, t_d, reduction='sum') / mask.sum().item()

    def make_mask(self, attention_mask, layer):
        #attention mask -> b, len
        # mask -> b, len, 6, 6
        mask = attention_mask.unsqueeze(2).unsqueeze(3)
        return mask.repeat(1, 1, layer, layer).float()

class LTR_Angle(nn.Module):
    def __init__(self):
        super(LTR_Angle, self).__init__()
    def forward(self, t_embed, s_embed, attention_mask, loss):
        bsz, layer_num, max_len, dim = t_embed.size()
        bsz, layer_num, max_len, sdim = s_embed.size()
        t_embed = t_embed.transpose(1,2).reshape(-1, layer_num, dim)
        s_embed = s_embed.transpose(1,2).reshape(-1, layer_num, sdim)
        mask  = self.make_mask(attention_mask, layer_num)
        mask = mask.view(-1, layer_num, layer_num, layer_num)
        with torch.no_grad():
            #1441
            t_sub = (t_embed.unsqueeze(1) - t_embed.unsqueeze(2))   #1873
            t_sub = F.normalize(t_sub, p=2, dim=3).view(-1,layer_num,dim) #2305
            t_angle = torch.bmm(t_sub, t_sub.transpose(1,2)).view(-1, layer_num, layer_num, layer_num)
            t_angle = t_angle * mask

        s_sub = (s_embed.unsqueeze(1) - s_embed.unsqueeze(2))   #2737
        s_sub = F.normalize(s_sub, p=2, dim=3).view(-1, layer_num, sdim)   #3169
        s_angle = torch.bmm(s_sub, s_sub.transpose(1,2)).view(-1, layer_num,layer_num,layer_num)   #3385
        s_angle = s_angle * mask
        if loss == "l1loss":
            return F.l1_loss(s_angle, t_angle, reduction='sum') / mask.sum().item()
        elif loss == "l2loss":
            return F.mse_loss(s_angle, t_angle, reduction='sum') / mask.sum().item()
        elif loss == "smoothl1":
            return F.smooth_l1_loss(s_angle, t_angle, reduction='sum') / mask.sum().item()

    def make_mask(self, attention_mask, layers):
        mask = attention_mask.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        return mask.repeat(1,1,layers,layers,layers).float()

class Hidden_mse(nn.Module):
    def __init__(self, student_size, teacher_size):
        super(Hidden_mse, self).__init__()
        #self.fit_dense = nn.Linear(student_size, teacher_size)

    def forward(self, s_embed, t_embed):
        bsz, layer_num, max_len, tdim = t_embed.size()
        bsz, layer_num, max_len, sdim = s_embed.size()
        t_embed = t_embed.view(-1, max_len, tdim)
        s_embed = s_embed.view(-1, max_len, sdim)
        #s_embed = self.fit_dense(s_embed)
        return F.mse_loss(s_embed, t_embed, reduction="mean")

class Attention_mse(nn.Module):
    def __init__(self):
        super(Attention_mse, self).__init__()
    def forward(self, student_atts, teacher_atts):
        bsz, layer_num, head, max_len, max_len = student_atts.size()
        student_atts = student_atts.view(-1, max_len, max_len)
        teacher_atts = teacher_atts.view(-1, max_len, max_len)
        student_atts = torch.where(student_atts <= -1e2, torch.zeros_like(student_atts).to(student_atts.device),
                                    student_atts)
        teacher_atts = torch.where(teacher_atts <= -1e2, torch.zeros_like(teacher_atts).to(student_atts.device),
                                    teacher_atts)
        return F.mse_loss(student_atts, teacher_atts)

class Embedding_mse(nn.Module):
    def __init__(self, student_size, teacher_size):
        super(Embedding_mse, self).__init__()
        #self.fit_dense = nn.Linear(student_size, teacher_size)
    def forward(self, s_embed, t_embed):
        bsz, num_layer, max_len, tdim = t_embed.size()
        bsz, num_layer, max_len, sdim = s_embed.size()
        t_embed = t_embed.view(-1, max_len, tdim)
        s_embed = s_embed.view(-1, max_len, sdim)
        #s_embed = self.fit_dense(s_embed)
        return F.mse_loss(s_embed, t_embed, reduction="mean")
