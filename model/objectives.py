import torch
import torch.nn as nn
import torch.nn.functional as F
# import build as get_triplets
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.distributed as dist
# import .utils
# from . import utils
# from . import utils
# from utils import all_gather_batch_with_grad

# def cos_distance(source, target):
#     cos_sim = F.cosine_similarity(source.unsqueeze(1), target, dim=-1)
#     distances = torch.clamp(1 - cos_sim, 0)
#     return distances

# def get_triplet_mask(s_labels, t_labels, opt):
#     flag = (opt.beta - 0.1) * opt.gamma
#     batch_size = s_labels.shape[0]
#     sim_origin = s_labels.mm(t_labels.t())
#     sim = (sim_origin > 0).float()
#     ideal_list = torch.sort(sim_origin, dim=1, descending=True)[0]
#     ph = torch.arange(0., batch_size) + 2
#     ph = ph.repeat(1, batch_size).reshape(batch_size, batch_size)
#     th = torch.log2(ph).to(opt.device)
#     Z = (((2 ** ideal_list - 1) / th).sum(axis=1)).reshape(-1, 1)
#     sim_origin = 2 ** sim_origin - 1
#     sim_origin = sim_origin / Z

#     i_equal_j = sim.unsqueeze(2)
#     i_equal_k = sim.unsqueeze(1)
#     sim_pos = sim_origin.unsqueeze(2)
#     sim_neg = sim_origin.unsqueeze(1)
#     weight = (sim_pos - sim_neg) * (flag + 0.1)
#     mask = i_equal_j * (1 - i_equal_k) * (flag + 0.1)

#     return mask, weight

# class TripletLoss(nn.Module):
#     def __init__(self, opt, reduction='mean'):
#         super(TripletLoss, self).__init__()
#         self.reduction = reduction
#         self.opt = opt

#     # tri_i2t = tri_loss(h_i, labels, target=h_t, margin=opt.margin)
#     def forward(self, source, s_labels, target=None, t_labels=None, margin=0):
#         if target is None:
#             target = source
#         if t_labels is None:
#             t_labels = s_labels

#         pairwise_dist = cos_distance(source, target)

#         # shape (batch_size, batch_size, 1)
#         anchor_positive_dist = pairwise_dist.unsqueeze(2)
#         # shape (batch_size, 1, batch_size)
#         anchor_negative_dist = pairwise_dist.unsqueeze(1)

#         triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

#         # Put to zero the invalid triplets
#         # (where label(a) != label(p) or label(n) == label(a) or a == p)
#         mask, weight = get_triplet_mask(s_labels, t_labels, self.opt)
#         if self.opt.alpha == 10:
#             triplet_loss = 10 * weight * mask * triplet_loss
#         else:
#             triplet_loss = mask * triplet_loss

#         # Remove negative losses (i.e. the easy triplets)
#         triplet_loss = triplet_loss.clamp(0)

#         # Count number of positive triplets (where triplet_loss > 0)
#         valid_triplets = triplet_loss.gt(1e-16).float()
#         num_positive_triplets = valid_triplets.sum()

#         if self.reduction == 'mean':
#             triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)
#         elif self.reduction == 'sum':
#             triplet_loss = triplet_loss.sum()

#         return triplet_loss

# class EntLoss2(nn.Module):
#     """Triplet loss with hard positive/negative mining.
    
#     Reference:
#     Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
#     Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    
#     Args:
#     - margin (float): margin for triplet.
#     """
    
#     # def __init__(self, batch_size, margin=0.3):
#     #     super(EntLoss, self).__init__()
#     #     self.margin = margin
#     #     self.ranking_loss = nn.MarginRankingLoss(margin=margin)
#     #     self.last_local_batch_size = None

#     def forward(self, i_feats, t_feats, labels):
#         q_a=i_feats
#         q_b=t_feats
#         q_a = F.normalize(q_a, dim=-1, p=2)
#         q_b = F.normalize(q_b, dim=-1, p=2)

#         local_batch_size = q_a.size(0)

#         k_a, k_b = utils.all_gather_batch_with_grad([q_a, q_b])

#         if local_batch_size != None:
#             labels = local_batch_size * utils.get_rank() + torch.arange(
#                 local_batch_size, device=q_a.device
#             )
#             total_batch_size = local_batch_size * utils.get_world_size()
#             self.masks = F.one_hot(labels, total_batch_size) * 1e9
#             # self.last_local_batch_size = local_batch_size

#         logits_aa = torch.matmul(q_a, k_a.transpose(0, 1)) / 0.001
#         logits_aa = logits_aa - self.masks
#         logits_bb = torch.matmul(q_b, k_b.transpose(0, 1)) / 0.001
#         logits_bb = logits_bb - self.masks
#         logits_ab = torch.matmul(q_a, k_b.transpose(0, 1)) / 0.001
#         logits_ba = torch.matmul(q_b, k_a.transpose(0, 1)) / 0.001

#         loss_a = F.cross_entropy(torch.cat([logits_ab, logits_aa], dim=1), labels)
#         loss_b = F.cross_entropy(torch.cat([logits_ba, logits_bb], dim=1), labels)
#         loss = (loss_a + loss_b) / 2
#         return loss

# def SupMMConLoss(feature_a, feature_b, labels, temperature=0.01):
#         # compute the mask matrix
#         labels = labels.contiguous().view(-1, 1)
#         # mask = torch.eq(labels, labels.T).float() - torch.eye(feature_a.shape[0], feature_a.shape[0])
#         mask = torch.eq(labels, labels.T).float()

#         # compute logits
#         logits = torch.div(torch.matmul(feature_a, feature_b.T), temperature)
#         logits_max, _ = torch.max(logits, dim=1, keepdim=True)
#         logits = logits - logits_max.detach()

#         exp_logits = torch.exp(logits.to(mask.device)) * mask
#         log_prob = logits.to(mask.device) - torch.log(exp_logits.sum(1, keepdim=True))

#         mean_log_pos = -(mask * log_prob).sum(1) / mask.sum(1)

#         return mean_log_pos.mean()


# def UniSMMConLoss( feature_a, feature_b, predict_a, predict_b, labels, temperature=0.07):
#         feature_a_ = feature_a.detach()
#         feature_b_ = feature_b.detach()

#         a_pre = predict_a.eq(labels)  # a True or not
#         a_pre_ = ~a_pre
#         b_pre = predict_b.eq(labels)  # b True or not
#         b_pre_ = ~b_pre

#         a_b_pre = torch.gt(a_pre | b_pre, 0)  # For mask ((P: TT, nP: TF & FT)=T, (N: FF)=F)
#         a_b_pre_ = torch.gt(a_pre & b_pre, 0) # For computing nP, ((P: TT)=T, (nP: TF & FT, N: FF)=F)

#         a_ = a_pre_ | a_b_pre_  # For locating nP not gradient of a
#         b_ = b_pre_ | a_b_pre_  # For locating nP not gradient of b

#         if True not in a_b_pre:
#             a_b_pre = ~a_b_pre
#             a_ = ~a_
#             b_ = ~b_
#         mask = a_b_pre.float()
# #
#         feature_a_f = [feature_a[i].clone() for i in range(feature_a.shape[0])]
#         for i in range(feature_a.shape[0]):
#             if not a_[i]:
#                 feature_a_f[i] = feature_a_[i].clone()
#         feature_a_f = torch.stack(feature_a_f)

#         feature_b_f = [feature_b[i].clone() for i in range(feature_b.shape[0])] # feature_b  # [[0,1]])
#         for i in range(feature_b.shape[0]):
#             if not b_[i]:
#                 feature_b_f[i] = feature_b_[i].clone()
#         feature_b_f = torch.stack(feature_b_f)

#         # compute logits
#         logits = torch.div(torch.matmul(feature_a_f, feature_b_f.T), temperature)
#         logits_max, _ = torch.max(logits, dim=1, keepdim=True)

#         # compute log_prob
#         exp_logits = torch.exp(logits-logits_max.detach())[0]
#         mean_log_pos = - torch.log(((mask * exp_logits).sum() / exp_logits.sum()) / mask.sum())# + 1e-6

#         return mean_log_pos

# def UnSupMMConLoss(feature_a, feature_b, temperature=0.01):

#         # compute the mask matrix
#         mask = torch.eye(feature_a.shape[0], dtype=torch.float32).to(feature_a.device)

#         # compute logits
#         logits = torch.div(torch.matmul(feature_a, feature_b.T), temperature)
#         logits_max, _ = torch.max(logits, dim=1, keepdim=True)
#         logits = logits - logits_max.detach()

#         exp_logits = torch.exp(logits) * mask
#         log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

#         mean_log_pos = -(mask * log_prob).sum(1) / mask.sum(1)
#         mean_log_pos = mean_log_pos.mean()
#         print(mean_log_pos)

#         return mean_log_pos

class EntLoss(nn.Module):
    def __init__(self, args):
        super(EntLoss, self).__init__()
        # self.lam1 = -0.1
        self.lam1 = -0.8
        # self.lam1 = -0.8
        self.lam2 = 0.1
        # self.lam2 = 1.0
        self.pqueue = None
        self.args = args
        # self.args.tau = 1.0
    
    def forward(self, feat1, feat2):
        probs1 = torch.nn.functional.softmax(feat1, dim=-1)
        probs2 = torch.nn.functional.softmax(feat2, dim=-1)

        probs1 = torch.mul(probs1,F.softmax(probs1, dim=1))
        probs2 = torch.mul(probs2, F.softmax(probs2, dim=1))

        loss = dict()
        loss['kl'] = 0.5*(KL(probs1, probs2, self.args) + KL(probs2, probs1, self.args))
        # loss['kl'] = 0.5 * (KL(F.softmax(probs1, dim=1), F.softmax(probs2, dim=1), self.args) + KL(F.softmax(probs1, dim=1), F.softmax(probs2, dim=1), self.args))

        sharpened_probs1 = torch.nn.functional.softmax(feat1/1.0, dim=-1)
        sharpened_probs2 = torch.nn.functional.softmax(feat2/1.0, dim=-1)
        sharpened_probs1 = torch.mul(sharpened_probs1,F.softmax(sharpened_probs1, dim=1))
        sharpened_probs2 = torch.mul(sharpened_probs2, F.softmax(sharpened_probs2, dim=1))
        loss['eh'] = 0.5*(EH(sharpened_probs1, self.args) + EH(sharpened_probs2, self.args))
        # loss['eh'] = 0.5 * (EH(F.softmax(sharpened_probs1, dim=1), self.args) + EH(F.softmax(sharpened_probs2, dim=1), self.args))

        # whether use historical data
        loss['he'] = 0.5*(HE(sharpened_probs1, self.args) + HE(sharpened_probs2, self.args))

        loss['final'] = loss['kl'] + ((1+self.lam1)*loss['eh'] - self.lam2*loss['he'])
        # print(f"KL Loss: {loss['kl'].item()}, EH Loss: {loss['eh'].item()}, HE Loss: {loss['he'].item()}")

        # print(loss['final'])
        # print("xpool loss", loss['final'])
        return loss['final']

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def KL(probs1, probs2, args):
    kl = (probs1 * (probs1 + 1e-8).log() - probs1 * (probs2 + 1e-6).log()).sum(dim=1)
    kl = kl.mean()
    # torch.distributed.all_reduce(kl)
    return kl

def CE(probs1, probs2, args):
    ce = - (probs1 * (probs2 + 1e-8).log()).sum(dim=1)
    ce = ce.mean()
    # torch.distributed.all_reduce(ce)
    return ce

def HE(probs, args): 
    mean = probs.mean(dim=0)
    # torch.distributed.all_reduce(mean)
    ent  = - (mean * (mean + get_world_size() * 1e-6).log()).sum()
    return ent

def EH(probs, args):
    ent = - (probs * (probs + 1e-8).log()).sum(dim=1)
    mean = ent.mean()
    # torch.distributed.all_reduce(mean)
    return mean



"""
    Used for self-labeling, the code is from SCAN: Learning to classify images without lables
    https://github.com/wvangansbeke/Unsupervised-Classification/blob/master/losses/losses.py
"""
# class MaskedCrossEntropyLoss(nn.Module):
#     def __init__(self):
#         super(MaskedCrossEntropyLoss, self).__init__()

#     def forward(self, input, target, mask, weight, reduction='mean'):
#         if not (mask != 0).any():
#             return 0 * input.sum()
#             raise ValueError('Mask in MaskedCrossEntropyLoss is all zeros.')
#         target = torch.masked_select(target, mask)
#         b, c = input.size()
#         n = target.size(0)
#         input = torch.masked_select(input, mask.view(b, 1)).view(n, c)
#         return torch.nn.functional.cross_entropy(input, target, weight = weight, reduction = reduction)

# """
#     Used for self-labeling, the code is from SCAN: Learning to classify images without lables
#     https://github.com/wvangansbeke/Unsupervised-Classification/blob/master/losses/losses.py
# """
# class ConfidenceBasedCE(nn.Module):
#     def __init__(self, threshold, apply_class_balancing):
#         super(ConfidenceBasedCE, self).__init__()
#         self.loss = MaskedCrossEntropyLoss()
#         self.softmax = nn.Softmax(dim = 1)
#         self.threshold = threshold
#         self.apply_class_balancing = apply_class_balancing

#     def forward(self, anchors_weak, anchors_strong):
#         """
#         Loss function during self-labeling
#         input: logits for original samples and for its strong augmentations
#         output: cross entropy
#         """
#         # Retrieve target and mask based on weakly augmentated anchors
#         weak_anchors_prob = self.softmax(anchors_weak)
#         max_prob, target = torch.max(weak_anchors_prob, dim = 1)
#         mask = max_prob > self.threshold
#         b, c = weak_anchors_prob.size()
#         target_masked = torch.masked_select(target, mask.squeeze())
#         n = target_masked.size(0)

#         # Inputs are strongly augmented anchors
#         input_ = anchors_strong

#         # Class balancing weights
#         if self.apply_class_balancing:
#             idx, counts = torch.unique(target_masked, return_counts = True)
#             freq = 1/(counts.float()/n)
#             weight = torch.ones(c).cuda()
#             weight[idx] = freq

#         else:
#             weight = None

#         # Loss
#         loss = self.loss(input_, target, mask, weight = weight, reduction='mean')

#         return loss
class TripletLoss(nn.Module):
    """
    Compute triplet loss
    """

    def __init__(self, margin=0, max_violation=False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.max_violation = max_violation
    def forward(self, scores):
        # Find the maximum score for each image-text pair
        max_score_per_image = scores.max(dim=1, keepdim=True)[0]  # (64, 1)
        max_score_per_text = scores.max(dim=0, keepdim=True)[0]  # (1, 16)

        # Compute margin-based losses
        cost_s = (self.margin + scores - max_score_per_image).clamp(min=0)
        cost_im = (self.margin + scores - max_score_per_text).clamp(min=0)

        # Sum up the losses
        return cost_s.sum() + cost_im.sum()
    # def forward(self, scores):
    #     # compute image-sentence score matrix
    #     print(scores.shape)
    #     scores = scores.mean(dim=1).view(-1, 1) 
    #     diagonal = scores.diag().view(scores.size(0), 1)
    #     print(diagonal.shape)
    #     d1 = diagonal.expand_as(scores)
    #     d2 = diagonal.t().expand_as(scores)

    #     # compare every diagonal score to scores in its column
    #     # caption retrieval
    #     cost_s = (self.margin + scores - d1).clamp(min=0)
    #     # compare every diagonal score to scores in its row
    #     # image retrieval
    #     cost_im = (self.margin + scores - d2).clamp(min=0)

    #     # clear diagonals
    #     mask = torch.eye(scores.size(0)) > .5
    #     I = Variable(mask)
    #     if torch.cuda.is_available():
    #         I = I.cuda()
    #     cost_s = cost_s.masked_fill_(I, 0)
    #     cost_im = cost_im.masked_fill_(I, 0)

    #     # keep the maximum violating negative for each query
    #     if self.max_violation:
    #         cost_s = cost_s.max(1)[0]
    #         cost_im = cost_im.max(0)[0]

    #     return cost_s.sum() + cost_im.sum()
    
def compute_cra(l, m, T=1, neg_num=None):
    '''Computes the noise contrastive estimation-based loss
    Args:
        l: [B,dim,n_l],keys
        m: [B,dim,n_g],query
        neg_num: the number of negatives from other pair
        neg_mask: if
    Returns:
        torch.Tensor: Loss.
    '''
    N, units, n_locals = l.size()
    _, _ , n_multis = m.size()

    # First we make the input tensors the right shape.
    l_p = l.permute(0, 2, 1) # [bt,n_loclas,dim]
    m_p = m.permute(0, 2, 1) # [bt,n_multis,dim]

    l_n = l_p.reshape(-1, units) # [bt*n_locals,dim]
    m_n = m_p.reshape(-1, units) # [bt*1,dim]

    # Inner product for positive samples. Outer product for negative. We need to do it this way
    # for the multiclass loss. For the outer product, we want a N x N x n_local x n_multi tensor.
    u_p = torch.matmul(l_p, m).unsqueeze(2) # [B,n_locals,dim] * [B,dim,1] ->[bt,n_locals,1,1]
    u_n = torch.mm(m_n, l_n.t()) # [B*1,dim] * [dim,B*36] ->[bt*1,b*n_locals]

    # add apply tao
    u_p = u_p / T
    u_n = u_n / T

    u_n = u_n.reshape(N, n_multis, N, n_locals).permute(0, 2, 3, 1) # N,N,n_locals,n_multis

    # We need to mask the diagonal part of the negative tensor.
    mask = torch.eye(N)[:, :, None, None].to(l.device) # [bt,bt,1,1]
    n_mask = 1 - mask
    n_mask = n_mask.expand(-1,-1,n_locals,n_multis)

    # Masking is done by shifting the diagonal before exp.
    u_n = (n_mask * u_n) - (10. * (1 - n_mask))  # mask out "self" examples
    u_n = u_n.reshape(N, N * n_locals, n_multis).unsqueeze(dim=1).expand(-1, n_locals, -1, -1)

    # Since this is multiclass, we concat the positive along the class dimension before performing log softmax.
    pred_lgt = torch.cat([u_p, u_n], dim=2)

    # hard negatives
    if neg_num:
        sort_u_n,_ = torch.topk(u_n,dim=2,k=neg_num)
        pred_lgt = torch.cat([u_p,sort_u_n],dim=2)

    pred_log = F.log_softmax(pred_lgt, dim=2) # [bt,n_locals,neg_num+1,n_multis]


    # The positive score is the first element of the log softmax.
    loss = -pred_log[:, :, 0].mean()

    return loss


def l2norm(x):
        """L2-normalize columns of x"""
        norm = torch.pow(x, 2).sum(dim=-1, keepdim=True).sqrt()
        return torch.div(x, norm)

class Global_Loss(nn.Module):
    def __init__(self):
        super(Global_Loss, self).__init__()
        self.CMPM = True
        self.CMPC = True
        # self.epsilon = 1e-10
        self.epsilon = 1e-8
        self.num_classes = 11003
        self.W = Parameter(torch.randn(512, 11003))
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.W.data, gain=1)

    def diversity_loss(self, x):
        x = l2norm(x)  # Columns of x MUST be l2-normalized
        gram_x = x.bmm(x.transpose(1, 2))
        I = torch.autograd.Variable(
            (torch.eye(x.size(1)) > 0.5).repeat(gram_x.size(0), 1, 1)
        )
        if torch.cuda.is_available():
            I = I.cuda()
        gram_x.masked_fill_(I, 0.0)
        loss = torch.stack([torch.norm(g, p=2) for g in gram_x]) / (x.size(1) ** 2)
        return loss.mean()

    def compute_cmpc_loss(self, image_embeddings, text_embeddings, labels):
        """
        Cross-Modal Projection Classfication loss(CMPC)
        :param image_embeddings: Tensor with dtype torch.float32
        :param text_embeddings: Tensor with dtype torch.float32
        :param labels: Tensor with dtype torch.int32
        :return:
        """
        
        criterion = nn.CrossEntropyLoss(reduction="mean")
        self.W_norm = self.W / self.W.norm(dim=0)
        # labels_onehot = one_hot_coding(labels, self.num_classes).float()
        image_norm = image_embeddings / (image_embeddings.norm(dim=1, keepdim=True)+ self.epsilon)
        text_norm = text_embeddings / (text_embeddings.norm(dim=1, keepdim=True)+ self.epsilon)
        # print("image_norm:", image_norm.min(), image_norm.max())
        # print("text_norm:", text_norm.min(), text_norm.max())

        image_proj_text = (
            torch.sum(image_embeddings * text_norm, dim=1, keepdim=True) * text_norm
        )
        text_proj_image = (
            torch.sum(text_embeddings * image_norm, dim=1, keepdim=True) * image_norm
        )

        image_logits = torch.matmul(image_proj_text.float(), self.W_norm)
        text_logits = torch.matmul(text_proj_image.float(), self.W_norm)
        # print("image_proj_text:", image_proj_text.min(), image_proj_text.max())
        # print("text_proj_image:", text_proj_image.min(), text_proj_image.max())

        # labels_one_hot = one_hot_coding(labels, num_classes)
        """
        ipt_loss = criterion(input=image_logits, target=labels)
        tpi_loss = criterion(input=text_logits, target=labels)
        cmpc_loss = ipt_loss + tpi_loss
        """
        cmpc_loss = criterion(image_logits, labels) + criterion(text_logits, labels)

        image_pred = torch.argmax(image_logits, dim=1)
        text_pred = torch.argmax(text_logits, dim=1)

        image_precision = torch.mean((image_pred == labels).float())
        text_precision = torch.mean((text_pred == labels).float())

        return cmpc_loss, image_precision, text_precision

    def compute_cmpm_loss(self, image_embeddings, text_embeddings, labels):
        """
        Cross-Modal Projection Matching Loss(CMPM)
        :param image_embeddings: Tensor with dtype torch.float32
        :param text_embeddings: Tensor with dtype torch.float32
        :param labels: Tensor with dtype torch.int32
        :return:
            i2t_loss: cmpm loss for image projected to text
            t2i_loss: cmpm loss for text projected to image
            pos_avg_sim: average cosine-similarity for positive pairs
            neg_avg_sim: averate cosine-similarity for negative pairs
        """
        # print(i_feats)
        batch_size = image_embeddings.shape[0]
        
        labels_reshape = torch.reshape(labels, (batch_size, 1))
        labels_dist = labels_reshape - labels_reshape.t()
        labels_mask = labels_dist == 0

        # print("image_embeddings shape:", t_feats.shape)

        image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
        text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
        # print('text_norm111:',text_norm.shape)   torch.Size([64, 512])
        # print('image_embeddings:',image_embeddings.shape)   torch.Size([64, 512])
        # print('text_embeddings222:',text_embeddings.norm(dim=1, keepdim=True))
        image_proj_text = torch.matmul(image_embeddings, text_norm.t())
        # print('image_proj_text111:',image_proj_text)
        text_proj_image = torch.matmul(text_embeddings, image_norm.t())

        # normalize the true matching distribution
        labels_mask_norm = labels_mask.float() / labels_mask.float().norm(dim=1)

        i2t_pred = F.softmax(image_proj_text, dim=1)
        # print('i2t_pred:',i2t_pred)
        # i2t_loss = i2t_pred * torch.log((i2t_pred + self.epsilon)/ (labels_mask_norm + self.epsilon))
        i2t_loss = i2t_pred * (
            F.log_softmax(image_proj_text, dim=1)
            - torch.log(labels_mask_norm + self.epsilon)
        )
        

        t2i_pred = F.softmax(text_proj_image, dim=1)
        # t2i_loss = t2i_pred * torch.log((t2i_pred + self.epsilon)/ (labels_mask_norm + self.epsilon))
        t2i_loss = t2i_pred * (
            F.log_softmax(text_proj_image, dim=1)
            - torch.log(labels_mask_norm + self.epsilon)
        )
        # print("i2t_loss:", i2t_loss.min(), i2t_loss.max())
        # print("t2i_loss:", t2i_loss.min(), t2i_loss.max())


        cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(
            torch.sum(t2i_loss, dim=1)
        )

        sim_cos = torch.matmul(image_norm, text_norm.t())

        pos_avg_sim = torch.mean(torch.masked_select(sim_cos, labels_mask))
        neg_avg_sim = torch.mean(torch.masked_select(sim_cos, labels_mask == 0))
        # return sim_cos,pos_avg_sim, neg_avg_sim

        return cmpm_loss, pos_avg_sim, neg_avg_sim

    def forward(
        self, image_embeddings, text_embeddings, img_f, text_f, labels
    ):
        # self.diversity_loss2(img_f)
        cmpm_loss = 0.0
        cmpc_loss = 0.0
        # print(type(image_embeddings))
        # print(image_embeddings)

        # print(image_embeddings.min(), image_embeddings.max())
        # image_precision = 0.0
        # text_precision = 0.0
        # neg_avg_sim = 0.0
        # pos_avg_sim = 0.0

        # if self.CMPM:
        #     cmpm_loss = 0
        #     for i in range(len(image_embeddings)):
        #         # print("1",image_embeddings[i].shape)
        #         cmpm_loss1, pos_avg_sim, neg_avg_sim = self.compute_cmpm_loss(
        #             image_embeddings[i], text_embeddings[i], labels
        #         )
        #         if i == 0 or i == 1:
        #             cmpm_loss += cmpm_loss1
        #         else:
        #             cmpm_loss += cmpm_loss1*0.1   # *1.0/(len(image_embeddings)-1)

        # if self.CMPC:
        #     cmpc_loss = 0
        #     for i in range(len(image_embeddings)):
        #         cmpc_loss1, image_precision, text_precision = self.compute_cmpc_loss(
        #             image_embeddings[i], text_embeddings[i], labels
        #         )
                # if i==0:
                # cmpc_loss += cmpc_loss1
                # else:
                #     cmpc_loss += cmpc_loss1*1.0/(len(image_embeddings)-1)

        # print(cmpm_loss)
        # text_f = torch.stack(text_embeddings, dim=1)
        # img_f = torch.stack(image_embeddings, dim=1)
        if self.CMPM:
            cmpm_loss = 0
            # print('image_embeddings1234:',image_embeddings)
            # print('text_embeddings1234:',text_embeddings)
            for i in range(len(image_embeddings)):
                cmpm_loss1, pos_avg_sim, neg_avg_sim = self.compute_cmpm_loss(
                    image_embeddings[i], text_embeddings[i], labels
                )
                # print('4444',cmpm_loss1)
                # 对单个批次的损失进行归一化处理
                # cmpm_loss1 = cmpm_loss1 / (cmpm_loss1.detach().item() + self.epsilon)

                if i == 0 or i == 1:
                    cmpm_loss += cmpm_loss1
                else:
                    cmpm_loss += cmpm_loss1*0.1#    *1.0/(len(image_embeddings)-1)#*0.1

        if self.CMPC:
            cmpc_loss = 0
            for i in range(len(image_embeddings)):
                cmpc_loss1, image_precision, text_precision = self.compute_cmpc_loss(
                    image_embeddings[i], text_embeddings[i], labels
                )
                # 对单个批次的损失进行归一化处理
                # cmpc_loss1 = cmpc_loss1 / (cmpc_loss1.detach().item() + self.epsilon)
                # print('3333',cmpc_loss1)
                cmpc_loss += cmpc_loss1
                # if i==0:
                #     cmpc_loss += cmpc_loss1
                # else:
                #     cmpc_loss += cmpc_loss1*1.0/(len(image_embeddings)-1)
        # print("1111",cmpc_loss)
        # print("222",cmpm_loss)
        loss = cmpm_loss + cmpc_loss
        # cmpm_loss1, pos_avg_sim, neg_avg_sim = self.compute_cmpm_loss(
        #             image_embeddings, text_embeddings, labels
        #         )
        # cmpc_loss1, image_precision, text_precision = self.compute_cmpc_loss(
        #             image_embeddings, text_embeddings, labels
                # )
        # print("1111",cmpc_loss1)
        # print("222",cmpm_loss1)
        # loss = cmpm_loss1 + cmpc_loss1
        loss += 0.2 * (
            self.diversity_loss(img_f[:,1:]) + self.diversity_loss(text_f[:,1:])
        )
        return loss


def compute_js_sdm(image_fetures, text_fetures, pid, logit_scale, image_id=None, factor=0.3, epsilon=1e-8):
    """
    Similarity Distribution Matching
    """
    batch_size = image_fetures.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    if image_id != None:
        # print("Mix PID and ImageID to create soft label.")
        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        labels = (labels - image_id_mask) * factor + image_id_mask
        # labels = (labels + image_id_mask) / 2

    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

    t2i_cosine_theta = text_norm @ image_norm.t()
    i2t_cosine_theta = t2i_cosine_theta.t()

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta

    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1)
    labels_distribute = labels_distribute + epsilon

    i2t_pred = F.softmax(image_proj_text, dim=1)
    js_distribute1 = (i2t_pred + labels_distribute) / 2
    i2t_loss = (0.5 * i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(js_distribute1)) + 0.5 * labels_distribute * (torch.log(labels_distribute) - torch.log(js_distribute1)))/0.04
    #i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))


    t2i_pred = F.softmax(text_proj_image, dim=1)
    js_distribute2 = (t2i_pred + labels_distribute) / 2
    t2i_loss = (0.5 * t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(js_distribute2)) + 0.5 * labels_distribute * (torch.log(labels_distribute) - torch.log(js_distribute2)))/0.04
    #t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return loss

def compute_mlm_6(scores, labels):
    ce = nn.CrossEntropyLoss()
    return ce(scores, labels)


def compute_cmpc_loss(image_embeddings, text_embeddings, labels):
        
        """
        Cross-Modal Projection Classfication loss(CMPC)
        :param image_embeddings: Tensor with dtype torch.float32
        :param text_embeddings: Tensor with dtype torch.float32
        :param labels: Tensor with dtype torch.int32
        :return:
        """
        # self.W = Parameter(torch.randn(768, 11003))
        criterion = nn.CrossEntropyLoss(reduction="mean")
        # device = image_embeddings.device
        # self.W = self.W.to(device)
        # device = image_embeddings.device
        # self.W_norm = Parameter(torch.randn(768, 11003)) / Parameter(torch.randn(768, 11003)).norm(dim=0)
        # labels_onehot = one_hot_coding(labels, self.num_classes).float()
        image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
        text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)

        image_proj_text = (
            torch.sum(image_embeddings * text_norm, dim=1, keepdim=True) * text_norm
        )
        text_proj_image = (
            torch.sum(text_embeddings * image_norm, dim=1, keepdim=True) * image_norm
        )
        # print(image_proj_text.device)
        # device = image_embeddings.device
        # image_proj_text = image_proj_text.to(device)
        # text_proj_image = text_proj_image.to(device)
        # print(image_proj_text.device)
        # print(image_embeddings.device)
        # device = image_proj_text.device  # 获取 image_proj_text 所在的设备
        # param = Parameter(torch.randn(768, 11003, device=device).to(torch.float16))  # 将参数放在同一设备上
        # image_logits = torch.matmul(image_proj_text, param / param.norm(dim=0))device = image_proj_text.device  # 获取image_proj_text所在的设备
        # device = image_proj_text.device
        # rand_tensor = Parameter(torch.randn(768, 11003, device=device))  # 在相同设备上创建随机张量
        # image_logits = torch.matmul(image_proj_text.float(), rand_tensor.float() / rand_tensor.float().norm(dim=0))

        # print(image_proj_text.device)
        # print(image_logits.device)




   
        image_logits = torch.matmul(image_proj_text, Parameter(torch.randn(768, 11003)) / Parameter(torch.randn(768, 11003)).norm(dim=0))
        
        text_logits = torch.matmul(text_proj_image, Parameter(torch.randn(768, 11003))/ Parameter(torch.randn(768, 11003)).norm(dim=0))

        # labels_one_hot = one_hot_coding(labels, num_classes)
        """
        ipt_loss = criterion(input=image_logits, target=labels)
        tpi_loss = criterion(input=text_logits, target=labels)
        cmpc_loss = ipt_loss + tpi_loss
        """
        cmpc_loss = criterion(image_logits, labels) + criterion(text_logits, labels)

        image_pred = torch.argmax(image_logits, dim=1)
        text_pred = torch.argmax(text_logits, dim=1)

        image_precision = torch.mean((image_pred == labels).float())
        text_precision = torch.mean((text_pred == labels).float())

        return cmpc_loss, image_precision, text_precision

def compute_cmpm_loss(img_global, text_global, labels):
        """
        Cross-Modal Projection Matching Loss(CMPM)
        :param image_embeddings: Tensor with dtype torch.float32
        :param text_embeddings: Tensor with dtype torch.float32
        :param labels: Tensor with dtype torch.int32
        :return:
            i2t_loss: cmpm loss for image projected to text
            t2i_loss: cmpm loss for text projected to image
            pos_avg_sim: average cosine-similarity for positive pairs
            neg_avg_sim: averate cosine-similarity for negative pairs
        """
        # print(labels.shape) 64
        # img_output = (img_global,)
        # text_output = (text_global,)

        batch_size = img_global.shape[0]
        labels_reshape = torch.reshape(labels, (batch_size, 1))
        # batch_size = labels.size(0)  # 64
        # labels_reshape = labels.view(batch_size, 1)

        labels_dist = labels_reshape - labels_reshape.t()
        labels_mask = labels_dist == 0

        image_norm = img_global / img_global.norm(dim=1, keepdim=True)
        text_norm = text_global / text_global.norm(dim=1, keepdim=True)
        image_proj_text = torch.matmul(img_global, text_norm.t())
        text_proj_image = torch.matmul(text_global, image_norm.t())

        # normalize the true matching distribution
        labels_mask_norm = labels_mask.float() / labels_mask.float().norm(dim=1)

        i2t_pred = F.softmax(image_proj_text, dim=1)
        # i2t_loss = i2t_pred * torch.log((i2t_pred + self.epsilon)/ (labels_mask_norm + self.epsilon))
        i2t_loss = i2t_pred * (
            F.log_softmax(image_proj_text, dim=1)
            - torch.log(labels_mask_norm + 1e-8)
        )

        t2i_pred = F.softmax(text_proj_image, dim=1)
        # t2i_loss = t2i_pred * torch.log((t2i_pred + self.epsilon)/ (labels_mask_norm + self.epsilon))
        t2i_loss = t2i_pred * (
            F.log_softmax(text_proj_image, dim=1)
            - torch.log(labels_mask_norm + 1e-8)
        )

        cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(
            torch.sum(t2i_loss, dim=1)
        )

        sim_cos = torch.matmul(image_norm, text_norm.t())
        # print(cmpm_loss)

        # pos_avg_sim = torch.mean(torch.masked_select(sim_cos, labels_mask))
        # neg_avg_sim = torch.mean(torch.masked_select(sim_cos, labels_mask == 0))
       

        # return cmpm_loss
        return sim_cos
# def xpool(image_feats, image_feats_selected_all,text_feats,text_feats_selected_all,image_feats_selected1,text_feats_selected1):

#     G_img_token = image_feats[:, 0, :].unsqueeze(1)
#     L_img_token = image_feats_selected_all
#     B = L_img_token.size(0)
#     G_text_token = text_feats[:, 0, :].unsqueeze(1)
#     L_text_token = text_feats_selected_all
#     print(G_text_token.shape)
#     print(L_text_token.shape)

#     G_img_token_norm = G_img_token / G_img_token.norm(dim=-1, keepdim=True)
#     L_img_token_norm = L_img_token / L_img_token.norm(dim=-1, keepdim=True)
#     G_text_token_norm = G_text_token / G_text_token.norm(dim=-1, keepdim=True)
#     L_text_token_norm = L_text_token / L_text_token.norm(dim=-1, keepdim=True)

#             # image-word sim
#     G_img_token_norm_l = G_img_token_norm.unsqueeze(1).repeat(1, B, 1, 1)
#     L_text_token_norm_r = L_text_token_norm.unsqueeze(0).repeat(B, 1, 1, 1)
#     print(G_img_token_norm_l.shape)

#     sim_iw = torch.matmul(G_img_token_norm_l, L_text_token_norm_r.transpose(-2, -1)) / 0.01
#     weight_iw = F.softmax(sim_iw, dim=-1)
#     sim_iw = torch.mul(sim_iw, weight_iw)
#     sim_iw = torch.sum(sim_iw, dim=-1).squeeze()

#             # piexl-text sim
#     L_img_token_norm_l = L_img_token_norm.unsqueeze(1).repeat(1, B, 1, 1)
#     G_text_token_norm_r = G_text_token_norm.unsqueeze(0).repeat(B, 1, 1, 1)

#     sim_pt = torch.matmul(L_img_token_norm_l, G_text_token_norm_r.transpose(-2, -1)) / 0.01
#     weight_pt = F.softmax(sim_pt, dim=2)
#     sim_pt = torch.mul(sim_pt, weight_pt)
#     sim_pt = torch.sum(sim_pt, dim=2).squeeze()

#     sim_cs = (sim_iw + sim_pt) / 2

#     # Correspondence Discovery
#     L_img_token1 = image_feats_selected1[:, 1:, :]
#     L_text_token1 = text_feats_selected1[:, 1:, :]
#     L_img_token_norm1 = L_img_token1 / L_img_token1.norm(dim=-1, keepdim=True)
#     L_text_token_norm1 = L_text_token1 / L_text_token1.norm(dim=-1, keepdim=True)
#     _b, _n, _c = L_text_token1.shape
#     vidwordSim = torch.bmm(L_img_token_norm1, L_text_token_norm1.permute(0, 2, 1))
#     vidwordSim = self.similarityNorm(vidwordSim)
#             # ---------- posWord -------
#     _, idxWord = vidwordSim.topk(3, dim=2, largest=True, sorted=True)
#     posWord = []
#     for _batch in range(idxWord.shape[0]):
#         posWord.append(L_text_token1[_batch, idxWord[_batch], :])
#     posWord = torch.stack(posWord)
#     posWord = torch.mean(posWord, dim=2)
#     posWord_norm = posWord / posWord.norm(dim=-1, keepdim=True)
#             # ---------- posClip -------
#     _, idxVid = vidwordSim.topk(3, dim=1, largest=True, sorted=True)
#     posClip = []
#     for _batch in range(idxVid.shape[0]):
#         posClip.append(L_img_token1[_batch, idxVid[_batch], :])
#     posClip = torch.stack(posClip)
#     posClip = torch.mean(posClip, dim=1)
#     posClip_norm = posClip / posClip.norm(dim=-1, keepdim=True)

#     # piexl_word sim
#     L_img_token_norm_l1 = L_img_token_norm1.unsqueeze(1).repeat(1, _b, 1, 1)
#     posWord_norm_r = posWord_norm.unsqueeze(0).repeat(_b, 1, 1, 1)
#     posClip_norm_l = posClip_norm.unsqueeze(1).repeat(1, _b, 1, 1)
#     L_text_token_norm_r1 = L_text_token_norm1.unsqueeze(0).repeat(_b, 1, 1, 1)

#     sim_pw0 = torch.matmul(L_img_token_norm_l1, posWord_norm_r.transpose(-2, -1)) / 0.01
#     sim_pw0 = torch.diagonal(sim_pw0, dim1=-2, dim2=-1)
#     sim_pw0 = torch.mean(sim_pw0, dim=2)

#     sim_pw1 = torch.matmul(posClip_norm_l, L_text_token_norm_r1.transpose(-2, -1)) / 0.01
#     sim_pw1 = torch.diagonal(sim_pw1, dim1=-2, dim2=-1)
#     sim_pw1 = torch.mean(sim_pw1, dim=2)

#     sim_cd = (sim_pw0 + sim_pw1) / 2    
#     return sim_cs+sim_cd


#TODO tgrs1
def compute_sdm(image_fetures, text_fetures, pid, logit_scale, image_id=None, factor=0.3, epsilon=1e-8):
    """
    Similarity Distribution Matching
    """
    batch_size = image_fetures.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    if image_id != None:
        # print("Mix PID and ImageID to create soft label.")
        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        labels = (labels - image_id_mask) * factor + image_id_mask
        # labels = (labels + image_id_mask) / 2

    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

    t2i_cosine_theta = text_norm @ image_norm.t()
    i2t_cosine_theta = t2i_cosine_theta.t()

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta

    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))
    # print("cmpm loss", loss)

    return loss


# def compute_mlm(scores, labels):
#     ce = nn.CrossEntropyLoss(ignore_index=0)
#     return ce(scores, labels)

# def compute_sdm(i_feats, t_feats, image_id=None, factor=0.1, epsilon=1e-8, temperature=0.2):
#     """
#     Similarity Distribution Matching //scanloss
#     """
#     # TODO dee+convd layer // +softmax 
#     # i_fetures = F.softmax(i_fetures,dim=1)
#     # t_fetures = F.softmax(t_fetures,dim=1)
#     i_feats = torch.mul(i_feats,F.softmax(i_feats, dim=1))
#     t_feats = torch.mul(t_feats, F.softmax(t_feats, dim=1))
#     z1 = F.normalize(i_feats,dim=1)
#     z2 = F.normalize(t_feats,dim=1)
#     # z1 = F.normalize(F.softmax(i_feats,dim=1))
#     # z2 = F.normalize(F.softmax(t_feats,dim=1))
#     N, Z = z1.shape
#         # N, Z = z1.size(0), z1.size(1)
 
#     device = z1.device 
#     # print("z1",z1.shape)  z1 torch.Size([64, 77, 512])
#     # print("z2",z2.shape)  z2 torch.Size([64, 64, 512])
#     # 计算需要填充的零的数量
#     # padding_size = z1.shape[1] - z2.shape[1]

#     # # 创建零张量并将其移动到设备上
#     # zeros_tensor = torch.zeros(z2.shape[0], z2.shape[1], padding_size, 512).to(z2.device)

#     # # 在第三个维度上连接 z2 和零张量
#     # z2 = F.pad(z2, (0, 0, 0, padding_size))
# #     print("z1 shape:", z1.shape)
# #     print("z2 shape:", z2.shape)
# #     z2 = z2[:, :128]
#     representations = torch.cat([z1, z2], dim=0)
#     #ccccccc
#     # representations = torch.mul(representations,F.softmax(representations,dim=1))
#     similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=-1)
# #     print("similarity_matrix shape11:", similarity_matrix.shape)

# #     l_pos = torch.diag(similarity_matrix, N)
# #     r_pos = torch.diag(similarity_matrix, -N)
#     # l_pos = torch.diag(similarity_matrix, diagonal=N)
#     # r_pos = torch.diag(similarity_matrix, diagonal=-N)
   

#     # 拼接 l_pos 和 r_pos，然后修改视图
#     # positives = torch.cat([l_pos, r_pos], dim=0)
#     l_pos = torch.diag(similarity_matrix, N)
#     r_pos = torch.diag(similarity_matrix, -N)
#     positives = torch.cat([l_pos, r_pos]).view(2 * N, 1)

#     # positives = positives.view(-1, 1)
# #     print("l_pos shape:", l_pos.shape)
# #     print("r_pos shape:", r_pos.shape)
# #     print("positives shape:", positives.shape)



#     # positives = torch.cat([l_pos, r_pos], dim=0).view(-1, 1)

#     # positives = torch.cat([l_pos, r_pos]).view(-1, 1)
#     # print("N:", N)
#     # print("similarity_matrix shape:", similarity_matrix.shape)

#     # diag = torch.eye(2 * N, dtype=torch.bool, device=device)
#     # diag[N:,:N] = diag[:N,N:] = diag[:N,:N]
#     # print("diag",diag.shape)
#     # print("N",N)
# #     print("l_pos shape:", l_pos.shape)
# #     print("r_pos shape:", r_pos.shape)
# #     if l_pos.shape == (15,) and r_pos.shape == (15,):
# #         l_pos = l_pos.repeat(3)  # 3是45除以15得到的结果
# #         r_pos = r_pos.repeat(3)
# #     print("l_pos shape2:", l_pos.shape)
# #     print("r_pos shape2:", r_pos.shape)

# #     positives = torch.cat([l_pos, r_pos]).view(2 * N, 1)
# #     padding_size = max(0, 64 - l_pos.shape[0])  # Adjust 64 based on your batch size

# #     l_pos = F.pad(l_pos, (0, padding_size), value=5)  # Choose an appropriate value for padding
# #     r_pos = F.pad(r_pos, (0, padding_size), value=5)

#         # Now both l_pos and r_pos should have the same shape
# #     positives = torch.cat([l_pos, r_pos], dim=0).view(2 * N, 1)

#     diag = torch.eye(2*N, dtype=torch.bool, device=device)
#     diag[N:,:N] = diag[:N,N:] = diag[:N,:N]

#     negatives = similarity_matrix[~diag].view(2*N, -1)
# #     print("positives shape:", positives.shape)
# #     print("negatives shape:", negatives.shape)

#     # negatives = negatives.view(-1, 1)
#     # 调整negatives的维度以匹配positives的batch size
#     # negatives = negatives[:positives.size(0), :]

#     # 将positives和negatives拼接在一起
#     # logits = torch.cat([positives, negatives], dim=0)
#     # 将positives和negatives拼接在一起
#     logits = torch.cat([positives, negatives], dim=1)
#     # print("positives shape:", positives.shape)
#     # print("negatives shape:", negatives.shape)

#     # logits = torch.cat([positives, negatives], dim=1)
#     logits /= temperature

#     labels = torch.zeros(2*N, device=device, dtype=torch.int64)
#     # labels = torch.zeros(logits.size(0), device=device, dtype=torch.int64)
#     # logits = torch.mul(logits,F.softmax(logits,dim=1))
#     # labels = torch.mul(labels,F.softmax(labels,dim=1))



#     loss = F.cross_entropy(logits, labels, reduction='sum')
#     return loss / (2 * N)

    # batch_size = image_fetures.shape[0]
    # pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    # pid_dist = pid - pid.t()
    # labels = (pid_dist == 0).float()

    # if image_id != None:
    #     # print("Mix PID and ImageID to create soft label.")
    #     image_id = image_id.reshape((-1, 1))
    #     image_id_dist = image_id - image_id.t()
    #     image_id_mask = (image_id_dist == 0).float()
    #     labels = (labels - image_id_mask) * factor + image_id_mask
    #     # labels = (labels + image_id_mask) / 2

    # image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    # text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

    # t2i_cosine_theta = text_norm @ image_norm.t()
    # i2t_cosine_theta = t2i_cosine_theta.t()

    # text_proj_image = logit_scale * t2i_cosine_theta
    # image_proj_text = logit_scale * i2t_cosine_theta

    # # normalize the true matching distribution
    # labels_distribute = labels / labels.sum(dim=1)

    # i2t_pred = F.softmax(image_proj_text, dim=1)
    # i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    # t2i_pred = F.softmax(text_proj_image, dim=1)
    # t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    # loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    # return loss


# def compute_mlm(image_feats,scores,labels):
#     # print("scores",scores.shape)
#     # print("logit",labels.shape)
#     batch_size = image_feats.shape[0]
#     # pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
#     # pid_dist = pid - pid.t()
#     # labels = (pid_dist == 0).float()
#     logprobs = torch.nn.functional.log_softmax(scores, dim=-1)
#     labels = torch.arange(start=0, end=batch_size, dtype=torch.int64)
#     # print("logit_scale shape:", logit_scale)
#     labels = labels.to('cuda:0')

#     # if len(logit_scale.shape) == 0:
#     #     logit_scale = logit_scale.unsqueeze(0)

#     # nll_loss = -logprobs.gather(dim=-1, index=logit_scale.unsqueeze(1))
#     # nll_loss = -logprobs.gather(dim=-1, index=logit_scale.unsqueeze(1).to(torch.int64))
#     # nll_loss = -logprobs.gather(dim=-1, index=logit_scale.unsqueeze(0).long())
#     # nll_loss = -logprobs.gather(dim=-1, index=logit_scale.unsqueeze(0).long())
#     # print("logit_scale:", logit_scale1)
#     nll_loss = -logprobs[..., labels.long()]

#     # nll_loss = nll_loss.squeeze(1)
#     # smooth_loss = -logprobs.mean(dim=-1)
#     smooth_loss = -logprobs.mean(dim=-1, keepdim=True)

#     # print("nll_loss shape:", nll_loss.shape)
#     # print("smooth_loss shape:", smooth_loss.shape)
#     loss = nll_loss

#     # loss = 0.9 * nll_loss + 0.1 * smooth_loss
#     # return loss
#     return loss.mean()








#确保 labels 是一维张量
    # labels = labels.squeeze()

    # 将 labels 转换为整数张量
    # labels = torch.tensor(labels, dtype=torch.long)
    # if labels.shape==(15):
    #      labels = labels[:14]


    # scores = scores[:,:,:,32]
  
    # # print("labels1",labels.shape)
    # # print("scores1",scores.shape)


    
    # labels = labels.view(-1, 2)

    # # 假设 num_classes=11003
    # num_classes = 64

#  # 检查 labels 是否在有效范围内
#     assert (labels >= 0).all() and (labels < num_classes).all(), "Invalid class index in labels."

    # 打印 labels 张量中的值
    # print("Minimum label value:", labels.min().item())
    # print("Maximum label value:", labels.max().item())

    # ce = nn.CrossEntropyLoss()
    # # scores = scores[32,:,:]
    # scores = scores[:32, ...]
    # # print("labels2",labels.shape)
    # # print("scores2",scores.shape)
    # loss = ce (scores, labels)
    # return loss


def compute_itc(image_features, text_features, logit_scale):
    """
    image-text contrastive (ITC) loss, InfoNCE
    """
    # print(image_features.shape)  ([64, 512])
    # print(text_features.shape)  ([64, 512])
    batch_size = image_features.shape[0]
    # batch_size = len(image_features)
    # print("image_features.shape",image_features.shape,image_features.device)
    # print("text_features.shape",text_features.shape)
    labels = torch.arange(start=0, end=batch_size, dtype=torch.int64)
    # print("===== 调试信息 =====")
    # print(f"image_features: shape={image_features.shape}, device={image_features.device}, dtype={image_features.dtype}")
    # print(f"labels: shape={labels.shape}, device={labels.device}, dtype={labels.dtype}")
    # print(f"labels 取值范围: [{labels.min()}, {labels.max()}]")
    # print(f"batch_size (声明): {batch_size}, 实际: {image_features.shape[0]}")
    # print("labels:",labels.min(),labels.max(),batch_size)
    labels = labels.to(image_features.device)
    # print("labels:",labels.shape,labels.device,labels.dtype)

    
    # normalized features
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)
    # print(image_norm.shape)  torch.Size([64, 512])
    # print(text_norm.shape)  torch.Size([64, 512])
    # print(logit_scale.shape)  # torch.Size([64])

    # cosine similarity as logits
    logits_per_image = logit_scale * image_norm @ text_norm.t()
    logits_per_text = logits_per_image.t()

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t =F.cross_entropy(logits_per_text, labels)
    loss = (loss_i +  loss_t)/2

    return loss

def SoftCrossEntropy(inputs, target, reduction='average'):
    log_likelihood = -F.log_softmax(inputs, dim=1)
    batch = inputs.shape[0]
    if reduction == 'average':
        loss = torch.sum(torch.mul(log_likelihood, target)) / batch
    else:
        loss = torch.sum(torch.mul(log_likelihood, target))
    return loss

def compute_soft_id(image_logits, text_logits, label_i, lable_t):

    loss = SoftCrossEntropy(image_logits,label_i)+ SoftCrossEntropy(text_logits,lable_t)
    return loss/2

def compute_id(image_logits, text_logits, labels):
    """
    Instance loss proposed at http://arxiv.org/abs/1711.05535
    """
    criterion = nn.CrossEntropyLoss(reduction="mean")

    loss = criterion(image_logits, labels) + criterion(text_logits, labels)
    
    return loss / 2

#CFine--Cross-Similarity
def compute_cmpm(G_img_token_norm, L_text_token_norm, L_img_token_norm, G_text_token_norm, L_img_token_norm_l1,posWord_norm_r,posClip_norm_l, L_text_token_norm_r1):
# def compute_cmpm(G_img_token_norm, L_text_token_norm, L_img_token_norm, G_text_token_norm):
                # image-word sim

    B = L_text_token_norm.size(0)
    L_text_token_norm = L_text_token_norm.to(torch.float16)
    G_text_token_norm = G_text_token_norm.to(torch.float16)


    G_img_token_norm_l = G_img_token_norm.unsqueeze(1).repeat(1, B, 1, 1)
    L_text_token_norm_r = L_text_token_norm.unsqueeze(0).repeat(B, 1, 1, 1)


#sim_iw全局图像和局部文本---跨模态跨粒度   跨粒度特征细化模块
    sim_iw = torch.matmul(G_img_token_norm_l, L_text_token_norm_r.transpose(-2, -1)) / 0.07
    weight_iw = F.softmax(sim_iw, dim=-1)
    sim_iw = torch.mul(sim_iw, weight_iw)
    sim_iw = torch.sum(sim_iw, dim=-1).squeeze()

            # piexl-text sim
    L_img_token_norm_l = L_img_token_norm.unsqueeze(1).repeat(1, B, 1, 1)
    G_text_token_norm_r = G_text_token_norm.unsqueeze(0).repeat(B, 1, 1, 1)
#sim_pt局部图像和全局文本---跨模态跨粒度  跨粒度特征细化模块
    sim_pt = torch.matmul(L_img_token_norm_l, G_text_token_norm_r.transpose(-2, -1)) / 0.07
    weight_pt = F.softmax(sim_pt, dim=2)
    sim_pt = torch.mul(sim_pt, weight_pt)
    sim_pt = torch.sum(sim_pt, dim=2).squeeze()
    # 找到较小的维度
    # min_dim = min(sim_iw.size(2), sim_pt.size(2))

    # # 裁剪两个张量使得第三个维度相同
    # sim_iw = sim_iw[:, :, :min_dim]
    # sim_pt = sim_pt[:, :, :min_dim]

    # 现在可以相加
    # sim_cs = (sim_iw + sim_pt) / 2

    # print('sim_iw',sim_iw.shape)
    # print('sim_pt',sim_pt.shape)
    sim_cs = (sim_iw + sim_pt) / 2
    # return sim_cs

# piexl_word sim
#sim_pw0局部图像和选择的单词---跨模态同粒度  细粒度对应关系模块
    posWord_norm_r=posWord_norm_r.to(torch.float16)
    sim_pw0 = torch.matmul(L_img_token_norm_l1, posWord_norm_r.transpose(-2, -1)) / 0.01
    sim_pw0 = torch.diagonal(sim_pw0, dim1=-2, dim2=-1)
    sim_pw0 = torch.mean(sim_pw0, dim=2)
#sim_pw1选择的局部图像和局部文本---跨模态同粒度  细粒度对应关系模块
    sim_pw1 = torch.matmul(posClip_norm_l, L_text_token_norm_r1.transpose(-2, -1)) / 0.01
    sim_pw1 = torch.diagonal(sim_pw1, dim1=-2, dim2=-1)
    sim_pw1 = torch.mean(sim_pw1, dim=2)

    sim_cd = (sim_pw0 + sim_pw1) / 2    


    return sim_cs+sim_cd

#cfine 跨模态跨粒度
def compute_sim_cs (G_img_token_norm, L_text_token_norm, L_img_token_norm, G_text_token_norm):
            # image-word sim
        B = L_text_token_norm.size(0)
        G_img_token_norm_l = G_img_token_norm.unsqueeze(1).repeat(1, B, 1, 1)
        L_text_token_norm_r = L_text_token_norm.unsqueeze(0).repeat(B, 1, 1, 1)
            #计算图像全局特征和文本局部特征的相似度
        sim_iw = torch.matmul(G_img_token_norm_l, L_text_token_norm_r.transpose(-2, -1)) / 0.07
        weight_iw = F.softmax(sim_iw, dim=-1)
        sim_iw = torch.mul(sim_iw, weight_iw)
        sim_iw = torch.sum(sim_iw, dim=-1).squeeze()

            # piexl-text sim,这样做是为了在相似度计算时，图像的每一个局部特征都能与文本的每一个全局特征进行匹配。
        L_img_token_norm_l = L_img_token_norm.unsqueeze(1).repeat(1, B, 1, 1)
        G_text_token_norm_r = G_text_token_norm.unsqueeze(0).repeat(B, 1, 1, 1)
            #计算图像局部特征和文本全局特征的相似度
        sim_pt = torch.matmul(L_img_token_norm_l, G_text_token_norm_r.transpose(-2, -1)) / 0.07
        weight_pt = F.softmax(sim_pt, dim=2)
        sim_pt = torch.mul(sim_pt, weight_pt)
        sim_pt = torch.sum(sim_pt, dim=2).squeeze() 
            #得到全局特征和局部特征之间的综合相似度   cross-grained feature refinement 跨模态跨粒度
        sim_cs = (sim_iw + sim_pt) / 2 
        return sim_cs  

#跨模态同粒度
def compute_sim_cd (L_img_token_norm_l1, posWord_norm_r, posClip_norm_l, L_text_token_norm_r1):
       

        sim_pw0 = torch.matmul(L_img_token_norm_l1, posWord_norm_r.transpose(-2, -1)) / 0.07
        sim_pw0 = torch.diagonal(sim_pw0, dim1=-2, dim2=-1)
        sim_pw0 = torch.mean(sim_pw0, dim=2)

        sim_pw1 = torch.matmul(posClip_norm_l, L_text_token_norm_r1.transpose(-2, -1)) / 0.07
        sim_pw1 = torch.diagonal(sim_pw1, dim1=-2, dim2=-1)
        sim_pw1 = torch.mean(sim_pw1, dim=2)

        #计算局部特征之间的相似度，包括关键特征与局部特征的相似度  fine-grained correspondence discovery  跨模态同粒度
        sim_cd = (sim_pw0 + sim_pw1) / 2
        return sim_cd

def phrase_region_score(R, W, tau1=0.07, tau2=0.07):
    B = R.size(0)
    R = R[0].T
    W = W[0].T
    W=W.to(torch.float16)
    # print("W.shape ",W.shape)
    # print("W  ",W)
    # print("R.shape ",R.shape)
    # print("R ",R)
    # 计算点积相似度矩阵 S
    S = torch.matmul(W.T, R)  # 结果形状为 (T-1, 196)
    # 对每个子区域（列）归一化相似度
    S_norm = F.softmax(S, dim=0)  # 在词（行）维度上进行softmax归一化
    # print("")
    # print("S_norm.shape ",S_norm.shape)
    # 计算 α，形状为 (T-1, 196)
    alpha = F.softmax(S_norm / tau1, dim=1)
    # 转置 R 的维度，使其与 alpha 匹配，形状变为 (196, 256)
    R_transposed = R.T
    # print("")
    # print("alpha.shape ",alpha.shape)
    # print("R_transposed.shape ",R_transposed.shape)
    # 计算注意力加权的子区域特征 r_attn，形状为 (T-1, 256)
    r_attn = torch.matmul(alpha, R_transposed)
    # print("")
    # print("r_attn.shape ",r_attn.shape)
    # print("W.shape ",W.shape)
    # dot_products = torch.sum(r_attn * W, dim=1)  # 结果形状为 (T-1,)
    dot_products = torch.sum(r_attn * W.T, dim=1)  # 结果形状为 (T-1,)
    dot_products = dot_products.unsqueeze(0).repeat(B,1)
    # dot_products = dot_products[:,:64]
    # 假设 dot_products 的形状为 (B, N)
    N1, N2 = dot_products.shape
    # 随机选择 64 个不重复的索引
    indices = torch.randperm(N2)[:64]  # 生成从 0 到 N-1 的随机排列，取前 64 个
    # 使用随机索引选择 dot_products 的第二维度的元素
    dot_products_random = dot_products[:, indices]

    # print("")
    # print("dot_products.shape ",dot_products.shape)
    # 计算匹配得分(所有单词和所有图片子区域，所以是一个标量)
    # exp_scores = torch.exp(dot_products / tau2) ** tau2   # 标量
    # print("")
    # print("exp_scores.shape ",exp_scores.shape)
    # print("exp_scores ", exp_scores)
    # exp_scores = torch.clamp(exp_scores, max=10)
    # print("")
    # print("exp_scores.shape ",exp_scores.shape)
    # print("exp_scores ", exp_scores)
    # 求和并取对数
    # matching_score = torch.log(torch.sum(exp_scores))

    # return matching_score
    return dot_products_random

import numpy as np
def get_similarity(patch_part, word_part):
    patch_part = patch_part.to(torch.float16)
    word_part = word_part.to(torch.float16)
    # print("patch_part.device ", patch_part.device)
    # print("word_part.device ", word_part.device)

    # logit_scale = np.log(1 / 0.07).exp()
    # logit_scale = np.exp(np.log(1 / 0.07))
    logit_scale = np.exp(np.log(1 / 0.7))

    total_logits = []
    # print("patch_part.shape ", patch_part.shape)    # [64, 25, 512]
    # print("word_part.shape ", word_part.shape)  # [64, 257, 512]
    patch_word_logits = logit_scale * aggregation_fine_grained_similarity(patch_part=patch_part, word_part=word_part)
    # print("patch_word_logits ", patch_word_logits)
    # print("patch_word_logits.shape ", patch_word_logits.shape)
    total_logits.append(patch_word_logits)

    sim_i_2_t = sum(total_logits) / len(total_logits)
    
    return sim_i_2_t

def aggregation_fine_grained_similarity(patch_part, word_part):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # local_mat_weight = nn.parameter.Parameter(torch.eye(768), requires_grad=True).to(device=device).to(torch.float16)
    # patch_mat_weight =  nn.parameter.Parameter(torch.eye(49), requires_grad=True).to(device=device).to(torch.float16)
    # word_mat_weight = nn.parameter.Parameter(torch.eye(76), requires_grad = True).to(device=device).to(torch.float16)
    # patch_mat_weight2 = nn.parameter.Parameter(torch.eye(49), requires_grad=True).to(device=device).to(torch.float16)
    # word_mat_weight2 = nn.parameter.Parameter(torch.eye(76), requires_grad=True).to(device=device).to(torch.float16)    
    local_mat_weight = nn.parameter.Parameter(torch.eye(512), requires_grad=True).to(device=device).to(torch.float16)
    patch_mat_weight =  nn.parameter.Parameter(torch.eye(25), requires_grad=True).to(device=device).to(torch.float16)
    word_mat_weight = nn.parameter.Parameter(torch.eye(257), requires_grad = True).to(device=device).to(torch.float16)
    patch_mat_weight2 = nn.parameter.Parameter(torch.eye(25), requires_grad=True).to(device=device).to(torch.float16)
    word_mat_weight2 = nn.parameter.Parameter(torch.eye(257), requires_grad=True).to(device=device).to(torch.float16)
    # print("local_mat_weight.device ", local_mat_weight.device)

    bs_img, num_patch, dim = patch_part.shape
    bs_text, max_length_batch, dim = word_part.shape
    # fine_grained_sim_scores = torch.matmul(patch_part.view(-1, dim), word_part.view(-1, dim).t()).view(bs_img, num_patch, bs_text, max_length_batch) 
    fine_grained_sim_scores = torch.matmul(patch_part.reshape(-1, dim), word_part.reshape(-1, dim).t()).reshape(bs_img, num_patch, bs_text, max_length_batch) 
    softmax_t = 1e-2
    # fine_grained_sim_scores = torch.matmul(torch.matmul(patch_part.view(-1, dim), local_mat_weight), word_part.view(-1, dim).t()).view(bs_img, num_patch, bs_text, max_length_batch)    #[bs_img, num_patch, bs_text, max_length_batch]        
    fine_grained_sim_scores = torch.matmul(torch.matmul(patch_part.reshape(-1, dim), local_mat_weight), word_part.reshape(-1, dim).t()).reshape(bs_img, num_patch, bs_text, max_length_batch)    #[bs_img, num_patch, bs_text, max_length_batch]        
    word_level_logit = torch.sum(torch.matmul(torch.softmax(fine_grained_sim_scores/softmax_t, dim=1).permute(0,2,3,1), patch_mat_weight).permute(0,3,1,2) * fine_grained_sim_scores, dim = 1) #[bs_img, bs_text, max_length_batch]
    patch_level_logit = torch.sum(torch.matmul(torch.softmax(fine_grained_sim_scores/softmax_t, dim = -1), word_mat_weight) * fine_grained_sim_scores, dim = -1)      #[bs_img, num_patch, bs_text]

    word_level_logit2 = torch.sum(torch.matmul(torch.softmax(word_level_logit/softmax_t, dim=-1), word_mat_weight2) * word_level_logit, dim=-1)                                         #[bs_img, bs_text]
    patch_level_logit2 = torch.sum(torch.matmul(torch.softmax(patch_level_logit/softmax_t, dim=1).permute(0,2,1), patch_mat_weight2).permute(0,2,1) * patch_level_logit, dim=1)     #[bs_img, bs_text]

    return (word_level_logit2 + patch_level_logit2) / 2

def align_loss(sim_matrix):
    logpt = F.log_softmax(sim_matrix, dim = -1)

    logpt = torch.diag(logpt)       #输出对角线的值构成一个一维tensor
    nce_loss = -logpt

    sim_loss = nce_loss.mean()
    return sim_loss



# def phrase_region_score(R, W, tau1=0.07, tau2=0.07):
#     # print("R.type ",R.dtype)
#     # print(W.dtype)
#     print("W.shape ",W.shape)
#     print("R.shape ",R.shape)
#     W=W.to(torch.float16)
#     # 计算点积相似度矩阵 S
#     print("W.T.shape ",W.shape)
#     print("R.shape ",R.shape)
#     S = torch.matmul(W.T, R)  # 结果形状为 (T-1, 196)t
#     # S = torch.matmul(R, W.T)  # 结果形状为 (T-1, 196)t
#     # S = torch.matmul(W, R.T)  # 结果形状为 (T-1, 196)
#     # 对每个子区域（列）归一化相似度
#     S_norm = F.softmax(S, dim=0)  # 在词（行）维度上进行softmax归一化
#     # 计算 α，形状为 (T-1, 196)
#     alpha = F.softmax(S_norm / tau1, dim=1)
#     # 转置 R 的维度，使其与 alpha 匹配，形状变为 (196, 256)
#     R_transposed = R.T
#     # 计算注意力加权的子区域特征 r_attn，形状为 (T-1, 256)
#     alpha = alpha.to(torch.float32)
#     R_transposed = R_transposed.to(torch.float32)
#     print("alpha.shape ",alpha.shape)
#     print("R_transposed.shape ",R_transposed.shape)
#     # r_attn = torch.matmul(alpha, R_transposed)
#     r_attn = torch.matmul(alpha, R_transposed.T)
#     print("r_attn.shape ",r_attn.shape)
#     print("W.shape ",W.shape)
#     dot_products = torch.sum(r_attn * W, dim=1)  # 结果形状为 (T-1,)
#     # 计算匹配得分
#     exp_scores = torch.exp(dot_products / tau2) ** tau2
#     # 求和并取对数
#     matching_score = torch.log(torch.sum(exp_scores))
#     return matching_score

# 计算 sim_cs (全局图像和全局文本相似度)
def compute_cmp(G_img_token_norm, G_text_token_norm):
    B = G_text_token_norm.size(0)
    # G_text_token_norm = G_text_token_norm.to(torch.float16)
    G_img_token_norm_l = G_img_token_norm.unsqueeze(1).repeat(1, B, 1, 1)
    G_text_token_norm_r = G_text_token_norm.unsqueeze(0).repeat(B, 1, 1, 1)

    # sim_iw = torch.matmul(G_img_token_norm_l, G_text_token_norm_r.transpose(-2, -1)) / 0.09 #原来0.07 可调
    sim_iw = torch.matmul(F.softmax(G_img_token_norm_l, dim=1), F.softmax(G_text_token_norm_r, dim=1).transpose(-2, -1)) / 0.07 #原来0.07 可调

    weight_iw = F.softmax(sim_iw, dim=-1)
    sim_iw = torch.mul(sim_iw, weight_iw)
    sim_iw = torch.sum(sim_iw, dim=-1).squeeze()
    
    return sim_iw

# 计算 sim_pt (局部图像和全局文本相似度)
def compute_cmm(L_img_token_norm, G_text_token_norm):
    B = L_img_token_norm.size(0)
    L_img_token_norm = L_img_token_norm.to(torch.float16)
    G_text_token_norm = G_text_token_norm.to(torch.float16)

    L_img_token_norm_l = L_img_token_norm.unsqueeze(1).repeat(1, B, 1, 1)
    G_text_token_norm_r = G_text_token_norm.unsqueeze(0).repeat(B, 1, 1, 1)

    sim_pt = torch.matmul(L_img_token_norm_l, G_text_token_norm_r.transpose(-2, -1)) / 0.07
    weight_pt = F.softmax(sim_pt, dim=2)
    sim_pt = torch.mul(sim_pt, weight_pt)
    sim_pt = torch.sum(sim_pt, dim=2).squeeze()
    
    return sim_pt
#cfine
# def compute_id(L_img_token_norm_l1, posWord_norm_r, posClip_norm_l, L_text_token_norm_r1):
#     """
#     Instance loss proposed at http://arxiv.org/abs/1711.05535
#     """
#     # print(L_img_token_norm_l1.dtype)
#     # print(posWord_norm_r.dtype)
#     posWord_norm_r=posWord_norm_r.to(torch.float16)
#     sim_pw0 = torch.matmul(L_img_token_norm_l1, posWord_norm_r.transpose(-2, -1)) / 0.01
#     sim_pw0 = torch.diagonal(sim_pw0, dim1=-2, dim2=-1)
#     sim_pw0 = torch.mean(sim_pw0, dim=2)

#     sim_pw1 = torch.matmul(posClip_norm_l, L_text_token_norm_r1.transpose(-2, -1)) / 0.01
#     sim_pw1 = torch.diagonal(sim_pw1, dim1=-2, dim2=-1)
#     sim_pw1 = torch.mean(sim_pw1, dim=2)

#     sim_cd = (sim_pw0 + sim_pw1) / 2
    
#     return sim_cd

# def compute_cmpm(image_embeddings, text_embeddings, labels, epsilon=1e-8):
#     """
#     Cross-Modal Projection Matching Loss(CMPM)
#     :param image_embeddings: Tensor with dtype torch.float32
#     :param text_embeddings: Tensor with dtype torch.float32
#     :param labels: Tensor with dtype torch.int32
#     :return:
#         i2t_loss: cmpm loss for image projected to text
#         t2i_loss: cmpm loss for text projected to image
#         pos_avg_sim: average cosine-similarity for positive pairs
#         neg_avg_sim: averate cosine-similarity for negative pairs
#     """

#     batch_size = image_embeddings.shape[0]
#     labels_reshape = torch.reshape(labels, (batch_size, 1))
#     labels_dist = labels_reshape - labels_reshape.t()
#     labels_mask = (labels_dist == 0).float()

#     image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
#     text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
#     image_proj_text = torch.matmul(image_embeddings, text_norm.t())
#     text_proj_image = torch.matmul(text_embeddings, image_norm.t())

#     # normalize the true matching distribution
#     labels_mask_norm = labels_mask / labels_mask.norm(dim=1)

#     i2t_pred = F.softmax(image_proj_text, dim=1)
#     i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_mask_norm + epsilon))
#     t2i_pred = F.softmax(text_proj_image, dim=1)
#     t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_mask_norm + epsilon))

#     cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

#     return cmpm_loss

# def compute_mlm(image_embeddings, text_embeddings, labels, epsilon=1e-8):
#     """
#     Cross-Modal Projection Matching Loss(CMPM)
#     :param image_embeddings: Tensor with dtype torch.float32
#     :param text_embeddings: Tensor with dtype torch.float32
#     :param labels: Tensor with dtype torch.int32
#     :return:
#         i2t_loss: cmpm loss for image projected to text
#         t2i_loss: cmpm loss for text projected to image
#         pos_avg_sim: average cosine-similarity for positive pairs
#         neg_avg_sim: averate cosine-similarity for negative pairs
#     """
#     batch_size = image_embeddings.shape[0]
#     if labels.size(0) == 15:
#          batch_size = batch_size[:15]
#     print("batch_size",batch_size)
#     print("labels",labels.shape)
#     labels_reshape = torch.reshape(labels, (batch_size, 1))
#     labels_dist = labels_reshape - labels_reshape.t()
#     labels_mask = (labels_dist == 0).float()
#     # print("image_embeddings",image_embeddings.shape)
#     # print("text_embeddings",text_embeddings.shape)

#     image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
#     text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
#     # print("image_norm",image_norm.shape)
#     # print("text_norm",text_norm.shape)
#     image_proj_text = torch.matmul(image_embeddings, text_norm.t())
#     # image_proj_text = torch.matmul(image_embeddings, text_norm.permute(0, 1, 3, 2))

#     text_proj_image = torch.matmul(text_embeddings, image_norm.t())
#     # text_proj_image = torch.matmul(text_embeddings, image_norm.permute(0, 1, 3, 2))

#     # normalize the true matching distribution
#     labels_mask_norm = labels_mask / labels_mask.norm(dim=1)

#     i2t_pred = F.softmax(image_proj_text, dim=1)
#     i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_mask_norm + epsilon))
#     t2i_pred = F.softmax(text_proj_image, dim=1)
#     t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_mask_norm + epsilon))

#     cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))
#     # print(cmpm_loss)

#     return cmpm_loss

# def compute_mlm(sim,label):
#         image_triplets, img_margin = get_triplets(similarity, label, margin, semi)
#         text_triplets, txt_margin = get_triplets(similarity.t(), label, margin, semi)

#         image_anchor_loss = F.relu(img_margin
#                                    - similarity[image_triplets[:, 0], image_triplets[:, 1]]
#                                    + similarity[image_triplets[:, 0], image_triplets[:, 2]])

#         similarity = similarity.t()
#         text_anchor_loss = F.relu(txt_margin
#                                   - similarity[text_triplets[:, 0], text_triplets[:, 1]]
#                                   + similarity[text_triplets[:, 0], text_triplets[:, 2]])

#         loss = torch.sum(image_anchor_loss) + torch.sum(text_anchor_loss)

#         return loss




def xpool(sims, logit_scale1):
        """
        Inputs: cosine similarities
            sims: n x n (text is dim-0)
            logit_scale: 1 x 1
        """
        # print(sims.shape)
        # print(logit_scale1.shape)
        logits = sims * logit_scale1
        
        t2v_log_sm = F.log_softmax(logits, dim=1)
        t2v_neg_ce = torch.diag(t2v_log_sm)
        t2v_loss = -t2v_neg_ce.mean()

        v2t_log_sm = F.log_softmax(logits, dim=0)
        v2t_neg_ce = torch.diag(v2t_log_sm)
        v2t_loss = -v2t_neg_ce.mean()

        return (t2v_loss + v2t_loss) / 2.0

# def xpool2(sims, logit_scale1):
#         """
#         Inputs: cosine similarities
#             sims: n x n (text is dim-0)
#             logit_scale: 1 x 1
#         """
#         print(sims.shape)
#         print(logit_scale1.shape)
#         logits = sims * logit_scale1
        
#         t2v_log_sm = F.log_softmax(logits, dim=1)
#         t2v_neg_ce = torch.diag(t2v_log_sm)
#         t2v_loss = -t2v_neg_ce.mean()

#         v2t_log_sm = F.log_softmax(logits, dim=0)
#         v2t_neg_ce = torch.diag(v2t_log_sm)
#         v2t_loss = -v2t_neg_ce.mean()

#         return (t2v_loss + v2t_loss) / 2.0


        # print("sims",sims) sims tensor([[[ 0.5371,  1.0459, -1.2832,  ..., -0.8447, -1.9336, -0.0293],
        # print("logit_scale",logit_scale)  logit_scale tensor(50.)
        # logit_scale = logit_scale.exp()
        # logits = sims * logit_scale
        # # print("logit_scale size:", logit_scale.size())
        # batch_size = image_feats.shape[0]
        # labels = torch.arange(start=0, end=batch_size, dtype=torch.int64)
        # labels = labels.to(image_feats.device)

        # sims_softmax = F.softmax(sims, dim=1)

        # # 选择每个样本中对应标签的预测概率
        # predicted_probs = sims_softmax[:, labels]
        # ce = nn.CrossEntropyLoss(sims_softmax,predicted_probs)
        # return ce
                                                                                                                                                                                                                                       

        # logit_scale = torch.clip(logit_scale, min=-100, max=100)  # 根据实际情况调整范围  这边
        # print(logit_scale)
        # print(sims.shape)
        # logit_scale = logit_scale.exp()
        # logits = sims * logit_scale
        # t2v_log_sm = F.log_softmax(logits, dim=1)
        # t2v_log_sm_2d = t2v_log_sm.view(t2v_log_sm.size(0), t2v_log_sm.size(1), -1)
        #             #cccccc
        #     # t2v_log_sm_2d = torch.mul(t2v_log_sm_2d, F.softmax(t2v_log_sm_2d,dim=1))
        # t2v_neg_ce = torch.diagonal(t2v_log_sm_2d, dim1=1, dim2=2)
        # t2v_loss = -t2v_neg_ce.mean()
        # v2t_log_sm = F.log_softmax(logits, dim=0)


        # v2t_log_sm_2d = v2t_log_sm.view(v2t_log_sm.size(0), v2t_log_sm.size(1), -1)
        #             #ccccc
        #     # v2t_log_sm_2d = torch.mul(v2t_log_sm_2d,F.softmax(v2t_log_sm_2d,dim=1))
        # v2t_neg_ce = torch.diagonal(v2t_log_sm_2d, dim1=1, dim2=2)
        # v2t_loss = -v2t_neg_ce.mean()

        # t2v_log_sm = F.log_softmax(logits, dim=1)
        # logits_2d = logits.view(-1, logits.size(-1))  # 将最后一维压缩，得到2D矩阵   
        
        # logits_2d = torch.mul(logits_2d,F.softmax(logits_2d,dim=1))    
        # t2v_neg_ce = torch.diag(F.softmax(logits_2d,dim=1))
        # t2v_loss = -t2v_neg_ce.mean()

        # # v2t_log_sm = F.log_softmax(logits, dim=0)
        # logits_2d = logits.view(-1, logits.size(-1))  # 将最后一维压缩，得到2D矩阵   
        # v2t_neg_ce = torch.diag(logits_2d)
        # v2t_loss = -v2t_neg_ce.mean()
        # loss = (t2v_loss + v2t_loss) / 2.0

        # return loss
        # a=torch.tensor(1)
        # logit_scale =a.exp()
        # print(logit_scale1)










# def CTLoss(output, labels): 
#         batch_size = output.size(0) 
#         labels = Variable(labels.cuda())
#         # print("batch size",batch_size)
#         # print("logit_scale1",logit_scale1)
#         # targets_expand = logit_scale1.view(batch_size, 1).expand(batch_size, output.size(1)) 
#         targets_expand = torch.full((batch_size, 1), int(labels), dtype=torch.long, device=output.device)
#         # centers_batch = self.centers.gather(0, targets_expand) 
#         centers_batch = nn.Parameter(torch.randn(11003, 11003, device=output.device), requires_grad=True).gather(0, targets_expand) 


#         centers_batch_bz = torch.stack([centers_batch]*batch_size) 
        
#         inputs_bz = torch.stack([output]*batch_size).transpose(0, 1) 
#         centers_batch_bz_expanded = centers_batch_bz.unsqueeze(3).unsqueeze(4).expand_as(inputs_bz)
#         # print("centers_batch_bz",centers_batch_bz)
#         # print("inputs_bz",inputs_bz)
#         dist = torch.sum((centers_batch_bz_expanded -inputs_bz)**2, 2).squeeze() 
#         dist = dist.clamp(min=1e-12).sqrt()  
#         mask = labels.expand(batch_size, batch_size).eq(labels.expand(batch_size, batch_size).t())

#         dist_ap, dist_an = [], [] 
#         for i in range(batch_size): 
#             # 计算 dist_ap
#             dist_ap.append(dist[i][mask[i]].max()) 
            
#             # 添加条件检查，确保至少有一个元素存在
#             if (mask[i] == 0).sum() > 0:
#                 dist_an_i = dist[i][mask[i] == 0].min(dim=some_dim).values
#                 # 处理 dist_an_i 为零维张量的情况
#                 if dist_an_i.dim() == 0:
#                     dist_an_i = dist_an_i.unsqueeze(0)  # 将其转换为一维张量
#                 dist_an.append(dist_an_i)
#             else:
#                 # 在这里处理 dist_an 为空的情况，例如设置一个默认值
#                 dist_an.append(torch.tensor(0.0))  # 替换为适当的默认值

#         # 在这里将 dist_ap 和 dist_an 拼接为一维张量
#         dist_ap = torch.stack(dist_ap)
#         dist_an = torch.stack(dist_an)

#         y = torch.empty_like(dist_an.data)
        
#         y.resize_as_(dist_an.data)
#         # y.fill_(1)
#         y = torch.zeros_like(dist_an)
#         dist_an = dist_an.to("cuda")
#         dist_ap = dist_ap.to("cuda")
#         y = y.to("cuda")

#         loss = F.margin_ranking_loss(dist_an, dist_ap, y,margin=0.02)
#         # print(loss)

#         # prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
#         return loss


# def CTLoss(output, labels): 
#     batch_size = output.size(0) 
#     # print("labels",labels.shape)
#     # print("batch_size",batch_size)
#     # print("output",output.shape)

#     targets_expand = labels.view(batch_size, 1).expand(batch_size, output.size(1)) 
#     centers_batch = nn.Parameter(torch.randn(128, 128, device=output.device), requires_grad=True).gather(0, targets_expand) 
#     centers_batch_expanded = centers_batch.unsqueeze(2).unsqueeze(2)

#     centers_batch_bz = torch.stack([centers_batch_expanded]*batch_size)
#     inputs_bz = torch.stack([output]*batch_size).transpose(0, 1)
#     dist = torch.sum((centers_batch_bz - inputs_bz)**2, 2).squeeze()
#     dist = dist.clamp(min=1e-12).sqrt()  
#     mask = labels.expand(batch_size, batch_size).eq(labels.expand(batch_size, batch_size).t())
#     dist_ap, dist_an = [], [] 
#     for i in range(batch_size): 
#         dist_ap.append(dist[i][mask[i]].max()) 
#         dist_an.append(dist[i][mask[i]==0].min()) 
#     # 在连接之前检查列表，如果有零维张量，则将其转换为具有一个维度的张量
#     dist_ap = [item.view(1) if item.dim() == 0 else item for item in dist_ap]

#     dist_ap = torch.cat(dist_ap)
#     # 在连接之前检查列表，如果有零维张量，则将其转换为具有一个维度的张量
#     dist_an = [item.view(1) if item.dim() == 0 else item for item in dist_an]
#     dist_an = torch.cat(dist_an)
#     y = torch.empty_like(dist_an).fill_(1)

#     dist_an = torch.mul (dist_an, F.softmax(dist_an, dim=0))
#     dist_ap = torch.mul (dist_ap, F.softmax(dist_ap, dim=0))

#     # loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=0.02)
#     loss = F.margin_ranking_loss(F.softmax(dist_an,dim=0), F.softmax(dist_ap,dim=0), y, margin=0.02)
#     # loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=0.5)
#     # print(loss)

#     return loss