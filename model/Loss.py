# -*- coding: utf-8 -*-
"""

@author: zifyloo
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter

def l2norm(x):
    """L2-normalize columns of x"""
    norm = torch.pow(x, 2).sum(dim=-1, keepdim=True).sqrt()
    return torch.div(x, norm)

class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()
        # self.CMPM = args.CMPM
        # self.CMPC = args.CMPC
        self.epsilon = args.epsilon
        self.num_classes = 11003
        self.W = Parameter(torch.randn(args.feature_size, args.num_classes))
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
        image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
        text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)

        image_proj_text = (
            torch.sum(image_embeddings * text_norm, dim=1, keepdim=True) * text_norm
        )
        text_proj_image = (
            torch.sum(text_embeddings * image_norm, dim=1, keepdim=True) * image_norm
        )

        image_logits = torch.matmul(image_proj_text, self.W_norm)
        text_logits = torch.matmul(text_proj_image, self.W_norm)

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

        batch_size = image_embeddings.shape[0]
        labels_reshape = torch.reshape(labels, (batch_size, 1))
        labels_dist = labels_reshape - labels_reshape.t()
        labels_mask = labels_dist == 0

        image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
        text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
        image_proj_text = torch.matmul(image_embeddings, text_norm.t())
        text_proj_image = torch.matmul(text_embeddings, image_norm.t())

        # normalize the true matching distribution
        labels_mask_norm = labels_mask.float() / labels_mask.float().norm(dim=1)

        i2t_pred = F.softmax(image_proj_text, dim=1)
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

        cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(
            torch.sum(t2i_loss, dim=1)
        )

        sim_cos = torch.matmul(image_norm, text_norm.t())

        pos_avg_sim = torch.mean(torch.masked_select(sim_cos, labels_mask))
        neg_avg_sim = torch.mean(torch.masked_select(sim_cos, labels_mask == 0))

        return cmpm_loss, pos_avg_sim, neg_avg_sim

    def forward(
        self, image_embeddings, text_embeddings, img_f, text_f, labels, lambda_diversity
    ):
        # self.diversity_loss2(img_f)
        cmpm_loss = 0.0
        cmpc_loss = 0.0
        image_precision = 0.0
        text_precision = 0.0
        neg_avg_sim = 0.0
        pos_avg_sim = 0.0
        if self.CMPM:
            cmpm_loss = 0
            for i in range(len(image_embeddings)):
                cmpm_loss1, pos_avg_sim, neg_avg_sim = self.compute_cmpm_loss(
                    image_embeddings[i], text_embeddings[i], labels
                )
                if i == 0 or i == 1:
                    cmpm_loss += cmpm_loss1
                else:
                    cmpm_loss += cmpm_loss1*0.1   # *1.0/(len(image_embeddings)-1)

        if self.CMPC:
            cmpc_loss = 0
            for i in range(len(image_embeddings)):
                cmpc_loss1, image_precision, text_precision = self.compute_cmpc_loss(
                    image_embeddings[i], text_embeddings[i], labels
                )
                # if i==0:
                cmpc_loss += cmpc_loss1
                # else:
                #     cmpc_loss += cmpc_loss1*1.0/(len(image_embeddings)-1)

        loss = cmpm_loss + cmpc_loss
        loss += lambda_diversity * (
            self.diversity_loss(img_f[:,1:]) + self.diversity_loss(text_f[:,1:])
        )
        return (
            cmpm_loss,
            cmpc_loss,
            loss,
            image_precision,
            text_precision,
            pos_avg_sim,
            neg_avg_sim,
        )

