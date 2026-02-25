# Copyright (c) OpenMMLab. All rights reserved.
import copy
import re
import warnings
from typing import Dict, Optional, Tuple, Union, List

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.nn import functional as F
from mmengine.runner.amp import autocast
from torch import Tensor

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType
from mmdet.models.layers import Mask2FormerTransformerDecoder, SinePositionalEncoding
from mmdet.models.layers.transformer.grounding_dino_layers import (
    GroundingDinoTransformerDecoder, GroundingDinoTransformerDecoderLayer, GroundingDinoTransformerEncoder)
from mmdet.models.layers.transformer.utils import coordinate_to_encoding
from mmdet.models.detectors.dino import DINO
from mmdet.models.detectors.grounding_dino import GroundingDINO
from mmdet.models.detectors.glip import (create_positive_map, create_positive_map_label_to_token,
                   run_ner)


from mmcv.cnn import build_norm_layer, Linear, ConvModule, Conv2d
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.cnn.bricks import DropPath
from mmcv.ops import MultiScaleDeformableAttention, batched_nms, point_sample
from mmengine.model import ModuleList
from mmengine.structures import InstanceData, PixelData
from torch import Tensor

from mmdet.models.utils.vlfuse_helper import SingleScaleBiAttentionBlock
from mmdet.utils import InstanceList, OptInstanceList, reduce_mean, ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.layers.transformer.deformable_detr_layers import (DeformableDetrTransformerDecoderLayer,
                                     DeformableDetrTransformerEncoder,
                                     DeformableDetrTransformerEncoderLayer)
from mmdet.models.layers.transformer.detr_layers import DetrTransformerEncoderLayer
from mmdet.models.layers.transformer.dino_layers import DinoTransformerDecoder
from mmdet.models.layers.transformer.utils import MLP, get_text_sine_pos_embed
from mmdet.models.dense_heads import GroundingDINOHead, DeformableDETRHead
from mmdet.models.losses import QualityFocalLoss
from mmdet.structures import SampleList, DetDataSample
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh, bbox_overlaps, bbox2roi
from mmdet.models.layers import inverse_sigmoid
from mmdet.models.utils import multi_apply
from mmseg.structures import SegDataSample
from mmseg.models.utils import resize
from mmseg.models import Mask2FormerHead
from mmengine.model import BaseModule

from mmengine.runner.checkpoint import load_checkpoint


from transformers import SamModel, SamProcessor
from PIL import Image

from mmdet.models.utils import unpack_gt_instances, preprocess_panoptic_gt, get_uncertain_point_coords_with_randomness
from contextlib import contextmanager
from mmdet.models.dense_heads.atss_vlfusion_head import convert_grounding_to_cls_scores
from mmdet.models.dense_heads.grounding_dino_head import ContrastiveEmbed
from mmdet.models.task_modules.samplers import SamplingResult

try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except Exception:
    checkpoint_wrapper = None

def clean_label_name(name: str) -> str:
    name = re.sub(r'\(.*\)', '', name)
    name = re.sub(r'_', ' ', name)
    name = re.sub(r'  ', ' ', name)
    return name


def chunks(lst: list, n: int) -> list:
    """Yield successive n-sized chunks from lst."""
    all_ = []
    for i in range(0, len(lst), n):
        data_index = lst[i:i + n]
        all_.append(data_index)
    counter = 0
    for i in all_:
        counter += len(i)
    assert (counter == len(lst))

    return all_


# modify
def create_positive_map_plus_object(tokenized,
                                tokens_positive: list,
                                max_num_entities: int = 256) -> Tensor:
    """construct a map such that positive_map[i,j] = True
    if box i is associated to token j

    Args:
        tokenized: The tokenized input.
        tokens_positive (list): A list of token ranges
            associated with positive boxes.
        max_num_entities (int, optional): The maximum number of entities.
            Defaults to 256.

    Returns:
        torch.Tensor: The positive map.

    Raises:
        Exception: If an error occurs during token-to-char mapping.
    """
    positive_map = torch.zeros((len(tokens_positive), max_num_entities),
                               dtype=torch.float)
    positive_map[:, 1] = 1

    for j, tok_list in enumerate(tokens_positive):
        for (beg, end) in tok_list:
            try:
                beg_pos = tokenized.char_to_token(beg)
                end_pos = tokenized.char_to_token(end - 1)
            except Exception as e:
                print('beg:', beg, 'end:', end)
                print('token_positive:', tokens_positive)
                raise e
            if beg_pos is None:
                try:
                    beg_pos = tokenized.char_to_token(beg + 1)
                    if beg_pos is None:
                        beg_pos = tokenized.char_to_token(beg + 2)
                except Exception:
                    beg_pos = None
            if end_pos is None:
                try:
                    end_pos = tokenized.char_to_token(end - 2)
                    if end_pos is None:
                        end_pos = tokenized.char_to_token(end - 3)
                except Exception:
                    end_pos = None
            if beg_pos is None or end_pos is None:
                continue

            assert beg_pos is not None and end_pos is not None
            positive_map[j, beg_pos:end_pos + 1].fill_(1)
    return positive_map / (positive_map.sum(-1)[:, None] + 1e-6)


class MyNeck(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.convs = ModuleList([nn.Conv2d(in_c, in_c, 3, 1, 1) for in_c in in_channels])
    
    def forward(self, xs):
        assert len(xs) == len(self.convs)
        res = []
        for x, c in zip(xs, self.convs):
            res.append(c(x))
        return res


@MODELS.register_module()
class GroundingDINOHeadIoU(GroundingDINOHead):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_iou_pred = MODELS.build(dict(type='L1Loss', loss_weight=1.0))
    
    def _init_layers(self) -> None:
        super()._init_layers()
        # modify
        self.cls_branches.requires_grad_(False) 
        fc_cls_bbyy = Linear(self.embed_dims, 1)

        if self.share_pred_layer:
            self.cls_branches_bbyy = nn.ModuleList(
                [fc_cls_bbyy for _ in range(self.num_pred_layer)])
        else:
            self.cls_branches_bbyy = nn.ModuleList(
                [copy.deepcopy(fc_cls_bbyy) for _ in range(self.num_pred_layer)])
        
        iou_branch = []
        for _ in range(self.num_reg_fcs):
            iou_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            iou_branch.append(nn.ReLU())
        iou_branch.append(nn.Linear(self.embed_dims, 1))
        iou_branch = nn.Sequential(*iou_branch)
        if self.share_pred_layer:
            self.iou_branches = nn.ModuleList(
                [iou_branch for _ in range(self.num_pred_layer-1)])
        else:
            self.iou_branches = nn.ModuleList(
                [copy.deepcopy(iou_branch) for _ in range(self.num_pred_layer-1)])
    
    def forward(
        self,
        hidden_states: Tensor,
        references: List[Tensor],
        memory_text: Tensor,
        text_token_mask: Tensor,
    ) -> Tuple[Tensor]:
        
        all_layers_outputs_classes = []
        all_layers_outputs_coords = []
        # modify
        all_layers_outputs_classes_bbyy = []
        all_layers_outputs_ious = []
        for layer_id in range(hidden_states.shape[0]):
            reference = inverse_sigmoid(references[layer_id])
            # NOTE The last reference will not be used.
            hidden_state = hidden_states[layer_id]
            outputs_class = self.cls_branches[layer_id](hidden_state,
                                                        memory_text,
                                                        text_token_mask)
            tmp_reg_preds = self.reg_branches[layer_id](hidden_state)
            if reference.shape[-1] == 4:
                # When `layer` is 0 and `as_two_stage` of the detector
                # is `True`, or when `layer` is greater than 0 and
                # `with_box_refine` of the detector is `True`.
                tmp_reg_preds += reference
            else:
                # When `layer` is 0 and `as_two_stage` of the detector
                # is `False`, or when `layer` is greater than 0 and
                # `with_box_refine` of the detector is `False`.
                assert reference.shape[-1] == 2
                tmp_reg_preds[..., :2] += reference
            outputs_coord = tmp_reg_preds.sigmoid()
            all_layers_outputs_classes.append(outputs_class)
            all_layers_outputs_coords.append(outputs_coord)
            # modify
            outputs_class_bbyy = self.cls_branches_bbyy[layer_id](hidden_state)
            all_layers_outputs_classes_bbyy.append(outputs_class_bbyy)
            outputs_iou = self.iou_branches[layer_id](hidden_state)
            all_layers_outputs_ious.append(outputs_iou)

        all_layers_outputs_classes = torch.stack(all_layers_outputs_classes)
        all_layers_outputs_coords = torch.stack(all_layers_outputs_coords)
        all_layers_outputs_classes_bbyy = torch.stack(all_layers_outputs_classes_bbyy)
        all_layers_outputs_ious = torch.stack(all_layers_outputs_ious)


        return all_layers_outputs_classes, all_layers_outputs_classes_bbyy, all_layers_outputs_coords, all_layers_outputs_ious

    def predict(self,
                hidden_states: Tensor,
                references: List[Tensor],
                memory_text: Tensor,
                text_token_mask: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> InstanceList:
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        batch_token_positive_maps = [
            data_samples.token_positive_map
            for data_samples in batch_data_samples
        ]

        outs = self(hidden_states, references, memory_text, text_token_mask)

        predictions = self.predict_by_feat(
            *outs,
            batch_img_metas=batch_img_metas,
            batch_token_positive_maps=batch_token_positive_maps,
            rescale=rescale)
        return predictions
    
    def predict_by_feat(self,
                        all_layers_outputs_classes: Tensor,
                        all_layers_outputs_classes_bbyy: Tensor,
                        all_layers_bbox_preds: Tensor,
                        all_layers_outputs_ious: Tensor,
                        batch_img_metas: List[Dict],
                        batch_token_positive_maps: Optional[List[dict]] = None,
                        rescale: bool = False) -> InstanceList:
        # cls_scores = all_layers_cls_scores[-1][:, :self.num_queries]
        # cls_scores_bbyy = all_layers_outputs_classes_bbyy[-1][:, -self.num_queries_bbyy:]
        cls_scores_bbyy = all_k=all_layers_outputs_ious[-1][:, -self.num_queries_bbyy:]
        bbox_preds = all_layers_bbox_preds[-1]
        result_list = []
        for img_id in range(len(batch_img_metas)):
            # cls_score = cls_scores[img_id]
            cls_score_bbyy = cls_scores_bbyy[img_id]
            bbox_pred = bbox_preds[img_id]
            img_meta = batch_img_metas[img_id]
            # token_positive_maps = batch_token_positive_maps[img_id]
            # results = self._predict_by_feat_single(cls_score, bbox_pred,
            #                                        token_positive_maps,
            #                                        img_meta, rescale)
            results = self._predict_by_feat_single_bbyy(cls_score_bbyy, bbox_pred[-self.num_queries_bbyy:],
                                                   img_meta, rescale)
            result_list.append(results)
        return result_list
    
    def _predict_by_feat_single_bbyy(self,
                                cls_score: Tensor,
                                bbox_pred: Tensor,
                                img_meta: dict,
                                rescale: bool = True) -> InstanceData:
        assert len(cls_score) == len(bbox_pred)  # num_queries
        max_per_img = self.test_cfg.get('max_per_img', len(cls_score))
        img_shape = img_meta['img_shape']
        # exclude background
        num_classes = 1
        if self.loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
            scores, indexes = cls_score.view(-1).topk(max_per_img)
            det_labels = indexes % num_classes
            bbox_index = indexes // num_classes
            bbox_pred = bbox_pred[bbox_index]
        else:
            scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(max_per_img)
            bbox_pred = bbox_pred[bbox_index]
            det_labels = det_labels[bbox_index]

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            assert img_meta.get('scale_factor') is not None
            det_bboxes /= det_bboxes.new_tensor(
                img_meta['scale_factor']).repeat((1, 2))

        results = InstanceData()
        results.bboxes = det_bboxes
        results.scores = scores
        results.labels = det_labels
        return results
    
    def loss_by_feat(
        self,
        all_layers_cls_scores: Tensor,
        all_layers_cls_scores_bbyy: Tensor,
        all_layers_bbox_preds: Tensor,
        all_layers_ious: Tensor,
        enc_cls_scores: Tensor,
        enc_bbox_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        dn_meta: Dict[str, int],
        batch_gt_instances_ignore = None
    ) -> Dict[str, Tensor]:
        # extract denoising and matching part of outputs
        (all_layers_matching_cls_scores, all_layers_matching_cls_scores_bbyy, all_layers_matching_bbox_preds, all_layers_matching_ious,
         all_layers_denoising_cls_scores, all_layers_denoising_cls_scores_bbyy, all_layers_denoising_bbox_preds, all_layers_denoising_ious) = \
            self.split_outputs(
                all_layers_cls_scores, all_layers_cls_scores_bbyy, all_layers_bbox_preds, all_layers_ious, dn_meta)

        assert batch_gt_instances_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            'for batch_gt_instances_ignore setting to None.'

        # modify
        # ======================================================================
        _, all_layers_matching_cls_scores_bbyy = torch.split(
                                        all_layers_matching_cls_scores_bbyy, [self.num_queries, self.num_queries_bbyy], dim=2)
        _, all_layers_matching_bbox_preds_bbyy = torch.split(
                                        all_layers_matching_bbox_preds, [self.num_queries, self.num_queries_bbyy], dim=2)
        _, all_layers_matching_ious_bbyy = torch.split(
                                        all_layers_matching_ious, [self.num_queries, self.num_queries_bbyy], dim=2)
        # ======================================================================
        
        losses_cls, losses_bbox, losses_iou, losses_iou_pred = multi_apply(
            self.loss_by_feat_single,
            all_layers_matching_cls_scores_bbyy,
            all_layers_matching_bbox_preds_bbyy,
            all_layers_matching_ious_bbyy,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_iou'] = losses_iou[-1]
        loss_dict['loss_iou_pred'] = losses_iou_pred[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i, loss_iou_pred in \
                zip(losses_cls[:-1], losses_bbox[:-1], losses_iou[:-1], losses_iou_pred[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
            loss_dict[f'd{num_dec_layer}.loss_iou_pred'] = loss_iou_pred
            num_dec_layer += 1
        
        # NOTE DETRHead.loss_by_feat but not DeformableDETRHead.loss_by_feat
        # is called, because the encoder loss calculations are different
        # between DINO and DeformableDETR.

        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            # NOTE The enc_loss calculation of the DINO is
            # different from that of Deformable DETR.
            enc_loss_cls, enc_losses_bbox, enc_losses_iou, _ = \
                self.loss_by_feat_single(
                    enc_cls_scores, enc_bbox_preds, None,
                    batch_gt_instances=batch_gt_instances,
                    batch_img_metas=batch_img_metas)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox
            loss_dict['enc_loss_iou'] = enc_losses_iou

        # if all_layers_denoising_cls_scores is not None:
        #     # calculate denoising loss from all decoder layers
        #     dn_losses_cls, dn_losses_bbox, dn_losses_iou, dn_losses_iou_pred = self.loss_dn(
        #         all_layers_denoising_cls_scores,
        #         all_layers_denoising_bbox_preds,
        #         all_layers_denoising_ious, 
        #         batch_gt_instances=batch_gt_instances,
        #         batch_img_metas=batch_img_metas,
        #         dn_meta=dn_meta)
        #     # collate denoising loss
        #     loss_dict['dn_loss_cls'] = dn_losses_cls[-1]
        #     loss_dict['dn_loss_bbox'] = dn_losses_bbox[-1]
        #     loss_dict['dn_loss_iou'] = dn_losses_iou[-1]
        #     loss_dict['dn_loss_iou_pred'] = dn_losses_iou_pred[-1]
        #     for num_dec_layer, (loss_cls_i, loss_bbox_i, loss_iou_i, loss_iou_pred_i) in \
        #             enumerate(zip(dn_losses_cls[:-1], dn_losses_bbox[:-1],
        #                           dn_losses_iou[:-1], dn_losses_iou_pred[:-1])):
        #         loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i
        #         loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i
        #         loss_dict[f'd{num_dec_layer}.dn_loss_iou'] = loss_iou_i
        #         loss_dict[f'd{num_dec_layer}.dn_loss_iou_pred'] = loss_iou_pred_i
        return loss_dict

    def loss_by_feat_single(self, cls_scores: Tensor, bbox_preds: Tensor, iou_preds: Tensor,
                            batch_gt_instances: InstanceList,
                            batch_img_metas: List[dict]) -> Tuple[Tensor]:
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        with torch.no_grad():
            cls_reg_targets = self.get_targets(cls_scores_list,
                                               bbox_preds_list,
                                               batch_gt_instances,
                                               batch_img_metas)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.stack(labels_list, 0)
        label_weights = torch.stack(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)    

        # ===== this change =====
        # Loss is not computed for the padded regions of the text.
        assert (self.text_masks.dim() == 2)
        text_masks = self.text_masks.new_zeros(
            (self.text_masks.size(0), self.max_text_len))
        text_masks[:, :self.text_masks.size(1)] = self.text_masks
        text_mask = (text_masks > 0).unsqueeze(1)
        text_mask = text_mask.repeat(1, cls_scores.size(1), 1)
        cls_scores = torch.masked_select(cls_scores, text_mask).contiguous()

        labels = torch.masked_select(labels, text_mask)
        label_weights = label_weights[...,
                                      None].repeat(1, 1, text_mask.size(-1))
        label_weights = torch.masked_select(label_weights, text_mask)
        

        # classification loss
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        if isinstance(self.loss_cls, QualityFocalLoss):
            raise NotImplementedError(
                'QualityFocalLoss for GroundingDINOHead is not supported yet.')
        else:
            loss_cls = self.loss_cls(
                cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, bbox_preds):
            img_h, img_w, = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        
        # modify
        if iou_preds is not None:
            ious_gt = bbox_overlaps(bboxes, bboxes_gt, is_aligned=True, eps=1e-6)
            loss_iou_pred = self.loss_iou_pred(iou_preds.sigmoid().flatten(), ious_gt, bbox_weights[:, 0], avg_factor=num_total_pos)
        else:
            loss_iou_pred = None
        return loss_cls, loss_bbox, loss_iou, loss_iou_pred

    def loss_dn(self, all_layers_denoising_cls_scores: Tensor,
                all_layers_denoising_bbox_preds: Tensor,
                all_layers_denoising_iou_preds: Tensor,
                batch_gt_instances: InstanceList, batch_img_metas: List[dict],
                dn_meta: Dict[str, int]) -> Tuple[List[Tensor]]:
        return multi_apply(
            self._loss_dn_single,
            all_layers_denoising_cls_scores,
            all_layers_denoising_bbox_preds,
            all_layers_denoising_iou_preds,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas,
            dn_meta=dn_meta)

    def _loss_dn_single(self, dn_cls_scores: Tensor, dn_bbox_preds: Tensor, dn_iou_preds: Tensor,
                        batch_gt_instances: InstanceList,
                        batch_img_metas: List[dict],
                        dn_meta: Dict[str, int]) -> Tuple[Tensor]:
        cls_reg_targets = self.get_dn_targets(batch_gt_instances,
                                              batch_img_metas, dn_meta)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.stack(labels_list, 0)
        label_weights = torch.stack(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        # ===== this change =====
        # Loss is not computed for the padded regions of the text.
        assert (self.text_masks.dim() == 2)
        text_masks = self.text_masks.new_zeros(
            (self.text_masks.size(0), self.max_text_len))
        text_masks[:, :self.text_masks.size(1)] = self.text_masks
        text_mask = (text_masks > 0).unsqueeze(1)
        text_mask = text_mask.repeat(1, dn_cls_scores.size(1), 1)
        cls_scores = torch.masked_select(dn_cls_scores, text_mask).contiguous()
        labels = torch.masked_select(labels, text_mask)
        label_weights = label_weights[...,
                                      None].repeat(1, 1, text_mask.size(-1))
        label_weights = torch.masked_select(label_weights, text_mask)
        # =======================

        # classification loss
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = \
            num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        if len(cls_scores) > 0:
            if isinstance(self.loss_cls, QualityFocalLoss):
                raise NotImplementedError('QualityFocalLoss is not supported')
            else:
                loss_cls = self.loss_cls(
                    cls_scores,
                    labels,
                    label_weights,
                    avg_factor=cls_avg_factor)
        else:
            loss_cls = torch.zeros(
                1, dtype=cls_scores.dtype, device=cls_scores.device)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, dn_bbox_preds):
            img_h, img_w = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = dn_bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        
        ious_gt = bbox_overlaps(bboxes, bboxes_gt, is_aligned=True, eps=1e-6)
        loss_iou_pred = self.loss_iou_pred(dn_iou_preds.sigmoid().flatten(), ious_gt, bbox_weights[:, 0], avg_factor=num_total_pos)

        return loss_cls, loss_bbox, loss_iou, loss_iou_pred

    @staticmethod
    def split_outputs(all_layers_cls_scores: Tensor,
                      all_layers_cls_scores_bbyy: Tensor,
                      all_layers_bbox_preds: Tensor,
                      all_layers_iou_preds: Tensor,
                      dn_meta: Dict[str, int]) -> Tuple[Tensor]:        
        num_denoising_queries = dn_meta['num_denoising_queries']
        if dn_meta is not None:
            all_layers_denoising_cls_scores = \
                all_layers_cls_scores[:, :, : num_denoising_queries, :]
            all_layers_denoising_cls_scores_bbyy = \
                all_layers_cls_scores_bbyy[:, :, : num_denoising_queries, :]
            all_layers_denoising_bbox_preds = \
                all_layers_bbox_preds[:, :, : num_denoising_queries, :]
            all_layers_matching_cls_scores = \
                all_layers_cls_scores[:, :, num_denoising_queries:, :]
            all_layers_matching_cls_scores_bbyy = \
                all_layers_cls_scores_bbyy[:, :, num_denoising_queries:, :]
            all_layers_matching_bbox_preds = \
                all_layers_bbox_preds[:, :, num_denoising_queries:, :]
            # modify
            all_layers_denoising_ious = \
                all_layers_iou_preds[:, :, : num_denoising_queries, :]
            all_layers_matching_ious = \
                all_layers_iou_preds[:, :, num_denoising_queries:, :]
        else:
            all_layers_denoising_cls_scores = None
            all_layers_denoising_cls_scores_bbyy = None
            all_layers_denoising_bbox_preds = None
            all_layers_matching_cls_scores = all_layers_cls_scores
            all_layers_matching_cls_scores_bbyy = all_layers_cls_scores_bbyy
            all_layers_matching_bbox_preds = all_layers_bbox_preds
            # modify
            all_layers_denoising_ious = None
            all_layers_matching_ious = all_layers_iou_preds
        return (all_layers_matching_cls_scores, all_layers_matching_cls_scores_bbyy, all_layers_matching_bbox_preds, all_layers_matching_ious, 
                all_layers_denoising_cls_scores, all_layers_denoising_cls_scores_bbyy, all_layers_denoising_bbox_preds, all_layers_denoising_ious)
    
    
    def loss(self, hidden_states: Tensor, references: List[Tensor],
             memory_text: Tensor, text_token_mask: Tensor,
             enc_outputs_class: Tensor, enc_outputs_coord: Tensor,
             batch_data_samples: SampleList, dn_meta: Dict[str, int]) -> dict:
        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            gt_instances = data_sample.gt_instances
            gt_instances.labels = torch.zeros_like(gt_instances.labels)
            batch_gt_instances.append(gt_instances)

        outs = self(hidden_states, references, memory_text, text_token_mask)
        self.text_masks = text_token_mask
        loss_inputs = outs + (enc_outputs_class, enc_outputs_coord,
                              batch_gt_instances, batch_img_metas, dn_meta)
        losses = self.loss_by_feat(*loss_inputs)
        return losses



# 添加contrastive loss
@MODELS.register_module()
class Mask2FormerHeadAnomaly(Mask2FormerHead):
    """Implements the Mask2Former head.

    See `Mask2Former: Masked-attention Mask Transformer for Universal Image
    Segmentation <https://arxiv.org/abs/2112.01527>`_ for details.

    Args:
        num_classes (int): Number of classes. Default: 150.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        ignore_index (int): The label index to be ignored. Default: 255.
    """

    def __init__(self,
                 loss_contrastive: ConfigType = None, 
                 **kwargs):
        super().__init__(**kwargs)

        self.loss_contrastive = None
        if loss_contrastive is not None:
            self.loss_contrastive = MODELS.build(loss_contrastive)    
    
    def loss_by_feat(self, all_cls_scores: Tensor, all_mask_preds: Tensor,
                     batch_gt_instances: List[InstanceData],
                     batch_img_metas: List[dict]) -> Dict[str, Tensor]:
        """Loss function.

        Args:
            all_cls_scores (Tensor): Classification scores for all decoder
                layers with shape (num_decoder, batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes
                background.
            all_mask_preds (Tensor): Mask scores for all decoder layers with
                shape (num_decoder, batch_size, num_queries, h, w).
            batch_gt_instances (list[obj:`InstanceData`]): each contains
                ``labels`` and ``masks``.
            batch_img_metas (list[dict]): List of image meta information.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_dec_layers = len(all_cls_scores)
        batch_gt_instances_list = [
            batch_gt_instances for _ in range(num_dec_layers)
        ]
        img_metas_list = [batch_img_metas for _ in range(num_dec_layers)]
        losses_cls, losses_mask, losses_dice, losses_contrastive = multi_apply(
            self._loss_by_feat_single, all_cls_scores, all_mask_preds,
            batch_gt_instances_list, img_metas_list)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_mask'] = losses_mask[-1]
        loss_dict['loss_dice'] = losses_dice[-1]
        
        
        if losses_contrastive[0] is not None:
            loss_dict['loss_contrastive'] = losses_contrastive[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_mask_i, loss_dice_i, loss_contrastive in zip(
                losses_cls[:-1], losses_mask[:-1], losses_dice[:-1], losses_contrastive[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_mask'] = loss_mask_i
            loss_dict[f'd{num_dec_layer}.loss_dice'] = loss_dice_i
            if loss_contrastive is not None:
                loss_dict[f'd{num_dec_layer}.loss_contrastive'] = loss_contrastive
            num_dec_layer += 1
        return loss_dict
    
    
    def _loss_by_feat_single(self, cls_scores: Tensor, mask_preds: Tensor,
                             batch_gt_instances: List[InstanceData],
                             batch_img_metas: List[dict]) -> Tuple[Tensor]:
        """Loss function for outputs from a single decoder layer.

        Args:
            cls_scores (Tensor): Mask score logits from a single decoder layer
                for all images. Shape (batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes
                background.
            mask_preds (Tensor): Mask logits for a pixel decoder for all
                images. Shape (batch_size, num_queries, h, w).
            batch_gt_instances (list[obj:`InstanceData`]): each contains
                ``labels`` and ``masks``.
            batch_img_metas (list[dict]): List of image meta information.

        Returns:
            tuple[Tensor]: Loss components for outputs from a single \
                decoder layer.
        """
        
        loss_contrastive = None
        if self.loss_contrastive is not None:
            loss_contrastive = self.loss_contrastive(cls_scores, \
                                                    mask_preds, batch_gt_instances, batch_img_metas)

        loss_cls, loss_mask, loss_dice = super()._loss_by_feat_single(cls_scores, \
                                                    mask_preds, batch_gt_instances, batch_img_metas)

        return loss_cls, loss_mask, loss_dice, loss_contrastive


@MODELS.register_module()
class GroundingDINOPT(GroundingDINO):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._freeze_modules()
        self.sam = SamModel.from_pretrained("./sam-vit-base")
        self.processor = SamProcessor.from_pretrained("./sam-vit-base")
        for p in self.sam.parameters():
            p.requires_grad = False
    
    def sam_predict_hf(self, raw_image, boxes):
        raw_image = Image.open(raw_image).convert("RGB")
        inputs = self.processor(raw_image, return_tensors="pt").to(self.sam.device)
        image_embeddings = self.sam.get_image_embeddings(inputs["pixel_values"])
        inputs = self.processor(raw_image, input_boxes=[boxes.cpu().numpy().tolist()], return_tensors="pt").to(self.sam.device)
        inputs.pop("pixel_values", None)
        inputs.update({"image_embeddings": image_embeddings})
        with torch.no_grad():
            outputs = self.sam(**inputs)
        masks = self.processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
        return masks

    def predict(self, batch_inputs, batch_data_samples, rescale: bool = True):
        text_prompts = []
        enhanced_text_prompts = []
        tokens_positives = []
        for data_samples in batch_data_samples:
            text_prompts.append(data_samples.text)
            if 'caption_prompt' in data_samples:
                enhanced_text_prompts.append(data_samples.caption_prompt)
            else:
                enhanced_text_prompts.append(None)
            tokens_positives.append(data_samples.get('tokens_positive', None))

        if 'custom_entities' in batch_data_samples[0]:
            # Assuming that the `custom_entities` flag
            # inside a batch is always the same. For single image inference
            custom_entities = batch_data_samples[0].custom_entities
        else:
            custom_entities = False
        if len(text_prompts) == 1:
            # All the text prompts are the same,
            # so there is no need to calculate them multiple times.
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(
                    text_prompts[0], custom_entities, enhanced_text_prompts[0],
                    tokens_positives[0])
            ] * len(batch_inputs)
        else:
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(text_prompt,
                                                     custom_entities,
                                                     enhanced_text_prompt,
                                                     tokens_positive)
                for text_prompt, enhanced_text_prompt, tokens_positive in zip(
                    text_prompts, enhanced_text_prompts, tokens_positives)
            ]
        token_positive_maps, text_prompts, _, entities = zip(
            *_positive_maps_and_prompts)

        # image feature extraction
        visual_feats = self.extract_feat(batch_inputs)

        if isinstance(text_prompts[0], list):
            # chunked text prompts, only bs=1 is supported
            assert len(batch_inputs) == 1
            count = 0
            results_list = []

            entities = [[item for lst in entities[0] for item in lst]]

            for b in range(len(text_prompts[0])):
                text_prompts_once = [text_prompts[0][b]]
                token_positive_maps_once = token_positive_maps[0][b]
                text_dict = self.language_model(text_prompts_once)
                # text feature map layer
                if self.text_feat_map is not None:
                    text_dict['embedded'] = self.text_feat_map(
                        text_dict['embedded'])

                batch_data_samples[
                    0].token_positive_map = token_positive_maps_once

                head_inputs_dict = self.forward_transformer(
                    copy.deepcopy(visual_feats), text_dict, batch_data_samples)
                pred_instances = self.bbox_head.predict(
                    **head_inputs_dict,
                    rescale=rescale,
                    batch_data_samples=batch_data_samples)[0]

                if len(pred_instances) > 0:
                    pred_instances.labels += count
                count += len(token_positive_maps_once)
                results_list.append(pred_instances)
            results_list = [results_list[0].cat(results_list)]
            is_rec_tasks = [False] * len(results_list)
        else:
            # extract text feats
            text_dict = self.language_model(list(text_prompts))
            # text feature map layer
            if self.text_feat_map is not None:
                text_dict['embedded'] = self.text_feat_map(
                    text_dict['embedded'])

            is_rec_tasks = []
            for i, data_samples in enumerate(batch_data_samples):
                if token_positive_maps[i] is not None:
                    is_rec_tasks.append(False)
                else:
                    is_rec_tasks.append(True)
                data_samples.token_positive_map = token_positive_maps[i]

            head_inputs_dict = self.forward_transformer(
                visual_feats, text_dict, batch_data_samples)
            results_list = self.bbox_head.predict(
                **head_inputs_dict,
                rescale=rescale,
                batch_data_samples=batch_data_samples)

        for data_sample, pred_instances, entity, is_rec_task in zip(
                batch_data_samples, results_list, entities, is_rec_tasks):
            if len(pred_instances) > 0:
                label_names = []
                for labels in pred_instances.labels:
                    if is_rec_task:
                        label_names.append(entity)
                        continue
                    if labels >= len(entity):
                        warnings.warn(
                            'The unexpected output indicates an issue with '
                            'named entity recognition. You can try '
                            'setting custom_entities=True and running '
                            'again to see if it helps.')
                        label_names.append('unobject')
                    else:
                        label_names.append(entity[labels])
                # for visualization
                pred_instances.label_names = label_names
            data_sample.pred_instances = pred_instances
            data_sample.pred_instances = data_sample.pred_instances[(data_sample.pred_instances.scores > 0.2) & (data_sample.pred_instances.labels == 0)]
        
        
        for input_img, data_sample in zip(batch_inputs, batch_data_samples):
            masks = torch.zeros_like(input_img[:1])
            ori_shape = data_sample.metainfo['ori_shape']
            if len(data_sample.pred_instances) > 0:
                masks = self.sam_predict_hf(data_sample.metainfo['img_path'], data_sample.pred_instances.bboxes)
                masks = masks[0][:, 0].bool().float()
            masks = F.interpolate(masks.unsqueeze(1), size=(ori_shape[0], ori_shape[1]), mode='bilinear').to(torch.int32)
            data_sample.set_data({
                'pred_sem_seg':
                PixelData(**{'sem_seg': masks.sum(dim=0).bool().float()}),
                # 'seg_logits':
                # PixelData(**{'data': seg_logits_ori_shape.squeeze(0)}),
                'pred_masks':
                PixelData(**{'sem_seg': masks.squeeze(1)}),
            })

        return batch_data_samples

    def _freeze_modules(self):  
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        for p in self.memory_trans_fc.parameters():
            p.requires_grad_(False)
        for p in self.memory_trans_norm.parameters():
            p.requires_grad_(False)
        for p in self.query_embedding.parameters():
            p.requires_grad_(False)
        for p in self.neck.parameters():
            p.requires_grad_(False)

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.encoder = GroundingDinoTransformerEncoder(**self.encoder)
        self.decoder = GroundingDinoTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)
        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

        # text modules
        self.language_model = MODELS.build(self.language_model_cfg)
        self.text_feat_map = nn.Linear(
            self.language_model.language_backbone.body.language_dim,
            self.embed_dims,
            bias=True)
        
        # modify
        # =============================================================

        # self.decoder = GroundingDinoTransformerDecoderPT(**self.decoder)

        self.num_queries_bbyy = 300
        self.query_embedding_bbyy = nn.Embedding(self.num_queries_bbyy, self.embed_dims)
        self.bbox_head.num_queries = self.num_queries
        self.bbox_head.num_queries_bbyy = self.num_queries_bbyy
        self.decoder.num_queries = self.num_queries
        self.decoder.num_queries_bbyy = self.num_queries_bbyy
        self.bbyy_embedding = nn.Embedding(1, 256)
        self.memory_trans_fc_bbyy = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm_bbyy = nn.LayerNorm(self.embed_dims)
        # =============================================================
    

    def forward_transformer(
        self,
        img_feats: Tuple[Tensor],
        text_dict: Dict,
        batch_data_samples: OptSampleList = None,
    ) -> Dict:
        
        # modify
        text_dict['embedded'][:, 1:2] = text_dict['embedded'][:, 1:2] + self.bbyy_embedding.weight.unsqueeze(0)
        
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)

        encoder_outputs_dict = self.forward_encoder(
            **encoder_inputs_dict, text_dict=text_dict)

        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        return head_inputs_dict

    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        memory_text: Tensor,
        text_token_mask: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        bs, _, c = memory.shape

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)

        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers](output_memory, memory_text,
                                     text_token_mask)
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers].max_text_len
        enc_outputs_coord_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers](output_memory) + output_proposals

        # NOTE The DINO selects top-k proposals according to scores of
        # multi-class classification, while DeformDETR, where the input
        # is `enc_outputs_class[..., 0]` selects according to scores of
        # binary classification.
        topk_indices = torch.topk(
            enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1)[1]

        topk_score = torch.gather(
            enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()
        
        # modify
        # =============================================================
        output_memory_bbyy, output_proposals_bbyy = self.gen_encoder_output_proposals_bbyy(
            memory, memory_mask, spatial_shapes)
        enc_outputs_class_bbyy = self.bbox_head.cls_branches_bbyy[
                        self.decoder.num_layers](output_memory_bbyy)
        topk_indices_bbyy = torch.topk(
            enc_outputs_class_bbyy.max(-1)[0], k=self.num_queries_bbyy, dim=1)[1]

        topk_score_bbyy = torch.gather(
            enc_outputs_class_bbyy, 1,
            topk_indices_bbyy.unsqueeze(-1).repeat(1, 1, 1))
        topk_coords_unact_bbyy = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices_bbyy.unsqueeze(-1).repeat(1, 1, 4))
        topk_coords_bbyy = topk_coords_unact_bbyy.sigmoid()
        topk_coords_unact_bbyy = topk_coords_unact_bbyy.detach()
        # =============================================================

        # modify
        # =============================================================
        query = torch.cat((self.query_embedding.weight, self.query_embedding_bbyy.weight), dim=0)[:, None, :]
        query = query.repeat(1, bs, 1).transpose(0, 1)
        if self.training:
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                self.dn_query_generator(batch_data_samples)
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact, topk_coords_unact_bbyy],
                                         dim=1)
            dn_mask_extend = dn_mask.new_zeros((query.size(1), query.size(1)))
            dn_mask_extend[:dn_mask.size(0), :dn_mask.size(1)] = dn_mask
            dn_mask_extend[dn_label_query.size(1):-self.num_queries_bbyy, -self.num_queries_bbyy:] = True
            dn_mask_extend[-self.num_queries_bbyy:, :-self.num_queries_bbyy] = True
        else:
            reference_points = torch.cat([topk_coords_unact, topk_coords_unact_bbyy], dim=1)
            dn_mask, dn_meta = None, None
            dn_mask_extend = torch.zeros((query.size(1), query.size(1)), device=query.device, dtype=torch.bool)
            dn_mask_extend[:self.num_queries, self.num_queries:] = True
            dn_mask_extend[self.num_queries:, :self.num_queries] = True
        
        
        query_text_mask = dn_mask_extend.new_zeros((query.size(1), text_token_mask.size(1)))
        query_text_mask[-self.num_queries_bbyy:, 3:] = True
        dn_mask = dn_mask_extend
        topk_score = topk_score_bbyy
        topk_coords = topk_coords_bbyy
        # topk_score = torch.cat((topk_score, topk_score_bbyy), dim=1)
        # topk_coords = torch.cat((topk_coords, topk_coords_bbyy), dim=1)
        # =============================================================
        reference_points = reference_points.sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
            reference_points=reference_points,
            dn_mask=dn_mask,
            memory_text=memory_text,
            text_attention_mask=~text_token_mask,
            query_text_mask=query_text_mask
        )
        # NOTE DINO calculates encoder losses on scores and coordinates
        # of selected top-k encoder queries, while DeformDETR is of all
        # encoder queries.
        head_inputs_dict = dict(
            enc_outputs_class=topk_score,
            enc_outputs_coord=topk_coords,
            dn_meta=dn_meta) if self.training else dict()
        # append text_feats to head_inputs_dict
        head_inputs_dict['memory_text'] = memory_text
        head_inputs_dict['text_token_mask'] = text_token_mask
        return decoder_inputs_dict, head_inputs_dict

    def gen_encoder_output_proposals_bbyy(
            self, memory: Tensor, memory_mask: Tensor,
            spatial_shapes: Tensor) -> Tuple[Tensor, Tensor]:
        bs = memory.size(0)
        proposals = []
        _cur = 0  # start index in the sequence of the current level
        for lvl, HW in enumerate(spatial_shapes):
            H, W = HW

            if memory_mask is not None:
                mask_flatten_ = memory_mask[:, _cur:(_cur + H * W)].view(
                    bs, H, W, 1)
                valid_H = torch.sum(~mask_flatten_[:, :, 0, 0],
                                    1).unsqueeze(-1)
                valid_W = torch.sum(~mask_flatten_[:, 0, :, 0],
                                    1).unsqueeze(-1)
                scale = torch.cat([valid_W, valid_H], 1).view(bs, 1, 1, 2)
            else:
                if not isinstance(HW, torch.Tensor):
                    HW = memory.new_tensor(HW)
                scale = HW.unsqueeze(0).flip(dims=[0, 1]).view(1, 1, 1, 2)
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(
                    0, H - 1, H, dtype=torch.float32, device=memory.device),
                torch.linspace(
                    0, W - 1, W, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
            grid = (grid.unsqueeze(0).expand(bs, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)
            proposal = torch.cat((grid, wh), -1).view(bs, -1, 4)
            proposals.append(proposal)
            _cur += (H * W)
        output_proposals = torch.cat(proposals, 1)
        # do not use `all` to make it exportable to onnx
        output_proposals_valid = (
            (output_proposals > 0.01) & (output_proposals < 0.99)).sum(
                -1, keepdim=True) == output_proposals.shape[-1]
        # inverse_sigmoid
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        if memory_mask is not None:
            output_proposals = output_proposals.masked_fill(
                memory_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(
            ~output_proposals_valid, float('inf'))

        output_memory = memory
        if memory_mask is not None:
            output_memory = output_memory.masked_fill(
                memory_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid,
                                                  float(0))
        output_memory = self.memory_trans_fc_bbyy(output_memory)
        output_memory = self.memory_trans_norm_bbyy(output_memory)
        # [bs, sum(hw), 2]
        return output_memory, output_proposals


class GroundingDinoTransformerDecoderPT(GroundingDinoTransformerDecoder):

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([
            GroundingDinoTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             f'{self._get_name()}')
        self.ref_point_head = MLP(self.embed_dims * 2, self.embed_dims,
                                  self.embed_dims, 2)
        self.norm = nn.LayerNorm(self.embed_dims)

        # modify
        # =============================================================
        self.bbyy_layer1 = GroundingDinoTransformerDecoderLayer(**self.layer_cfg)
        # self.bbyy_layer2 = GroundingDinoTransformerDecoderLayer(**self.layer_cfg)
        # =============================================================

    def forward(self, query: Tensor, value: Tensor, key_padding_mask: Tensor,
                self_attn_mask: Tensor, reference_points: Tensor,
                spatial_shapes: Tensor, level_start_index: Tensor,
                valid_ratios: Tensor, reg_branches: nn.ModuleList,
                **kwargs) -> Tuple[Tensor]:
        intermediate = []
        intermediate_reference_points = [reference_points]

        # modify
        # =============================================================
        query, query_bbyy = query.split((query.size(1) - self.num_queries_bbyy, self.num_queries_bbyy), dim=1)
        reference_points_bbyy = reference_points[:, -self.num_queries_bbyy:]
        if reference_points_bbyy.shape[-1] == 4:
                reference_points_input = \
                    reference_points_bbyy[:, :, None] * torch.cat(
                        [valid_ratios, valid_ratios], -1)[:, None]
        else:
            assert reference_points_bbyy.shape[-1] == 2
            reference_points_input = \
                reference_points_bbyy[:, :, None] * valid_ratios[:, None]

        query_sine_embed = coordinate_to_encoding(
            reference_points_input[:, :, 0, :])
        query_pos = self.ref_point_head(query_sine_embed)

        query_bbyy = self.bbyy_layer1(
            query_bbyy,
            query_pos=query_pos,
            value=value,
            key_padding_mask=key_padding_mask,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reference_points=reference_points_input,
            **kwargs)

        query = torch.cat((query, query_bbyy), dim=1)
        # =============================================================

        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = \
                    reference_points[:, :, None] * torch.cat(
                        [valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = \
                    reference_points[:, :, None] * valid_ratios[:, None]

            query_sine_embed = coordinate_to_encoding(
                reference_points_input[:, :, 0, :])
            query_pos = self.ref_point_head(query_sine_embed)

            query = layer(
                query,
                query_pos=query_pos,
                value=value,
                key_padding_mask=key_padding_mask,
                self_attn_mask=self_attn_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points_input,
                **kwargs)

            if reg_branches is not None:
                tmp = reg_branches[lid](query)
                assert reference_points.shape[-1] == 4
                new_reference_points = tmp + inverse_sigmoid(
                    reference_points, eps=1e-3)
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(self.norm(query))
                intermediate_reference_points.append(new_reference_points)
                # NOTE this is for the "Look Forward Twice" module,
                # in the DeformDETR, reference_points was appended.

        # modify
        # =============================================================
        # query, query_bbyy = query.split((query.size(1) - self.num_queries_bbyy, self.num_queries_bbyy), dim=1)
        # reference_points_bbyy = reference_points[:, -self.num_queries_bbyy:]
        # if reference_points_bbyy.shape[-1] == 4:
        #         reference_points_input = \
        #             reference_points_bbyy[:, :, None] * torch.cat(
        #                 [valid_ratios, valid_ratios], -1)[:, None]
        # else:
        #     assert reference_points_bbyy.shape[-1] == 2
        #     reference_points_input = \
        #         reference_points_bbyy[:, :, None] * valid_ratios[:, None]

        # query_sine_embed = coordinate_to_encoding(
        #     reference_points_input[:, :, 0, :])
        # query_pos = self.ref_point_head(query_sine_embed)

        # query_bbyy = self.bbyy_layer2(
        #     query_bbyy,
        #     query_pos=query_pos,
        #     value=value,
        #     key_padding_mask=key_padding_mask,
        #     spatial_shapes=spatial_shapes,
        #     level_start_index=level_start_index,
        #     valid_ratios=valid_ratios,
        #     reference_points=reference_points_input,
        #     **kwargs)

        # query = torch.cat((query, query_bbyy), dim=1)
        # =============================================================

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return query, reference_points


@MODELS.register_module()
class GroundingDINOPTSeg(GroundingDINOPT):
    def __init__(self, 
                 seg_decoder: OptConfigType = None,
                 roi_head: OptConfigType = None,
                 sam: OptConfigType = None,
                 loss_contrastive: ConfigType = None,
                 **kwargs):
        self.seg_decoder = seg_decoder
        self.roi_head = roi_head
        self.sam = sam
        
        if loss_contrastive is not None:
            self.loss_contrastive = MODELS.build(loss_contrastive)

        super().__init__(**kwargs)

        self.neck_seg = MyNeck([128, 256, 512, 1024])

        # self.neck_seg = MODELS.build(dict(
        #                     type='ChannelMapper',
        #                     in_channels=[128, 256, 512, 1024],
        #                     kernel_size=1,
        #                     out_channels=256,
        #                     act_cfg=None,
        #                     bias=True,
        #                     norm_cfg=dict(type='GN', num_groups=32),
        #                     num_outs=4),)


        self._freeze_modules()
        

    def _freeze_modules(self):
        for m in self.backbone.parameters():
            m.requires_grad = False
        for n, p in self.neck.named_parameters():
            if '3' not in n:
                p.requires_grad = False
        for m in self.encoder.parameters():
            m.requires_grad = False
        for n, p in self.decoder.named_parameters():
            p.requires_grad = False
        for m in self.bbox_head.parameters():
            m.requires_grad = False
        for m in self.dn_query_generator.parameters():
            m.requires_grad = False

        for m in self.query_embedding.parameters():
            m.requires_grad = False
        for m in self.query_embedding_bbyy.parameters():
            m.requires_grad = False
        for m in self.bbyy_embedding.parameters():
            m.requires_grad = False
        self.level_embed.requires_grad = False
        for m in self.memory_trans_fc.parameters():
            m.requires_grad = False
        for m in self.memory_trans_norm.parameters():
            m.requires_grad = False
        for m in self.memory_trans_fc_bbyy.parameters():
            m.requires_grad = False
        for m in self.memory_trans_norm_bbyy.parameters():
            m.requires_grad = False
        for m in self.language_model.parameters():
            m.requires_grad = False
        for m in self.text_feat_map.parameters():
            m.requires_grad = False

    def _init_layers(self) -> None:
        super()._init_layers()
        
        if self.seg_decoder is not None:
            self.seg_decoder = MODELS.build(self.seg_decoder)
            self.align_corners = self.seg_decoder.align_corners
        if self.roi_head is not None:
            self.roi_head = MODELS.build(self.roi_head)
        if self.sam is not None:

            self.sam = SamModel.from_pretrained("./sam-vit-base")
            self.processor = SamProcessor.from_pretrained("./sam-vit-base")

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        x = self.backbone(batch_inputs) # [4, 8, 16, 32]
        if self.with_neck:
            x_uc = self.neck(x[1:])

        x = self.neck_seg(x)

        return x, x_uc


    def predict(self, batch_inputs, batch_data_samples, rescale: bool = True):
        
        text_prompts = []
        enhanced_text_prompts = []
        tokens_positives = []
        for data_samples in batch_data_samples:
            text_prompts.append(data_samples.text)
            if 'caption_prompt' in data_samples:
                enhanced_text_prompts.append(data_samples.caption_prompt)
            else:
                enhanced_text_prompts.append(None)
            tokens_positives.append(data_samples.get('tokens_positive', None))

        if 'custom_entities' in batch_data_samples[0]:
            # Assuming that the `custom_entities` flag
            # inside a batch is always the same. For single image inference
            custom_entities = batch_data_samples[0].custom_entities
        else:
            custom_entities = False
        if len(text_prompts) == 1:
            # All the text prompts are the same,
            # so there is no need to calculate them multiple times.
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(
                    text_prompts[0], custom_entities, enhanced_text_prompts[0],
                    tokens_positives[0])
            ] * len(batch_inputs)
        else:
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(text_prompt,
                                                     custom_entities,
                                                     enhanced_text_prompt,
                                                     tokens_positive)
                for text_prompt, enhanced_text_prompt, tokens_positive in zip(
                    text_prompts, enhanced_text_prompts, tokens_positives)
            ]
        token_positive_maps, text_prompts, _, entities = zip(
            *_positive_maps_and_prompts)

        # image feature extraction
        backbone_feats, visual_feats = self.extract_feat(batch_inputs)
        
        batch_img_metas = [data_sample.metainfo for data_sample in batch_data_samples]
        seg_logits = self.seg_decoder.predict(backbone_feats, batch_img_metas, None)
        ori_shape = batch_img_metas[0]['ori_shape']
        seg_logits_ori_shape = F.interpolate(seg_logits, ori_shape, mode='bilinear', align_corners=False)
        seg_preds = seg_logits_ori_shape.argmax(dim=1)


        # batch_data_samples =  self.postprocess_result(seg_logits, batch_data_samples)

        # return batch_data_samples

        # anomaly_scores = -torch.max(seg_logits_ori_shape[:, :19], dim=1)[0].unsqueeze(1)
        anomaly_scores = -torch.sum(seg_logits_ori_shape[:, :19].tanh(), dim=1).unsqueeze(1)    # RbA anomaly score

        if isinstance(text_prompts[0], list):
            # chunked text prompts, only bs=1 is supported
            assert len(batch_inputs) == 1
            count = 0
            results_list = []

            entities = [[item for lst in entities[0] for item in lst]]

            for b in range(len(text_prompts[0])):
                text_prompts_once = [text_prompts[0][b]]
                token_positive_maps_once = token_positive_maps[0][b]
                text_dict = self.language_model(text_prompts_once)
                # text feature map layer
                if self.text_feat_map is not None:
                    text_dict['embedded'] = self.text_feat_map(
                        text_dict['embedded'])

                batch_data_samples[
                    0].token_positive_map = token_positive_maps_once

                head_inputs_dict = self.forward_transformer(
                    copy.deepcopy(visual_feats), text_dict, batch_data_samples)
                pred_instances = self.bbox_head.predict(
                    **head_inputs_dict,
                    rescale=rescale,
                    batch_data_samples=batch_data_samples)[0]

                if len(pred_instances) > 0:
                    pred_instances.labels += count
                count += len(token_positive_maps_once)
                results_list.append(pred_instances)
            results_list = [results_list[0].cat(results_list)]
            is_rec_tasks = [False] * len(results_list)
        else:
            # extract text feats
            text_dict = self.language_model(list(text_prompts))
            # text feature map layer
            if self.text_feat_map is not None:
                text_dict['embedded'] = self.text_feat_map(
                    text_dict['embedded'])

            is_rec_tasks = []
            for i, data_samples in enumerate(batch_data_samples):
                if token_positive_maps[i] is not None:
                    is_rec_tasks.append(False)
                else:
                    is_rec_tasks.append(True)
                data_samples.token_positive_map = token_positive_maps[i]

            head_inputs_dict = self.forward_transformer(
                visual_feats, text_dict, batch_data_samples)
            results_list = self.bbox_head.predict(
                **head_inputs_dict,
                rescale=rescale,
                batch_data_samples=batch_data_samples)

        for data_sample, pred_instances, entity, is_rec_task, seg_pred, anomaly_score, img_metas in zip(
                batch_data_samples, results_list, entities, is_rec_tasks, seg_preds, anomaly_scores, batch_img_metas):
            if len(pred_instances) > 0:
                label_names = []
                for labels in pred_instances.labels:
                    if is_rec_task:
                        label_names.append(entity)
                        continue
                    if labels >= len(entity):
                        warnings.warn(
                            'The unexpected output indicates an issue with '
                            'named entity recognition. You can try '
                            'setting custom_entities=True and running '
                            'again to see if it helps.')
                        label_names.append('unobject')
                    else:
                        label_names.append(entity[labels])
                # for visualization
                pred_instances.label_names = label_names
            data_sample.pred_instances = pred_instances

            labels = data_sample.pred_instances.labels
            scores = data_sample.pred_instances.scores
            bboxes = data_sample.pred_instances.bboxes

            mask_id = torch.ones(ori_shape).to(batch_inputs.device)
            mask_road = torch.ones(ori_shape).to(batch_inputs.device)
            bboxes_id_mask = (torch.isin(labels, torch.arange(18, 20).to(batch_inputs.device)) & (scores > 0.8)) | \
                            (torch.isin(labels, torch.arange(3, 12).to(batch_inputs.device)) & (scores > 0.5)) | \
                            (torch.isin(labels, torch.arange(12, 18).to(batch_inputs.device)) & (scores > 0.5))
            bboxes_id = bboxes[bboxes_id_mask]
            bboxes_road = bboxes[scores > 0.2][torch.isin(labels[scores > 0.2], torch.arange(3).to(batch_inputs.device))].int()

            y, x = torch.meshgrid(torch.arange(ori_shape[0], device=batch_inputs.device), 
                                    torch.arange(ori_shape[1], device=batch_inputs.device),
                                    indexing='ij')
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
            bboxes_id = bboxes_id.unsqueeze(1).unsqueeze(1)
            bboxes_road = bboxes_road.unsqueeze(1).unsqueeze(1)
            
            mask_id = mask_id * ((x >= bboxes_id[..., 0]) & (x < bboxes_id[..., 2]) & (y >= bboxes_id[..., 1]) & (y < bboxes_id[..., 3])).any(dim=0)
            mask_road = mask_road * ((x >= bboxes_road[..., 0]) & (x < bboxes_road[..., 2]) & (y >= bboxes_road[..., 1]) & (y < bboxes_road[..., 3])).any(dim=0)
            # seg_pred = seg_pred * mask_id
            mask_road = mask_road * torch.isin(seg_pred, torch.arange(0, 2).to(batch_inputs.device))
            # mask_road = mask_road.cpu().numpy()
            # cv2.floodFill(mask_road, None, (0, 0), 1)
            # mask_road = torch.from_numpy(mask_road).to(batch_inputs.device)

            bbox_road_overlap = self.roi_head([mask_road.unsqueeze(0).unsqueeze(0)], [data_sample.pred_instances], [data_sample], False)
            bbox_road_overlap = bbox_road_overlap.view(len(results_list), -1, *bbox_road_overlap.shape[2:])
            bboxes[:, 0].clamp_(0, ori_shape[1] - 1)
            bboxes[:, 1].clamp_(0, ori_shape[0] - 1)
            bboxes[:, 2].clamp_(0, ori_shape[1] - 1)
            bboxes[:, 3].clamp_(0, ori_shape[0] - 1)
            bboxes_mask = (bbox_road_overlap[0].mean(dim=-1).mean(dim=-1).flatten() > 0.4) | (mask_road[bboxes.int()[:, 1], bboxes.int()[:, 0]].bool() & mask_road[bboxes.int()[:, 3], bboxes.int()[:, 2]].bool())
            # data_sample.pred_instances = data_sample.pred_instances[bboxes_mask]

            data_sample.pred_instances = data_sample.pred_instances[data_sample.pred_instances.labels == 0]


            
            bbox_anomaly_score = self.roi_head([anomaly_score.unsqueeze(0)], [data_sample.pred_instances], [data_sample], False)
            bbox_anomaly_score = bbox_anomaly_score.view(len(results_list), -1, *bbox_anomaly_score.shape[2:])[0].mean(dim=-1).mean(dim=-1).flatten()
            bboxes = data_sample.pred_instances.bboxes
            scores = data_sample.pred_instances.scores
                # data_sample.pred_instances.scores[(mask_road[bboxes.int()[:, 1], bboxes.int()[:, 0]].bool() & 
                #                                     mask_road[bboxes.int()[:, 3], bboxes.int()[:, 2]].bool() &
                #                                     mask_road[bboxes.int()[:, 3], bboxes.int()[:, 0]].bool() &
                #                                     mask_road[bboxes.int()[:, 1], bboxes.int()[:, 2]].bool())] = 0.9

            # data_sample.pred_instances = data_sample.pred_instances[(bbox_anomaly_score > -0.7) & (scores > 0.3)]
            # data_sample.pred_instances = data_sample.pred_instances[(scores > 0.3)]
            # data_sample.pred_instances.scores = bbox_anomaly_score[scores > 0.3]
            # data_sample.pred_instances.scores = torch.maximum(data_sample.pred_instances.scores, 1 + bbox_anomaly_score)


            # data_sample.pred_instances = data_sample.pred_instances[(data_sample.pred_instances.scores > 0.1) & (1 + bbox_anomaly_score > 0.8)]
            data_sample.pred_instances = data_sample.pred_instances[(data_sample.pred_instances.scores > 0.2)]

            # anomaly_score = F.interpolate(anomaly_score.unsqueeze(0), ori_shape, mode='bilinear', align_corners=False).squeeze(0)
            # print(anomaly_score.max(), anomaly_score.min())
            # print(anomaly_score.shape, ori_shape)

            bboxes_anomaly = data_sample.pred_instances.bboxes.unsqueeze(1).unsqueeze(1)
            objectness = torch.ones(ori_shape).to(batch_inputs.device) * 0.1
            objectness[((x >= bboxes_anomaly[..., 0]) & (x < bboxes_anomaly[..., 2]) & (y >= bboxes_anomaly[..., 1]) & (y < bboxes_anomaly[..., 3])).any(dim=0)] = 1
            anomaly_score = anomaly_score + objectness

            data_sample.set_data({
                'anomaly_scores':
                PixelData(**{'data': anomaly_score.squeeze(0)}),
                'pred_masks':
                PixelData(**{'sem_seg': anomaly_score.squeeze(0)})
            })


            # data_sample.pred_instances = data_sample.pred_instances[(data_sample.pred_instances.scores > 0.38)]
            # data_sample.pred_instances = data_sample.pred_instances[(data_sample.pred_instances.labels == 0)]


            # if len(data_sample.pred_instances) > 0:
            #     masks = self.sam_predict_hf(data_sample.metainfo['img_path'], data_sample.pred_instances.bboxes)
            #     masks = masks[0][:, 0].bool().float().to(anomaly_score.device)
            #     for s, m in zip(data_sample.pred_instances.scores, masks):
            #         anomaly_score += m.unsqueeze(0)


            data_sample.set_data({
                'anomaly_scores':
                PixelData(**{'data': anomaly_score.squeeze(0)}),
            })


            # all_masks = []
            # for input_img, data_sample in zip(batch_inputs, batch_data_samples):
            #     masks = torch.zeros_like(input_img[:1])
            #     # input_img = F.interpolate(input_img.unsqueeze(0), size=(1024,1024), mode='bilinear')
            #     if len(data_sample.pred_instances) > 0:
            #         # masks = self.sam_predict(input_img, data_sample.pred_instances.bboxes, ori_shape).to(torch.float32)
            #         masks = self.sam_predict_hf(data_sample.metainfo['img_path'], data_sample.pred_instances.bboxes)
            #         # masks = masks[0][:, 0].sum(dim=0).unsqueeze(0).bool().float()
            #         masks = masks[0][:, 0].bool().float()
            #     masks = F.interpolate(masks.unsqueeze(1), size=(ori_shape[0], ori_shape[1]), mode='bilinear').to(torch.int32)
            #     data_sample.set_data({
            #         'pred_sem_seg':
            #         PixelData(**{'sem_seg': masks.sum(dim=0).bool().float()}),
            #         'seg_logits':
            #         PixelData(**{'data': seg_logits_ori_shape.squeeze(0)}),
            #         'pred_masks':
            #         PixelData(**{'sem_seg': masks.squeeze(1)}),
            #     })
            #     all_masks.append(masks)
            # all_masks = torch.stack(all_masks)


            # data_sample.set_data({
            #     'seg_logits':
            #     PixelData(**{'data': torch.stack((mask_road.squeeze(), 1-mask_road.squeeze()))}),
            #     'pred_sem_seg':
            #     PixelData(**{'sem_seg': mask_road.squeeze()})
            # })
            
        return batch_data_samples

    def postprocess_result(self,
                           seg_logits: Tensor,
                           data_samples: OptSampleList = None) -> SampleList:
        batch_size, C, H, W = seg_logits.shape

        if data_samples is None:
            data_samples = [SegDataSample() for _ in range(batch_size)]
            only_prediction = True
        else:
            only_prediction = False

        for i in range(batch_size):
            if not only_prediction:
                img_meta = data_samples[i].metainfo
                # remove padding area
                if 'img_padding_size' not in img_meta:
                    padding_size = img_meta.get('padding_size', [0] * 4)
                else:
                    padding_size = img_meta['img_padding_size']
                padding_left, padding_right, padding_top, padding_bottom =\
                    padding_size
                # i_seg_logits shape is 1, C, H, W after remove padding
                i_seg_logits = seg_logits[i:i + 1, :,
                                          padding_top:H - padding_bottom,
                                          padding_left:W - padding_right]

                flip = img_meta.get('flip', None)
                if flip:
                    flip_direction = img_meta.get('flip_direction', None)
                    assert flip_direction in ['horizontal', 'vertical']
                    if flip_direction == 'horizontal':
                        i_seg_logits = i_seg_logits.flip(dims=(3, ))
                    else:
                        i_seg_logits = i_seg_logits.flip(dims=(2, ))

                # resize as original shape
                i_seg_logits = resize(
                    i_seg_logits,
                    size=img_meta['ori_shape'],
                    mode='bilinear',
                    align_corners=self.align_corners,
                    warning=False).squeeze(0)
            else:
                i_seg_logits = seg_logits[i]

            if C > 1:
                i_seg_pred = i_seg_logits.argmax(dim=0, keepdim=True)
            else:
                i_seg_logits = i_seg_logits.sigmoid()
                i_seg_pred = (i_seg_logits >
                              self.decode_head.threshold).to(i_seg_logits)
            data_samples[i].set_data({
                'seg_logits':
                PixelData(**{'data': i_seg_logits}),
                'pred_sem_seg':
                PixelData(**{'sem_seg': i_seg_pred})
            })

        return data_samples

    def sam_predict_hf(self, raw_image, boxes):
        raw_image = Image.open(raw_image).convert("RGB")
        inputs = self.processor(raw_image, return_tensors="pt").to(self.sam.device)
        image_embeddings = self.sam.get_image_embeddings(inputs["pixel_values"])
        inputs = self.processor(raw_image, input_boxes=[boxes.cpu().numpy().tolist()], return_tensors="pt").to(self.sam.device)
        inputs.pop("pixel_values", None)
        inputs.update({"image_embeddings": image_embeddings})
        with torch.no_grad():
            outputs = self.sam(**inputs)
        masks = self.processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
        return masks


    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        text_prompts = [
            data_samples.text for data_samples in batch_data_samples
        ]

        gt_labels = [
            data_samples.gt_instances.labels
            for data_samples in batch_data_samples
        ]

        if 'tokens_positive' in batch_data_samples[0]:
            tokens_positive = [
                data_samples.tokens_positive
                for data_samples in batch_data_samples
            ]
            positive_maps = []
            for token_positive, text_prompt, gt_label in zip(
                    tokens_positive, text_prompts, gt_labels):
                tokenized = self.language_model.tokenizer(
                    [text_prompt],
                    padding='max_length'
                    if self.language_model.pad_to_max else 'longest',
                    return_tensors='pt')
                new_tokens_positive = [
                    token_positive[label.item()] for label in gt_label
                ]
                _, positive_map = self.get_positive_map(
                    tokenized, new_tokens_positive)
                positive_maps.append(positive_map)
            new_text_prompts = text_prompts
        else:
            new_text_prompts = []
            positive_maps = []
            if len(set(text_prompts)) == 1:
                # All the text prompts are the same,
                # so there is no need to calculate them multiple times.
                tokenized, caption_string, tokens_positive, _ = \
                    self.get_tokens_and_prompts(
                        text_prompts[0], True)
                new_text_prompts = [caption_string] * len(batch_inputs)
                for gt_label in gt_labels:
                    new_tokens_positive = [
                        tokens_positive[label] for label in gt_label
                    ]
                    _, positive_map = self.get_positive_map(
                        tokenized, new_tokens_positive)
                    positive_maps.append(positive_map)
            else:
                for text_prompt, gt_label in zip(text_prompts, gt_labels):
                    tokenized, caption_string, tokens_positive, _ = \
                        self.get_tokens_and_prompts(
                            text_prompt, True)
                    new_tokens_positive = [
                        tokens_positive[label] for label in gt_label
                    ]
                    
                    _, positive_map = self.get_positive_map(
                        tokenized, new_tokens_positive)
                    positive_maps.append(positive_map)
                    new_text_prompts.append(caption_string)
        
        text_dict = self.language_model(new_text_prompts)
        if self.text_feat_map is not None:
            text_dict['embedded'] = self.text_feat_map(text_dict['embedded'])

        for i, data_samples in enumerate(batch_data_samples):
            positive_map = positive_maps[i].to(
                batch_inputs.device).bool().float()
            text_token_mask = text_dict['text_token_mask'][i]
            data_samples.gt_instances.positive_maps = positive_map
            data_samples.gt_instances.text_token_mask = \
                text_token_mask.unsqueeze(0).repeat(
                    len(positive_map), 1)
        if self.use_autocast:
            with autocast(enabled=True):
                backbone_features, visual_features = self.extract_feat(batch_inputs)
        else:
            backbone_features, visual_features = self.extract_feat(batch_inputs)
        # head_inputs_dict, _ = self.forward_transformer(visual_features, text_dict,
        #                                             batch_data_samples)

        # losses = self.bbox_head.loss(
        #     **head_inputs_dict, batch_data_samples=batch_data_samples)
        
        losses = dict()
        # losses.update(self.bbox_head.loss(**head_inputs_dict, batch_data_samples=batch_data_samples))
        for data_samples in batch_data_samples:
            gt_sem_seg = data_samples.gt_sem_seg.sem_seg
            data_samples.gt_sem_seg = PixelData(sem_seg=gt_sem_seg, data=gt_sem_seg.long())
        losses.update(self.seg_decoder.loss(backbone_features, batch_data_samples, None))
        
        return losses

@MODELS.register_module()
class GroundingDINOHeadPT(GroundingDINOHead):
    """Head of the Grounding DINO: Marrying DINO with Grounded Pre-Training for
    Open-Set Object Detection.

    Args:
        contrastive_cfg (dict, optional): Contrastive config that contains
          keys like ``max_text_len``. Defaults to dict(max_text_len=256).
    """

    def _init_layers(self) -> None:
        """Initialize classification branch and regression branch of head."""
        super()._init_layers()
        self.cls_branches.requires_grad_(False)
        # modify
        # ==================================================================
        fc_cls_bbyy = Linear(self.embed_dims, 1)

        if self.share_pred_layer:
            self.cls_branches_bbyy = nn.ModuleList(
                [fc_cls_bbyy for _ in range(self.num_pred_layer)])
        else:
            self.cls_branches_bbyy = nn.ModuleList(
                [copy.deepcopy(fc_cls_bbyy) for _ in range(self.num_pred_layer)])
        # ==================================================================

    

    def _get_targets_single(self, cls_score: Tensor, bbox_pred: Tensor,
                            gt_instances: InstanceData,
                            img_meta: dict) -> tuple:
        img_h, img_w = img_meta['img_shape']
        factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        num_bboxes = bbox_pred.size(0)
        # convert bbox_pred from xywh, normalized to xyxy, unnormalized
        bbox_pred = bbox_cxcywh_to_xyxy(bbox_pred)
        bbox_pred = bbox_pred * factor

        pred_instances = InstanceData(scores=cls_score, bboxes=bbox_pred)
        # assigner and sampler
        assign_result = self.assigner.assign(
            pred_instances=pred_instances,
            gt_instances=gt_instances,
            img_meta=img_meta)

        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds.long(), :]

        # modify
        # ==================================================================
        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    1,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)
        # ==================================================================

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred, dtype=gt_bboxes.dtype)
        bbox_weights = torch.zeros_like(bbox_pred, dtype=gt_bboxes.dtype)
        bbox_weights[pos_inds] = 1.0

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        pos_gt_bboxes_normalized = pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds)

    def forward(
        self,
        hidden_states: Tensor,
        references: List[Tensor],
        memory_text: Tensor,
        text_token_mask: Tensor,
    ) -> Tuple[Tensor]:
        all_layers_outputs_classes = []
        all_layers_outputs_classes_bbyy = []
        all_layers_outputs_coords = []

        for layer_id in range(hidden_states.shape[0]):
            reference = inverse_sigmoid(references[layer_id])
            # NOTE The last reference will not be used.
            hidden_state = hidden_states[layer_id]
            outputs_class = self.cls_branches[layer_id](hidden_state,
                                                        memory_text,
                                                        text_token_mask)
            outputs_class_bbyy = self.cls_branches_bbyy[layer_id](hidden_state)
            tmp_reg_preds = self.reg_branches[layer_id](hidden_state)
            if reference.shape[-1] == 4:
                # When `layer` is 0 and `as_two_stage` of the detector
                # is `True`, or when `layer` is greater than 0 and
                # `with_box_refine` of the detector is `True`.
                tmp_reg_preds += reference
            else:
                # When `layer` is 0 and `as_two_stage` of the detector
                # is `False`, or when `layer` is greater than 0 and
                # `with_box_refine` of the detector is `False`.
                assert reference.shape[-1] == 2
                tmp_reg_preds[..., :2] += reference
            outputs_coord = tmp_reg_preds.sigmoid()
            all_layers_outputs_classes.append(outputs_class)
            all_layers_outputs_classes_bbyy.append(outputs_class_bbyy)
            all_layers_outputs_coords.append(outputs_coord)

        all_layers_outputs_classes = torch.stack(all_layers_outputs_classes)
        all_layers_outputs_classes_bbyy = torch.stack(all_layers_outputs_classes_bbyy)
        all_layers_outputs_coords = torch.stack(all_layers_outputs_coords)

        return all_layers_outputs_classes, all_layers_outputs_classes_bbyy, all_layers_outputs_coords

    def predict(self,
                hidden_states: Tensor,
                references: List[Tensor],
                memory_text: Tensor,
                text_token_mask: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> InstanceList:
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        batch_token_positive_maps = [
            data_samples.token_positive_map
            for data_samples in batch_data_samples
        ]

        outs = self(hidden_states, references, memory_text, text_token_mask)

        predictions = self.predict_by_feat(
            *outs,
            batch_img_metas=batch_img_metas,
            batch_token_positive_maps=batch_token_positive_maps,
            rescale=rescale)
        return predictions
    
    def predict_by_feat(self,
                        all_layers_cls_scores: Tensor,
                        all_layers_outputs_classes_bbyy: Tensor,
                        all_layers_bbox_preds: Tensor,
                        batch_img_metas: List[Dict],
                        batch_token_positive_maps: Optional[List[dict]] = None,
                        rescale: bool = False) -> InstanceList:
        cls_scores = all_layers_cls_scores[-1][:, :self.num_queries]
        cls_scores_bbyy = all_layers_outputs_classes_bbyy[-1][:, -self.num_queries_bbyy:]
        bbox_preds = all_layers_bbox_preds[-1]
        result_list = []
        for img_id in range(len(batch_img_metas)):
            cls_score = cls_scores[img_id]
            cls_score_bbyy = cls_scores_bbyy[img_id]
            bbox_pred = bbox_preds[img_id]
            img_meta = batch_img_metas[img_id]
            token_positive_maps = batch_token_positive_maps[img_id]
            results_id = self._predict_by_feat_single(cls_score, bbox_pred[:self.num_queries],
                                                   token_positive_maps,
                                                   img_meta, rescale)
            results_uni = self._predict_by_feat_single_bbyy(cls_score_bbyy, bbox_pred[-self.num_queries_bbyy:],
                                                   img_meta, rescale)
            bboxes = torch.cat((results_id.bboxes, results_uni.bboxes), dim=0)
            scores = torch.cat((results_id.scores, results_uni.scores), dim=0)
            labels = torch.cat((results_id.labels, results_uni.labels), dim=0)

            # results_id = results_id[results_id.scores > 0.225]

            # bboxes = results_id.bboxes
            # scores = results_id.scores
            # labels = results_id.labels
            

            thres_dict = {1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2, 6: 0.2, 7: 0.2, 8: 0.2, 9: 0.2, 10: 0.2, 
                          11: 0.2, 12: 0.5, 13: 0.5, 14: 0.5, 15: 0.5, 16: 0.5, 17: 0.5, 18: 0.5, 19: 0.5}
            thres = labels.new_ones(labels.shape).float()

            for lbl in range(1, 20):
                thres[labels == lbl] = thres_dict[lbl]

            mask = (labels != 0) & (scores > thres)
            # mask = (scores > thres)

            scores[mask] *= 2
            det_bboxes, keep = batched_nms(bboxes, scores, labels,  
                                                nms_cfg=dict(iou_threshold=0.5), class_agnostic=True)
            results = InstanceData()
            results.bboxes = det_bboxes[:, :-1]
            results.scores = det_bboxes[:, -1]
            results.labels = labels[keep]
            results.scores[mask[keep]] /= 2

            # results_id.scores[mask] = scores[mask] * 1.5
            # result_list.append(results_id)

            result_list.append(results)
            
        return result_list
    
    def _predict_by_feat_single_bbyy(self,
                                cls_score: Tensor,
                                bbox_pred: Tensor,
                                img_meta: dict,
                                rescale: bool = True) -> InstanceData:
        assert len(cls_score) == len(bbox_pred)  # num_queries
        max_per_img = self.test_cfg.get('max_per_img', len(cls_score))
        img_shape = img_meta['img_shape']
        # exclude background
        num_classes = 1
        # if self.loss_cls.use_sigmoid:
        if True:
            cls_score = cls_score.sigmoid()
            scores, indexes = cls_score.view(-1).topk(max_per_img)
            det_labels = indexes % num_classes
            bbox_index = indexes // num_classes
            bbox_pred = bbox_pred[bbox_index]
        else:
            scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(max_per_img)
            bbox_pred = bbox_pred[bbox_index]
            det_labels = det_labels[bbox_index]

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            assert img_meta.get('scale_factor') is not None
            det_bboxes /= det_bboxes.new_tensor(
                img_meta['scale_factor']).repeat((1, 2))

        results = InstanceData()
        results.bboxes = det_bboxes
        results.scores = scores
        results.labels = det_labels
        
        return results

    def loss(self, hidden_states: Tensor, references: List[Tensor],
             memory_text: Tensor, text_token_mask: Tensor,
             enc_outputs_class: Tensor, enc_outputs_coord: Tensor,
             batch_data_samples: SampleList, dn_meta: Dict[str, int]) -> dict:
        batch_gt_instances = []
        batch_img_metas = []

        # modify: all gt labels are set to 0
        # ==================================================================
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            gt_instances = data_sample.gt_instances
            gt_instances.labels = torch.zeros_like(gt_instances.labels)
            batch_gt_instances.append(gt_instances)
        # ==================================================================

        outs = self(hidden_states, references, memory_text, text_token_mask)
        self.text_masks = text_token_mask
        loss_inputs = outs + (enc_outputs_class, enc_outputs_coord,
                              batch_gt_instances, batch_img_metas, dn_meta)
        losses = self.loss_by_feat(*loss_inputs)
        return losses
    
    def loss_by_feat(
        self,
        all_layers_cls_scores: Tensor,
        all_layers_cls_scores_bbyy: Tensor, 
        all_layers_bbox_preds: Tensor,
        enc_cls_scores: Tensor,
        enc_bbox_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        dn_meta: Dict[str, int],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        # extract denoising and matching part of outputs
        (all_layers_matching_cls_scores, all_layers_matching_cls_scores_bbyy, all_layers_matching_bbox_preds,
         all_layers_denoising_cls_scores, all_layers_denoising_cls_scores_bbyy, all_layers_denoising_bbox_preds) = \
            self.split_outputs(
                all_layers_cls_scores, all_layers_cls_scores_bbyy, all_layers_bbox_preds, dn_meta)

        # modify
        # ======================================================================
        _, all_layers_matching_cls_scores_bbyy = torch.split(
                                        all_layers_matching_cls_scores_bbyy, [self.num_queries, self.num_queries_bbyy], dim=2)
        _, all_layers_matching_bbox_preds_bbyy = torch.split(
                                        all_layers_matching_bbox_preds, [self.num_queries, self.num_queries_bbyy], dim=2)
        # ======================================================================
        loss_dict = super(DeformableDETRHead, self).loss_by_feat(
            all_layers_matching_cls_scores_bbyy, all_layers_matching_bbox_preds_bbyy,
            batch_gt_instances, batch_img_metas, batch_gt_instances_ignore)
        # NOTE DETRHead.loss_by_feat but not DeformableDETRHead.loss_by_feat
        # is called, because the encoder loss calculations are different
        # between DINO and DeformableDETR.

        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            # NOTE The enc_loss calculation of the DINO is
            # different from that of Deformable DETR.
            enc_loss_cls, enc_losses_bbox, enc_losses_iou = \
                self.loss_by_feat_single(
                    enc_cls_scores, enc_bbox_preds,
                    batch_gt_instances=batch_gt_instances,
                    batch_img_metas=batch_img_metas)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox
            loss_dict['enc_loss_iou'] = enc_losses_iou
        return loss_dict    
    
    def loss_by_feat_single(self, cls_scores: Tensor, bbox_preds: Tensor,
                            batch_gt_instances: InstanceList,
                            batch_img_metas: List[dict]) -> Tuple[Tensor]:
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           batch_gt_instances, batch_img_metas)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        # modify
        cls_scores = cls_scores.reshape(-1, 1)
        # cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        if isinstance(self.loss_cls, QualityFocalLoss):
            bg_class_ind = self.num_classes
            pos_inds = ((labels >= 0)
                        & (labels < bg_class_ind)).nonzero().squeeze(1)
            scores = label_weights.new_zeros(labels.shape)
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_decode_bbox_targets = bbox_cxcywh_to_xyxy(pos_bbox_targets)
            pos_bbox_pred = bbox_preds.reshape(-1, 4)[pos_inds]
            pos_decode_bbox_pred = bbox_cxcywh_to_xyxy(pos_bbox_pred)
            scores[pos_inds] = bbox_overlaps(
                pos_decode_bbox_pred.detach(),
                pos_decode_bbox_targets,
                is_aligned=True)
            loss_cls = self.loss_cls(
                cls_scores, (labels, scores),
                label_weights,
                avg_factor=cls_avg_factor)
        else:
            
            loss_cls = self.loss_cls(
                cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, bbox_preds):
            img_h, img_w, = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        return loss_cls, loss_bbox, loss_iou

    @staticmethod
    def split_outputs(all_layers_cls_scores: Tensor,
                      all_layers_cls_scores_bbyy: Tensor,
                      all_layers_bbox_preds: Tensor,
                      dn_meta: Dict[str, int]) -> Tuple[Tensor]:
        num_denoising_queries = dn_meta['num_denoising_queries']
        if dn_meta is not None:
            all_layers_denoising_cls_scores = \
                all_layers_cls_scores[:, :, : num_denoising_queries, :]
            all_layers_denoising_cls_scores_bbyy = \
                all_layers_cls_scores_bbyy[:, :, : num_denoising_queries, :]
            all_layers_denoising_bbox_preds = \
                all_layers_bbox_preds[:, :, : num_denoising_queries, :]
            all_layers_matching_cls_scores = \
                all_layers_cls_scores[:, :, num_denoising_queries:, :]
            all_layers_matching_cls_scores_bbyy = \
                all_layers_cls_scores_bbyy[:, :, num_denoising_queries:, :]
            all_layers_matching_bbox_preds = \
                all_layers_bbox_preds[:, :, num_denoising_queries:, :]
        else:
            all_layers_denoising_cls_scores = None
            all_layers_denoising_cls_scores_bbyy = None
            all_layers_denoising_bbox_preds = None
            all_layers_matching_cls_scores = all_layers_cls_scores
            all_layers_matching_cls_scores_bbyy = all_layers_cls_scores_bbyy
            all_layers_matching_bbox_preds = all_layers_bbox_preds
        return (all_layers_matching_cls_scores, all_layers_matching_cls_scores_bbyy, 
                all_layers_matching_bbox_preds,
                all_layers_denoising_cls_scores, all_layers_denoising_cls_scores_bbyy, 
                all_layers_denoising_bbox_preds)


@MODELS.register_module()
class GroundingDINOPTSegAnomaly(GroundingDINOPTSeg):
    def predict(self, batch_inputs, batch_data_samples, rescale: bool = True):
        
        text_prompts = []
        enhanced_text_prompts = []
        tokens_positives = []
        for data_samples in batch_data_samples:
            text_prompts.append(data_samples.text)
            if 'caption_prompt' in data_samples:
                enhanced_text_prompts.append(data_samples.caption_prompt)
            else:
                enhanced_text_prompts.append(None)
            tokens_positives.append(data_samples.get('tokens_positive', None))

        if 'custom_entities' in batch_data_samples[0]:
            # Assuming that the `custom_entities` flag
            # inside a batch is always the same. For single image inference
            custom_entities = batch_data_samples[0].custom_entities
        else:
            custom_entities = False
        if len(text_prompts) == 1:
            # All the text prompts are the same,
            # so there is no need to calculate them multiple times.
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(
                    text_prompts[0], custom_entities, enhanced_text_prompts[0],
                    tokens_positives[0])
            ] * len(batch_inputs)
        else:
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(text_prompt,
                                                     custom_entities,
                                                     enhanced_text_prompt,
                                                     tokens_positive)
                for text_prompt, enhanced_text_prompt, tokens_positive in zip(
                    text_prompts, enhanced_text_prompts, tokens_positives)
            ]
        token_positive_maps, text_prompts, _, entities = zip(
            *_positive_maps_and_prompts)

        # image feature extraction
        backbone_feats, visual_feats = self.extract_feat(batch_inputs)
        
        batch_img_metas = [data_sample.metainfo for data_sample in batch_data_samples]
        seg_logits = self.seg_decoder.predict(backbone_feats, batch_img_metas, None)
        ori_shape = batch_img_metas[0]['ori_shape']
        seg_logits_ori_shape = F.interpolate(seg_logits, ori_shape, mode='bilinear', align_corners=False)
        seg_preds = seg_logits_ori_shape.argmax(dim=1)

        # anomaly_scores = -torch.max(seg_logits_ori_shape[:, :19], dim=1)[0].unsqueeze(1)
        anomaly_scores = -torch.sum(seg_logits_ori_shape[:, :19].tanh(), dim=1).unsqueeze(1)

        # load anomaly score maps
        # anomaly_scores = torch.from_numpy(np.stack([img_metas['anomaly_score_map'] for img_metas in batch_img_metas])).to(batch_inputs.device).unsqueeze(1)

        # batch_data_samples = self.postprocess_result(seg_logits, batch_data_samples)

        if isinstance(text_prompts[0], list):
            # chunked text prompts, only bs=1 is supported
            assert len(batch_inputs) == 1
            count = 0
            results_list = []

            entities = [[item for lst in entities[0] for item in lst]]

            for b in range(len(text_prompts[0])):
                text_prompts_once = [text_prompts[0][b]]
                token_positive_maps_once = token_positive_maps[0][b]
                text_dict = self.language_model(text_prompts_once)
                # text feature map layer
                if self.text_feat_map is not None:
                    text_dict['embedded'] = self.text_feat_map(
                        text_dict['embedded'])

                batch_data_samples[
                    0].token_positive_map = token_positive_maps_once

                head_inputs_dict = self.forward_transformer(
                    copy.deepcopy(visual_feats), text_dict, batch_data_samples)
                pred_instances = self.bbox_head.predict(
                    **head_inputs_dict,
                    rescale=rescale,
                    batch_data_samples=batch_data_samples)[0]

                if len(pred_instances) > 0:
                    pred_instances.labels += count
                count += len(token_positive_maps_once)
                results_list.append(pred_instances)
            results_list = [results_list[0].cat(results_list)]
            is_rec_tasks = [False] * len(results_list)
        else:
            # extract text feats
            text_dict = self.language_model(list(text_prompts))
            # text feature map layer
            if self.text_feat_map is not None:
                text_dict['embedded'] = self.text_feat_map(
                    text_dict['embedded'])

            is_rec_tasks = []
            for i, data_samples in enumerate(batch_data_samples):
                if token_positive_maps[i] is not None:
                    is_rec_tasks.append(False)
                else:
                    is_rec_tasks.append(True)
                data_samples.token_positive_map = token_positive_maps[i]

            head_inputs_dict = self.forward_transformer(
                visual_feats, text_dict, batch_data_samples)
            results_list = self.bbox_head.predict(
                **head_inputs_dict,
                rescale=rescale,
                batch_data_samples=batch_data_samples)

        for data_sample, pred_instances, entity, is_rec_task, seg_pred, anomaly_score, img_metas in zip(
                batch_data_samples, results_list, entities, is_rec_tasks, seg_preds, anomaly_scores, batch_img_metas):
            if len(pred_instances) > 0:
                label_names = []
                for labels in pred_instances.labels:
                    if is_rec_task:
                        label_names.append(entity)
                        continue
                    if labels >= len(entity):
                        warnings.warn(
                            'The unexpected output indicates an issue with '
                            'named entity recognition. You can try '
                            'setting custom_entities=True and running '
                            'again to see if it helps.')
                        label_names.append('unobject')
                    else:
                        label_names.append(entity[labels])
                # for visualization
                pred_instances.label_names = label_names
            data_sample.pred_instances = pred_instances

            labels = data_sample.pred_instances.labels
            scores = data_sample.pred_instances.scores
            bboxes = data_sample.pred_instances.bboxes

            mask_id = torch.ones(ori_shape).to(batch_inputs.device)
            mask_road = torch.ones(ori_shape).to(batch_inputs.device)
            bboxes_id_mask = (torch.isin(labels, torch.arange(18, 20).to(batch_inputs.device)) & (scores > 0.8)) | \
                            (torch.isin(labels, torch.arange(3, 12).to(batch_inputs.device)) & (scores > 0.5)) | \
                            (torch.isin(labels, torch.arange(12, 18).to(batch_inputs.device)) & (scores > 0.5))
            bboxes_id = bboxes[bboxes_id_mask]
            bboxes_road = bboxes[scores > 0.2][torch.isin(labels[scores > 0.2], torch.arange(3).to(batch_inputs.device))].int()

            y, x = torch.meshgrid(torch.arange(ori_shape[0], device=batch_inputs.device), 
                                    torch.arange(ori_shape[1], device=batch_inputs.device),
                                    indexing='ij')
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
            bboxes_id = bboxes_id.unsqueeze(1).unsqueeze(1)
            bboxes_road = bboxes_road.unsqueeze(1).unsqueeze(1)
            
            mask_id = mask_id * ((x >= bboxes_id[..., 0]) & (x < bboxes_id[..., 2]) & (y >= bboxes_id[..., 1]) & (y < bboxes_id[..., 3])).any(dim=0)
            mask_road = mask_road * ((x >= bboxes_road[..., 0]) & (x < bboxes_road[..., 2]) & (y >= bboxes_road[..., 1]) & (y < bboxes_road[..., 3])).any(dim=0)
            # seg_pred = seg_pred * mask_id
            mask_road = mask_road * torch.isin(seg_pred, torch.arange(0, 2).to(batch_inputs.device))
            # mask_road = mask_road.cpu().numpy()
            # cv2.floodFill(mask_road, None, (0, 0), 1)
            # mask_road = torch.from_numpy(mask_road).to(batch_inputs.device)

            bbox_road_overlap = self.roi_head([mask_road.unsqueeze(0).unsqueeze(0)], [data_sample.pred_instances], [data_sample], False)
            bbox_road_overlap = bbox_road_overlap.view(len(results_list), -1, *bbox_road_overlap.shape[2:])
            bboxes[:, 0].clamp_(0, ori_shape[1] - 1)
            bboxes[:, 1].clamp_(0, ori_shape[0] - 1)
            bboxes[:, 2].clamp_(0, ori_shape[1] - 1)
            bboxes[:, 3].clamp_(0, ori_shape[0] - 1)
            bboxes_mask = (bbox_road_overlap[0].mean(dim=-1).mean(dim=-1).flatten() > 0.4) | (mask_road[bboxes.int()[:, 1], bboxes.int()[:, 0]].bool() & mask_road[bboxes.int()[:, 3], bboxes.int()[:, 2]].bool())
            # data_sample.pred_instances = data_sample.pred_instances[bboxes_mask]

            data_sample.pred_instances = data_sample.pred_instances[data_sample.pred_instances.labels == 0]


            
            bbox_anomaly_score = self.roi_head([anomaly_score.unsqueeze(0)], [data_sample.pred_instances], [data_sample], False)
            bbox_anomaly_score = bbox_anomaly_score.view(len(results_list), -1, *bbox_anomaly_score.shape[2:])[0].mean(dim=-1).mean(dim=-1).flatten()
            bboxes = data_sample.pred_instances.bboxes
            scores = data_sample.pred_instances.scores
                # data_sample.pred_instances.scores[(mask_road[bboxes.int()[:, 1], bboxes.int()[:, 0]].bool() & 
                #                                     mask_road[bboxes.int()[:, 3], bboxes.int()[:, 2]].bool() &
                #                                     mask_road[bboxes.int()[:, 3], bboxes.int()[:, 0]].bool() &
                #                                     mask_road[bboxes.int()[:, 1], bboxes.int()[:, 2]].bool())] = 0.9

            # data_sample.pred_instances = data_sample.pred_instances[(bbox_anomaly_score > -0.7) & (scores > 0.3)]
            # data_sample.pred_instances = data_sample.pred_instances[(scores > 0.3)]
            # data_sample.pred_instances.scores = bbox_anomaly_score[scores > 0.3]
            # data_sample.pred_instances.scores = torch.maximum(data_sample.pred_instances.scores, 1 + bbox_anomaly_score)


            data_sample.pred_instances = data_sample.pred_instances[(data_sample.pred_instances.scores > 0.2) & (1 + bbox_anomaly_score > 0.8)]
            # data_sample.pred_instances = data_sample.pred_instances[(data_sample.pred_instances.scores > 0.1)]


            bboxes_anomaly = data_sample.pred_instances.bboxes.unsqueeze(1).unsqueeze(1)
            objectness = torch.ones(ori_shape).to(batch_inputs.device) * 0.1
            objectness[((x >= bboxes_anomaly[..., 0]) & (x < bboxes_anomaly[..., 2]) & (y >= bboxes_anomaly[..., 1]) & (y < bboxes_anomaly[..., 3])).any(dim=0)] = 0.15
            anomaly_score = anomaly_score + objectness


            # data_sample.pred_instances = data_sample.pred_instances[(data_sample.pred_instances.scores > 0.38)]
            # data_sample.pred_instances = data_sample.pred_instances[(data_sample.pred_instances.labels == 0)]

            data_sample.set_data({
                'anomaly_scores':
                PixelData(**{'data': anomaly_score.squeeze(0)}),
            })


            # all_masks = []
            # for input_img, data_sample in zip(batch_inputs, batch_data_samples):
            #     masks = torch.zeros_like(input_img[:1])
            #     # input_img = F.interpolate(input_img.unsqueeze(0), size=(1024,1024), mode='bilinear')
            #     if len(data_sample.pred_instances) > 0:
            #         # masks = self.sam_predict(input_img, data_sample.pred_instances.bboxes, ori_shape).to(torch.float32)
            #         masks = self.sam_predict_hf(data_sample.metainfo['img_path'], data_sample.pred_instances.bboxes)
            #         # masks = masks[0][:, 0].sum(dim=0).unsqueeze(0).bool().float()
            #         masks = masks[0][:, 0].bool().float()
            #     masks = F.interpolate(masks.unsqueeze(1), size=(ori_shape[0], ori_shape[1]), mode='bilinear').to(torch.int32)
            #     data_sample.set_data({
            #         'pred_sem_seg':
            #         PixelData(**{'sem_seg': masks.sum(dim=0).bool().float()}),
            #         'seg_logits':
            #         PixelData(**{'data': seg_logits_ori_shape.squeeze(0)}),
            #         'pred_masks':
            #         PixelData(**{'sem_seg': masks.squeeze(1)}),
            #     })
            #     all_masks.append(masks)
            # all_masks = torch.stack(all_masks)
            
        return batch_data_samples

# det + seg + sam
@MODELS.register_module()
class GroundingDINOPTDetSegSAM(GroundingDINOPTSeg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sam = SamModel.from_pretrained("./sam-vit-base")
        self.processor = SamProcessor.from_pretrained("./sam-vit-base")


    def predict(self, batch_inputs, batch_data_samples, rescale: bool = True):
        text_prompts = []
        enhanced_text_prompts = []
        tokens_positives = []
        for data_samples in batch_data_samples:
            text_prompts.append(data_samples.text)
            if 'caption_prompt' in data_samples:
                enhanced_text_prompts.append(data_samples.caption_prompt)
            else:
                enhanced_text_prompts.append(None)
            tokens_positives.append(data_samples.get('tokens_positive', None))

        if 'custom_entities' in batch_data_samples[0]:
            # Assuming that the `custom_entities` flag
            # inside a batch is always the same. For single image inference
            custom_entities = batch_data_samples[0].custom_entities
        else:
            custom_entities = False
        if len(text_prompts) == 1:
            # All the text prompts are the same,
            # so there is no need to calculate them multiple times.
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(
                    text_prompts[0], custom_entities, enhanced_text_prompts[0],
                    tokens_positives[0])
            ] * len(batch_inputs)
        else:
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(text_prompt,
                                                     custom_entities,
                                                     enhanced_text_prompt,
                                                     tokens_positive)
                for text_prompt, enhanced_text_prompt, tokens_positive in zip(
                    text_prompts, enhanced_text_prompts, tokens_positives)
            ]
        token_positive_maps, text_prompts, _, entities = zip(
            *_positive_maps_and_prompts)

        # image feature extraction
        backbone_feats, visual_feats = self.extract_feat(batch_inputs)
        batch_img_metas = [data_sample.metainfo for data_sample in batch_data_samples]
        seg_logits = self.seg_decoder.predict(backbone_feats, batch_img_metas, None)
        ori_shape = batch_img_metas[0]['ori_shape']
        seg_logits_ori_shape = F.interpolate(seg_logits, ori_shape, mode='bilinear', align_corners=False)
        seg_preds = seg_logits_ori_shape.argmax(dim=1)
        anomaly_scores = -torch.sum(seg_logits_ori_shape[:, :19].tanh(), dim=1).unsqueeze(1)

        if isinstance(text_prompts[0], list):
            # chunked text prompts, only bs=1 is supported
            assert len(batch_inputs) == 1
            count = 0
            results_list = []

            entities = [[item for lst in entities[0] for item in lst]]

            for b in range(len(text_prompts[0])):
                text_prompts_once = [text_prompts[0][b]]
                token_positive_maps_once = token_positive_maps[0][b]
                text_dict = self.language_model(text_prompts_once)
                # text feature map layer
                if self.text_feat_map is not None:
                    text_dict['embedded'] = self.text_feat_map(
                        text_dict['embedded'])

                batch_data_samples[
                    0].token_positive_map = token_positive_maps_once

                head_inputs_dict = self.forward_transformer(
                    copy.deepcopy(visual_feats), text_dict, batch_data_samples)
                pred_instances = self.bbox_head.predict(
                    **head_inputs_dict,
                    rescale=rescale,
                    batch_data_samples=batch_data_samples)[0]

                if len(pred_instances) > 0:
                    pred_instances.labels += count
                count += len(token_positive_maps_once)
                results_list.append(pred_instances)
            results_list = [results_list[0].cat(results_list)]
            is_rec_tasks = [False] * len(results_list)
        else:
            # extract text feats
            text_dict = self.language_model(list(text_prompts))
            # text feature map layer
            if self.text_feat_map is not None:
                text_dict['embedded'] = self.text_feat_map(
                    text_dict['embedded'])

            is_rec_tasks = []
            for i, data_samples in enumerate(batch_data_samples):
                if token_positive_maps[i] is not None:
                    is_rec_tasks.append(False)
                else:
                    is_rec_tasks.append(True)
                data_samples.token_positive_map = token_positive_maps[i]

            head_inputs_dict = self.forward_transformer(
                visual_feats, text_dict, batch_data_samples)
            results_list = self.bbox_head.predict(
                **head_inputs_dict,
                rescale=rescale,
                batch_data_samples=batch_data_samples)

        for data_sample, pred_instances, entity, is_rec_task, seg_pred, anomaly_score, img_metas in zip(
                batch_data_samples, results_list, entities, is_rec_tasks, seg_preds, anomaly_scores, batch_img_metas):
            if len(pred_instances) > 0:
                label_names = []
                for labels in pred_instances.labels:
                    if is_rec_task:
                        label_names.append(entity)
                        continue
                    if labels >= len(entity):
                        warnings.warn(
                            'The unexpected output indicates an issue with '
                            'named entity recognition. You can try '
                            'setting custom_entities=True and running '
                            'again to see if it helps.')
                        label_names.append('unobject')
                    else:
                        label_names.append(entity[labels])
                # for visualization
                pred_instances.label_names = label_names
            data_sample.pred_instances = pred_instances
            anomaly_score += 0.3
            bbox_anomaly_score = self.roi_head([anomaly_score.unsqueeze(0)], [data_sample.pred_instances], [data_sample], False)
            bbox_anomaly_score = bbox_anomaly_score.view(len(results_list), -1, *bbox_anomaly_score.shape[2:])[0].mean(dim=-1).mean(dim=-1).flatten()
            bboxes = data_sample.pred_instances.bboxes
            areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            scores = data_sample.pred_instances.scores
            data_sample.pred_instances.scores[(scores > 0.1) & (areas < 1024)] = bbox_anomaly_score[(scores > 0.1) & (areas < 1024)]
            # data_sample.pred_instances = data_sample.pred_instances[areas < 1024]
            # data_sample.pred_instances = data_sample.pred_instances[(data_sample.pred_instances.scores > 0.25) & (data_sample.pred_instances.labels == 0)]
        
        
        # for input_img, data_sample in zip(batch_inputs, batch_data_samples):
        #     masks = torch.zeros_like(input_img[:1])
        #     ori_shape = data_sample.metainfo['ori_shape']
        #     if len(data_sample.pred_instances) > 0:
        #         masks = self.sam_predict_hf(data_sample.metainfo['img_path'], data_sample.pred_instances.bboxes)
        #         masks = masks[0][:, 0].bool().float()
        #     masks = F.interpolate(masks.unsqueeze(1), size=(ori_shape[0], ori_shape[1]), mode='bilinear').to(torch.int32)
        #     data_sample.set_data({
        #         'pred_sem_seg':
        #         PixelData(**{'sem_seg': masks.sum(dim=0).bool().float()}),
        #         # 'seg_logits':
        #         # PixelData(**{'data': seg_logits_ori_shape.squeeze(0)}),
        #         'pred_masks':
        #         PixelData(**{'sem_seg': masks.squeeze(1)}),
        #     })

        return batch_data_samples


    def sam_predict_hf(self, raw_image, boxes):
        raw_image = Image.open(raw_image).convert("RGB")
        inputs = self.processor(raw_image, return_tensors="pt").to(self.sam.device)
        image_embeddings = self.sam.get_image_embeddings(inputs["pixel_values"])
        inputs = self.processor(raw_image, input_boxes=[boxes.cpu().numpy().tolist()], return_tensors="pt").to(self.sam.device)
        inputs.pop("pixel_values", None)
        inputs.update({"image_embeddings": image_embeddings})
        with torch.no_grad():
            outputs = self.sam(**inputs)
        masks = self.processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
        return masks


@MODELS.register_module()
class GroundingDINOPTSegSAM(GroundingDINOPT):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

        self.sam = SamModel.from_pretrained("./sam-vit-base")
        self.processor = SamProcessor.from_pretrained("./sam-vit-base")
    

    def predict(self, batch_inputs, batch_data_samples, rescale: bool = True):
        text_prompts = []
        enhanced_text_prompts = []
        tokens_positives = []
        for data_samples in batch_data_samples:
            text_prompts.append(data_samples.text)
            if 'caption_prompt' in data_samples:
                enhanced_text_prompts.append(data_samples.caption_prompt)
            else:
                enhanced_text_prompts.append(None)
            tokens_positives.append(data_samples.get('tokens_positive', None))

        if 'custom_entities' in batch_data_samples[0]:
            # Assuming that the `custom_entities` flag
            # inside a batch is always the same. For single image inference
            custom_entities = batch_data_samples[0].custom_entities
        else:
            custom_entities = False
        if len(text_prompts) == 1:
            # All the text prompts are the same,
            # so there is no need to calculate them multiple times.
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(
                    text_prompts[0], custom_entities, enhanced_text_prompts[0],
                    tokens_positives[0])
            ] * len(batch_inputs)
        else:
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(text_prompt,
                                                     custom_entities,
                                                     enhanced_text_prompt,
                                                     tokens_positive)
                for text_prompt, enhanced_text_prompt, tokens_positive in zip(
                    text_prompts, enhanced_text_prompts, tokens_positives)
            ]
        token_positive_maps, text_prompts, _, entities = zip(
            *_positive_maps_and_prompts)

        # image feature extraction
        visual_feats = self.extract_feat(batch_inputs)

        if isinstance(text_prompts[0], list):
            # chunked text prompts, only bs=1 is supported
            assert len(batch_inputs) == 1
            count = 0
            results_list = []

            entities = [[item for lst in entities[0] for item in lst]]

            for b in range(len(text_prompts[0])):
                text_prompts_once = [text_prompts[0][b]]
                token_positive_maps_once = token_positive_maps[0][b]
                text_dict = self.language_model(text_prompts_once)
                # text feature map layer
                if self.text_feat_map is not None:
                    text_dict['embedded'] = self.text_feat_map(
                        text_dict['embedded'])

                batch_data_samples[
                    0].token_positive_map = token_positive_maps_once

                head_inputs_dict = self.forward_transformer(
                    copy.deepcopy(visual_feats), text_dict, batch_data_samples)
                pred_instances = self.bbox_head.predict(
                    **head_inputs_dict,
                    rescale=rescale,
                    batch_data_samples=batch_data_samples)[0]

                if len(pred_instances) > 0:
                    pred_instances.labels += count
                count += len(token_positive_maps_once)
                results_list.append(pred_instances)
            results_list = [results_list[0].cat(results_list)]
            is_rec_tasks = [False] * len(results_list)
        else:
            # extract text feats
            text_dict = self.language_model(list(text_prompts))
            # text feature map layer
            if self.text_feat_map is not None:
                text_dict['embedded'] = self.text_feat_map(
                    text_dict['embedded'])

            is_rec_tasks = []
            for i, data_samples in enumerate(batch_data_samples):
                if token_positive_maps[i] is not None:
                    is_rec_tasks.append(False)
                else:
                    is_rec_tasks.append(True)
                data_samples.token_positive_map = token_positive_maps[i]

            head_inputs_dict = self.forward_transformer(
                visual_feats, text_dict, batch_data_samples)
            results_list = self.bbox_head.predict(
                **head_inputs_dict,
                rescale=rescale,
                batch_data_samples=batch_data_samples)

        for data_sample, pred_instances, entity, is_rec_task in zip(
                batch_data_samples, results_list, entities, is_rec_tasks):
            if len(pred_instances) > 0:
                label_names = []
                for labels in pred_instances.labels:
                    if is_rec_task:
                        label_names.append(entity)
                        continue
                    if labels >= len(entity):
                        warnings.warn(
                            'The unexpected output indicates an issue with '
                            'named entity recognition. You can try '
                            'setting custom_entities=True and running '
                            'again to see if it helps.')
                        label_names.append('unobject')
                    else:
                        label_names.append(entity[labels])
                # for visualization
                pred_instances.label_names = label_names
            data_sample.pred_instances = pred_instances
            # data_sample.pred_instances = data_sample.pred_instances[(data_sample.pred_instances.scores > 0.35) & (data_sample.pred_instances.labels == 0)]
            data_sample.pred_instances = data_sample.pred_instances[(data_sample.pred_instances.labels == 0)]
        
        
        # for input_img, data_sample in zip(batch_inputs, batch_data_samples):
        #     masks = torch.zeros_like(input_img[:1])
        #     ori_shape = data_sample.metainfo['ori_shape']
        #     if len(data_sample.pred_instances) > 0:
        #         masks = self.sam_predict_hf(data_sample.metainfo['img_path'], data_sample.pred_instances.bboxes)
        #         masks = masks[0][:, 0].bool().float()
        #     masks = F.interpolate(masks.unsqueeze(1), size=(ori_shape[0], ori_shape[1]), mode='bilinear').to(torch.int32)
        #     data_sample.set_data({
        #         'pred_sem_seg':
        #         PixelData(**{'sem_seg': masks.sum(dim=0).bool().float()}),
        #         # 'seg_logits':
        #         # PixelData(**{'data': seg_logits_ori_shape.squeeze(0)}),
        #         'pred_masks':
        #         PixelData(**{'sem_seg': masks.squeeze(1)}),
        #     })

        return batch_data_samples


    def sam_predict_hf(self, raw_image, boxes):
        raw_image = Image.open(raw_image).convert("RGB")
        inputs = self.processor(raw_image, return_tensors="pt").to(self.sam.device)
        image_embeddings = self.sam.get_image_embeddings(inputs["pixel_values"])
        inputs = self.processor(raw_image, input_boxes=[boxes.cpu().numpy().tolist()], return_tensors="pt").to(self.sam.device)
        inputs.pop("pixel_values", None)
        inputs.update({"image_embeddings": image_embeddings})
        with torch.no_grad():
            outputs = self.sam(**inputs)
        masks = self.processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
        return masks


@MODELS.register_module()
class SimpleRoIHead(BaseModule):
    def __init__(self, bbox_roi_extractor: OptMultiConfig = None):
        super().__init__()
        self.bbox_roi_extractor = MODELS.build(bbox_roi_extractor)
    
    def forward(self,
                x: Tuple[Tensor],
                rpn_results_list: InstanceList,
                batch_data_samples: SampleList = None, 
                scale: bool = True) -> tuple:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            x (List[Tensor]): Multi-level features that may have different
                resolutions.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.

        Returns
            tuple: A tuple of features from ``bbox_head`` and ``mask_head``
            forward.
        """
        proposals = [rpn_results.bboxes for rpn_results in rpn_results_list]
        if scale:
            ori_shape = torch.stack([torch.tensor(data_sample.metainfo['ori_shape']) for data_sample in batch_data_samples], dim=0).repeat(1, 2).flip(dims=[1]).reshape(-1, 4)
            img_shape = torch.stack([torch.tensor(data_sample.metainfo['img_shape']) for data_sample in batch_data_samples], dim=0).repeat(1, 2).flip(dims=[1]).reshape(-1, 4)
            proposals = [bboxes.cpu() / ori_shape * img_shape for bboxes in proposals]
        rois = bbox2roi(proposals)
        bbox_results = self._bbox_forward(x, rois)
        return bbox_results

    def _bbox_forward(self, x: Tuple[Tensor], rois: Tensor) -> dict:
        """Box head forward function used in both training and testing.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            rois (Tensor): RoIs with the shape (n, 5) where the first
                column indicates batch id of each RoI.

        Returns:
             dict[str, Tensor]: Usually returns a dictionary with keys:

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `bbox_feats` (Tensor): Extract bbox RoI features.
        """
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        return bbox_feats

