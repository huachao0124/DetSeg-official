# Copyright (c) OpenMMLab. All rights reserved.
import copy
import re
import warnings
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from mmcv.cnn import Linear
from mmcv.ops import batched_nms
from mmengine.model import BaseModule, ModuleList
from mmengine.runner.amp import autocast
from mmengine.structures import InstanceData, PixelData
from mmseg.models import Mask2FormerHead
from PIL import Image
from transformers import SamModel, SamProcessor

from mmdet.models.dense_heads import DeformableDETRHead, GroundingDINOHead
from mmdet.models.detectors.grounding_dino import GroundingDINO
from mmdet.models.layers import SinePositionalEncoding, inverse_sigmoid
from mmdet.models.layers.transformer.grounding_dino_layers import (
    GroundingDinoTransformerDecoder, GroundingDinoTransformerEncoder)
from mmdet.models.losses import QualityFocalLoss
from mmdet.models.utils import multi_apply
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample, OptSampleList, SampleList
from mmdet.structures.bbox import (
    bbox2roi, bbox_cxcywh_to_xyxy, bbox_overlaps, bbox_xyxy_to_cxcywh)
from mmdet.utils import (
    ConfigType, InstanceList, OptConfigType, OptInstanceList, OptMultiConfig,
    reduce_mean)


try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except Exception:
    checkpoint_wrapper = None


# =============================================================================
# General Helpers
# =============================================================================


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


def create_positive_map_plus_object(tokenized,
                                    tokens_positive: list,
                                    max_num_entities: int = 256) -> Tensor:
    """Create token-positive maps anchored to the object token.

    The DetSeg prompt prepends an ``objects`` token at position 1. Assigning
    each target to that token lets the universal-query branch learn
    class-agnostic objectness while the normal text branch keeps ID labels.

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
            except Exception:
                raise
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


# =============================================================================
# MyNeck
# =============================================================================


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


# =============================================================================
# Mask2FormerHeadAnomaly
# =============================================================================


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
            loss_contrastive = self.loss_contrastive(
                cls_scores, mask_preds, batch_gt_instances, batch_img_metas)

        loss_cls, loss_mask, loss_dice = super()._loss_by_feat_single(
            cls_scores, mask_preds, batch_gt_instances, batch_img_metas)

        return loss_cls, loss_mask, loss_dice, loss_contrastive


# =============================================================================
# SamPromptMixin
# =============================================================================


class SamPromptMixin:
    """Shared SAM box-prompt utilities for DetSeg variants that use masks."""

    def _init_sam(
            self,
            sam_model_name: Optional[str] = 'facebook/sam-vit-base') -> None:
        self.sam_model_name = sam_model_name or 'facebook/sam-vit-base'
        object.__setattr__(self, '_sam_model', None)
        object.__setattr__(self, '_sam_processor', None)

    @property
    def sam(self) -> SamModel:
        """Return the external SAM model without registering it in state_dict."""
        sam_model = getattr(self, '_sam_model', None)
        if sam_model is None:
            sam_model = SamModel.from_pretrained(self.sam_model_name)
            sam_model.eval()
            for p in sam_model.parameters():
                p.requires_grad = False
            object.__setattr__(self, '_sam_model', sam_model)
        return sam_model

    @property
    def sam_processor(self) -> SamProcessor:
        """Return the external SAM processor."""
        processor = getattr(self, '_sam_processor', None)
        if processor is None:
            processor = SamProcessor.from_pretrained(self.sam_model_name)
            object.__setattr__(self, '_sam_processor', processor)
        return processor

    def _sam_on_device(self, device: torch.device) -> SamModel:
        """Move the unregistered SAM model to the inference device on demand."""
        sam_model = self.sam
        sam_device = next(sam_model.parameters()).device
        if sam_device != device:
            sam_model = sam_model.to(device)
            object.__setattr__(self, '_sam_model', sam_model)
        for p in sam_model.parameters():
            p.requires_grad = False
        sam_model.eval()
        return sam_model

    def sam_predict_hf(self, raw_image, boxes):
        device = boxes.device
        sam_model = self._sam_on_device(device)
        processor = self.sam_processor
        raw_image = Image.open(raw_image).convert('RGB')
        inputs = processor(raw_image, return_tensors='pt').to(device)
        image_embeddings = sam_model.get_image_embeddings(
            inputs['pixel_values'])
        inputs = processor(
            raw_image,
            input_boxes=[boxes.cpu().numpy().tolist()],
            return_tensors='pt').to(device)
        inputs.pop('pixel_values', None)
        inputs.update({'image_embeddings': image_embeddings})
        with torch.no_grad():
            outputs = sam_model(**inputs)
        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs['original_sizes'].cpu(),
            inputs['reshaped_input_sizes'].cpu())
        return masks

    def sam_masks_from_boxes(self, img_path: str, boxes: Tensor,
                             output_shape: tuple,
                             device: torch.device) -> Tensor:
        """Return SAM masks prompted by boxes, resized to ``output_shape``."""
        if len(boxes) == 0:
            return torch.zeros((1, *output_shape), device=device)

        masks = self.sam_predict_hf(img_path, boxes)
        masks = masks[0][:, 0].float().to(device)
        if masks.shape[-2:] != tuple(output_shape):
            masks = F.interpolate(
                masks.unsqueeze(1),
                size=output_shape,
                mode='bilinear',
                align_corners=False).squeeze(1)
        return masks


# =============================================================================
# GroundingDINOHeadWithUniversalObjectness
# =============================================================================


@MODELS.register_module()
class GroundingDINOHeadWithUniversalObjectness(GroundingDINOHead):
    """Grounding DINO head with DetSeg universal-query objectness branch.

    This head keeps the original text-conditioned ID branch and adds a
    class-agnostic branch for universal object queries. The class-agnostic
    branch is trained with all ground-truth boxes mapped to label 0 and is used
    to propose OoD candidate boxes for DetSeg/DetSeg-R.
    """

    def _init_layers(self) -> None:
        """Initialize classification branch and regression branch of head."""
        super()._init_layers()
        self.cls_branches.requires_grad_(False)
        objectness_branch = Linear(self.embed_dims, 1)

        if self.share_pred_layer:
            self.objectness_branches = nn.ModuleList(
                [objectness_branch for _ in range(self.num_pred_layer)])
        else:
            self.objectness_branches = nn.ModuleList(
                [copy.deepcopy(objectness_branch)
                 for _ in range(self.num_pred_layer)])

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

        # The universal objectness branch treats every GT box as foreground.
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    1,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

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
        all_layers_cls_scores = []
        all_layers_objectness_scores = []
        all_layers_bbox_preds = []

        for layer_id in range(hidden_states.shape[0]):
            reference = inverse_sigmoid(references[layer_id])
            # NOTE The last reference will not be used.
            hidden_state = hidden_states[layer_id]
            outputs_class = self.cls_branches[layer_id](hidden_state,
                                                        memory_text,
                                                        text_token_mask)
            objectness_scores = self.objectness_branches[layer_id](hidden_state)
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
            all_layers_cls_scores.append(outputs_class)
            all_layers_objectness_scores.append(objectness_scores)
            all_layers_bbox_preds.append(outputs_coord)

        all_layers_cls_scores = torch.stack(all_layers_cls_scores)
        all_layers_objectness_scores = torch.stack(all_layers_objectness_scores)
        all_layers_bbox_preds = torch.stack(all_layers_bbox_preds)

        return (all_layers_cls_scores, all_layers_objectness_scores,
                all_layers_bbox_preds)

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
                        all_layers_objectness_scores: Tensor,
                        all_layers_bbox_preds: Tensor,
                        batch_img_metas: List[Dict],
                        batch_token_positive_maps: Optional[List[dict]] = None,
                        rescale: bool = False) -> InstanceList:
        cls_scores = all_layers_cls_scores[-1][:, :self.num_queries]
        objectness_scores = all_layers_objectness_scores[
            -1][:, -self.num_universal_queries:]
        bbox_preds = all_layers_bbox_preds[-1]
        result_list = []
        for img_id in range(len(batch_img_metas)):
            cls_score = cls_scores[img_id]
            objectness_score = objectness_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            img_meta = batch_img_metas[img_id]
            token_positive_maps = batch_token_positive_maps[img_id]
            results_id = self._predict_by_feat_single(
                cls_score, bbox_pred[:self.num_queries],
                token_positive_maps, img_meta, rescale)
            results_universal = self._predict_universal_by_feat_single(
                objectness_score,
                bbox_pred[-self.num_universal_queries:],
                img_meta,
                rescale)
            bboxes = torch.cat(
                (results_id.bboxes, results_universal.bboxes), dim=0)
            scores = torch.cat(
                (results_id.scores, results_universal.scores), dim=0)
            labels = torch.cat(
                (results_id.labels, results_universal.labels), dim=0)
            result_list.append(self._ie_nms(bboxes, scores, labels))

        return result_list

    def _ie_nms(self, bboxes: Tensor, scores: Tensor,
                labels: Tensor) -> InstanceData:
        """Apply ID-enhanced NMS for ID and universal object proposals."""
        thres = labels.new_ones(labels.shape).float()
        id_thresholds = scores.new_tensor([0.2] * 11 + [0.5] * 8)
        id_mask = (labels > 0) & (labels <= len(id_thresholds))
        thres[id_mask] = id_thresholds[labels[id_mask] - 1]

        id_mask = (labels != 0) & (scores > thres)
        nms_scores = scores.clone()
        nms_scores[id_mask] *= 2
        det_bboxes, keep = batched_nms(
            bboxes,
            nms_scores,
            labels,
            nms_cfg=dict(iou_threshold=0.5),
            class_agnostic=True)

        results = InstanceData()
        results.bboxes = det_bboxes[:, :-1]
        results.scores = det_bboxes[:, -1]
        results.labels = labels[keep]
        results.scores[id_mask[keep]] /= 2
        return results

    def _predict_universal_by_feat_single(self,
                                          objectness_score: Tensor,
                                          bbox_pred: Tensor,
                                          img_meta: dict,
                                          rescale: bool = True
                                          ) -> InstanceData:
        assert len(objectness_score) == len(bbox_pred)
        max_per_img = self.test_cfg.get(
            'max_per_img', len(objectness_score))
        img_shape = img_meta['img_shape']
        num_classes = 1
        objectness_score = objectness_score.sigmoid()
        scores, indexes = objectness_score.view(-1).topk(max_per_img)
        det_labels = indexes % num_classes
        bbox_index = indexes // num_classes
        bbox_pred = bbox_pred[bbox_index]

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

        # Objectness supervision: all GT categories become foreground class 0.
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

    def loss_by_feat(
        self,
        all_layers_cls_scores: Tensor,
        all_layers_objectness_scores: Tensor,
        all_layers_bbox_preds: Tensor,
        enc_cls_scores: Tensor,
        enc_bbox_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        dn_meta: Dict[str, int],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        # extract denoising and matching part of outputs
        (_, matching_objectness_scores, matching_bbox_preds, _, _, _) = \
            self.split_outputs(
                all_layers_cls_scores, all_layers_objectness_scores,
                all_layers_bbox_preds, dn_meta)

        _, matching_objectness_scores = torch.split(
            matching_objectness_scores,
            [self.num_queries, self.num_universal_queries],
            dim=2)
        _, matching_universal_bbox_preds = torch.split(
            matching_bbox_preds,
            [self.num_queries, self.num_universal_queries],
            dim=2)
        loss_dict = super(DeformableDETRHead, self).loss_by_feat(
            matching_objectness_scores, matching_universal_bbox_preds,
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

        # Objectness logits are one-channel foreground/background scores.
        cls_scores = cls_scores.reshape(-1, 1)
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
                      all_layers_objectness_scores: Tensor,
                      all_layers_bbox_preds: Tensor,
                      dn_meta: Dict[str, int]) -> Tuple[Tensor]:
        if dn_meta is not None:
            num_denoising_queries = dn_meta['num_denoising_queries']
            all_layers_denoising_cls_scores = \
                all_layers_cls_scores[:, :, : num_denoising_queries, :]
            all_layers_denoising_objectness_scores = \
                all_layers_objectness_scores[:, :, : num_denoising_queries, :]
            all_layers_denoising_bbox_preds = \
                all_layers_bbox_preds[:, :, : num_denoising_queries, :]
            all_layers_matching_cls_scores = \
                all_layers_cls_scores[:, :, num_denoising_queries:, :]
            all_layers_matching_objectness_scores = \
                all_layers_objectness_scores[:, :, num_denoising_queries:, :]
            all_layers_matching_bbox_preds = \
                all_layers_bbox_preds[:, :, num_denoising_queries:, :]
        else:
            all_layers_denoising_cls_scores = None
            all_layers_denoising_objectness_scores = None
            all_layers_denoising_bbox_preds = None
            all_layers_matching_cls_scores = all_layers_cls_scores
            all_layers_matching_objectness_scores = all_layers_objectness_scores
            all_layers_matching_bbox_preds = all_layers_bbox_preds
        return (all_layers_matching_cls_scores,
                all_layers_matching_objectness_scores,
                all_layers_matching_bbox_preds,
                all_layers_denoising_cls_scores,
                all_layers_denoising_objectness_scores,
                all_layers_denoising_bbox_preds)


# =============================================================================
# RoadOverlapRoIPooler
# =============================================================================


@MODELS.register_module()
class RoadOverlapRoIPooler(BaseModule):
    """Pool road-mask values inside predicted boxes.

    DetSeg-R uses this lightweight wrapper for optional road-overlap
    filtering: the estimated road/drivable mask is passed as a single-channel
    map, and predicted boxes are pooled with RoIAlign. This is not a full
    detection RoI head because it has no bbox or mask prediction branch.

    Args:
        bbox_roi_extractor: RoI extractor config, usually
            ``SingleRoIExtractor`` with ``pool_mode='avg'``.
    """

    def __init__(self, bbox_roi_extractor: OptMultiConfig = None):
        super().__init__()
        self.bbox_roi_extractor = MODELS.build(bbox_roi_extractor)

    def forward(self,
                x: Tuple[Tensor],
                rpn_results_list: InstanceList,
                batch_data_samples: SampleList = None,
                scale: bool = True) -> Tensor:
        """Extract pooled RoI features from dense maps.

        Args:
            x: Dense maps/features to sample from, e.g. a road mask with
                shape ``(B, 1, H, W)``.
            rpn_results_list: Instances whose ``bboxes`` define RoIs.
            batch_data_samples: Provides ``ori_shape`` and ``img_shape`` when
                ``scale`` is enabled.
            scale: If true, convert boxes from original-image coordinates to
                resized image/map coordinates before RoIAlign.

        Returns:
            Tensor: Pooled RoI features with one entry per proposal.
        """
        proposals = [rpn_results.bboxes for rpn_results in rpn_results_list]
        if scale:
            assert batch_data_samples is not None
            scaled_proposals = []
            for bboxes, data_sample in zip(proposals, batch_data_samples):
                ori_shape = bboxes.new_tensor(
                    data_sample.metainfo['ori_shape']).repeat(2).flip(dims=[0])
                img_shape = bboxes.new_tensor(
                    data_sample.metainfo['img_shape']).repeat(2).flip(dims=[0])
                scaled_proposals.append(bboxes / ori_shape * img_shape)
            proposals = scaled_proposals
        rois = bbox2roi(proposals)
        roi_feats = self._extract_rois(x, rois)
        return roi_feats

    def _extract_rois(self, x: Tuple[Tensor], rois: Tensor) -> Tensor:
        """Run the configured RoI extractor on map tensors.

        Args:
            x: Dense map tensors.
            rois: RoIs with the shape ``(N, 5)`` where the first
                column indicates batch id of each RoI.

        Returns:
            Tensor: Extracted RoI features.
        """
        roi_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        return roi_feats


# =============================================================================
# DetSeg
# =============================================================================


@MODELS.register_module()
class DetSeg(GroundingDINO):
    """DetSeg detector with universal queries for class-agnostic OoD boxes."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._freeze_modules()

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
            pred_instances = data_sample.pred_instances
            keep = (pred_instances.scores > 0.2) & (
                pred_instances.labels == 0)
            data_sample.pred_instances = pred_instances[keep]

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

        self.num_universal_queries = 300
        self.universal_query_embedding = nn.Embedding(
            self.num_universal_queries, self.embed_dims)
        self.bbox_head.num_queries = self.num_queries
        self.bbox_head.num_universal_queries = self.num_universal_queries
        self.decoder.num_queries = self.num_queries
        self.decoder.num_universal_queries = self.num_universal_queries
        self.universal_embedding = nn.Embedding(1, self.embed_dims)
        self.universal_memory_trans_fc = nn.Linear(
            self.embed_dims, self.embed_dims)
        self.universal_memory_trans_norm = nn.LayerNorm(self.embed_dims)

    def forward_transformer(
        self,
        img_feats: Tuple[Tensor],
        text_dict: Dict,
        batch_data_samples: OptSampleList = None,
    ) -> Dict:

        # Position 1 is the prepended "objects" token used by universal
        # objectness queries.
        text_dict['embedded'][:, 1:2] = (
            text_dict['embedded'][:, 1:2] +
            self.universal_embedding.weight.unsqueeze(0))

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
        enc_outputs_coord_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers](output_memory) + output_proposals

        # NOTE The DINO selects top-k proposals according to scores of
        # multi-class classification, while DeformDETR, where the input
        # is `enc_outputs_class[..., 0]` selects according to scores of
        # binary classification.
        topk_indices = torch.topk(
            enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1)[1]

        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        topk_coords_unact = topk_coords_unact.detach()

        universal_output_memory, _ = \
            self.gen_universal_encoder_output_proposals(
                memory, memory_mask, spatial_shapes)
        enc_outputs_objectness = self.bbox_head.objectness_branches[
            self.decoder.num_layers](universal_output_memory)
        topk_universal_indices = torch.topk(
            enc_outputs_objectness.max(-1)[0],
            k=self.num_universal_queries,
            dim=1)[1]

        topk_universal_scores = torch.gather(
            enc_outputs_objectness, 1,
            topk_universal_indices.unsqueeze(-1).repeat(1, 1, 1))
        topk_universal_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_universal_indices.unsqueeze(-1).repeat(1, 1, 4))
        topk_universal_coords = topk_universal_coords_unact.sigmoid()
        topk_universal_coords_unact = topk_universal_coords_unact.detach()

        query = torch.cat(
            (self.query_embedding.weight,
             self.universal_query_embedding.weight),
            dim=0)[:, None, :]
        query = query.repeat(1, bs, 1).transpose(0, 1)
        if self.training:
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                self.dn_query_generator(batch_data_samples)
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points = torch.cat(
                [dn_bbox_query, topk_coords_unact,
                 topk_universal_coords_unact],
                dim=1)
            dn_mask_extend = dn_mask.new_zeros((query.size(1), query.size(1)))
            dn_mask_extend[:dn_mask.size(0), :dn_mask.size(1)] = dn_mask
            dn_mask_extend[
                dn_label_query.size(1):-self.num_universal_queries,
                -self.num_universal_queries:] = True
            dn_mask_extend[
                -self.num_universal_queries:,
                :-self.num_universal_queries] = True
        else:
            reference_points = torch.cat(
                [topk_coords_unact, topk_universal_coords_unact], dim=1)
            dn_mask, dn_meta = None, None
            dn_mask_extend = torch.zeros(
                (query.size(1), query.size(1)),
                device=query.device,
                dtype=torch.bool)
            dn_mask_extend[:self.num_queries, self.num_queries:] = True
            dn_mask_extend[self.num_queries:, :self.num_queries] = True

        query_text_mask = dn_mask_extend.new_zeros(
            (query.size(1), text_token_mask.size(1)))
        query_text_mask[-self.num_universal_queries:, 3:] = True
        dn_mask = dn_mask_extend
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
            enc_outputs_class=topk_universal_scores,
            enc_outputs_coord=topk_universal_coords,
            dn_meta=dn_meta) if self.training else dict()
        # append text_feats to head_inputs_dict
        head_inputs_dict['memory_text'] = memory_text
        head_inputs_dict['text_token_mask'] = text_token_mask
        return decoder_inputs_dict, head_inputs_dict

    def gen_universal_encoder_output_proposals(
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
        output_memory = self.universal_memory_trans_fc(output_memory)
        output_memory = self.universal_memory_trans_norm(output_memory)
        # [bs, sum(hw), 2]
        return output_memory, output_proposals


# =============================================================================
# DetSegR
# =============================================================================


@MODELS.register_module()
class DetSegR(SamPromptMixin, DetSeg):
    """DetSeg-R detector with optional internal anomaly-score generation.

    The class first predicts OoD candidate boxes with the universal-query
    detection branch, then refines a pixel-level anomaly score map inside the
    retained boxes.

    Two score-map sources are supported:
    - external maps loaded by the data pipeline as ``anomaly_score_map``;
    - internal maps generated by ``seg_decoder`` when no external map exists.

    External-score configs can set ``seg_decoder=None`` to run without the
    internal segmentation module. Internal-score configs must provide
    ``seg_decoder`` because no external anomaly map is available.

    Set ``use_sam_refine=True`` to boost only SAM object masks inside
    retained boxes instead of boosting the full bounding-box area. This option
    is disabled by default.

    In inference it writes ``pred_instances`` for retained OoD boxes and
    ``anomaly_scores`` for pixel-level evaluation. The refined score map is
    also written to ``pred_masks`` for oracle-threshold IoU sweeps used by
    comparison experiments.
    """

    def __init__(self,
                 seg_decoder: OptConfigType = None,
                 road_overlap_pooler: OptConfigType = None,
                 box_score_thr: float = 0.2,
                 strong_objectness_box: float = 1.0,
                 filter_by_road_overlap: bool = False,
                 use_sam_refine: bool = False,
                 sam_model_name: Optional[str] = 'facebook/sam-vit-base',
                 **kwargs):
        """Initialize DetSeg-R.

        Args:
            seg_decoder: Optional segmentation head used to generate internal
                anomaly score maps when external score maps are unavailable.
                Defaults to ``None``.
            road_overlap_pooler: Optional RoIAlign pooler used to measure how
                much each detected box overlaps the estimated road/drivable
                area. Required only when ``filter_by_road_overlap=True``.
                Defaults to ``None``.
            box_score_thr: Minimum score for keeping universal-query OoD boxes.
                Defaults to ``0.2``.
            strong_objectness_box: Score boost added inside retained DetSeg-R
                boxes, or inside SAM-refined masks when enabled.
                Defaults to ``1.0``.
            filter_by_road_overlap: If true, keep only boxes overlapping the
                estimated road/drivable area. Disabled for reproduced results.
                Defaults to ``False``.
            use_sam_refine: If true, use SAM masks instead of full boxes for
                score-map boosting. Defaults to ``False``.
            sam_model_name: HuggingFace model id or local path passed to
                ``SamModel.from_pretrained`` when SAM refinement is enabled.
                Defaults to ``'facebook/sam-vit-base'``.
            **kwargs: Remaining Grounding DINO detector arguments.
        """
        self.seg_decoder = seg_decoder
        self.road_overlap_pooler = road_overlap_pooler
        self.box_score_thr = box_score_thr
        self.strong_objectness_box = strong_objectness_box
        self.filter_by_road_overlap = filter_by_road_overlap
        self.use_sam_refine = use_sam_refine

        super().__init__(**kwargs)

        if self.use_sam_refine:
            self._init_sam(sam_model_name)

        self.neck_seg = None
        if self.seg_decoder is not None:
            self.neck_seg = MyNeck([128, 256, 512, 1024])

        self._freeze_modules()

    def _freeze_modules(self):
        """Freeze the Grounding DINO detection path for seg-branch training."""
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
        for m in self.universal_query_embedding.parameters():
            m.requires_grad = False
        for m in self.universal_embedding.parameters():
            m.requires_grad = False
        self.level_embed.requires_grad = False
        for m in self.memory_trans_fc.parameters():
            m.requires_grad = False
        for m in self.memory_trans_norm.parameters():
            m.requires_grad = False
        for m in self.universal_memory_trans_fc.parameters():
            m.requires_grad = False
        for m in self.universal_memory_trans_norm.parameters():
            m.requires_grad = False
        for m in self.language_model.parameters():
            m.requires_grad = False
        for m in self.text_feat_map.parameters():
            m.requires_grad = False

    def _init_layers(self) -> None:
        """Build optional segmentation decoder and road-overlap pooler."""
        super()._init_layers()

        if self.seg_decoder is not None:
            self.seg_decoder = MODELS.build(self.seg_decoder)
            self.align_corners = self.seg_decoder.align_corners
        if self.road_overlap_pooler is not None:
            self.road_overlap_pooler = MODELS.build(
                self.road_overlap_pooler)
        elif self.filter_by_road_overlap:
            raise ValueError(
                'road_overlap_pooler must be set when '
                'filter_by_road_overlap=True.')

    def extract_feat(
            self, batch_inputs: Tensor) -> Tuple[Optional[Tensor], Tensor]:
        """Return segmentation and detection features from the shared image."""
        x = self.backbone(batch_inputs)  # [4, 8, 16, 32]
        if self.seg_decoder is None:
            x_uc = self.neck(x) if self.with_neck else x
            return None, x_uc

        if self.with_neck:
            x_uc = self.neck(x[1:])

        x = (
            self.neck_seg(x)
            if self.seg_decoder is not None and self.neck_seg is not None
            else None)

        return x, x_uc

    def _boost_anomaly_score(self, anomaly_score: Tensor,
                             data_sample: DetDataSample, ori_shape: tuple,
                             x: Tensor, y: Tensor,
                             box_boost: float) -> Tensor:
        """Boost retained DetSeg-R regions with optional SAM mask refinement."""
        objectness = torch.zeros(ori_shape, device=anomaly_score.device)
        if len(data_sample.pred_instances) == 0:
            return anomaly_score + objectness

        if self.use_sam_refine:
            masks = self.sam_masks_from_boxes(
                data_sample.metainfo['img_path'],
                data_sample.pred_instances.bboxes,
                ori_shape,
                anomaly_score.device)
            objectness[masks.bool().any(dim=0)] = box_boost
            return anomaly_score + objectness

        bboxes_anomaly = data_sample.pred_instances.bboxes.unsqueeze(1).unsqueeze(1)
        objectness[((x >= bboxes_anomaly[..., 0])
                    & (x < bboxes_anomaly[..., 2])
                    & (y >= bboxes_anomaly[..., 1])
                    & (y < bboxes_anomaly[..., 3])).any(dim=0)] = box_boost
        return anomaly_score + objectness

    def _get_road_overlap_mask(self, mask_road: Tensor,
                               data_sample: DetDataSample,
                               results_list: InstanceList,
                               ori_shape: tuple) -> Tensor:
        """Return bbox mask for boxes overlapping the estimated road area."""
        bbox_road_overlap = self.road_overlap_pooler(
            [mask_road.unsqueeze(0).unsqueeze(0)],
            [data_sample.pred_instances], [data_sample], False)
        bbox_road_overlap = bbox_road_overlap.view(
            len(results_list), -1, *bbox_road_overlap.shape[2:])
        bboxes = data_sample.pred_instances.bboxes.clone()
        bboxes[:, 0].clamp_(0, ori_shape[1] - 1)
        bboxes[:, 1].clamp_(0, ori_shape[0] - 1)
        bboxes[:, 2].clamp_(0, ori_shape[1] - 1)
        bboxes[:, 3].clamp_(0, ori_shape[0] - 1)
        box_mean_on_road = bbox_road_overlap[0].mean(
            dim=-1).mean(dim=-1).flatten() > 0.4
        box_corners_on_road = (
            mask_road[bboxes.int()[:, 1], bboxes.int()[:, 0]].bool()
            & mask_road[bboxes.int()[:, 3], bboxes.int()[:, 2]].bool())
        return box_mean_on_road | box_corners_on_road

    def predict(self, batch_inputs, batch_data_samples, rescale: bool = True):
        """Predict DetSeg-R outputs.

        If every sample carries ``anomaly_score_map`` in metainfo, that map is
        used as the base anomaly score. Otherwise, ``seg_decoder`` predicts
        Cityscapes logits and the RbA-style score
        ``-sum(tanh(logits[:19]))`` is used as the internal anomaly map.

        The detection branch then keeps universal-query OoD boxes, optionally
        filters by road overlap, and boosts the score map inside retained
        boxes.
        """

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
        ori_shape = batch_img_metas[0]['ori_shape']
        external_score_maps = [
            img_metas.get('anomaly_score_map', None)
            for img_metas in batch_img_metas
        ]
        if all(score_map is not None for score_map in external_score_maps):
            anomaly_scores = []
            for score_map in external_score_maps:
                score_map = torch.as_tensor(
                    score_map, dtype=torch.float32, device=batch_inputs.device)
                score_map = score_map.squeeze()
                score_map = score_map[None, None]
                if score_map.shape[-2:] != tuple(ori_shape):
                    score_map = F.interpolate(
                        score_map,
                        size=ori_shape,
                        mode='bilinear',
                        align_corners=False)
                anomaly_scores.append(score_map.squeeze(0))
            anomaly_scores = torch.stack(anomaly_scores)
            seg_preds = torch.zeros(
                (len(batch_inputs), *ori_shape),
                dtype=torch.long,
                device=batch_inputs.device)
        else:
            if self.seg_decoder is None:
                raise RuntimeError(
                    'DetSegR requires external anomaly_score_map in metainfo '
                    'when seg_decoder is not configured.')
            seg_logits = self.seg_decoder.predict(
                backbone_feats, batch_img_metas, None)
            seg_logits_ori_shape = F.interpolate(
                seg_logits,
                ori_shape,
                mode='bilinear',
                align_corners=False)
            seg_preds = seg_logits_ori_shape.argmax(dim=1)
            anomaly_scores = -torch.sum(
                seg_logits_ori_shape[:, :19].tanh(), dim=1).unsqueeze(1)

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

        for data_sample, pred_instances, entity, is_rec_task, seg_pred, anomaly_score in zip(
                batch_data_samples, results_list, entities, is_rec_tasks,
                seg_preds, anomaly_scores):
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

            mask_road = torch.ones(ori_shape).to(batch_inputs.device)
            high_score_mask = scores > 0.2
            road_label_ids = torch.arange(3, device=batch_inputs.device)
            road_box_mask = torch.isin(
                labels[high_score_mask], road_label_ids)
            bboxes_road = bboxes[high_score_mask][road_box_mask].int()

            y, x = torch.meshgrid(
                torch.arange(ori_shape[0], device=batch_inputs.device),
                torch.arange(ori_shape[1], device=batch_inputs.device),
                indexing='ij')
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
            bboxes_road = bboxes_road.unsqueeze(1).unsqueeze(1)

            road_area_from_boxes = (
                (x >= bboxes_road[..., 0])
                & (x < bboxes_road[..., 2])
                & (y >= bboxes_road[..., 1])
                & (y < bboxes_road[..., 3])).any(dim=0)
            road_area_from_seg = torch.isin(
                seg_pred, torch.arange(2, device=batch_inputs.device))
            mask_road = mask_road * road_area_from_boxes
            mask_road = mask_road * road_area_from_seg

            if self.filter_by_road_overlap:
                bboxes_mask = self._get_road_overlap_mask(
                    mask_road, data_sample, results_list, ori_shape)
                data_sample.pred_instances = data_sample.pred_instances[bboxes_mask]

            ood_mask = data_sample.pred_instances.labels == 0
            data_sample.pred_instances = data_sample.pred_instances[ood_mask]

            data_sample.pred_instances = data_sample.pred_instances[
                data_sample.pred_instances.scores > self.box_score_thr]

            anomaly_score = self._boost_anomaly_score(
                anomaly_score, data_sample, ori_shape, x, y,
                self.strong_objectness_box)

            data_sample.set_data({
                'anomaly_scores':
                PixelData(**{'data': anomaly_score.squeeze(0)}),
                'pred_masks':
                PixelData(**{'sem_seg': anomaly_score.squeeze(0)})
            })
        return batch_data_samples

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Train only the internal segmentation branch for DetSeg-R scoring."""
        if self.seg_decoder is None:
            raise RuntimeError(
                'DetSegR.loss requires seg_decoder. Use an internal-score '
                'training config when training the segmentation branch.')

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

        losses = dict()
        for data_samples in batch_data_samples:
            gt_sem_seg = data_samples.gt_sem_seg.sem_seg
            data_samples.gt_sem_seg = PixelData(sem_seg=gt_sem_seg, data=gt_sem_seg.long())
        losses.update(self.seg_decoder.loss(backbone_features, batch_data_samples, None))

        return losses


# =============================================================================
# DetSegS
# =============================================================================


@MODELS.register_module()
class DetSegS(SamPromptMixin, DetSeg):
    """DetSeg-S threshold-free anomaly mask predictor.

    DetSeg-S reuses DetSeg's universal-query OoD boxes and applies SAM with
    those boxes as prompts. The resulting binary mask is written to
    ``pred_sem_seg`` for ``BinaryMaskAnomalyIoUMetric``.
    """

    def __init__(self,
                 sam_model_name: Optional[str] = 'facebook/sam-vit-base',
                 **kwargs):
        super().__init__(**kwargs)
        self._init_sam(sam_model_name)

    def predict(self, batch_inputs, batch_data_samples, rescale: bool = True):
        """Predict DetSeg boxes and convert them into SAM binary masks."""
        batch_data_samples = super().predict(
            batch_inputs, batch_data_samples, rescale=rescale)

        for data_sample in batch_data_samples:
            ori_shape = data_sample.metainfo['ori_shape']
            masks = self.sam_masks_from_boxes(
                data_sample.metainfo['img_path'],
                data_sample.pred_instances.bboxes,
                ori_shape,
                batch_inputs.device)
            mask_union = masks.bool().any(dim=0).float()
            data_sample.set_data({
                'pred_sem_seg':
                PixelData(**{'sem_seg': mask_union}),
                'pred_masks':
                PixelData(**{'sem_seg': masks.bool().float()}),
            })

        return batch_data_samples
