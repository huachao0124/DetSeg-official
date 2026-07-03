_base_ = './detseg-r_swin-b_internal.py'

# Reproduction entry for training DetSeg-R's internal segmentation branch.
# It inherits the Cityscapes+COCO-paste training setup and Fishyscapes Static
# validation from detseg-r_swin-b_internal.py.
load_from = 'ckpts/detseg_swin-b_coco_20260623-207453f5.pth'
