"""Kaggle Detectron2 Mask R-CNN weights -> torchvision Mask R-CNN.

The Kaggle hold dataset ships a trained Mask R-CNN (R50-FPN, 2 foreground
classes) as a Detectron2 checkpoint. This converts it to torchvision so it can
run as a pseudo-label teacher without the detectron2 package (which needs a
C++ toolchain to build).

Three conversion traps, all handled here:
1. D2 class order is [fg0, fg1, bg]; torchvision is [bg, fg0, fg1].
2. D2 bbox/mask predictors are fg-only; torchvision wants bg-padded channels.
3. D2 pixel mean/std are in 0-255 units; torchvision normalizes 0-1 inputs.

Verified: class 1 = holds, class 2 = volumes; masks match GT tightly.

Usage:
    python d2teacher.py --ckpt ~/data/kaggle-climbing/model/model_final.pth \
        --out /tmp/tv_hold_rcnn.pth [--verify IMAGE]
"""
import argparse
import re
from pathlib import Path

import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead


def convert(d2state):
    out = {}
    for k, v in d2state.items():
        n = None
        if k == 'backbone.bottom_up.stem.conv1.weight':
            n = 'backbone.body.conv1.weight'
        m = re.match(r'backbone\.bottom_up\.stem\.conv1\.norm\.(.*)', k)
        if m:
            n = f'backbone.body.bn1.{m.group(1)}'
        m = re.match(r'backbone\.bottom_up\.res(\d+)\.(\d+)\.conv(\d+)\.norm\.(.*)', k)
        if m:
            n = f'backbone.body.layer{int(m.group(1)) - 1}.{m.group(2)}.bn{m.group(3)}.{m.group(4)}'
        m = re.match(r'backbone\.bottom_up\.res(\d+)\.(\d+)\.conv(\d+)\.(weight)', k)
        if m:
            n = f'backbone.body.layer{int(m.group(1)) - 1}.{m.group(2)}.conv{m.group(3)}.weight'
        m = re.match(r'backbone\.bottom_up\.res(\d+)\.(\d+)\.shortcut\.weight', k)
        if m:
            n = f'backbone.body.layer{int(m.group(1)) - 1}.{m.group(2)}.downsample.0.weight'
        m = re.match(r'backbone\.bottom_up\.res(\d+)\.(\d+)\.shortcut\.norm\.(.*)', k)
        if m:
            n = f'backbone.body.layer{int(m.group(1)) - 1}.{m.group(2)}.downsample.1.{m.group(3)}'
        m = re.match(r'backbone\.fpn_lateral(\d+)\.(.*)', k)
        if m:
            n = f'backbone.fpn.inner_blocks.{int(m.group(1)) - 2}.0.{m.group(2)}'
        m = re.match(r'backbone\.fpn_output(\d+)\.(.*)', k)
        if m:
            n = f'backbone.fpn.layer_blocks.{int(m.group(1)) - 2}.0.{m.group(2)}'
        if k in ('proposal_generator.rpn_head.conv.weight',
                 'proposal_generator.rpn_head.conv.bias'):
            n = 'rpn.head.conv.0.0.' + k.rsplit('.', 1)[1]
        m = re.match(r'proposal_generator\.rpn_head\.objectness_logits\.(.*)', k)
        if m:
            n = f'rpn.head.cls_logits.{m.group(1)}'
        m = re.match(r'proposal_generator\.rpn_head\.anchor_deltas\.(.*)', k)
        if m:
            n = f'rpn.head.bbox_pred.{m.group(1)}'
        m = re.match(r'roi_heads\.box_head\.fc(\d+)\.(.*)', k)
        if m:
            n = f'roi_heads.box_head.fc{int(m.group(1)) + 5}.{m.group(2)}'
        if n is not None:
            out[n] = v
    # cls_score: D2 [fg0, fg1, bg] -> torchvision [bg, fg0, fg1].
    for suf in ('weight', 'bias'):
        d = d2state[f'roi_heads.box_predictor.cls_score.{suf}']
        out[f'roi_heads.box_predictor.cls_score.{suf}'] = d[[2, 0, 1]]
    # bbox_pred: D2 fg-only (8) -> torchvision bg+fg (12); zero bg rows.
    out['roi_heads.box_predictor.bbox_pred.weight'] = torch.cat(
        [torch.zeros(4, 1024), d2state['roi_heads.box_predictor.bbox_pred.weight']])
    out['roi_heads.box_predictor.bbox_pred.bias'] = torch.cat(
        [torch.zeros(4), d2state['roi_heads.box_predictor.bbox_pred.bias']])
    # mask predictor: D2 fg-only (2) -> torchvision (3); zero bg channel.
    out['roi_heads.mask_predictor.mask_fcn_logits.weight'] = torch.cat(
        [torch.zeros(1, 256, 1, 1), d2state['roi_heads.mask_head.predictor.weight']])
    out['roi_heads.mask_predictor.mask_fcn_logits.bias'] = torch.cat(
        [torch.zeros(1), d2state['roi_heads.mask_head.predictor.bias']])
    out['roi_heads.mask_predictor.conv5_mask.weight'] = d2state['roi_heads.mask_head.deconv.weight']
    out['roi_heads.mask_predictor.conv5_mask.bias'] = d2state['roi_heads.mask_head.deconv.bias']
    for i in range(1, 5):
        for suf in ('weight', 'bias'):
            out[f'roi_heads.mask_head.{i - 1}.0.{suf}'] = \
                d2state[f'roi_heads.mask_head.mask_fcn{i}.{suf}']
    return out


def build_model():
    """Torchvision Mask R-CNN matching the D2 training config (RPN + input)."""
    anchor_gen = AnchorGenerator(sizes=((32, 64),) * 5,
                                 aspect_ratios=((0.5, 1.0, 2.0),) * 5)
    head = RPNHead(256, anchor_gen.num_anchors_per_location()[0])
    model = maskrcnn_resnet50_fpn(
        weights=None, num_classes=3,
        image_mean=[123.675 / 255, 116.28 / 255, 103.53 / 255],
        image_std=[1.0 / 255] * 3,
        rpn_pre_nms_top_n_train=12000, rpn_pre_nms_top_n_test=12000,
        rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=2000,
        rpn_nms_thresh=0.7,
        box_score_thresh=0.05, box_nms_thresh=0.5,
        box_detections_per_img=100)
    model.rpn.anchor_generator = anchor_gen
    model.rpn.head = head
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='D2 model_final.pth')
    ap.add_argument('--out', required=True, help='torchvision state_dict output')
    ap.add_argument('--verify', default='', help='optional image for a smoke test')
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=True)
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    tv = convert(state)
    model = build_model()
    own = model.state_dict()
    missing = [k for k in own if k not in tv]
    bad = [(k, tuple(tv[k].shape), tuple(own[k].shape))
           for k in tv if k in own and tv[k].shape != own[k].shape]
    assert not missing, missing
    assert not bad, bad
    model.load_state_dict(tv, strict=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(tv, args.out)
    print(f'saved {args.out} ({len(tv)} tensors)')

    if args.verify:
        import numpy as np
        from PIL import Image
        model.load_state_dict(tv, strict=True)
        model.eval()
        img = Image.open(args.verify).convert('RGB')
        x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        with torch.no_grad():
            out = model([x])[0]
        for cls in (1, 2):
            m = out['labels'] == cls
            sc = out['scores'][m]
            print(f'class {cls}: n={m.sum().item()} '
                  f'maxscore={sc.max().item() if len(sc) else 0:.3f}')


if __name__ == '__main__':
    main()
