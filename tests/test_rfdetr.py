"""End-to-end CPU checks for RF-DETR and the refinement branches."""
import unittest

import torch

from hsi_refine.config import ModelConfig
from hsi_refine.factory import build_rfdetr_model, load_detector_weights
from hsi_refine.geometry import cxcywh_to_xyxy, valid_xyxy, xyxy_to_cxcywh


class RFDETRIntegrationTests(unittest.TestCase):
    def test_coco_transfer_with_fewer_query_groups(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        source, _ = build_rfdetr_model(ModelConfig(
            num_classes=90, group_detr=13, detector_resolution=64,
            mode="baseline", full_iterative=False,
        ))
        state = source.detector.state_dict()
        for full_iterative in (False, True):
            with self.subTest(full_iterative=full_iterative):
                target, _ = build_rfdetr_model(ModelConfig(
                    num_classes=3, group_detr=2, detector_resolution=64,
                    mode="baseline", full_iterative=full_iterative,
                ))
                original_classifier = target.detector.class_embed.weight.detach().clone()
                load_detector_weights(target.detector, state, coco_pretrained=True)
                loaded = target.detector.state_dict()
                for key, value in loaded.items():
                    if key in {"refpoint_embed.weight", "query_feat.weight"}:
                        torch.testing.assert_close(value, state[key][:value.shape[0]])
                    elif key.startswith(("transformer.enc_out_bbox_embed.",
                                         "transformer.enc_output.", "transformer.enc_output_norm.")):
                        torch.testing.assert_close(value, state[key])
                torch.testing.assert_close(target.detector.class_embed.weight, original_classifier)

                # Extra groups may contain known head parameters only.
                for key, value in (
                    ("transformer.enc_out_bbox_embed.12.unexpected", torch.ones(1)),
                    ("transformer.enc_out_bbox_embed.12.layers.0.weight", torch.ones(1)),
                    ("transformer.enc_output.12.unexpected", torch.ones(1)),
                    ("transformer.enc_output_norm.12.weight", torch.ones(1)),
                ):
                    bad = dict(state)
                    bad[key] = value
                    with self.assertRaises(ValueError):
                        load_detector_weights(target.detector, bad, coco_pretrained=True)
                missing = dict(state)
                missing.pop("transformer.enc_out_bbox_embed.0.layers.0.weight")
                with self.assertRaises(ValueError):
                    load_detector_weights(target.detector, missing, coco_pretrained=True)

    def test_training_gradients_and_inference_query_groups(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        config = ModelConfig(num_classes=3, class_offset=1, group_detr=2,
                             detector_resolution=64, fusion_channels=8,
                             roi_size=8, boundary_bins=32, train_topk=4, eval_topk=4)
        model, criterion = build_rfdetr_model(config, "cpu", initialize_pretrained=False)
        model.train()
        cubes = torch.rand(1, 16, 64, 128)
        # Use a coarse proposal so the target is covered by the refinement ROI.
        with torch.no_grad():
            initial = model(cubes)["coarse_outputs"]
            box = valid_xyxy(cxcywh_to_xyxy(initial["pred_boxes"][0, :1]))
        targets = [{"boxes": xyxy_to_cxcywh(box), "labels": torch.tensor([1])}]
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            outputs = model(cubes, targets)
            losses, stats = criterion(outputs, targets)
            self.assertEqual(stats["matched_rois"], config.group_detr)
            self.assertGreaterEqual(stats["supervised_rois"], 1)
            loss = criterion.total(losses)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            optimizer.step()
        for name in ("adapter", "global_projection", "highres_stem", "fusion", "head"):
            with self.subTest(module=name):
                gradients = [p.grad for p in getattr(model, name).parameters() if p.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0)
        self.assertFalse(model.detector.backbone._forward_hooks)
        model.eval()
        criterion.eval()
        with torch.no_grad():
            inference = model(cubes)
            output = model(cubes, targets)
            losses, stats = criterion(output, targets)
        self.assertEqual(output["pred_boxes"].shape[1] * config.group_detr,
                         initial["pred_boxes"].shape[1])
        torch.testing.assert_close(output["pred_boxes"], inference["pred_boxes"], rtol=0, atol=0)
        torch.testing.assert_close(output["pred_logits"], inference["pred_logits"], rtol=0, atol=0)
        torch.testing.assert_close(output["refinement"]["pairs"], inference["refinement"]["pairs"])
        self.assertEqual(len(output["refinement"]["pairs"]), config.eval_topk)
        self.assertIn(stats["matched_rois"], (0, 1))
        self.assertTrue(torch.isfinite(criterion.total(losses)))
        with self.assertRaisesRegex(ValueError, r"model\(cubes, targets\)"):
            criterion(inference, targets)
        model.train()
        criterion.train()
        with torch.no_grad():
            output = model(cubes, targets)
            losses, stats = criterion(output, targets)
        self.assertEqual(output["pred_boxes"].shape[1], initial["pred_boxes"].shape[1])
        self.assertEqual(stats["matched_rois"], config.group_detr)
        self.assertTrue(torch.isfinite(criterion.total(losses)))
