"""CPU tests for module behavior and detector interfaces."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel.scatter_gather import gather

from hsi_refine.config import ModelConfig
from hsi_refine.model import HSIBoundaryDetector
from hsi_refine.modules import SpectralAdapter
from hsi_refine.losses import RefinementCriterion, interpolated_boundary_loss
from hsi_refine.geometry import crop_features, cxcywh_to_xyxy, paired_giou
from hsi_refine.factory import load_detector_weights


class BackboneFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, stride=8, padding=1)

    def forward(self, image):
        feature = self.conv(image)
        return [SimpleNamespace(tensors=feature)], [torch.zeros_like(feature)]


class DetectorFixture(nn.Module):
    def __init__(self, classes=3, queries=8, groups=1):
        super().__init__()
        self.backbone = BackboneFixture()
        self.class_embed = nn.Linear(16, classes + 1)
        self.boxes = nn.Parameter(torch.tensor([[0.4, 0.4, 0.3, 0.3]]).repeat(queries * groups, 1))
        self.queries, self.groups = queries, groups

    def forward(self, image):
        features, _ = self.backbone(image)
        logits = self.class_embed(features[0].tensors.mean((-2, -1)))
        count = self.queries * self.groups if self.training else self.queries
        logits = logits[:, None].expand(-1, count, -1)
        return {"pred_logits": logits, "pred_boxes": self.boxes[:count][None].expand(len(image), -1, -1),
                "aux_outputs": [{"pred_logits": logits, "pred_boxes": self.boxes[:count][None].expand(len(image), -1, -1)}]}


class MatcherFixture(nn.Module):
    def forward(self, outputs, targets, group_detr=1):
        result = []
        per_group = outputs["pred_boxes"].shape[1] // group_detr
        for target in targets:
            n = min(len(target["boxes"]), per_group)
            query = torch.cat([torch.arange(n) + g * per_group for g in range(group_detr)])
            gt = torch.arange(n).repeat(group_detr)
            result.append((query, gt))
        return result


class RankedDetectorFixture(DetectorFixture):
    """Keep matched queries below the inference TopK, without score ties."""
    def forward(self, image):
        output = super().forward(image)
        logits = output["pred_logits"]
        rank = torch.arange(logits.shape[1], device=logits.device, dtype=logits.dtype)
        output["pred_logits"] = rank[None, :, None].expand_as(logits)
        return output


class CriterionFixture(nn.Module):
    weight_dict = {"loss_ce": 1.0, "loss_bbox": 1.0}

    def forward(self, outputs, targets):
        return {"loss_ce": outputs["pred_logits"].square().mean() * 0.01,
                "loss_bbox": outputs["pred_boxes"].square().mean() * 0.01}


def make_model(mode="combined", groups=1, frozen=False, chunk=4, ranked=False):
    config = ModelConfig(num_classes=3, detector_resolution=64, group_detr=groups,
                         global_channels=16, fusion_channels=8, roi_size=8,
                         boundary_bins=32, mode=mode, train_topk=2, eval_topk=4,
                         roi_chunk_size=chunk, freeze_detector=frozen)
    detector = RankedDetectorFixture if ranked else DetectorFixture
    return HSIBoundaryDetector(detector(groups=groups), config, MatcherFixture())


def targets(batch=2, empty=False):
    return [{"boxes": torch.empty(0, 4) if empty else torch.tensor([[0.415, 0.4, 0.28, 0.3]]),
             "labels": torch.empty(0, dtype=torch.long) if empty else torch.tensor([1]),
             "image_id": torch.tensor(b), "orig_size": torch.tensor([64, 128])} for b in range(batch)]


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(7)

    def test_spectral_projection_uses_all_bands_from_initialization(self):
        adapter = SpectralAdapter()
        cubes = torch.rand(2, 16, 32, 64, requires_grad=True)
        features = adapter(cubes, 64)
        self.assertEqual(features.shape, (2, 3, 64, 64))
        self.assertTrue(torch.isfinite(features).all())
        features.square().mean().backward()
        self.assertTrue(torch.isfinite(cubes.grad).all())
        self.assertTrue((cubes.grad.abs().sum(dim=(0, 2, 3)) > 0).all())
        with self.assertRaisesRegex(ValueError, "Xavier uniform"):
            ModelConfig(adapter_init="unsupported")

    def test_all_modes_preserve_coarse_boxes_at_refiner_initialization(self):
        cube = torch.rand(2, 16, 64, 128)
        for mode in ("baseline", "highres", "locnet", "combined"):
            with self.subTest(mode=mode):
                model = make_model(mode).eval()
                output = model(cube)
                self.assertEqual(output["pred_boxes"].shape, (2, 8, 4))
                self.assertTrue(torch.isfinite(output["pred_boxes"]).all())
                torch.testing.assert_close(output["pred_boxes"], output["coarse_outputs"]["pred_boxes"], atol=2e-6, rtol=0)
                self.assertIs(output["pred_logits"], output["coarse_outputs"]["pred_logits"])
                self.assertIs(output["aux_outputs"], output["coarse_outputs"]["aux_outputs"])
                self.assertEqual(len(model.detector.backbone._forward_hooks), 0)

    def test_identity_at_image_borders(self):
        model = make_model().eval()
        model.config.eval_topk = 8  # Ensure both border boxes really enter the refiner.
        with torch.no_grad():
            model.detector.boxes[0] = torch.tensor([0.10, 0.10, 0.20, 0.20])
            model.detector.boxes[1] = torch.tensor([0.90, 0.90, 0.20, 0.20])
        output = model(torch.rand(1, 16, 64, 128))
        torch.testing.assert_close(output["pred_boxes"], output["coarse_outputs"]["pred_boxes"], atol=2e-6, rtol=0)

    def test_group_positive_queries_are_retained(self):
        model = make_model(groups=3).train()
        output = model(torch.rand(2, 16, 64, 128), targets())
        ref = output["refinement"]
        for b in range(2):
            queries = ref["pairs"][ref["pairs"][:, 0] == b, 1].tolist()
            self.assertTrue({0, 8, 16} <= set(queries))
        self.assertEqual(int((ref["gt_indices"] >= 0).sum()), 6)

    def test_eval_predictions_are_independent_of_targets(self):
        cube = torch.rand(1, 16, 64, 128)
        for mode in ("baseline", "highres", "locnet", "combined"):
            for frozen in (False, True):
                with self.subTest(mode=mode, frozen=frozen):
                    model = make_model(mode, frozen=frozen, chunk=1, ranked=True).eval()
                    model.config.eval_topk = 2
                    with torch.no_grad():
                        # A nonzero refiner ensures a changed candidate set would
                        # change the prediction, unlike the identity initialization.
                        if mode == "highres":
                            model.head.net[-1].bias.fill_(0.4)
                        elif mode != "baseline":
                            for head in (model.head.x_head, model.head.y_head):
                                head[-1].weight.normal_(0, 0.2)
                        prediction = model(cube)
                        if mode != "baseline":
                            difference = prediction["pred_boxes"] - prediction["coarse_outputs"]["pred_boxes"]
                            self.assertGreater(float(difference.abs().max()), 1e-6)
                        for batch_targets in (targets(1), targets(1, empty=True)):
                            output = model(cube, batch_targets)
                            torch.testing.assert_close(output["pred_boxes"], prediction["pred_boxes"], rtol=0, atol=0)
                            torch.testing.assert_close(output["pred_logits"], prediction["pred_logits"], rtol=0, atol=0)
                            if mode != "baseline":
                                self.assertEqual(output["refinement"]["pairs"].tolist(), [[0, 7], [0, 6]])
                            losses, stats = RefinementCriterion(CriterionFixture()).eval()(output, batch_targets)
                            self.assertEqual(stats["matched_rois"], 0)
                            self.assertTrue(all(torch.isfinite(value) for value in losses.values()))

    def test_eval_loss_supervises_only_matched_topk_candidates(self):
        model = make_model(ranked=True).eval()
        model.config.eval_topk = 2
        batch_targets = targets(1)
        batch_targets[0]["boxes"] = batch_targets[0]["boxes"].repeat(7, 1)
        batch_targets[0]["labels"] = batch_targets[0]["labels"].repeat(7)
        with torch.no_grad():
            output = model(torch.rand(1, 16, 64, 128), batch_targets)
            losses, stats = RefinementCriterion(CriterionFixture()).eval()(output, batch_targets)
        # Matcher assigns queries 0..6. Only query 6 is inside TopK [7, 6].
        self.assertEqual(output["refinement"]["pairs"].tolist(), [[0, 7], [0, 6]])
        self.assertEqual(output["refinement"]["gt_indices"].tolist(), [-1, 6])
        self.assertEqual(stats, {"matched_rois": 1, "supervised_rois": 1})
        self.assertGreater(float(losses["loss_boundary"]), 0)

    def test_criterion_rejects_missing_forward_targets(self):
        cube = torch.rand(1, 16, 64, 128)
        for mode in ("highres", "locnet", "combined"):
            for training in (True, False):
                with self.subTest(mode=mode, training=training):
                    model = make_model(mode).train(training)
                    criterion = RefinementCriterion(CriterionFixture()).train(training)
                    output = model(cube)
                    for empty in (False, True):
                        with self.assertRaisesRegex(ValueError, r"model\(cubes, targets\)"):
                            criterion(output, targets(1, empty=empty))

    def test_baseline_loss_accepts_targets_only_in_criterion(self):
        model = make_model("baseline")
        criterion = RefinementCriterion(CriterionFixture())
        losses, stats = criterion(model(torch.rand(1, 16, 64, 128)), targets(1))
        self.assertTrue(torch.isfinite(criterion.total(losses)))
        self.assertEqual(stats, {"matched_rois": 0, "supervised_rois": 0})

    def test_assignment_metadata_survives_output_gather(self):
        model = make_model().eval()
        cube = torch.rand(1, 16, 64, 128)
        with torch.no_grad():
            flags = [model(cube, batch_targets)["refinement"]["has_target_assignments"]
                     for batch_targets in (targets(1), targets(1, empty=True), None)]
        # Exercise PyTorch's actual container recursion on CPU. Only the CUDA
        # tensor transfer is replaced; Python bool leaves still fail here.
        def concatenate(_device, dim, *tensors):
            return torch.cat(tensors, dim=dim)
        with patch("torch.nn.parallel.scatter_gather.Gather.apply", side_effect=concatenate):
            merged = gather([{"refinement": {"has_target_assignments": flag}}
                             for flag in flags], target_device="cpu")
        flag = merged["refinement"]["has_target_assignments"]
        self.assertEqual(flag.dtype, torch.bool)
        self.assertEqual(flag.tolist(), [True, True, False])

    def test_loss_checks_all_gathered_assignment_flags(self):
        model = make_model()
        batch_targets = targets(1)
        output = model(torch.rand(1, 16, 64, 128), batch_targets)
        criterion = RefinementCriterion(CriterionFixture())
        for flags in ([True, True], [True, False], [False, True], [False, False]):
            with self.subTest(flags=flags):
                output["refinement"]["has_target_assignments"] = torch.tensor(flags)
                if all(flags):
                    losses, stats = criterion(output, batch_targets)
                    self.assertEqual(stats["supervised_rois"], 1)
                    self.assertTrue(torch.isfinite(criterion.total(losses)))
                else:
                    with self.assertRaisesRegex(ValueError, r"model\(cubes, targets\)"):
                        criterion(output, batch_targets)

    def test_refinement_targets_with_mixed_counts_and_grouped_matches(self):
        model = make_model(groups=2).train()
        batch_targets = targets(batch=3, empty=True)
        batch_targets[0].update(boxes=torch.tensor([[0.385, 0.395, 0.27, 0.29]]),
                                labels=torch.tensor([0]))
        batch_targets[2].update(boxes=torch.tensor([[0.415, 0.4, 0.28, 0.3],
                                                   [0.38, 0.42, 0.26, 0.27]]),
                                labels=torch.tensor([1, 2]))
        output = model(torch.rand(3, 16, 64, 128), batch_targets)
        losses, stats = RefinementCriterion(CriterionFixture())(output, batch_targets)
        self.assertEqual(stats, {"matched_rois": 6, "supervised_rois": 6})
        ref = output["refinement"]
        positive = ref["gt_indices"] >= 0
        expected_boxes = torch.stack([
            batch_targets[int(batch)]["boxes"][int(index)]
            for batch, index in zip(ref["pairs"][positive, 0], ref["gt_indices"][positive])
        ])
        expected_xyxy = cxcywh_to_xyxy(expected_boxes)
        refined = ref["raw_refined_xyxy"][positive]
        torch.testing.assert_close(losses["loss_refine_l1"], F.l1_loss(refined, expected_xyxy))
        torch.testing.assert_close(losses["loss_refine_giou"],
                                   (1 - paired_giou(refined, expected_xyxy)).mean())

    def test_new_branches_receive_gradients_after_zero_head_warm_start(self):
        model = make_model().train()
        criterion = RefinementCriterion(CriterionFixture())
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
        cubes = torch.rand(2, 16, 64, 128)
        for _ in range(2):
            optimizer.zero_grad()
            output = model(cubes, targets())
            losses, stats = criterion(output, targets())
            loss = criterion.total(losses)
            self.assertTrue(torch.isfinite(loss))
            self.assertEqual(stats["supervised_rois"], 2)
            loss.backward()
            optimizer.step()
        for module in (model.global_projection, model.highres_stem, model.fusion, model.head):
            gradients = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0)
        output = model(cubes)
        boxes = cxcywh_to_xyxy(output["pred_boxes"])
        self.assertTrue((boxes[..., 2:] > boxes[..., :2]).all())
        self.assertTrue((boxes >= 0).all() and (boxes <= 1).all())

    def test_empty_targets_produce_finite_loss(self):
        for mode in ("highres", "locnet", "combined"):
            with self.subTest(mode=mode):
                model = make_model(mode).train()
                criterion = RefinementCriterion(CriterionFixture())
                output = model(torch.rand(2, 16, 64, 128), targets(empty=True))
                losses, stats = criterion(output, targets(empty=True))
                self.assertEqual(stats["supervised_rois"], 0)
                self.assertTrue(torch.isfinite(criterion.total(losses)))
                criterion.total(losses).backward()

    def test_outside_search_region_is_not_clamped_into_false_supervision(self):
        model = make_model().train()
        bad = targets(batch=1)
        bad[0]["boxes"] = torch.tensor([[0.9, 0.9, 0.1, 0.1]])
        output = model(torch.rand(1, 16, 64, 128), bad)
        losses, stats = RefinementCriterion(CriterionFixture())(output, bad)
        self.assertEqual(stats, {"matched_rois": 1, "supervised_rois": 0})
        self.assertEqual(float(losses["loss_boundary"].detach()), 0.0)

    def test_frozen_detector_and_adapter_remain_frozen(self):
        model = make_model(groups=3, frozen=True)
        cubes, batch_targets = torch.rand(1, 16, 64, 128), targets(batch=1)
        # Check the first forward as well as later train/eval transitions.
        for mode in (None, True, False, True):
            if mode is not None:
                model.train(mode)
            self.assertFalse(model.detector.training)
            self.assertFalse(model.adapter.training)
            output = model(cubes, batch_targets)
            self.assertEqual(output["pred_boxes"].shape, (1, 8, 4))
            selected = output["refinement"]["pairs"][:, 1].tolist()
            self.assertEqual(int((output["refinement"]["gt_indices"] >= 0).sum()), int(0 in selected))
            if model.training:
                self.assertIn(0, selected)
        criterion = RefinementCriterion(CriterionFixture())
        losses, _ = criterion(output, batch_targets)
        loss = criterion.total(losses)
        loss.backward()
        self.assertTrue(all(p.grad is None for p in model.detector.parameters()))
        self.assertTrue(all(p.grad is None for p in model.adapter.parameters()))
        self.assertTrue(any(p.grad is not None for p in model.head.parameters()))

    def test_chunking_has_same_predictions(self):
        model = make_model(chunk=1).eval()
        clone = make_model(chunk=100).eval()
        model.config.eval_topk = clone.config.eval_topk = 8
        with torch.no_grad():
            for head in (model.head.x_head, model.head.y_head):
                head[-1].weight.normal_(0, 0.2)
                head[-1].bias.normal_(0, 0.1)
            model.detector.boxes.copy_(torch.tensor([
                [0.2, 0.2, 0.2, 0.2], [0.5, 0.5, 0.3, 0.3],
                [0.8, 0.8, 0.2, 0.2], [0.3, 0.7, 0.2, 0.3],
            ]).repeat(2, 1))
        clone.load_state_dict(model.state_dict())
        cube = torch.rand(2, 16, 64, 128)
        with torch.no_grad():
            output, other = model(cube), clone(cube)
        self.assertGreater(float((output["pred_boxes"] - output["coarse_outputs"]["pred_boxes"]).detach().abs().max()), 1e-5)
        torch.testing.assert_close(output["pred_boxes"], other["pred_boxes"], atol=1e-6, rtol=0)

    def test_feature_maps_with_different_aspects_are_aligned(self):
        def coordinate_map(h, w):
            x = (torch.arange(w) + 0.5) / w
            y = (torch.arange(h) + 0.5) / h
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            return torch.stack((xx, yy))[None]
        region = torch.tensor([[0.2, 0.3, 0.8, 0.7]])
        batch = torch.tensor([0])
        one = crop_features(coordinate_map(16, 16), region, batch, 8)
        two = crop_features(coordinate_map(32, 64), region, batch, 8)
        torch.testing.assert_close(one, two, atol=1e-6, rtol=0)

    def test_class_offset_and_unused_channel_are_respected(self):
        model = make_model().eval()
        model.config.class_offset, model.config.eval_topk = 1, 1
        coarse = {"pred_logits": torch.tensor([[[100.0, 0.0, 0.0, 0.0],
                                                [0.0, 4.0, 0.0, 0.0]]])}
        pairs, gt_indices = model._select(coarse, matches=None)
        torch.testing.assert_close(pairs, torch.tensor([[0, 1]]))
        torch.testing.assert_close(gt_indices, torch.tensor([-1]))

    def test_boundary_labels_are_continuous(self):
        logits = torch.zeros(1, 4, 8, requires_grad=True)
        loss = interpolated_boundary_loss(logits, torch.full((1, 4), 0.37))
        loss.backward()
        # 0.37 lies between bins 2 and 3; both receive positive target mass.
        self.assertLess(float(logits.grad[0, 0, 2]), 0)
        self.assertLess(float(logits.grad[0, 0, 3]), 0)

    def test_paired_giou_identity_and_disjoint(self):
        box = torch.tensor([[0.1, 0.1, 0.3, 0.3]])
        torch.testing.assert_close(paired_giou(box, box), torch.ones(1))
        self.assertLess(float(paired_giou(box, box + 0.5)), 0)

    def test_detector_import_rejects_unknown_or_mismatched_tensors(self):
        detector = DetectorFixture()
        state = detector.state_dict()
        bad = dict(state)
        bad["boxes"] = torch.rand(9, 4)
        with self.assertRaises(RuntimeError):
            load_detector_weights(detector, bad)
        bad = dict(state)
        bad["unexplained"] = torch.ones(1)
        with self.assertRaises(RuntimeError):
            load_detector_weights(detector, bad)


if __name__ == "__main__":
    unittest.main()
