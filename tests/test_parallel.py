"""Image-wise scatter, concurrent feature capture and full-output gather."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Barrier
import unittest
from unittest.mock import patch

import torch

from hsi_refine import HSIDataParallel
from hsi_refine.losses import RefinementCriterion
from test_model import CriterionFixture, DetectorFixture, make_model, targets


def cpu_scatter(_devices, _sizes, dim, tensor):
    return tuple(torch.chunk(tensor, len(_devices), dim=dim))


def cpu_gather(_device, dim, *tensors):
    return torch.cat(tensors, dim=dim)


def cpu_broadcast(tensors, devices, detach=False):
    return [[tensor.detach().clone() for tensor in tensors] for _ in devices]


def cpu_wrapper(model):
    # Keep the CPU regressions runnable even on a host with visible GPUs.
    with patch('torch.nn.parallel.data_parallel._get_available_device_type', return_value=None):
        return HSIDataParallel(model)


class SynchronizedDetector(DetectorFixture):
    def __init__(self, barrier):
        super().__init__()
        self.barrier = barrier

    def forward(self, images):
        # Both forwards have installed their hooks before either backbone runs.
        self.barrier.wait(timeout=10)
        outputs = super().forward(images)
        self.barrier.wait(timeout=10)
        return outputs


def varied_targets():
    batch = targets(3, empty=True)
    batch[0].update(boxes=torch.tensor([[.30, .40, .20, .30]]), labels=torch.tensor([1]))
    batch[2].update(boxes=torch.tensor([[.51, .40, .12, .30],
                                      [.38, .42, .26, .27],
                                      [.42, .38, .24, .25]]), labels=torch.tensor([2, 0, 1]))
    return batch


def activate_head(model):
    with torch.no_grad():
        if model.config.mode == "highres":
            model.head.net[-1].weight.normal_(0, .05)
        elif model.refiner_enabled:
            for head in (model.head.x_head, model.head.y_head):
                head[-1].weight.normal_(0, .05)


class ParallelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(7)

    def assert_output_close(self, expected, actual):
        if isinstance(expected, dict):
            self.assertEqual(set(expected), set(actual))
            for key in expected:
                if key == "has_target_assignments":
                    self.assertTrue((actual[key] == expected[key][0]).all())
                else:
                    self.assert_output_close(expected[key], actual[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(expected), len(actual))
            for left, right in zip(expected, actual):
                self.assert_output_close(left, right)
        elif isinstance(expected, torch.Tensor):
            torch.testing.assert_close(expected, actual, atol=2e-6, rtol=1e-5)
        else:
            self.assertEqual(expected, actual)

    def test_scatter_keeps_variable_targets_empty_boxes_and_metadata_together(self):
        wrapper = cpu_wrapper(make_model())
        cubes = torch.rand(5, 16, 64, 128)
        batch = targets(5, empty=True)
        for index, (target, count) in enumerate(zip(batch, (0, 1, 3, 2, 4))):
            target['boxes'] = torch.full((count, 4), float(index))
            target['labels'] = torch.arange(count)
            target['metadata'] = {'id': [torch.tensor(index), ('image', torch.tensor(index))]}
        with patch('torch.nn.parallel.scatter_gather.Scatter.apply', side_effect=cpu_scatter):
            inputs, kwargs = wrapper.scatter((cubes, batch), {}, [0, 1])
        self.assertEqual([len(shard[0]) for shard in inputs], [3, 2])
        self.assertEqual(kwargs, ({}, {}))
        first = 0
        for images, local_targets in inputs:
            torch.testing.assert_close(images, cubes[first:first + len(images)])
            self.assertEqual(len(local_targets), len(images))
            for local, original in zip(local_targets, batch[first:first + len(images)]):
                self.assertIsNot(local, original)
                self.assert_output_close(original, local)
                self.assertEqual(local['image_id'].ndim, 0)
            first += len(images)

    def test_scatter_handles_more_devices_than_images_and_inference(self):
        wrapper = cpu_wrapper(make_model())
        cubes = torch.rand(1, 16, 64, 128)
        with patch('torch.nn.parallel.scatter_gather.Scatter.apply', side_effect=cpu_scatter):
            inputs, kwargs = wrapper.scatter((cubes, None), {}, [0, 1, 2])
        self.assertEqual(len(inputs), 1)
        self.assertIsNone(inputs[0][1])
        torch.testing.assert_close(inputs[0][0], cubes)
        self.assertEqual(kwargs, ({},))

    def test_concurrent_hooks_are_scoped_to_replica_and_thread(self):
        reference = make_model().eval()
        activate_head(reference)
        cubes = [torch.rand(1, 16, 64, 128), torch.rand(1, 16, 64, 128) * .1]
        with torch.no_grad():
            expected = [reference(cube) for cube in cubes]
        for replicated in (False, True):
            with self.subTest(replicated=replicated):
                model = make_model().eval()
                model.detector = SynchronizedDetector(Barrier(2)).eval()
                model.load_state_dict(reference.state_dict())
                wrapper = cpu_wrapper(model)
                with torch.no_grad(), patch('torch.nn.parallel.replicate._broadcast_coalesced_reshape',
                                            side_effect=cpu_broadcast):
                    replicas = wrapper.replicate(model, [0, 1]) if replicated else [model, model]
                self.assertIs(replicas[0].detector.backbone._forward_hooks,
                              replicas[1].detector.backbone._forward_hooks)

                def forward(args):
                    replica, cube = args
                    with torch.no_grad():
                        return replica(cube)

                # Repeated calls must not retain callbacks or captured features.
                for _ in range(2):
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        actual = list(pool.map(forward, zip(replicas, cubes)))
                    for left, right in zip(expected, actual):
                        self.assert_output_close(left, right)
                    self.assertFalse(model.detector.backbone._forward_hooks)

    def test_gather_matches_full_batch_losses_and_gradients(self):
        cubes, batch = torch.rand(3, 16, 64, 128), varied_targets()
        for mode in ('baseline', 'highres', 'locnet', 'combined'):
            with self.subTest(mode=mode):
                full_model = make_model(mode=mode, groups=2)
                activate_head(full_model)
                split_model = deepcopy(full_model)
                criterion = RefinementCriterion(CriterionFixture())
                wrapper = cpu_wrapper(split_model)
                expected = full_model(cubes, batch)
                shards = [split_model(cubes[:2], batch[:2]), split_model(cubes[2:], batch[2:])]
                original_pairs = [out['refinement']['pairs'].clone() for out in shards] if mode != 'baseline' else []
                with patch('torch.nn.parallel.scatter_gather.Gather.apply', side_effect=cpu_gather):
                    actual = wrapper.gather(shards, 'cpu')
                self.assert_output_close(expected, actual)
                for shard, pairs in zip(shards, original_pairs):
                    torch.testing.assert_close(shard['refinement']['pairs'], pairs)
                left_loss, left_stats = criterion(expected, batch)
                right_loss, right_stats = criterion(actual, batch)
                self.assertEqual(left_stats, right_stats)
                self.assert_output_close(left_loss, right_loss)
                if mode != 'baseline':
                    self.assertEqual(left_stats['supervised_rois'], 8)
                    self.assertEqual(actual['refinement']['has_target_assignments'].tolist(), [True, True])
                criterion.total(left_loss).backward()
                criterion.total(right_loss).backward()
                for (name, left), (_, right) in zip(full_model.named_parameters(), split_model.named_parameters()):
                    with self.subTest(parameter=name):
                        if left.grad is None:
                            self.assertIsNone(right.grad)
                        else:
                            torch.testing.assert_close(left.grad, right.grad, atol=2e-5, rtol=2e-4)
                if mode != 'baseline':
                    self.assertGreater(sum(p.grad.abs().sum().item() for p in split_model.head.parameters()), 0)

    def test_gather_inference_and_all_empty_targets(self):
        model = make_model().eval()
        activate_head(model)
        wrapper = cpu_wrapper(model)
        cubes = torch.rand(3, 16, 64, 128)
        for batch in (None, targets(3, empty=True)):
            with self.subTest(targets_present=batch is not None), torch.no_grad():
                expected = model(cubes, batch)
                shards = [model(cubes[:2], None if batch is None else batch[:2]),
                          model(cubes[2:], None if batch is None else batch[2:])]
                with patch('torch.nn.parallel.scatter_gather.Gather.apply', side_effect=cpu_gather):
                    actual = wrapper.gather(shards, 'cpu')
                self.assert_output_close(expected, actual)
                if batch is not None:
                    losses, stats = RefinementCriterion(CriterionFixture())(actual, batch)
                    self.assertEqual(stats, {'matched_rois': 0, 'supervised_rois': 0})
                    self.assertTrue(all(torch.isfinite(value) for value in losses.values()))

    def test_cpu_fallback_and_target_batch_validation(self):
        wrapper = cpu_wrapper(make_model().eval())
        cubes, batch = torch.rand(2, 16, 64, 128), targets(2)
        with torch.no_grad():
            self.assert_output_close(wrapper.module(cubes, batch), wrapper(cubes=cubes, targets=batch))
        for invalid in ([], batch[:1], batch + batch, [None, None]):
            with self.assertRaisesRegex(ValueError, 'one dictionary per image'):
                wrapper(cubes, invalid)
        with self.assertRaisesRegex(ValueError, 'nonempty'):
            wrapper(cubes[:0])

    def test_stock_data_parallel_replica_gives_actionable_error(self):
        replica = make_model()._replicate_for_data_parallel()
        with self.assertRaisesRegex(RuntimeError, 'hsi_refine.HSIDataParallel'):
            replica(torch.rand(1, 16, 64, 128))

    def test_real_rfdetr_gather_preserves_grouped_loss(self):
        from hsi_refine import ModelConfig
        from hsi_refine.factory import build_rfdetr_model
        from hsi_refine.geometry import cxcywh_to_xyxy, valid_xyxy, xyxy_to_cxcywh

        model, criterion = build_rfdetr_model(ModelConfig(
            num_classes=3, group_detr=2, detector_resolution=64,
            fusion_channels=8, roi_size=8, boundary_bins=32,
            train_topk=4, eval_topk=4,
        ), initialize_pretrained=False)
        model.train()
        criterion.train()
        wrapper = cpu_wrapper(model)
        cubes = torch.rand(3, 16, 64, 96)
        with torch.no_grad():
            coarse = model(cubes)['coarse_outputs']
            batch = targets(3, empty=True)
            for index, count in enumerate((1, 0, 2)):
                boxes = valid_xyxy(cxcywh_to_xyxy(coarse['pred_boxes'][index, :count]))
                batch[index].update(boxes=xyxy_to_cxcywh(boxes), labels=torch.arange(count))
            expected = model(cubes, batch)
            shards = [model(cubes[:2], batch[:2]), model(cubes[2:], batch[2:])]
            with patch('torch.nn.parallel.scatter_gather.Gather.apply', side_effect=cpu_gather):
                actual = wrapper.gather(shards, 'cpu')
            # Check real auxiliary/encoder outputs and Hungarian assignments,
            # not just the fixture's final predictions or boolean flag.
            self.assert_output_close(expected, actual)
            left, left_stats = criterion(expected, batch)
            right, right_stats = criterion(actual, batch)
            self.assertEqual(left_stats, right_stats)
            self.assertGreater(left_stats['supervised_rois'], 0)
            self.assert_output_close(left, right)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, 'requires two CUDA devices')
    def test_two_cuda_devices_with_different_output_device(self):
        model = make_model(groups=2)
        activate_head(model)
        reference = deepcopy(model).to('cuda:1')
        wrapper = HSIDataParallel(model.to('cuda:0'), device_ids=[0, 1], output_device=1)
        cubes = torch.rand(3, 16, 64, 128)
        batch = [{key: value.to('cuda:1') for key, value in target.items()} for target in varied_targets()]
        expected = reference(cubes.to('cuda:1'), batch)
        actual = wrapper(cubes, batch)
        self.assert_output_close(expected, actual)
        criterion = RefinementCriterion(CriterionFixture()).to('cuda:1')
        left, left_stats = criterion(expected, batch)
        right, right_stats = criterion(actual, batch)
        self.assertEqual(left_stats, right_stats)
        self.assert_output_close(left, right)
        criterion.total(left).backward()
        criterion.total(right).backward()
        for expected_param, actual_param in zip(reference.parameters(), model.parameters()):
            if expected_param.grad is not None:
                torch.testing.assert_close(expected_param.grad.cpu(), actual_param.grad.cpu(), atol=2e-5, rtol=2e-4)


if __name__ == '__main__':
    unittest.main()
