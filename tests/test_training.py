# Copyright 2026 Alibaba Group Holding Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import unittest

import torch

from qwen_drive.training import (
    make_flow_batch,
    masked_endpoint_mse,
    prefill_conditioned_vlm,
)


class TrainingHelpersTest(unittest.TestCase):
    def test_flow_batch_interpolates_endpoints(self) -> None:
        target = torch.ones(2, 4, 3)
        noise = torch.zeros_like(target)
        noisy, time, returned_noise = make_flow_batch(
            target,
            flow_time=torch.tensor([0.0, 1.0]),
            noise=noise,
        )
        self.assertTrue(torch.equal(noisy[0], noise[0]))
        self.assertTrue(torch.equal(noisy[1], target[1]))
        self.assertTrue(torch.equal(time, torch.tensor([0.0, 1.0])))
        self.assertTrue(torch.equal(returned_noise, noise))

    def test_masked_loss_ignores_invalid_points(self) -> None:
        prediction = torch.zeros(1, 3, 3, requires_grad=True)
        target = torch.zeros_like(prediction)
        target[:, 0] = 2.0
        target[:, 1] = 100.0
        loss = masked_endpoint_mse(prediction, target, torch.tensor([[1, 0, 0]]))
        self.assertAlmostEqual(float(loss), 4.0, places=6)
        loss.backward()
        self.assertTrue(torch.equal(prediction.grad[:, 1:], torch.zeros(1, 2, 3)))

    def test_all_invalid_loss_is_differentiable_zero(self) -> None:
        prediction = torch.randn(1, 3, 3, requires_grad=True)
        loss = masked_endpoint_mse(prediction, torch.zeros_like(prediction), torch.zeros(1, 3))
        self.assertEqual(float(loss), 0.0)
        loss.backward()
        self.assertTrue(torch.equal(prediction.grad, torch.zeros_like(prediction)))

    def test_nonfinite_invalid_target_does_not_poison_loss(self) -> None:
        prediction = torch.zeros(1, 2, 3, requires_grad=True)
        target = torch.tensor([[[1.0, 0.0, 0.0], [float("nan"), 0.0, 0.0]]])
        loss = masked_endpoint_mse(prediction, target, torch.tensor([[1, 0]]))
        self.assertAlmostEqual(float(loss), 1.0 / 3.0, places=6)
        self.assertTrue(torch.isfinite(loss))

    def test_reasoning_prefill_uses_inference_cache_and_detaches_it(self) -> None:
        class FakeVLM:
            def eval(self):
                return self

        class FakeModel:
            vlm = FakeVLM()

            def _prefill_with_reasoning(self, inputs, max_new_tokens):
                self.inputs = inputs
                self.max_new_tokens = max_new_tokens
                cache = [(torch.ones(1, requires_grad=True), torch.ones(1, requires_grad=True))]
                anchor = torch.zeros(1, 1, 1, requires_grad=True)
                return cache, anchor, "yield to the pedestrian"

        model = FakeModel()
        inputs = {
            "input_ids": torch.zeros(1, 2, dtype=torch.long),
            "pixel_values": torch.zeros(1, 3),
            "image_grid_thw": torch.ones(1, 3, dtype=torch.long),
        }
        cache, anchor, reasoning = prefill_conditioned_vlm(
            model,
            inputs,
            conditioning_mode="reasoning",
            max_reasoning_tokens=64,
        )
        self.assertEqual(model.max_new_tokens, 64)
        self.assertEqual(reasoning, "yield to the pedestrian")
        self.assertFalse(cache[0][0].requires_grad)
        self.assertFalse(cache[0][1].requires_grad)
        self.assertFalse(anchor.requires_grad)

    def test_reasoning_prefill_requires_positive_token_cap(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_reasoning_tokens"):
            prefill_conditioned_vlm(object(), {}, conditioning_mode="reasoning")


if __name__ == "__main__":
    unittest.main()
