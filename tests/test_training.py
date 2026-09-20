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

from qwen_drive.training import make_flow_batch, masked_endpoint_mse


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


if __name__ == "__main__":
    unittest.main()
