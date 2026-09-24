# Copyright 2026 Alibaba Group Holding Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

from qwen_drive.multitask_data import AOKVQAData, DRIVELM_CATEGORIES, DriveLMData


class MultitaskDataTest(unittest.TestCase):
    def test_drivelm_uses_scene_level_holdout_and_formats_context(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            annotations = root / "annotations" / "drivelm.json"
            annotations.parent.mkdir()
            scenes = {}
            for scene_id in ("scene_train", "scene_val"):
                scenes[scene_id] = {
                    "scene_description": "A car approaches an intersection.",
                    "key_frames": {
                        "frame_0": {
                            "image_paths": {
                                view: f"samples/{view}/frame.jpg"
                                for view in (
                                    "CAM_FRONT",
                                    "CAM_FRONT_LEFT",
                                    "CAM_FRONT_RIGHT",
                                    "CAM_BACK",
                                    "CAM_BACK_LEFT",
                                    "CAM_BACK_RIGHT",
                                )
                            },
                            "QA": {
                                category: [
                                    {
                                        "Q": "What is object <c1,CAM_FRONT,10,20> doing?",
                                        "A": "It is crossing.",
                                    }
                                ]
                                for category in DRIVELM_CATEGORIES
                            },
                            "key_object_infos": {
                                "<c1,CAM_FRONT,10,20>": {
                                    "Category": "pedestrian",
                                    "Status": "moving",
                                    "Visual_description": "wearing a dark coat",
                                }
                            },
                        }
                    },
                }
            annotations.write_text(json.dumps(scenes))

            train = DriveLMData(
                annotations,
                split="train",
                validation_fraction=0.5,
                validate_images=False,
            )
            validation = DriveLMData(
                annotations,
                split="validation",
                validation_fraction=0.5,
                validate_images=False,
            )
            self.assertEqual(len(train) + len(validation), 8)
            train_sample = train.sample(random.Random(3))
            val_sample = validation.sample(random.Random(3))
            self.assertTrue(train_sample.sample_id.startswith("drivelm:scene_"))
            self.assertTrue(val_sample.sample_id.startswith("drivelm:scene_"))
            self.assertEqual(len(train_sample.images), 6)
            self.assertIn("Six synchronized surround-view images", train_sample.question)
            self.assertIn("pedestrian", train_sample.question)
            self.assertEqual(train_sample.answer, "It is crossing.")
            self.assertNotEqual(train_sample.sample_id.split(":")[1], val_sample.sample_id.split(":")[1])
            indexed = validation.sample_by_index(0)
            self.assertEqual(indexed.sample_id, validation.sample_by_index(0).sample_id)
            self.assertIn(":perception:0", indexed.sample_id)

    def test_aokvqa_formats_answer_as_labeled_choice(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            annotation = root / "aok.json"
            annotation.write_text(
                json.dumps(
                    [
                        {
                            "question_id": "q1",
                            "image_id": 23,
                            "question": "What is in the picture?",
                            "choices": ["a dog", "a bus", "a tree"],
                            "correct_choice_idx": 1,
                        }
                    ]
                )
            )
            dataset = AOKVQAData(
                annotation,
                root / "images",
                split="train",
                validate_images=False,
            )
            sample = dataset.sample(random.Random(4))
            self.assertEqual(sample.sample_id, "aokvqa:q1")
            self.assertEqual(sample.answer, "B. a bus")
            self.assertIn("A. a dog", sample.question)
            self.assertEqual(sample.images[0].name, "000000000023.jpg")
            self.assertEqual(dataset.sample_by_index(0), sample)


if __name__ == "__main__":
    unittest.main()
