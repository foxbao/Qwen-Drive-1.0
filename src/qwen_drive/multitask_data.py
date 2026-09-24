# Copyright 2026 Alibaba Group Holding Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Lightweight adapters for DriveLM and A-OKVQA multitask SFT data."""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

DRIVELM_CATEGORIES = ("perception", "prediction", "planning", "behavior")
DRIVELM_IMAGE_ORDER = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)
OBJECT_TAG = re.compile(r"<c\d+,[^>]+>")


@dataclass(frozen=True)
class VQASample:
    sample_id: str
    dataset: str
    images: tuple[Path, ...]
    question: str
    answer: str


class DriveLMData:
    """Random-access QA view over the nested DriveLM key-frame annotations.

    The split is held out by scene ID (rather than QA row) to avoid near-identical
    camera images crossing train and validation. A category is sampled uniformly first,
    so the 160k perception QAs do not swamp the much smaller behavior category.
    """

    def __init__(
        self,
        annotation_file: str | Path,
        *,
        split: str = "train",
        validation_fraction: float = 0.05,
        validate_images: bool = True,
    ) -> None:
        self.annotation_file = Path(annotation_file)
        self.root = self.annotation_file.parent.parent
        self.image_base = self.root / "images" / "nuscenes"
        with self.annotation_file.open() as handle:
            self.scenes = json.load(handle)

        if split not in {"train", "validation"}:
            raise ValueError("DriveLM split must be 'train' or 'validation'")
        if not 0.0 <= validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1)")
        self.by_category: dict[str, list[tuple[str, str, int]]] = {
            category: [] for category in DRIVELM_CATEGORIES
        }
        for scene_id, scene in self.scenes.items():
            is_validation = _stable_fraction(scene_id) < validation_fraction
            if is_validation != (split == "validation"):
                continue
            for frame_id, frame in scene.get("key_frames", {}).items():
                image_paths = frame.get("image_paths", {})
                if validate_images:
                    missing_views = [view for view in DRIVELM_IMAGE_ORDER if view not in image_paths]
                    if missing_views:
                        raise ValueError(
                            f"DriveLM {scene_id}/{frame_id} missing image views: {missing_views}"
                        )
                    missing_files = [
                        str(self.resolve_image(image_paths[view]))
                        for view in DRIVELM_IMAGE_ORDER
                        if not self.resolve_image(image_paths[view]).is_file()
                    ]
                    if missing_files:
                        raise FileNotFoundError(
                            f"DriveLM {scene_id}/{frame_id} has missing images: {missing_files[:3]}"
                        )
                for category in DRIVELM_CATEGORIES:
                    for qa_index, qa in enumerate(frame.get("QA", {}).get(category, [])):
                        if qa.get("Q") and qa.get("A"):
                            self.by_category[category].append((scene_id, frame_id, qa_index))

        self.nonempty_categories = [
            category for category, entries in self.by_category.items() if entries
        ]
        if not self.nonempty_categories:
            raise ValueError(f"no DriveLM QA samples found for split={split!r}")
        # Keep a deterministic flat view for evaluation. Rebuilding it for every indexed
        # sample turns a 26k-question held-out evaluation into quadratic Python work.
        self.entries = [
            (category, scene_id, frame_id, qa_index)
            for category in DRIVELM_CATEGORIES
            for scene_id, frame_id, qa_index in self.by_category[category]
        ]

    def __len__(self) -> int:
        return len(self.entries)

    def resolve_image(self, image_path: str) -> Path:
        return (self.image_base / image_path).resolve()

    def sample(self, rng: random.Random) -> VQASample:
        category = rng.choice(self.nonempty_categories)
        scene_id, frame_id, qa_index = rng.choice(self.by_category[category])
        return self.sample_by_key(scene_id, frame_id, category, qa_index)

    def sample_by_index(self, index: int) -> VQASample:
        category, scene_id, frame_id, qa_index = self.entries[index]
        return self.sample_by_key(scene_id, frame_id, category, qa_index)

    def sample_by_key(self, scene_id: str, frame_id: str, category: str, qa_index: int) -> VQASample:
        scene = self.scenes[scene_id]
        frame = scene["key_frames"][frame_id]
        qa = frame["QA"][category][qa_index]
        images = tuple(
            self.resolve_image(frame["image_paths"][view]) for view in DRIVELM_IMAGE_ORDER
        )

        question = str(qa["Q"]).strip()
        object_context = []
        seen = set()
        for tag in OBJECT_TAG.findall(question):
            if tag in seen:
                continue
            seen.add(tag)
            info = frame.get("key_object_infos", {}).get(tag)
            if info:
                details = [
                    str(info.get(key, "")).strip()
                    for key in ("Category", "Status", "Visual_description")
                    if info.get(key)
                ]
                if info.get("2d_bbox"):
                    details.append(f"2D bbox={info['2d_bbox']}")
                if details:
                    object_context.append(f"{tag}: " + "; ".join(details))

        context = [
            "Six synchronized surround-view images are provided in this order: "
            + ", ".join(DRIVELM_IMAGE_ORDER)
            + ".",
        ]
        if scene.get("scene_description"):
            context.append("Scene summary: " + str(scene["scene_description"]).strip())
        if object_context:
            context.append("Referenced objects: " + " | ".join(object_context))
        context.append(f"Question: {question}\nAnswer concisely.")

        return VQASample(
            sample_id=f"drivelm:{scene_id}:{frame_id}:{category}:{qa_index}",
            dataset="drivelm",
            images=images,
            question="\n".join(context),
            answer=str(qa["A"]).strip(),
        )


class AOKVQAData:
    """Multiple-choice A-OKVQA adapter returning the labeled choice as text."""

    def __init__(
        self,
        annotation_file: str | Path,
        image_root: str | Path,
        *,
        split: str,
        validate_images: bool = True,
    ) -> None:
        self.annotation_file = Path(annotation_file)
        self.image_root = Path(image_root)
        self.split = split
        with self.annotation_file.open() as handle:
            self.rows = json.load(handle)
        if not isinstance(self.rows, list):
            raise ValueError(f"A-OKVQA annotations must be a list: {self.annotation_file}")
        if validate_images:
            missing = []
            for row in self.rows:
                path = self.image_path(row)
                if not path.is_file():
                    missing.append(str(path))
                    if len(missing) >= 5:
                        break
            if missing:
                raise FileNotFoundError(f"missing A-OKVQA images, e.g.: {missing}")

    def __len__(self) -> int:
        return len(self.rows)

    def image_path(self, row: dict) -> Path:
        return self.image_root / f"{int(row['image_id']):012d}.jpg"

    def sample(self, rng: random.Random) -> VQASample:
        return self.sample_by_index(rng.randrange(len(self.rows)))

    def sample_by_index(self, index: int) -> VQASample:
        row = self.rows[index]
        choices = row["choices"]
        correct = int(row["correct_choice_idx"])
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if not 0 <= correct < len(choices) or len(choices) > len(letters):
            raise ValueError(f"invalid A-OKVQA choice index in {row.get('question_id')}")
        choice_text = "\n".join(
            f"{letters[index]}. {choice}" for index, choice in enumerate(choices)
        )
        question = (
            "Answer the visual question by selecting the best choice.\n"
            f"Question: {row['question']}\nChoices:\n{choice_text}\n"
            "Respond with the option letter and answer."
        )
        return VQASample(
            sample_id=f"aokvqa:{row['question_id']}",
            dataset="aokvqa",
            images=(self.image_path(row),),
            question=question,
            answer=f"{letters[correct]}. {choices[correct]}",
        )


def _stable_fraction(identifier: str) -> float:
    digest = hashlib.sha1(identifier.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)
