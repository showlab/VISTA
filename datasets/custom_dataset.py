from torch.utils.data import Dataset
import os
import re
from decord import VideoReader
from torchvision import transforms
import numpy as np
from PIL import Image
import torch


class CustomDataset(Dataset):
    def __init__(
        self,
        video_root,
        video_root2,
        first_root,
        height=512,
        width=512,
        sample_n_frames=49,
        is_one2three=False,
        training_len=-1,
        caption_ext=".txt",
    ):
        self.training_len = training_len
        self.is_one2three = is_one2three

        self.video_root = video_root
        self.video_root2 = video_root2
        self.caption_ext = caption_ext

        def _natural_key(name: str):
            parts = re.split(r"(\d+)", name)
            return [int(p) if p.isdigit() else p.lower() for p in parts]

        img_exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
        if first_root and os.path.isdir(first_root):
            first_list_all = sorted(os.listdir(first_root), key=_natural_key)
            first_list = [x for x in first_list_all if x.lower().endswith(img_exts)]
            if len(first_list) > 0:
                self.first_root = first_root
                self.first_paths = [os.path.join(first_root, x) for x in first_list]
                self.use_first = True
            else:
                self.first_root = None
                self.first_paths = []
                self.use_first = False
        else:
            self.first_root = None
            self.first_paths = []
            self.use_first = False

        print(
            f"CustomDataset: video_root: {video_root}, "
            f"video_root2: {video_root2}, first_root: {self.first_root or ''}"
        )

        video_exts = (".mp4", ".avi", ".mov", ".mkv")

        video_list = sorted(
            [x for x in os.listdir(self.video_root) if x.lower().endswith(video_exts)],
            key=_natural_key,
        )
        video_list2 = sorted(
            [x for x in os.listdir(self.video_root2) if x.lower().endswith(video_exts)],
            key=_natural_key,
        )

        self.video_paths = [os.path.join(self.video_root, v) for v in video_list]
        self.video_paths2 = [os.path.join(self.video_root2, v) for v in video_list2]

        self.height = height
        self.width = width
        self.train_video_transforms = transforms.Compose(
            [
                transforms.CenterCrop((height, width)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        self.sample_n_frames = sample_n_frames

        self.len_videos = len(self.video_paths)
        self.len_videos2 = len(self.video_paths2)
        self.len_firsts = len(self.first_paths)

        assert self.len_videos == self.len_videos2, "mismatch in first videos and third videos"

    def __len__(self):
        if self.training_len != -1:
            return self.training_len
        if self.use_first:
            return min(self.len_videos, self.len_videos2, self.len_firsts)
        else:
            return min(self.len_videos, self.len_videos2)

    def _caption_path_for(self, video2_path: str):
        stem, _ = os.path.splitext(video2_path)
        return stem + self.caption_ext

    def _load_caption(self, video2_path: str) -> str:
        cap_path = self._caption_path_for(video2_path)
        if os.path.exists(cap_path) and os.path.isfile(cap_path):
            try:
                with open(cap_path, "r", encoding="utf-8") as f:
                    return f.read().strip()
            except Exception:
                return ""
        return ""

    def __getitem__(self, index):
        if self.use_first:
            min_len = min(self.len_videos, self.len_videos2, self.len_firsts)
        else:
            min_len = min(self.len_videos, self.len_videos2)

        if min_len <= 0:
            raise RuntimeError(
                f"No valid samples: "
                f"len_videos={self.len_videos}, len_videos2={self.len_videos2}, len_firsts={self.len_firsts}."
            )

        index = index % min_len

        video_path = self.video_paths[index]
        video_reader = VideoReader(video_path)
        video_length = len(video_reader)

        video_path2 = self.video_paths2[index]
        video_reader2 = VideoReader(video_path2)
        video_length2 = len(video_reader2)

        first_frame = None
        if self.use_first:
            first_frame_path = self.first_paths[index]
            first_frame = Image.open(first_frame_path).convert("RGB")
            first_frame = self.train_video_transforms(first_frame)

        assert video_length == video_length2, "video lengths do not match"
        assert self.sample_n_frames <= video_length, "sample_n_frames > video length"

        stride = 1
        available = video_length - (self.sample_n_frames - 1) * stride
        available = max(available, 1)
        if available <= 4:
            start_index = 0
        else:
            start_index = np.random.randint(0, available - 3)
        frame_indices = start_index + np.arange(self.sample_n_frames) * stride

        video = video_reader.get_batch(frame_indices).asnumpy()
        video = [Image.fromarray(frame) for frame in video]
        pixel_values = [self.train_video_transforms(frame) for frame in video]
        pixel_values = torch.stack(pixel_values)

        video2 = video_reader2.get_batch(frame_indices).asnumpy()
        video2 = [Image.fromarray(frame) for frame in video2]
        pixel_values2 = [self.train_video_transforms(frame) for frame in video2]
        pixel_values2 = torch.stack(pixel_values2)

        prompt = self._load_caption(video_path2)

        sample = {
            "pixel_values": pixel_values.permute(1, 0, 2, 3),
            "pixel_values2": pixel_values2.permute(1, 0, 2, 3),
            "prompts": prompt,
        }
        if self.use_first:
            sample["first_frames"] = first_frame

        return sample
