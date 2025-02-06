from abc import ABC, abstractmethod
from typing import Dict, Any
import torch
import os
import json
import numpy as np
import bittensor as bt
from urllib.parse import urlparse
import requests
import random
import zipfile
from PIL import Image
import torchvision.transforms as T
import cv2
from tqdm import tqdm


class ImageTransforms:
    @staticmethod
    def get_transform(transform_name: str, params: Dict[str, Any]) -> T.Compose:
        if transform_name == "resize":
            return T.Compose(
                [
                    T.Resize((params.get("height", 224), params.get("width", 224))),
                    T.ToTensor(),
                ]
            )
        elif transform_name == "normalize":
            return T.Compose(
                [
                    T.ToTensor(),
                    T.Normalize(
                        mean=params.get("mean", [0.485, 0.456, 0.406]),
                        std=params.get("std", [0.229, 0.224, 0.225]),
                    ),
                ]
            )
        else:
            bt.logging.warning(f"Unknown transform {transform_name}, using default")
            return T.Compose([T.ToTensor()])


class CompetitionDataProcessor(ABC):
    @abstractmethod
    def process_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        pass


class DefaultDataProcessor(CompetitionDataProcessor):
    def process_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs


class CompetitionDataSource(ABC):
    def __init__(
        self, competition_directory: str, processor: CompetitionDataProcessor = None
    ):
        self.competition_directory = competition_directory
        self.processor = processor or DefaultDataProcessor()
        self._load_config()

    def _load_config(self):
        config_path = os.path.join(
            self.competition_directory, "competition_config.json"
        )
        try:
            with open(config_path) as f:
                self.config = json.load(f)
        except Exception as e:
            bt.logging.error(f"Error loading competition config: {e}")
            self.config = {}

    @abstractmethod
    def get_benchmark_data(self) -> torch.Tensor:
        pass

    @abstractmethod
    def sync_data(self) -> bool:
        pass


class RandomDataSource(CompetitionDataSource):
    def get_benchmark_data(self) -> torch.Tensor:
        input_shape = tuple(
            self.config.get("circuit_settings", {}).get("input_shape", [1, 10])
        )
        inputs = torch.randn(*input_shape)
        return self.processor.process_inputs(inputs)

    def sync_data(self) -> bool:
        return True


class RemoteDataSource(CompetitionDataSource):
    def __init__(
        self, competition_directory: str, processor: CompetitionDataProcessor = None
    ):
        super().__init__(competition_directory, processor)
        self.data_cache = {}
        self.data_config = self.config.get("data_source", {})
        self.data_url = self.data_config.get("url")
        self.local_cache_path = os.path.join(self.competition_directory, "data_cache")
        self.format = self.data_config.get("format", "npz")
        self.transform = None
        if self.data_config.get("input_transform"):
            self.transform = ImageTransforms.get_transform(
                self.data_config["input_transform"],
                self.data_config.get("transform_params", {}),
            )

    def _validate_url(self, url: str) -> bool:
        try:
            parsed = urlparse(url)
            return any(
                domain in parsed.netloc
                for domain in [
                    "r2.cloudflarestorage.com",
                    "s3.amazonaws.com",
                    ".r2.dev",
                ]
            )
        except Exception:
            return False

    def sync_data(self) -> bool:
        try:
            processed_path = os.path.join(self.competition_directory, "processed_64")
            zip_path = os.path.join(self.competition_directory, "age.zip")
            extracted_path = os.path.join(self.competition_directory, "extracted")

            if not os.path.exists(processed_path):
                os.makedirs(processed_path, exist_ok=True)
                os.makedirs(extracted_path, exist_ok=True)

                url = "https://storage.omron.ai/age.zip"

                print("Downloading dataset...")
                response = requests.get(url, stream=True)
                total_size = int(response.headers.get("content-length", 0))

                with (
                    open(zip_path, "wb") as f,
                    tqdm(
                        desc="Downloading", total=total_size, unit="iB", unit_scale=True
                    ) as pbar,
                ):
                    for data in response.iter_content(chunk_size=1024):
                        size = f.write(data)
                        pbar.update(size)

                print("Extracting zip...")
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(extracted_path)

                print("Processing images to 64x64...")
                for img_name in tqdm(os.listdir(extracted_path)):
                    if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                        img_path = os.path.join(extracted_path, img_name)
                        try:
                            img = cv2.imread(img_path)
                            if img is not None:
                                img = cv2.resize(img, (64, 64))
                                cv2.imwrite(os.path.join(processed_path, img_name), img)
                        except Exception as e:
                            bt.logging.warning(f"Failed to process {img_name}: {e}")

                os.remove(zip_path)
                import shutil

                shutil.rmtree(extracted_path)

                print(f"Dataset processed and saved to {processed_path}")

            return True

        except Exception as e:
            bt.logging.error(f"Failed to download/process dataset: {e}")
            return False

    def _load_image_batch(self, batch_size: int = 1) -> torch.Tensor:
        input_dir = os.path.join(
            self.local_cache_path, self.data_config.get("input_key", "")
        )
        pattern = self.data_config.get("input_pattern", "*")

        import glob

        image_files = glob.glob(os.path.join(input_dir, pattern))
        if not image_files:
            bt.logging.error(
                f"No images found matching pattern {pattern} in {input_dir}"
            )
            return None

        selected_files = random.sample(image_files, batch_size)
        images = []
        for img_path in selected_files:
            try:
                img = Image.open(img_path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                else:
                    img = T.ToTensor()(img)
                images.append(img)
            except Exception as e:
                bt.logging.error(f"Error loading image {img_path}: {e}")
                continue

        if not images:
            return None

        return torch.stack(images)

    def get_benchmark_data(self) -> torch.Tensor:
        if self.format == "npz":
            if not self.data_cache and os.path.exists(f"{self.local_cache_path}.npz"):
                try:
                    self.data_cache = np.load(f"{self.local_cache_path}.npz")
                except Exception as e:
                    bt.logging.error(f"Error loading cached data: {e}")
                    return None

            if not self.data_cache:
                if not self.sync_data():
                    return None

            try:
                inputs = torch.from_numpy(
                    self.data_cache[self.data_config.get("input_key", "inputs")]
                )
                return self.processor.process_inputs(inputs)
            except Exception as e:
                bt.logging.error(f"Error preparing benchmark data: {e}")
                return None
        else:  # zip/tar
            if not os.path.exists(self.local_cache_path):
                if not self.sync_data():
                    return None

            batch = self._load_image_batch()
            if batch is not None:
                return self.processor.process_inputs(batch)
            return None
