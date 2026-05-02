import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode, functional as TF


class AarizCephalometricDataset(Dataset):
    """
    Dataset wrapper for the Aariz Cephalometric dataset.

    It averages the junior and senior annotations and keeps a configurable
    subset of landmarks (26 by default, see `LANDMARK_INDEX_26`).
    """

    LANDMARK_INDEX_19 = np.array(
        [11, 5, 6, 16, 21, 3, 7, 4, 14, 15, 18, 22, 26, 25, 29, 28, 8, 2, 12],
        dtype=np.int32,
    ) - 1  # convert to zero-based

    # Keep the first 26 annotated landmarks (drops soft-tissue Nasion/Pogonion + Subnasale).
    LANDMARK_INDEX_26 = np.arange(26, dtype=np.int32)

    DEFAULT_AUGMENT_PARAMS: Dict[str, float] = {
        "max_rotate": 7.0,
        "max_translate": 0.05,  # fraction of width/height
        "min_scale": 0.9,
        "max_scale": 1.1,
        "max_shear": 5.0,
        "brightness": 0.15,
        "contrast": 0.2,
        "brightness_prob": 0.75,
        "contrast_prob": 0.75,
        "blur_prob": 0.2,
        "blur_radius": 1.0,
        "noise_prob": 0.5,
        "noise_std": 0.02,
    }

    def __init__(
        self,
        dataset_root: Path,
        split: str,
        image_size: int = 1024,
        heatmap_sigma: float = 1.5,
        heatmap_stride: int = 32,
        return_heatmap: bool = True,
        to_tensor: Optional[transforms.Compose] = None,
        landmark_mode: str = "26",
        augment: bool = False,
        augmentation_params: Optional[Dict[str, float]] = None,
        normalize: bool = False,
        preresized: bool = False,
    ) -> None:
        super().__init__()
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.image_size = image_size
        self.heatmap_sigma = heatmap_sigma
        self.heatmap_stride = heatmap_stride
        self.return_heatmap = return_heatmap
        self.augment = augment
        self.normalize = normalize
        self.landmark_indices = self._resolve_landmark_indices(landmark_mode)
        self.preresized = preresized
        self.original_sizes: Optional[Dict[str, Tuple[int, int]]] = None
        if preresized:
            sizes_path = self.dataset_root / "original_sizes.json"
            if not sizes_path.exists():
                raise FileNotFoundError(
                    f"preresized=True requires {sizes_path} from preresize_aariz.py"
                )
            with sizes_path.open() as f:
                raw = json.load(f)
            self.original_sizes = {k: (int(v[0]), int(v[1])) for k, v in raw.items()}
        self.resize = transforms.Resize(
            (image_size, image_size),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        self.to_tensor = to_tensor or transforms.Compose([transforms.ToTensor()])
        self.normalize_transform = transforms.Normalize(mean=[0.5], std=[0.5]) if normalize else None
        self.augmentation_params = {**self.DEFAULT_AUGMENT_PARAMS}
        if augmentation_params:
            self.augmentation_params.update(augmentation_params)

        self.image_dir = self.dataset_root / split / "Cephalograms"
        self.junior_dir = (
            self.dataset_root
            / split
            / "Annotations"
            / "Cephalometric Landmarks"
            / "Junior Orthodontists"
        )
        self.senior_dir = (
            self.dataset_root
            / split
            / "Annotations"
            / "Cephalometric Landmarks"
            / "Senior Orthodontists"
        )

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Missing images in {self.image_dir}")

        self.samples = sorted(
            [p for p in self.image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}]
        )
        if not self.samples:
            raise RuntimeError(f"No images found in {self.image_dir}")

        mapping_path = self.dataset_root / "cephalogram_machine_mappings.csv"
        if not mapping_path.exists():
            raise FileNotFoundError(f"Missing pixel size mapping at {mapping_path}")
        self.pixel_size_map: Dict[str, float] = {}
        with mapping_path.open() as f:
            for row in csv.DictReader(f):
                self.pixel_size_map[row["cephalogram_id"]] = float(row["pixel_size"])

        # Inspect one label to capture landmark names/order
        example_labels = self._load_annotations(self.samples[0].stem)
        self.landmark_names = example_labels["names"]
        self.num_landmarks = len(self.landmark_names)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path = self.samples[idx]
        labels = self._load_annotations(image_path.stem)
        coords = labels["coords"]

        image = Image.open(image_path).convert("L")

        if self.preresized:
            # Image already lives at target resolution; coords need pre-scaling
            # so augmentation + resize transforms remain consistent.
            try:
                base_w, base_h = self.original_sizes[image_path.stem]
            except KeyError as exc:
                raise KeyError(
                    f"original size missing for {image_path.stem} in preresized dataset"
                ) from exc
            pre_sx = self.image_size / float(base_w)
            pre_sy = self.image_size / float(base_h)
            coords = coords * np.array([pre_sx, pre_sy], dtype=np.float32)
            if self.augment:
                image, coords = self._apply_augmentations(image, coords)
            image = self.resize(image)  # no-op when already at target size
            resized_w, resized_h = image.size
            scale_x = resized_w / float(base_w)
            scale_y = resized_h / float(base_h)
            coords = self._clip_coords(coords, resized_w, resized_h)
        else:
            if self.augment:
                image, coords = self._apply_augmentations(image, coords)

            base_w, base_h = image.size
            image = self.resize(image)
            resized_w, resized_h = image.size

            scale_x = resized_w / float(base_w)
            scale_y = resized_h / float(base_h)

            coords = coords * np.array([scale_x, scale_y], dtype=np.float32)
            coords = self._clip_coords(coords, resized_w, resized_h)

        image_tensor = self.to_tensor(image)
        if self.normalize_transform is not None:
            image_tensor = self.normalize_transform(image_tensor)
        if self.augment and random.random() < self.augmentation_params["noise_prob"]:
            noise = torch.randn_like(image_tensor) * self.augmentation_params["noise_std"]
            image_tensor = (image_tensor + noise).clamp_(0.0, 1.0)

        coords_tensor = torch.from_numpy(coords).float()  # [K,2]

        try:
            pixel_size_mm = self.pixel_size_map[image_path.stem]
        except KeyError as exc:
            raise KeyError(f"Pixel size not found for {image_path.stem}") from exc

        meta_dict = {
            "image_id": image_path.stem,
            "original_width": float(base_w),
            "original_height": float(base_h),
            "scale_x": float(scale_x),
            "scale_y": float(scale_y),
            "pixel_size_mm": float(pixel_size_mm),
        }

        if not self.return_heatmap:
            return image_tensor, coords_tensor, meta_dict

        heatmap = self._generate_gaussian_heatmaps(coords_tensor, resized_h, resized_w)
        return image_tensor, coords_tensor, heatmap, meta_dict

    def _load_annotations(self, image_stem: str) -> Dict[str, np.ndarray]:
        junior_file = self.junior_dir / f"{image_stem}.json"
        senior_file = self.senior_dir / f"{image_stem}.json"
        if not junior_file.exists() or not senior_file.exists():
            raise FileNotFoundError(f"Missing annotation for {image_stem}")

        with junior_file.open() as f:
            junior = json.load(f)["landmarks"]
        with senior_file.open() as f:
            senior = json.load(f)["landmarks"]

        junior_coords, junior_names = self._extract_coords_and_names(junior)
        senior_coords, _ = self._extract_coords_and_names(senior)

        # Average and keep configured subset
        mean_coords = (junior_coords + senior_coords) * 0.5
        if self.landmark_indices is not None:
            idx = self.landmark_indices
            coords = mean_coords[idx]
            names = [junior_names[i] for i in idx]
        else:
            coords = mean_coords
            names = junior_names
        return {"coords": coords.astype(np.float32), "names": names}

    @staticmethod
    def _extract_coords_and_names(items: List[Dict]) -> Tuple[np.ndarray, List[str]]:
        coords = []
        names = []
        for item in items:
            coords.append([item["value"]["x"], item["value"]["y"]])
            names.append(item["title"])
        return np.asarray(coords, dtype=np.float32), names

    def _generate_gaussian_heatmaps(
        self,
        coords: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Generate Gaussian heatmaps at the stride expected by the model."""
        stride = self.heatmap_stride
        hm_h = height // stride
        hm_w = width // stride
        device = coords.device

        # Convert to heatmap coordinates
        coords_hm = coords / stride
        xs = coords_hm[:, 0].unsqueeze(-1).unsqueeze(-1)
        ys = coords_hm[:, 1].unsqueeze(-1).unsqueeze(-1)

        grid_y = torch.arange(hm_h, device=device).view(1, hm_h, 1)
        grid_x = torch.arange(hm_w, device=device).view(1, 1, hm_w)

        sigma = self.heatmap_sigma
        exp_term = (
            (grid_x - xs) ** 2 + (grid_y - ys) ** 2
        ) / (2 * sigma**2)
        heatmaps = torch.exp(-exp_term)
        heatmaps = heatmaps.clamp_(min=1e-6)  # avoid zeros for dice loss stability
        return heatmaps.float()

    def _resolve_landmark_indices(self, mode: str) -> Optional[np.ndarray]:
        if mode == "all":
            return None
        if mode == "19":
            return self.LANDMARK_INDEX_19
        if mode == "26":
            return self.LANDMARK_INDEX_26
        raise ValueError(f"Unsupported landmark_mode='{mode}'. Use one of: 19, 26, all.")

    @staticmethod
    def _clip_coords(coords: np.ndarray, width: float, height: float) -> np.ndarray:
        coords[:, 0] = np.clip(coords[:, 0], 0, width - 1)
        coords[:, 1] = np.clip(coords[:, 1], 0, height - 1)
        return coords

    def _apply_augmentations(self, image: Image.Image, coords: np.ndarray) -> Tuple[Image.Image, np.ndarray]:
        """Apply paired augmentations to image and coordinates."""
        params = self.augmentation_params
        w, h = image.size
        center = (w * 0.5, h * 0.5)

        angle = random.uniform(-params["max_rotate"], params["max_rotate"])
        translate_frac = params["max_translate"]
        translate = (
            random.uniform(-translate_frac, translate_frac) * w,
            random.uniform(-translate_frac, translate_frac) * h,
        )
        scale = random.uniform(params["min_scale"], params["max_scale"])
        shear_x = random.uniform(-params["max_shear"], params["max_shear"])
        shear_y = random.uniform(-params["max_shear"], params["max_shear"])
        shear = (shear_x, shear_y)

        image = TF.affine(
            image,
            angle=angle,
            translate=(int(translate[0]), int(translate[1])),
            scale=scale,
            shear=shear,
            fill=0,
        )
        coords = AarizCephalometricDataset._transform_coords_affine(coords, angle, translate, scale, shear, center)
        coords = self._clip_coords(coords, w, h)

        if random.random() < params["brightness_prob"]:
            factor = 1.0 + random.uniform(-params["brightness"], params["brightness"])
            image = TF.adjust_brightness(image, factor)
        if random.random() < params["contrast_prob"]:
            factor = 1.0 + random.uniform(-params["contrast"], params["contrast"])
            image = TF.adjust_contrast(image, factor)
        if random.random() < params["blur_prob"]:
            radius = random.uniform(0.1, params["blur_radius"])
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))

        return image, coords

    @staticmethod
    def _transform_coords_affine(
        coords: np.ndarray,
        angle: float,
        translate: Tuple[float, float],
        scale: float,
        shear: Tuple[float, float],
        center: Tuple[float, float],
    ) -> np.ndarray:
        # torchvision exposes the forward matrix with inverted=False
        matrix = TF._get_inverse_affine_matrix(
            center=center,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=shear,
            inverted=False,
        )
        a, b, c, d, e, f = matrix
        x = coords[:, 0]
        y = coords[:, 1]
        x_new = a * x + b * y + c
        y_new = d * x + e * y + f
        return np.stack([x_new, y_new], axis=1)


class CephAdoAduDataset(Dataset):
    """
    Dataset wrapper for the CephAdoAdu (Aariz_Cephalograms adolescent + adult) release.

    Layout:
        <root>/
            final_splits.json            # {"train":[[id, group], ...], "test":[...]}
            adult/dataset/<id>.jpg
            adult/txt/<id>.txt           # JSON list of {"data":[{"x":..,"y":..}], "type":"1"...}
            under_age/dataset/<id>.jpg
            under_age/txt/<id>.txt

    Returns items in the same 4-tuple shape as `AarizCephalometricDataset` so the
    existing training loop can be reused. MRE-in-mm assumes a constant
    `pixel_size_mm` (default 0.1 mm/px, typical for cephalometric radiographs).
    """

    DEFAULT_AUGMENT_PARAMS = AarizCephalometricDataset.DEFAULT_AUGMENT_PARAMS

    # Reuse the Aariz helpers as-is (they only need state that both classes expose).
    _apply_augmentations = AarizCephalometricDataset._apply_augmentations
    _generate_gaussian_heatmaps = AarizCephalometricDataset._generate_gaussian_heatmaps
    _transform_coords_affine = staticmethod(AarizCephalometricDataset._transform_coords_affine)
    _clip_coords = staticmethod(AarizCephalometricDataset._clip_coords)

    _VAL_FRACTION = 0.1
    _VAL_SEED = 42

    def __init__(
        self,
        dataset_root: Path,
        split: str,
        image_size: int = 1024,
        heatmap_sigma: float = 1.5,
        heatmap_stride: int = 32,
        return_heatmap: bool = True,
        to_tensor: Optional[transforms.Compose] = None,
        augment: bool = False,
        augmentation_params: Optional[Dict[str, float]] = None,
        normalize: bool = False,
        default_pixel_size_mm: float = 0.1,
        num_landmarks: int = 10,
    ) -> None:
        super().__init__()
        if split not in {"train", "valid", "test"}:
            raise ValueError(f"Unsupported split='{split}' for CephAdoAdu. Use train/valid/test.")
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.image_size = image_size
        self.heatmap_sigma = heatmap_sigma
        self.heatmap_stride = heatmap_stride
        self.return_heatmap = return_heatmap
        self.augment = augment
        self.normalize = normalize
        self.default_pixel_size_mm = float(default_pixel_size_mm)
        self.num_landmarks = int(num_landmarks)
        self.landmark_names = [str(i + 1) for i in range(self.num_landmarks)]

        self.resize = transforms.Resize(
            (image_size, image_size),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        self.to_tensor = to_tensor or transforms.Compose([transforms.ToTensor()])
        self.normalize_transform = transforms.Normalize(mean=[0.5], std=[0.5]) if normalize else None
        self.augmentation_params = {**self.DEFAULT_AUGMENT_PARAMS}
        if augmentation_params:
            self.augmentation_params.update(augmentation_params)

        splits_path = self.dataset_root / "final_splits.json"
        if not splits_path.exists():
            raise FileNotFoundError(f"Missing final_splits.json at {splits_path}")
        with splits_path.open() as f:
            raw_splits = json.load(f)

        if split == "test":
            entries = list(raw_splits["test"])
        else:
            train_entries = list(raw_splits["train"])
            rng = random.Random(self._VAL_SEED)
            indices = list(range(len(train_entries)))
            rng.shuffle(indices)
            val_count = max(1, int(round(len(train_entries) * self._VAL_FRACTION)))
            val_set = set(indices[:val_count])
            if split == "train":
                entries = [train_entries[i] for i in indices[val_count:]]
            else:  # valid
                entries = [train_entries[i] for i in indices[:val_count]]
            entries.sort()

        self.samples: List[Tuple[Path, Path, str]] = []
        missing = 0
        for image_id, group in entries:
            img_path = self.dataset_root / group / "dataset" / f"{image_id}.jpg"
            ann_path = self.dataset_root / group / "txt" / f"{image_id}.txt"
            if not img_path.exists() or not ann_path.exists():
                missing += 1
                continue
            self.samples.append((img_path, ann_path, image_id))

        if not self.samples:
            raise RuntimeError(f"No usable CephAdoAdu samples for split='{split}' at {self.dataset_root}")
        if missing:
            print(f"[CephAdoAdu:{split}] Skipped {missing} entries missing image or annotation.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, ann_path, image_id = self.samples[idx]
        coords = self._load_annotation(ann_path)

        image = Image.open(image_path).convert("L")
        if self.augment:
            image, coords = self._apply_augmentations(image, coords)

        base_w, base_h = image.size
        image = self.resize(image)
        resized_w, resized_h = image.size

        scale_x = resized_w / float(base_w)
        scale_y = resized_h / float(base_h)

        coords = coords * np.array([scale_x, scale_y], dtype=np.float32)
        coords = self._clip_coords(coords, resized_w, resized_h)

        image_tensor = self.to_tensor(image)
        if self.normalize_transform is not None:
            image_tensor = self.normalize_transform(image_tensor)
        if self.augment and random.random() < self.augmentation_params["noise_prob"]:
            noise = torch.randn_like(image_tensor) * self.augmentation_params["noise_std"]
            image_tensor = (image_tensor + noise).clamp_(0.0, 1.0)

        coords_tensor = torch.from_numpy(coords).float()

        meta_dict = {
            "image_id": image_id,
            "original_width": float(base_w),
            "original_height": float(base_h),
            "scale_x": float(scale_x),
            "scale_y": float(scale_y),
            "pixel_size_mm": float(self.default_pixel_size_mm),
        }

        if not self.return_heatmap:
            return image_tensor, coords_tensor, meta_dict

        heatmap = self._generate_gaussian_heatmaps(coords_tensor, resized_h, resized_w)
        return image_tensor, coords_tensor, heatmap, meta_dict

    def _load_annotation(self, path: Path) -> np.ndarray:
        with path.open() as f:
            records = json.load(f)
        # Sort by integer type so landmark order is stable
        records = sorted(records, key=lambda r: int(r["type"]))
        if len(records) < self.num_landmarks:
            raise ValueError(
                f"{path} has {len(records)} landmarks but expected at least {self.num_landmarks}"
            )
        coords = np.zeros((self.num_landmarks, 2), dtype=np.float32)
        for i in range(self.num_landmarks):
            point = records[i]["data"][0]
            coords[i, 0] = float(point["x"])
            coords[i, 1] = float(point["y"])
        return coords


class ISBI2015Dataset(Dataset):
    """
    Dataset wrapper for the ISBI 2015 Cephalometric Challenge release
    (figshare 37ec464af8e81ae6ebbf).

    Layout:
        <root>/
            RawImage/TrainingData/{001..150}.bmp
            RawImage/Test1Data/{151..300}.bmp
            RawImage/Test2Data/{301..400}.bmp
            400_junior/{001..400}.txt   # 19 lines "x,y" + trailing classification rows
            400_senior/{001..400}.txt

    19 landmarks per image. Annotations are averaged across the junior + senior
    orthodontists, matching the standard ISBI evaluation protocol.

    Splits:
        - "train"  -> RawImage/TrainingData (150 images, 001-150)
        - "valid"  -> RawImage/Test1Data    (150 images, 151-300)
        - "test1"  -> alias for valid
        - "test"   -> RawImage/Test2Data    (100 images, 301-400)
        - "test2"  -> alias for test

    The ISBI 2015 scanner pixel size is 0.1 mm (configurable via
    `default_pixel_size_mm`).
    """

    NUM_LANDMARKS = 19

    DEFAULT_AUGMENT_PARAMS = AarizCephalometricDataset.DEFAULT_AUGMENT_PARAMS

    _apply_augmentations = AarizCephalometricDataset._apply_augmentations
    _generate_gaussian_heatmaps = AarizCephalometricDataset._generate_gaussian_heatmaps
    _transform_coords_affine = staticmethod(AarizCephalometricDataset._transform_coords_affine)
    _clip_coords = staticmethod(AarizCephalometricDataset._clip_coords)

    _SPLIT_DIRS = {
        "train": "TrainingData",
        "valid": "Test1Data",
        "test1": "Test1Data",
        "test":  "Test2Data",
        "test2": "Test2Data",
    }

    def __init__(
        self,
        dataset_root: Path,
        split: str,
        image_size: int = 1024,
        heatmap_sigma: float = 1.5,
        heatmap_stride: int = 32,
        return_heatmap: bool = True,
        to_tensor: Optional[transforms.Compose] = None,
        augment: bool = False,
        augmentation_params: Optional[Dict[str, float]] = None,
        normalize: bool = False,
        default_pixel_size_mm: float = 0.1,
    ) -> None:
        super().__init__()
        if split not in self._SPLIT_DIRS:
            raise ValueError(
                f"Unsupported split='{split}' for ISBI2015. "
                f"Use one of: {sorted(self._SPLIT_DIRS)}."
            )
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.image_size = image_size
        self.heatmap_sigma = heatmap_sigma
        self.heatmap_stride = heatmap_stride
        self.return_heatmap = return_heatmap
        self.augment = augment
        self.normalize = normalize
        self.default_pixel_size_mm = float(default_pixel_size_mm)
        self.num_landmarks = self.NUM_LANDMARKS
        self.landmark_names = [str(i + 1) for i in range(self.num_landmarks)]

        self.resize = transforms.Resize(
            (image_size, image_size),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        self.to_tensor = to_tensor or transforms.Compose([transforms.ToTensor()])
        self.normalize_transform = transforms.Normalize(mean=[0.5], std=[0.5]) if normalize else None
        self.augmentation_params = {**self.DEFAULT_AUGMENT_PARAMS}
        if augmentation_params:
            self.augmentation_params.update(augmentation_params)

        image_dir = self.dataset_root / "RawImage" / self._SPLIT_DIRS[split]
        if not image_dir.exists():
            raise FileNotFoundError(f"Missing ISBI image folder: {image_dir}")
        self.image_dir = image_dir

        self.junior_dir = self.dataset_root / "400_junior"
        self.senior_dir = self.dataset_root / "400_senior"
        if not self.junior_dir.exists() or not self.senior_dir.exists():
            raise FileNotFoundError(
                f"Missing ISBI annotation folders at {self.junior_dir} / {self.senior_dir}"
            )

        self.samples = sorted(
            p for p in image_dir.iterdir()
            if p.suffix.lower() in {".bmp", ".png", ".jpg", ".jpeg"}
        )
        if not self.samples:
            raise RuntimeError(f"No ISBI images in {image_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path = self.samples[idx]
        image_id = image_path.stem
        coords = self._load_annotation(image_id)

        image = Image.open(image_path).convert("L")
        if self.augment:
            image, coords = self._apply_augmentations(image, coords)

        base_w, base_h = image.size
        image = self.resize(image)
        resized_w, resized_h = image.size

        scale_x = resized_w / float(base_w)
        scale_y = resized_h / float(base_h)

        coords = coords * np.array([scale_x, scale_y], dtype=np.float32)
        coords = self._clip_coords(coords, resized_w, resized_h)

        image_tensor = self.to_tensor(image)
        if self.normalize_transform is not None:
            image_tensor = self.normalize_transform(image_tensor)
        if self.augment and random.random() < self.augmentation_params["noise_prob"]:
            noise = torch.randn_like(image_tensor) * self.augmentation_params["noise_std"]
            image_tensor = (image_tensor + noise).clamp_(0.0, 1.0)

        coords_tensor = torch.from_numpy(coords).float()

        meta_dict = {
            "image_id": image_id,
            "original_width": float(base_w),
            "original_height": float(base_h),
            "scale_x": float(scale_x),
            "scale_y": float(scale_y),
            "pixel_size_mm": float(self.default_pixel_size_mm),
        }

        if not self.return_heatmap:
            return image_tensor, coords_tensor, meta_dict

        heatmap = self._generate_gaussian_heatmaps(coords_tensor, resized_h, resized_w)
        return image_tensor, coords_tensor, heatmap, meta_dict

    def _load_annotation(self, image_id: str) -> np.ndarray:
        junior_path = self.junior_dir / f"{image_id}.txt"
        senior_path = self.senior_dir / f"{image_id}.txt"
        if not junior_path.exists() or not senior_path.exists():
            raise FileNotFoundError(
                f"Missing ISBI annotation for {image_id} (junior={junior_path.exists()}, senior={senior_path.exists()})"
            )
        junior = self._parse_landmark_file(junior_path)
        senior = self._parse_landmark_file(senior_path)
        return ((junior + senior) * 0.5).astype(np.float32)

    def _parse_landmark_file(self, path: Path) -> np.ndarray:
        coords: List[List[float]] = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line or "," not in line:
                    continue
                parts = line.split(",")
                if len(parts) < 2:
                    continue
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                except ValueError:
                    continue
                coords.append([x, y])
                if len(coords) >= self.NUM_LANDMARKS:
                    break
        if len(coords) < self.NUM_LANDMARKS:
            raise ValueError(
                f"{path} has {len(coords)} landmarks, expected {self.NUM_LANDMARKS}"
            )
        return np.asarray(coords, dtype=np.float32)


def create_isbi2015_dataloaders(
    dataset_root: Path,
    batch_size: int = 2,
    num_workers: int = 2,
    image_size: int = 1024,
    heatmap_sigma: float = 1.5,
    augment_train: bool = False,
    augmentation_params: Optional[Dict[str, float]] = None,
    normalize: bool = False,
    default_pixel_size_mm: float = 0.1,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    train_ds = ISBI2015Dataset(
        dataset_root=dataset_root,
        split="train",
        image_size=image_size,
        heatmap_sigma=heatmap_sigma,
        augment=augment_train,
        augmentation_params=augmentation_params,
        normalize=normalize,
        default_pixel_size_mm=default_pixel_size_mm,
    )
    val_ds = ISBI2015Dataset(
        dataset_root=dataset_root,
        split="valid",
        image_size=image_size,
        heatmap_sigma=heatmap_sigma,
        augment=False,
        augmentation_params=augmentation_params,
        normalize=normalize,
        default_pixel_size_mm=default_pixel_size_mm,
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def create_cephadoadu_dataloaders(
    dataset_root: Path,
    batch_size: int = 2,
    num_workers: int = 2,
    image_size: int = 1024,
    heatmap_sigma: float = 1.5,
    augment_train: bool = False,
    augmentation_params: Optional[Dict[str, float]] = None,
    normalize: bool = False,
    default_pixel_size_mm: float = 0.1,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    train_ds = CephAdoAduDataset(
        dataset_root=dataset_root,
        split="train",
        image_size=image_size,
        heatmap_sigma=heatmap_sigma,
        augment=augment_train,
        augmentation_params=augmentation_params,
        normalize=normalize,
        default_pixel_size_mm=default_pixel_size_mm,
    )
    val_ds = CephAdoAduDataset(
        dataset_root=dataset_root,
        split="valid",
        image_size=image_size,
        heatmap_sigma=heatmap_sigma,
        augment=False,
        augmentation_params=augmentation_params,
        normalize=normalize,
        default_pixel_size_mm=default_pixel_size_mm,
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def create_aariz_dataloaders(
    dataset_root: Path,
    batch_size: int = 2,
    num_workers: int = 2,
    image_size: int = 1024,
    heatmap_sigma: float = 1.5,
    landmark_mode: str = "26",
    augment_train: bool = False,
    augmentation_params: Optional[Dict[str, float]] = None,
    normalize: bool = False,
    preresized: bool = False,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Convenience helper that returns (train_loader, val_loader).
    """
    train_ds = AarizCephalometricDataset(
        dataset_root=dataset_root,
        split="train",
        image_size=image_size,
        heatmap_sigma=heatmap_sigma,
        landmark_mode=landmark_mode,
        augment=augment_train,
        augmentation_params=augmentation_params,
        normalize=normalize,
        preresized=preresized,
    )
    val_ds = AarizCephalometricDataset(
        dataset_root=dataset_root,
        split="valid",
        image_size=image_size,
        heatmap_sigma=heatmap_sigma,
        landmark_mode=landmark_mode,
        augment=False,
        augmentation_params=augmentation_params,
        normalize=normalize,
        preresized=preresized,
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
