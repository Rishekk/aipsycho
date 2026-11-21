"""
Data-processing package.

Names are imported lazily (PEP 562) so that dependency-light modules such as
``src.data_processing.splits`` can be imported without torch / librosa / OpenCV.
``from src.data_processing import FER2013Dataset`` etc. keep working.
"""
import importlib

_LAZY = {
    # dataset_loader
    "FER2013Dataset": ".dataset_loader", "RAFDBDataset": ".dataset_loader",
    "RAVDESSDataset": ".dataset_loader", "PoseSequenceDataset": ".dataset_loader",
    "get_face_loaders": ".dataset_loader", "get_speech_loaders": ".dataset_loader",
    "get_pose_loaders": ".dataset_loader", "make_weighted_sampler": ".dataset_loader",
    "map_to_psych_state": ".dataset_loader",
    # preprocessor
    "preprocess_face": ".preprocessor", "preprocess_face_batch": ".preprocessor",
    "preprocess_audio": ".preprocessor", "audio_to_mfcc_tensor": ".preprocessor",
    "normalize_pose": ".pose_norm",
    "normalize_pose_sequence": ".pose_norm",
    # data_augmentation
    "get_face_train_transforms": ".data_augmentation",
    "get_face_val_transforms": ".data_augmentation",
    "AudioAugmentor": ".data_augmentation", "PoseAugmentor": ".data_augmentation",
    # synchronizer
    "AudioVideoSynchronizer": ".synchronizer", "MultimodalDatasetBuilder": ".synchronizer",
}

__all__ = list(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
