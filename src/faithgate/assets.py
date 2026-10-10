"""Pinned external assets: source, licence and the sha256 observed on first download."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetFile:
    name: str
    url: str
    sha256: str
    size: int


@dataclass(frozen=True)
class ModelFile:
    filename: str
    sha256: str
    size: int


RAGTRUTH_COMMIT = "c103204b9ce28d6bbad859304bf30de72b8ed8fe"
_RAGTRUTH_RAW = (
    f"https://raw.githubusercontent.com/ParticleMedia/RAGTruth/{RAGTRUTH_COMMIT}/dataset"
)

RAGTRUTH_FILES: tuple[DatasetFile, ...] = (
    DatasetFile(
        name="response.jsonl",
        url=f"{_RAGTRUTH_RAW}/response.jsonl",
        sha256="e4c2e4ac24fff676d8984cc61c35d791612fadc58015335d97dd632375e18073",
        size=21458735,
    ),
    DatasetFile(
        name="source_info.jsonl",
        url=f"{_RAGTRUTH_RAW}/source_info.jsonl",
        sha256="0dffc26ea9f3c1c3d7c7e8336b56ef1646e3cec876edffcca3c9c624d12d578b",
        size=15117971,
    ),
)
RAGTRUTH_LICENSE = (
    "MIT for the RAGTruth annotations and code (ParticleMedia/RAGTruth); the source passages come "
    "from MS MARCO, CNN/Daily Mail and recent news articles, and the Yelp Open Dataset, under "
    "their own terms"
)

NLI_REPO = "MoritzLaurer/xtremedistil-l6-h256-zeroshot-v1.1-all-33"
NLI_REVISION = "c07f66d9cbf781191bee66edfe8ad7856f045781"
NLI_LICENSE = "MIT"
NLI_FILES: tuple[ModelFile, ...] = (
    ModelFile(
        filename="onnx/model.onnx",
        sha256="89ca1f96830106de21fcea15b55685060dd6faafd7864a790594910f2e661e3f",
        size=51140063,
    ),
    ModelFile(
        filename="tokenizer.json",
        sha256="d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
        size=711396,
    ),
    ModelFile(
        filename="config.json",
        sha256="7056598267ea0cf462d3b35c5f42c1f0be2bcb06f5e301a1861d7d6f957ca228",
        size=882,
    ),
)
