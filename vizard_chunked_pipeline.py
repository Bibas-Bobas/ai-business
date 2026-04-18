from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import asdict, dataclass
from typing import Dict, Optional

import rasterio
from shapely.geometry import Polygon, box

from geotifs_manager import VizardDataManager
from paired_tif_chunking import (
    DEFAULT_CHUNK_SIZE,
    chunk_geotiff_pair,
    reconstruct_geotiff_pair_from_manifest,
)


@dataclass(frozen=True)
class ProcessedChunkManifest:
    output_dir: str
    image_chunks_dir: str
    mask_chunks_dir: str
    manifest_path: str
    chunk_count: int
    processing_kind: str


@dataclass(frozen=True)
class VizardChunkedPipelineResult:
    polygon_bounds: tuple[float, float, float, float]
    run_dir: str
    export_result: Dict[str, object]
    chunk_result: Dict[str, object]
    processed_chunk_result: ProcessedChunkManifest
    reconstruction_result: Dict[str, str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "polygon_bounds": self.polygon_bounds,
            "run_dir": self.run_dir,
            "export_result": self.export_result,
            "chunk_result": self.chunk_result,
            "processed_chunk_result": asdict(self.processed_chunk_result),
            "reconstruction_result": self.reconstruction_result,
        }


class VizardChunkedInferencePipeline:
    """
    Склеивает гео-подготовку, чанкование, инференс по чанкам и обратную сборку.

    Архитектура по шагам:
    1. `VizardDataManager` строит GeoTIFF пары `image + mask` по полигону.
    2. `paired_tif_chunking` режет пару на синхронные чанки.
    3. `run_model_on_chunk_pairs()` обрабатывает чанки. Сейчас это заглушка passthrough.
    4. `paired_tif_chunking` собирает обработанные чанки обратно в GeoTIFF.
    """

    def __init__(
        self,
        data_dir: str,
        *,
        history_depth: int = 5,
        buffer_fraction: float = 0.2,
        tile_size: int = 1024,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ):
        self.data_dir = os.path.abspath(data_dir)
        self.history_depth = history_depth
        self.buffer_fraction = buffer_fraction
        self.tile_size = tile_size
        self.chunk_size = chunk_size
        self.data_manager = VizardDataManager(data_dir=self.data_dir)

    def run(
        self,
        polygon: Polygon,
        output_dir: str,
        *,
        run_name: str = "polygon_chunked_pipeline",
    ) -> VizardChunkedPipelineResult:
        run_dir = os.path.join(os.path.abspath(output_dir), run_name)
        export_dir = os.path.join(run_dir, "01_export")
        chunks_dir = os.path.join(run_dir, "02_chunks")
        processed_dir = os.path.join(run_dir, "03_model_output")
        reconstructed_dir = os.path.join(run_dir, "04_reconstructed")
        os.makedirs(run_dir, exist_ok=True)

        export_result = self.export_pair_from_polygon(polygon, export_dir, run_name=run_name)
        chunk_result = self.chunk_exported_pair(export_result, chunks_dir)
        processed_chunk_result = self.run_model_on_chunk_pairs(
            chunk_result["manifest_path"],
            processed_dir,
        )
        reconstruction_result = reconstruct_geotiff_pair_from_manifest(
            processed_chunk_result.manifest_path,
            reconstructed_dir,
        )

        return VizardChunkedPipelineResult(
            polygon_bounds=tuple(polygon.bounds),
            run_dir=run_dir,
            export_result=export_result,
            chunk_result=chunk_result,
            processed_chunk_result=processed_chunk_result,
            reconstruction_result=reconstruction_result,
        )

    def export_pair_from_polygon(
        self,
        polygon: Polygon,
        output_dir: str,
        *,
        run_name: str,
    ) -> Dict[str, object]:
        os.makedirs(output_dir, exist_ok=True)
        export_prefix = os.path.join(output_dir, run_name)
        return self.data_manager.export_polygon_cropped_stack_and_mask(
            polygon,
            history_depth=self.history_depth,
            buffer_fraction=self.buffer_fraction,
            output_prefix=export_prefix,
            tile_size=self.tile_size,
        )

    def chunk_exported_pair(
        self,
        export_result: Dict[str, object],
        output_dir: str,
    ) -> Dict[str, object]:
        paths = export_result["paths"]
        return chunk_geotiff_pair(
            paths["stack_cropped_tif"],
            paths["class0_mask_tif"],
            output_dir,
            chunk_size=self.chunk_size,
            pad_edges=True,
        )

    def run_model_on_chunk_pairs(
        self,
        chunk_manifest_path: str,
        output_dir: str,
    ) -> ProcessedChunkManifest:
        """
        Extension point под реальную нейронку.

        Текущая реализация ничего не меняет в данных и просто копирует image/mask чанки
        в новую папку, после чего переписывает manifest под новые пути.
        """
        return self._run_passthrough_stub(chunk_manifest_path, output_dir)

    def _run_passthrough_stub(
        self,
        chunk_manifest_path: str,
        output_dir: str,
    ) -> ProcessedChunkManifest:
        chunk_manifest_path = os.path.abspath(chunk_manifest_path)
        output_dir = os.path.abspath(output_dir)
        image_dir = os.path.join(output_dir, "image_chunks")
        mask_dir = os.path.join(output_dir, "mask_chunks")
        manifest_path = os.path.join(output_dir, "chunk_manifest.json")
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        with open(chunk_manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        if manifest.get("kind") != "paired_geotiff_chunks":
            raise ValueError("Ожидался manifest формата paired_geotiff_chunks.")

        processed_chunks = []
        for chunk_info in manifest["chunks"]:
            source_image_path = chunk_info["image_path"]
            source_mask_path = chunk_info["mask_path"]
            target_image_path = os.path.join(image_dir, os.path.basename(source_image_path))
            target_mask_path = os.path.join(mask_dir, os.path.basename(source_mask_path))

            shutil.copy2(source_image_path, target_image_path)
            shutil.copy2(source_mask_path, target_mask_path)

            updated_chunk = dict(chunk_info)
            updated_chunk["image_path"] = target_image_path
            updated_chunk["mask_path"] = target_mask_path
            processed_chunks.append(updated_chunk)

        processed_manifest = dict(manifest)
        processed_manifest["image_chunks_dir"] = image_dir
        processed_manifest["mask_chunks_dir"] = mask_dir
        processed_manifest["chunks"] = processed_chunks
        processed_manifest["source_manifest_path"] = chunk_manifest_path
        processed_manifest["processing_kind"] = "passthrough_stub_copy"

        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(processed_manifest, handle, ensure_ascii=False, indent=2)

        return ProcessedChunkManifest(
            output_dir=output_dir,
            image_chunks_dir=image_dir,
            mask_chunks_dir=mask_dir,
            manifest_path=manifest_path,
            chunk_count=len(processed_chunks),
            processing_kind="passthrough_stub_copy",
        )


def build_polygon_around_tif(
    target_file: str,
    *,
    area_scale: float = 3.0,
) -> Polygon:
    """
    Строит прямоугольный полигон вокруг сцены с тем же центром.

    `area_scale=3.0` повторяет идею теста из `geotifs_manager.py`: область примерно
    в 3 раза больше исходного фрейма по площади.
    """
    if area_scale <= 0:
        raise ValueError("area_scale должен быть положительным.")

    target_file = os.path.abspath(target_file)
    with rasterio.open(target_file) as src:
        bounds = src.bounds
        center_x = (bounds.left + bounds.right) / 2.0
        center_y = (bounds.bottom + bounds.top) / 2.0
        scene_width = bounds.right - bounds.left
        scene_height = bounds.top - bounds.bottom
        half_width = scene_width * math.sqrt(area_scale) / 2.0
        half_height = scene_height * math.sqrt(area_scale) / 2.0
        return box(
            center_x - half_width,
            center_y - half_height,
            center_x + half_width,
            center_y + half_height,
        )


def run_file_polygon_pipeline_test(
    target_file: str,
    data_dir: Optional[str] = None,
    *,
    output_dir: str = "test_output3",
    run_name: str = "polygon_chunked_pipeline_demo",
    history_depth: int = 5,
    buffer_fraction: float = 0.2,
    tile_size: int = 1024,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Dict[str, object]:
    print("=== ЗАПУСК CHUNKED PIPELINE TEST ===")
    target_file = os.path.abspath(target_file)
    if data_dir is None:
        data_dir = os.path.dirname(target_file)
    else:
        data_dir = os.path.abspath(data_dir)

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Папка с данными не найдена: {data_dir}")

    polygon = build_polygon_around_tif(target_file, area_scale=3.0)
    with rasterio.open(target_file) as src:
        print(f"[PIPELINE TEST 0] data_dir: {data_dir}")
        print(f"[PIPELINE TEST 1] target_file: {target_file}")
        print(f"[PIPELINE TEST 1] CRS: {src.crs}")
        print(f"[PIPELINE TEST 1] source bounds: {src.bounds}")
        print(f"[PIPELINE TEST 1] polygon bounds: {polygon.bounds}")
        print(f"[PIPELINE TEST 1] shape: {src.height}x{src.width}, bands={src.count}")

    pipeline = VizardChunkedInferencePipeline(
        data_dir=data_dir,
        history_depth=history_depth,
        buffer_fraction=buffer_fraction,
        tile_size=tile_size,
        chunk_size=chunk_size,
    )
    result = pipeline.run(
        polygon,
        output_dir=output_dir,
        run_name=run_name,
    )

    summary = result.to_dict()
    print(f"[PIPELINE TEST 2] history dates: {summary['export_result']['history']}")
    print(f"[PIPELINE TEST 3] chunk count: {summary['chunk_result']['chunk_count']}")
    print(
        "[PIPELINE TEST 4] reconstructed image:",
        summary["reconstruction_result"]["image_reconstructed_path"],
    )
    print(
        "[PIPELINE TEST 4] reconstructed mask:",
        summary["reconstruction_result"]["mask_reconstructed_path"],
    )
    print("=== CHUNKED PIPELINE TEST ЗАВЕРШЕН ===")
    return summary


if __name__ == "__main__":
    TIFS_PATH = "Dataset_2025_IceClass/"
    result = run_file_polygon_pipeline_test(
        TIFS_PATH
        + "S1A_EW_GRDM_1SDH_20250218T031712_20250218T031812_057945_07267C_D886_fix_IceClass.tif"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
