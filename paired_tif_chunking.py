from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject
from rasterio.windows import Window, transform as window_transform


DEFAULT_CHUNK_SIZE = 384


@dataclass(frozen=True)
class ChunkPairRecord:
    """Информация об одной сохраненной паре чанков."""

    row_off: int
    col_off: int
    width: int
    height: int
    image_path: str
    mask_path: str


def _normalize_resampling(value: str | Resampling) -> Resampling:
    if isinstance(value, Resampling):
        return value
    try:
        return Resampling[str(value).lower()]
    except KeyError as error:
        supported = ", ".join(item.name for item in Resampling)
        raise ValueError(f"Неизвестный resampling '{value}'. Доступно: {supported}") from error


def _iter_chunk_windows(width: int, height: int, chunk_size: int) -> Iterable[Window]:
    if chunk_size <= 0:
        raise ValueError("chunk_size должен быть положительным.")
    for row_off in range(0, height, chunk_size):
        for col_off in range(0, width, chunk_size):
            yield Window(
                col_off=col_off,
                row_off=row_off,
                width=min(chunk_size, width - col_off),
                height=min(chunk_size, height - row_off),
            )


def _resolve_pad_value(nodata, dtype: np.dtype) -> float | int:
    if nodata is None:
        return 0.0 if np.issubdtype(dtype, np.floating) else 0
    if np.issubdtype(dtype, np.floating):
        return float(nodata)
    return int(nodata)


def _pad_array_to_size(data: np.ndarray, chunk_size: int, pad_value: float | int) -> np.ndarray:
    bands, height, width = data.shape
    if height == chunk_size and width == chunk_size:
        return data
    out = np.full((bands, chunk_size, chunk_size), pad_value, dtype=data.dtype)
    out[:, :height, :width] = data
    return out


def _pad_mask_to_size(mask: np.ndarray, chunk_size: int) -> np.ndarray:
    height, width = mask.shape
    if height == chunk_size and width == chunk_size:
        return mask
    out = np.zeros((chunk_size, chunk_size), dtype=mask.dtype)
    out[:height, :width] = mask
    return out


def _datasets_share_grid(left: rasterio.io.DatasetReader, right: rasterio.io.DatasetReader) -> bool:
    return (
        left.width == right.width
        and left.height == right.height
        and str(left.crs) == str(right.crs)
        and np.allclose(tuple(left.transform)[:6], tuple(right.transform)[:6], atol=1e-9)
    )


def _write_chunk(
    path: str,
    data: np.ndarray,
    valid_mask: np.ndarray,
    *,
    crs,
    transform: Affine,
    nodata,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    meta = {
        "driver": "GTiff",
        "height": int(data.shape[1]),
        "width": int(data.shape[2]),
        "count": int(data.shape[0]),
        "dtype": np.dtype(data.dtype).name,
        "crs": crs,
        "transform": transform,
        "compress": "LZW",
        "tiled": True,
        "BIGTIFF": "IF_SAFER",
    }
    if nodata is not None:
        meta["nodata"] = nodata
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(data)
        dst.write_mask(valid_mask.astype(np.uint8) * 255)


def _to_serializable_transform(transform: Affine) -> List[float]:
    return [float(value) for value in tuple(transform)[:6]]


def _from_serialized_transform(values: List[float]) -> Affine:
    if len(values) != 6:
        raise ValueError("В manifest transform должен содержать 6 чисел.")
    return Affine(*values)


def _write_chunk_manifest(path: str, payload: Dict[str, object]) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return path


def resize_geotiff(
    input_tif_path: str,
    output_tif_path: str,
    *,
    scale_factor: float = 0.5,
    resampling: str | Resampling = Resampling.nearest,
) -> str:
    """
    Уменьшает GeoTIFF по ширине/высоте с сохранением геопривязки.

    Для IceClass и масок по умолчанию используется `nearest`, чтобы не смешивать классы.
    """
    if not (0 < scale_factor <= 1):
        raise ValueError("scale_factor должен быть в диапазоне (0, 1].")

    resampling_method = _normalize_resampling(resampling)
    input_tif_path = os.path.abspath(input_tif_path)
    output_tif_path = os.path.abspath(output_tif_path)
    output_dir = os.path.dirname(output_tif_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with rasterio.open(input_tif_path) as src:
        new_width = max(1, int(math.ceil(src.width * scale_factor)))
        new_height = max(1, int(math.ceil(src.height * scale_factor)))
        scale_x = src.width / new_width
        scale_y = src.height / new_height
        dst_transform = src.transform * Affine.scale(scale_x, scale_y)

        dst_data = np.zeros((src.count, new_height, new_width), dtype=np.dtype(src.dtypes[0]))
        dst_valid_mask = np.zeros((new_height, new_width), dtype=np.uint8)

        for band_index in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, band_index),
                destination=dst_data[band_index - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=src.crs,
                src_nodata=src.nodata,
                dst_nodata=src.nodata,
                resampling=resampling_method,
            )

        reproject(
            source=src.dataset_mask(),
            destination=dst_valid_mask,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=src.crs,
            src_nodata=0,
            dst_nodata=0,
            resampling=Resampling.nearest,
        )

        meta = src.meta.copy()
        meta.update(
            {
                "height": new_height,
                "width": new_width,
                "transform": dst_transform,
                "compress": "LZW",
                "tiled": True,
                "BIGTIFF": "IF_SAFER",
            }
        )

        with rasterio.open(output_tif_path, "w", **meta) as dst:
            dst.write(dst_data)
            dst.write_mask((dst_valid_mask > 0).astype(np.uint8) * 255)

    return output_tif_path


def chunk_geotiff_pair(
    image_tif_path: str,
    mask_tif_path: str,
    output_dir: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    pad_edges: bool = True,
    image_prefix: str = "image",
    mask_prefix: str = "mask",
) -> Dict[str, object]:
    """
    Синхронно режет `image_tif_path` и `mask_tif_path` на одинаковые окна.

    Выходная структура:
    - `output_dir/image_chunks/*.tif`
    - `output_dir/mask_chunks/*.tif`
    - `output_dir/chunk_manifest.json`
    """
    image_tif_path = os.path.abspath(image_tif_path)
    mask_tif_path = os.path.abspath(mask_tif_path)
    output_dir = os.path.abspath(output_dir)
    image_dir = os.path.join(output_dir, "image_chunks")
    mask_dir = os.path.join(output_dir, "mask_chunks")
    manifest_path = os.path.join(output_dir, "chunk_manifest.json")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    records: List[ChunkPairRecord] = []
    with rasterio.open(image_tif_path) as image_src, rasterio.open(mask_tif_path) as mask_src:
        if not _datasets_share_grid(image_src, mask_src):
            raise ValueError(
                "TIFF и mask должны лежать на одной сетке: одинаковые width/height, CRS и transform."
            )

        manifest_payload: Dict[str, object] = {
            "kind": "paired_geotiff_chunks",
            "image_tif_path": image_tif_path,
            "mask_tif_path": mask_tif_path,
            "chunk_size": int(chunk_size),
            "pad_edges": bool(pad_edges),
            "source_grid": {
                "width": int(image_src.width),
                "height": int(image_src.height),
                "crs": str(image_src.crs) if image_src.crs is not None else None,
                "transform": _to_serializable_transform(image_src.transform),
            },
            "image_meta": {
                "count": int(image_src.count),
                "dtype": str(np.dtype(image_src.dtypes[0]).name),
                "nodata": image_src.nodata,
            },
            "mask_meta": {
                "count": int(mask_src.count),
                "dtype": str(np.dtype(mask_src.dtypes[0]).name),
                "nodata": mask_src.nodata,
            },
            "image_chunks_dir": image_dir,
            "mask_chunks_dir": mask_dir,
            "chunks": [],
        }
        image_pad_value = _resolve_pad_value(image_src.nodata, np.dtype(image_src.dtypes[0]))
        mask_pad_value = _resolve_pad_value(mask_src.nodata, np.dtype(mask_src.dtypes[0]))

        for window in _iter_chunk_windows(image_src.width, image_src.height, chunk_size):
            row_off = int(window.row_off)
            col_off = int(window.col_off)
            window_width = int(window.width)
            window_height = int(window.height)

            image_chunk = image_src.read(window=window)
            mask_chunk = mask_src.read(window=window)
            image_valid_mask = image_src.read_masks(window=window).all(axis=0) > 0
            mask_valid_mask = mask_src.read_masks(window=window).all(axis=0) > 0

            if pad_edges:
                image_chunk = _pad_array_to_size(image_chunk, chunk_size, image_pad_value)
                mask_chunk = _pad_array_to_size(mask_chunk, chunk_size, mask_pad_value)
                image_valid_mask = _pad_mask_to_size(image_valid_mask, chunk_size)
                mask_valid_mask = _pad_mask_to_size(mask_valid_mask, chunk_size)
                write_window = Window(col_off, row_off, chunk_size, chunk_size)
            else:
                if int(window.width) != chunk_size or int(window.height) != chunk_size:
                    continue
                write_window = window

            chunk_transform = window_transform(write_window, image_src.transform)
            base_name = f"r{row_off:06d}_c{col_off:06d}"
            image_chunk_path = os.path.join(image_dir, f"{image_prefix}_{base_name}.tif")
            mask_chunk_path = os.path.join(mask_dir, f"{mask_prefix}_{base_name}.tif")

            _write_chunk(
                image_chunk_path,
                image_chunk,
                image_valid_mask,
                crs=image_src.crs,
                transform=chunk_transform,
                nodata=image_src.nodata,
            )
            _write_chunk(
                mask_chunk_path,
                mask_chunk,
                mask_valid_mask,
                crs=mask_src.crs,
                transform=chunk_transform,
                nodata=mask_src.nodata,
            )

            records.append(
                ChunkPairRecord(
                    row_off=row_off,
                    col_off=col_off,
                    width=window_width,
                    height=window_height,
                    image_path=image_chunk_path,
                    mask_path=mask_chunk_path,
                )
            )
            manifest_payload["chunks"].append(
                {
                    "row_off": row_off,
                    "col_off": col_off,
                    "width": window_width,
                    "height": window_height,
                    "image_path": image_chunk_path,
                    "mask_path": mask_chunk_path,
                }
            )

    _write_chunk_manifest(manifest_path, manifest_payload)

    return {
        "output_dir": output_dir,
        "image_chunks_dir": image_dir,
        "mask_chunks_dir": mask_dir,
        "manifest_path": manifest_path,
        "chunk_size": chunk_size,
        "pad_edges": pad_edges,
        "chunk_count": len(records),
        "chunks": [record.__dict__ for record in records],
    }


def reconstruct_geotiff_pair_from_manifest(
    manifest_path: str,
    output_dir: Optional[str] = None,
) -> Dict[str, str]:
    """
    Собирает исходные TIFF обратно из `chunk_manifest.json`.

    Возвращает пути к восстановленным `image` и `mask`.
    """
    manifest_path = os.path.abspath(manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    if manifest.get("kind") != "paired_geotiff_chunks":
        raise ValueError("Manifest не похож на paired_geotiff_chunks.")

    base_output_dir = (
        os.path.abspath(output_dir)
        if output_dir is not None
        else os.path.join(os.path.dirname(manifest_path), "reconstructed")
    )
    os.makedirs(base_output_dir, exist_ok=True)

    source_grid = manifest["source_grid"]
    width = int(source_grid["width"])
    height = int(source_grid["height"])
    crs = source_grid["crs"]
    transform = _from_serialized_transform(source_grid["transform"])
    chunks = manifest["chunks"]

    outputs = {
        "image": {
            "path": os.path.join(base_output_dir, "image_reconstructed.tif"),
            "meta": manifest["image_meta"],
        },
        "mask": {
            "path": os.path.join(base_output_dir, "mask_reconstructed.tif"),
            "meta": manifest["mask_meta"],
        },
    }

    mask_buffers = {
        "image": np.zeros((height, width), dtype=np.uint8),
        "mask": np.zeros((height, width), dtype=np.uint8),
    }

    for kind, item in outputs.items():
        meta = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": int(item["meta"]["count"]),
            "dtype": item["meta"]["dtype"],
            "crs": crs,
            "transform": transform,
            "compress": "LZW",
            "tiled": True,
            "BIGTIFF": "IF_SAFER",
        }
        if item["meta"]["nodata"] is not None:
            meta["nodata"] = item["meta"]["nodata"]

        with rasterio.open(item["path"], "w", **meta) as dst:
            for chunk_info in chunks:
                row_off = int(chunk_info["row_off"])
                col_off = int(chunk_info["col_off"])
                real_width = int(chunk_info["width"])
                real_height = int(chunk_info["height"])
                chunk_path = chunk_info[f"{kind}_path"]

                with rasterio.open(chunk_path) as chunk_src:
                    chunk_data = chunk_src.read(window=Window(0, 0, real_width, real_height))
                    chunk_valid_mask = chunk_src.dataset_mask()[:real_height, :real_width]

                write_window = Window(col_off, row_off, real_width, real_height)
                dst.write(chunk_data, window=write_window)
                mask_buffers[kind][row_off : row_off + real_height, col_off : col_off + real_width] = (
                    chunk_valid_mask
                )

            dst.write_mask(mask_buffers[kind])

    return {
        "image_reconstructed_path": outputs["image"]["path"],
        "mask_reconstructed_path": outputs["mask"]["path"],
    }


def resize_and_chunk_geotiff_pair(
    image_tif_path: str,
    mask_tif_path: str,
    output_dir: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    scale_factor: float = 0.5,
    pad_edges: bool = True,
    image_resampling: str | Resampling = Resampling.nearest,
    mask_resampling: str | Resampling = Resampling.nearest,
) -> Dict[str, object]:
    """
    Тестовый end-to-end пайплайн:
    1. уменьшает исходный TIFF и его mask,
    2. режет уменьшенные версии на синхронные чанки.
    """
    output_dir = os.path.abspath(output_dir)
    resized_dir = os.path.join(output_dir, "resized")
    os.makedirs(resized_dir, exist_ok=True)

    resized_image_path = os.path.join(resized_dir, "image_resized.tif")
    resized_mask_path = os.path.join(resized_dir, "mask_resized.tif")

    resize_geotiff(
        image_tif_path,
        resized_image_path,
        scale_factor=scale_factor,
        resampling=image_resampling,
    )
    resize_geotiff(
        mask_tif_path,
        resized_mask_path,
        scale_factor=scale_factor,
        resampling=mask_resampling,
    )

    chunk_result = chunk_geotiff_pair(
        resized_image_path,
        resized_mask_path,
        os.path.join(output_dir, "chunks"),
        chunk_size=chunk_size,
        pad_edges=pad_edges,
    )

    return {
        "output_dir": output_dir,
        "resized_image_path": resized_image_path,
        "resized_mask_path": resized_mask_path,
        "chunk_result": chunk_result,
    }


def run_real_file_pair_smoke_test(
    *,
    image_tif_path: str = r"C:\Users\Kolya\Desktop\mvp_ai_buisness\test_output3\polygon_pipeline_demo_stack_cropped.tif",
    mask_tif_path: str = r"C:\Users\Kolya\Desktop\mvp_ai_buisness\test_output3\polygon_pipeline_demo_class0_mask.tif",
    output_dir: str = r"C:\Users\Kolya\Desktop\mvp_ai_buisness\test_output3\paired_tif_chunking_smoke",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    scale_factor: float = 0.5,
) -> Dict[str, object]:
    """
    Smoke-test на реальных файлах пользователя:
    1. проверяет, что исходный TIFF и mask лежат на одной сетке,
    2. режет исходную пару на чанки,
    3. уменьшает пару файлов,
    4. режет уменьшенную пару на чанки.
    """
    image_tif_path = os.path.abspath(image_tif_path)
    mask_tif_path = os.path.abspath(mask_tif_path)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    with rasterio.open(image_tif_path) as image_src, rasterio.open(mask_tif_path) as mask_src:
        if not _datasets_share_grid(image_src, mask_src):
            raise AssertionError("Исходный TIFF и mask не совпадают по сетке.")

        source_width = image_src.width
        source_height = image_src.height
        source_crs = image_src.crs
        source_transform = image_src.transform
        expected_source_chunk_count = math.ceil(source_width / chunk_size) * math.ceil(
            source_height / chunk_size
        )

    direct_chunk_result = chunk_geotiff_pair(
        image_tif_path,
        mask_tif_path,
        os.path.join(output_dir, "direct_chunks"),
        chunk_size=chunk_size,
        pad_edges=True,
    )
    assert direct_chunk_result["chunk_count"] == expected_source_chunk_count
    assert direct_chunk_result["chunks"], "После прямой нарезки не создано ни одного чанка."
    assert os.path.exists(direct_chunk_result["manifest_path"]), "Manifest для чанков не создан."

    first_direct_chunk = direct_chunk_result["chunks"][0]
    last_direct_chunk = direct_chunk_result["chunks"][-1]
    with rasterio.open(first_direct_chunk["image_path"]) as first_chunk_src:
        assert first_chunk_src.width == chunk_size
        assert first_chunk_src.height == chunk_size
        assert str(first_chunk_src.crs) == str(source_crs)
        assert np.allclose(tuple(first_chunk_src.transform)[:6], tuple(source_transform)[:6], atol=1e-9)

    with rasterio.open(last_direct_chunk["image_path"]) as last_chunk_src:
        assert last_chunk_src.width == chunk_size
        assert last_chunk_src.height == chunk_size

    reconstructed_result = reconstruct_geotiff_pair_from_manifest(
        direct_chunk_result["manifest_path"],
        os.path.join(output_dir, "reconstructed_from_direct_chunks"),
    )
    with rasterio.open(reconstructed_result["image_reconstructed_path"]) as reconstructed_image_src, rasterio.open(
        reconstructed_result["mask_reconstructed_path"]
    ) as reconstructed_mask_src, rasterio.open(image_tif_path) as source_image_src, rasterio.open(
        mask_tif_path
    ) as source_mask_src:
        assert _datasets_share_grid(reconstructed_image_src, source_image_src)
        assert _datasets_share_grid(reconstructed_mask_src, source_mask_src)

        sample_windows = [
            Window(0, 0, min(chunk_size, source_width), min(chunk_size, source_height)),
            Window(
                max(0, source_width - min(chunk_size, source_width)),
                max(0, source_height - min(chunk_size, source_height)),
                min(chunk_size, source_width),
                min(chunk_size, source_height),
            ),
        ]
        for sample_window in sample_windows:
            assert np.array_equal(
                source_image_src.read(window=sample_window),
                reconstructed_image_src.read(window=sample_window),
            )
            assert np.array_equal(
                source_mask_src.read(window=sample_window),
                reconstructed_mask_src.read(window=sample_window),
            )

    end_to_end_result = resize_and_chunk_geotiff_pair(
        image_tif_path,
        mask_tif_path,
        os.path.join(output_dir, "resized_pipeline"),
        chunk_size=chunk_size,
        scale_factor=scale_factor,
        pad_edges=True,
    )

    resized_image_path = end_to_end_result["resized_image_path"]
    resized_mask_path = end_to_end_result["resized_mask_path"]
    with rasterio.open(resized_image_path) as resized_image_src, rasterio.open(resized_mask_path) as resized_mask_src:
        assert _datasets_share_grid(resized_image_src, resized_mask_src)

        expected_resized_width = max(1, int(math.ceil(source_width * scale_factor)))
        expected_resized_height = max(1, int(math.ceil(source_height * scale_factor)))
        assert resized_image_src.width == expected_resized_width
        assert resized_image_src.height == expected_resized_height
        assert resized_mask_src.width == expected_resized_width
        assert resized_mask_src.height == expected_resized_height

        expected_resized_chunk_count = math.ceil(expected_resized_width / chunk_size) * math.ceil(
            expected_resized_height / chunk_size
        )

    resized_chunk_result = end_to_end_result["chunk_result"]
    assert resized_chunk_result["chunk_count"] == expected_resized_chunk_count
    assert resized_chunk_result["chunks"], "После resize+chunk не создано ни одного чанка."

    return {
        "image_tif_path": image_tif_path,
        "mask_tif_path": mask_tif_path,
        "output_dir": output_dir,
        "chunk_size": chunk_size,
        "scale_factor": scale_factor,
        "source_size": {"width": source_width, "height": source_height},
        "source_chunk_count": direct_chunk_result["chunk_count"],
        "resized_size": {
            "width": expected_resized_width,
            "height": expected_resized_height,
        },
        "resized_chunk_count": resized_chunk_result["chunk_count"],
        "sample_chunk_paths": {
            "first_direct_image_chunk": first_direct_chunk["image_path"],
            "last_direct_image_chunk": last_direct_chunk["image_path"],
            "resized_image_tif": resized_image_path,
            "resized_mask_tif": resized_mask_path,
        },
        "manifest_paths": {
            "direct_manifest": direct_chunk_result["manifest_path"],
            "resized_manifest": resized_chunk_result["manifest_path"],
        },
        "reconstructed_paths": reconstructed_result,
    }


if __name__ == "__main__":
    result = run_real_file_pair_smoke_test()
    print(json.dumps(result, ensure_ascii=False, indent=2))

