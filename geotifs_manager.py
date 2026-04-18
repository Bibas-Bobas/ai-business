from __future__ import annotations

import json
import os
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.features import geometry_mask as rasterio_geometry_mask
from rasterio.transform import array_bounds, from_origin
from rasterio.warp import reproject
from rasterio.windows import Window, from_bounds, transform as window_transform
from shapely.geometry import Polygon, box, mapping
from shapely.geometry.base import BaseGeometry


DEFAULT_INDEX_NAME = "vizard_index.gpkg"
# Размер стороны квадратного патча под модель (как --patch-size в prepare.py).
DEFAULT_MODEL_PATCH_SIZE_PX = 384
FILENAME_TIME_FORMAT = "%Y%m%dT%H%M%S"
INDEX_DATETIME_COLUMNS = ("start_time", "stop_time")
INDEX_SORT_COLUMNS = ("date", "start_time", "stop_time", "filepath")
# IceClass: класс 0 — RGB [1, 1, 1] («нет класса»), не путать с nodata.
EMPTY_CLASS_RGB = np.array([1, 1, 1], dtype=np.uint8)
REQUIRED_INDEX_COLUMNS = (
    "filepath",
    "mission",
    "start_time",
    "stop_time",
    "date",
    "orbit",
    "datatake_id",
    "product_id",
    "band_count",
    "dtype",
    "nodata",
    "geometry",
)


def _semantic_valid_mask(data: np.ndarray, observation_mask: np.ndarray) -> np.ndarray:
    """
    Возвращает маску полезных пикселей.

    Для IceClass пиксель [1, 1, 1] означает "нет класса" и не должен
    перекрывать реальные данные при наложении.
    """
    valid_mask = np.asarray(observation_mask, dtype=bool).copy()
    if data.ndim == 3 and data.shape[0] >= 3:
        empty_class_mask = valid_mask.copy()
        for band_index, channel_value in enumerate(EMPTY_CLASS_RGB):
            empty_class_mask &= data[band_index] == channel_value
        valid_mask &= ~empty_class_mask
    return valid_mask


def _iceclass_empty_mask_rgb(layer_data: np.ndarray, layer_observed: np.ndarray) -> np.ndarray:
    """True там, где в слое есть наблюдение и RGB совпадает с EMPTY_CLASS_RGB (первые 3 канала)."""
    empty = np.asarray(layer_observed, dtype=bool).copy()
    if layer_data.ndim != 3 or layer_data.shape[0] < 3:
        return np.zeros_like(empty, dtype=bool)
    for band_index, channel_value in enumerate(EMPTY_CLASS_RGB):
        empty &= layer_data[band_index] == channel_value
    return empty


def _pixels_inside_polygon_mask(
    polygon: BaseGeometry,
    height: int,
    width: int,
    transform,
    *,
    all_touched: bool = True,
) -> np.ndarray:
    """
    bool[H, W]: True для пикселей, пересекающих полигон (CRS должна совпадать с transform растра).
    """
    outside = rasterio_geometry_mask(
        [mapping(polygon)],
        out_shape=(height, width),
        transform=transform,
        invert=False,
        all_touched=all_touched,
    )
    return ~np.asarray(outside, dtype=bool)


def _overlay_arrays(
    base_data: np.ndarray,
    base_mask: np.ndarray,
    base_semantic_valid: np.ndarray,
    layer_data: np.ndarray,
    layer_mask: np.ndarray,
):
    """
    Накладывает layer на base в порядке от более старого снимка к более новому.

    Осмысленные классы льда перезаписывают то, что было; пиксели [1,1,1] не затирают
    уже записанный осмысленный класс, но попадают в результат там, где осмысленного
    класса ещё не было (иначе в базе оставались бы нули вместо «нет класса»).
    """
    layer_observed = np.asarray(layer_mask, dtype=bool)
    layer_semantic = _semantic_valid_mask(layer_data, layer_observed)

    for band_index in range(base_data.shape[0]):
        np.copyto(base_data[band_index], layer_data[band_index], where=layer_semantic)
    base_mask[layer_semantic] = True
    base_semantic_valid[layer_semantic] = True

    if layer_data.ndim == 3 and layer_data.shape[0] >= 3:
        layer_empty = _iceclass_empty_mask_rgb(layer_data, layer_observed)
        fill_empty = layer_empty & ~base_semantic_valid
        for band_index in range(base_data.shape[0]):
            np.copyto(base_data[band_index], layer_data[band_index], where=fill_empty)
        base_mask[fill_empty] = True


def _intersect_bounds(
    first: Tuple[float, float, float, float],
    second: Tuple[float, float, float, float],
) -> Optional[Tuple[float, float, float, float]]:
    """Пересечение двух bbox в формате (left, bottom, right, top)."""
    left = max(first[0], second[0])
    bottom = max(first[1], second[1])
    right = min(first[2], second[2])
    top = min(first[3], second[3])
    if right <= left or top <= bottom:
        return None
    return (left, bottom, right, top)


def _same_grid(left: HistoricalSnapshot, right: HistoricalSnapshot) -> bool:
    """Проверяет, что два слепка лежат на одной и той же пиксельной сетке."""
    if left.data.shape[1:] != right.data.shape[1:]:
        return False
    if str(left.crs) != str(right.crs):
        return False
    return bool(np.allclose(tuple(left.transform)[:6], tuple(right.transform)[:6], atol=1e-9))


def _iter_windows(width: int, height: int, tile_size: int) -> Iterable[Window]:
    """Итерируется по выходному растру тайлами фиксированного размера."""
    for row_off in range(0, height, tile_size):
        for col_off in range(0, width, tile_size):
            yield Window(
                col_off=col_off,
                row_off=row_off,
                width=min(tile_size, width - col_off),
                height=min(tile_size, height - row_off),
            )


def write_streaming_tile_manifest_json(
    manifest_path: str,
    width_px: int,
    height_px: int,
    tile_size: int,
) -> str:
    """
    Сохраняет список окон (col_off, row_off, width, height), которыми потоково
    пишется большой GeoTIFF при export. Это не патчи модели — см. iter_model_chunks_from_geotiff.
    """
    tiles: List[Dict[str, int]] = []
    for row_off in range(0, height_px, tile_size):
        for col_off in range(0, width_px, tile_size):
            tiles.append(
                {
                    "col_off": col_off,
                    "row_off": row_off,
                    "width": min(tile_size, width_px - col_off),
                    "height": min(tile_size, height_px - row_off),
                }
            )
    payload = {
        "kind": "streaming_geotiff_write_windows",
        "raster_width_px": width_px,
        "raster_height_px": height_px,
        "tile_size_px": tile_size,
        "tile_count": len(tiles),
        "tiles": tiles,
        "model_patches_note": (
            "Для нарезки под обучение используйте "
            f"VizardDataManager.iter_model_chunks_from_geotiff(stack_cropped_tif, patch_size={DEFAULT_MODEL_PATCH_SIZE_PX})."
        ),
    }
    directory = os.path.dirname(os.path.abspath(manifest_path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return manifest_path


def _default_model_chunk_stride(patch_size: int) -> int:
    """Шаг сетки якорей как в prepare.py: max(64, patch_size // 2)."""
    return max(64, patch_size // 2)


def _iter_model_chunk_anchors(
    height: int,
    width: int,
    patch_size: int,
    stride: int,
    pad_edges: bool,
) -> Iterable[Tuple[int, int]]:
    """Возвращает (row_off, col_off) верхнего левого угла каждого патча в пикселях."""
    if pad_edges:
        for row_off in range(0, max(1, height - patch_size + 1), stride):
            for col_off in range(0, max(1, width - patch_size + 1), stride):
                yield row_off, col_off
        return
    if height < patch_size or width < patch_size:
        return
    for row_off in range(0, height - patch_size + 1, stride):
        for col_off in range(0, width - patch_size + 1, stride):
            yield row_off, col_off


def _synthetic_chunk_test_raster(bands: int, height: int, width: int, seed: int = 42) -> np.ndarray:
    """
    Псевдорастр для smoke-теста нарезки: градиенты по X/Y/диагонали.

    Случайный шум в превью выглядит как «битые» данные; здесь намеренно гладкий паттерн.
    """
    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.int32),
        np.arange(width, dtype=np.int32),
        indexing="ij",
    )
    wx = max(width - 1, 1)
    wy = max(height - 1, 1)
    wsum = max(width + height - 2, 1)
    templates = [
        (xx * 255 // wx).astype(np.uint8),
        (yy * 255 // wy).astype(np.uint8),
        ((xx + yy) * 255 // wsum).astype(np.uint8),
    ]
    off = int(seed) % 32
    out = np.empty((bands, height, width), dtype=np.uint8)
    for i in range(bands):
        out[i] = np.clip(templates[i % len(templates)].astype(np.int16) + off + i * 3, 0, 255).astype(np.uint8)
    return out


def _crop_pad_multiband(
    data: np.ndarray,
    row_off: int,
    col_off: int,
    patch_size: int,
    pad_value: float,
) -> np.ndarray:
    """Вырезает [C, patch_size, patch_size] из [C, H, W]; дополняет pad_value у краёв."""
    if data.ndim != 3:
        raise ValueError("Ожидается массив формы (bands, height, width).")
    bands, height, width_r = data.shape
    out = np.full((bands, patch_size, patch_size), pad_value, dtype=data.dtype)
    row_end = min(height, row_off + patch_size)
    col_end = min(width_r, col_off + patch_size)
    out[:, : row_end - row_off, : col_end - col_off] = data[:, row_off:row_end, col_off:col_end]
    return out


def _resolve_chunk_pad_value(sample_value, dtype: np.dtype) -> float:
    if sample_value is None or (isinstance(sample_value, float) and np.isnan(sample_value)):
        return 0.0 if np.issubdtype(dtype, np.floating) else 0
    return float(sample_value) if np.issubdtype(dtype, np.floating) else int(sample_value)


def _model_chunk_base_name(prefix: str, row_off: int, col_off: int) -> str:
    return f"{prefix}_r{row_off:05d}_c{col_off:05d}"


def _write_model_chunk_geotiff(
    chunk: "ModelInputChunk",
    path: str,
    nodata: Optional[float] = None,
) -> None:
    """Записывает один чанк как GeoTIFF с геопривязкой."""
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    data = chunk.data
    meta: Dict = {
        "driver": "GTiff",
        "height": int(data.shape[1]),
        "width": int(data.shape[2]),
        "count": int(data.shape[0]),
        "dtype": np.dtype(data.dtype).name,
        "crs": chunk.crs,
        "transform": chunk.transform,
        "compress": "LZW",
        "tiled": True,
        "BIGTIFF": "IF_SAFER",
    }
    if nodata is not None:
        meta["nodata"] = nodata
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(data)
        dst.update_tags(row_off=str(chunk.row_off), col_off=str(chunk.col_off))


def _chunk_raster_to_png_u8(data: np.ndarray) -> np.ndarray:
    """Готовит массив H×W×3 (uint8) для предпросмотра PNG."""
    bands, _, _ = data.shape
    if bands >= 3:
        rgb = data[:3].astype(np.float32, copy=False)
        if np.issubdtype(data.dtype, np.integer):
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        else:
            flat = rgb.reshape(3, -1)
            out = np.empty_like(flat, dtype=np.uint8)
            for i in range(3):
                b = flat[i]
                mn, mx = float(np.nanmin(b)), float(np.nanmax(b))
                if mx > mn:
                    out[i] = np.clip((b - mn) / (mx - mn) * 255.0, 0, 255).astype(np.uint8)
                else:
                    out[i] = 0
            rgb = out.reshape(3, data.shape[1], data.shape[2])
            rgb = np.moveaxis(rgb, 0, -1)
            return rgb
        return np.moveaxis(rgb, 0, -1)
    band = data[0].astype(np.float32, copy=False)
    if np.issubdtype(data.dtype, np.integer):
        gray = np.clip(band, 0, 255).astype(np.uint8)
    else:
        mn, mx = float(np.nanmin(band)), float(np.nanmax(band))
        if mx > mn:
            gray = np.clip((band - mn) / (mx - mn) * 255.0, 0, 255).astype(np.uint8)
        else:
            gray = np.zeros(band.shape, dtype=np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def _write_model_chunk_png(chunk: "ModelInputChunk", path: str) -> None:
    """RGB-PNG предпросмотр (значения как в чанке, без смены семантики классов в TIF)."""
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    rgb = _chunk_raster_to_png_u8(chunk.data)
    Image.fromarray(rgb, mode="RGB").save(path)


def _grid_from_bounds(bounds: Tuple[float, float, float, float], reference) -> Tuple[int, int, object]:
    """Строит выходную сетку по bounds с разрешением reference raster."""
    left, bottom, right, top = bounds
    res_x = abs(reference.transform.a)
    res_y = abs(reference.transform.e)
    width = max(1, int(np.ceil((right - left) / res_x)))
    height = max(1, int(np.ceil((top - bottom) / res_y)))
    transform = from_origin(left, top, res_x, res_y)
    return width, height, transform


class HistoricalSnapshot:
    """Исторический слепок по полигону для конкретной даты."""

    def __init__(
        self,
        data: np.ndarray,
        transform,
        crs,
        timestamp,
        observation_mask: np.ndarray,
        nodata: Optional[float] = None,
        metadata: Optional[Dict] = None,
        source_files: Optional[Sequence[str]] = None,
    ):
        data = np.asarray(data)
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        if data.ndim != 3:
            raise ValueError("HistoricalSnapshot.data должен иметь форму (bands, height, width).")

        observation_mask = np.asarray(observation_mask, dtype=bool)
        if observation_mask.shape != data.shape[1:]:
            raise ValueError("Размер observation_mask должен совпадать с высотой и шириной data.")

        self.data = data
        self.transform = transform
        self.crs = crs
        self.timestamp = pd.Timestamp(timestamp).to_pydatetime()
        self.observation_mask = observation_mask
        self.nodata = nodata
        self.metadata = dict(metadata or {})
        self.source_files = list(source_files or [])

    @property
    def binary_mask(self) -> np.ndarray:
        """Бинарная маска наблюдений: 1 - есть данные, 0 - нет данных."""
        return self.observation_mask.astype(np.uint8)

    @property
    def meta(self) -> Dict:
        """Метаданные GeoTIFF, достаточные для записи результата на диск."""
        meta = {
            "driver": "GTiff",
            "height": self.data.shape[1],
            "width": self.data.shape[2],
            "count": self.data.shape[0],
            "dtype": str(self.data.dtype),
            "crs": self.crs,
            "transform": self.transform,
        }
        if self.nodata is not None:
            meta["nodata"] = self.nodata
        return meta

    def as_masked_array(self) -> np.ma.MaskedArray:
        """Возвращает данные как masked array с учетом observation_mask."""
        mask = np.broadcast_to(~self.observation_mask, self.data.shape)
        return np.ma.array(self.data, mask=mask)

    def show(
        self,
        band: Optional[int] = None,
        rgb_bands: Tuple[int, int, int] = (1, 2, 3),
        cmap: str = "viridis",
        figsize: Tuple[int, int] = (8, 8),
        title: Optional[str] = None,
    ):
        """Быстрый просмотр слепка через matplotlib."""
        masked = self.as_masked_array()
        fig, ax = plt.subplots(figsize=figsize)

        if self.data.shape[0] == 1 or band is not None:
            band_index = 0 if band is None else band - 1
            if band_index < 0 or band_index >= self.data.shape[0]:
                raise IndexError("Запрошенный band вне диапазона.")
            image = masked[band_index].astype(np.float32).filled(np.nan)
            ax.imshow(image, cmap=cmap)
        else:
            band_indexes = [item - 1 for item in rgb_bands]
            if any(idx < 0 or idx >= self.data.shape[0] for idx in band_indexes):
                raise IndexError("Один из RGB band вне диапазона.")
            rgb = np.moveaxis(masked[band_indexes].filled(0), 0, -1).astype(np.float32)
            max_value = float(np.nanmax(rgb)) if rgb.size else 0.0
            if max_value > 0:
                rgb /= max_value
            rgb[~self.observation_mask] = 0
            ax.imshow(rgb)

        ax.set_title(title or f"HistoricalSnapshot {self.timestamp:%Y-%m-%d}")
        ax.set_axis_off()
        plt.tight_layout()
        return ax

    def save(self, filename: str):
        """Сохраняет слепок как GeoTIFF с геопривязкой и внутренней маской."""
        directory = os.path.dirname(os.path.abspath(filename))
        if directory:
            os.makedirs(directory, exist_ok=True)

        write_data = self.data.copy()
        if self.nodata is not None:
            write_data[:, ~self.observation_mask] = self.nodata

        with rasterio.open(filename, "w", **self.meta) as dst:
            dst.write(write_data)
            dst.write_mask(self.binary_mask * 255)
            dst.update_tags(
                timestamp=self.timestamp.isoformat(),
                source_file_count=len(self.source_files),
            )

        return filename


@dataclass
class ModelInputChunk:
    """
    Квадратный фрагмент растра под вход модели (размер как в prepare.py).

    Пиксели не перекодируются: значения каналов совпадают с исходным GeoTIFF
    или HistoricalSnapshot (в т.ч. семантика IceClass RGB [1,1,1]).
    """

    data: np.ndarray
    col_off: int
    row_off: int
    transform: object
    crs: object
    bounds: Tuple[float, float, float, float]
    window: Window


def compose_stacked_historical_snapshot(
    snapshots: List[HistoricalSnapshot],
) -> Tuple[HistoricalSnapshot, Polygon]:
    """
    Строит один HistoricalSnapshot — стек истории без записи на диск.

    Возвращает (stacked_snapshot, query_polygon), где query_polygon — shapely box
    по bounds из metadata (или по охвату данных).
    """
    if not snapshots:
        raise ValueError("snapshots не должен быть пустым.")

    ordered = sorted(snapshots, key=lambda snapshot: snapshot.timestamp)
    crs_values = {str(snapshot.crs) for snapshot in ordered}
    if len(crs_values) != 1:
        raise ValueError("Все HistoricalSnapshot должны иметь одинаковый CRS.")

    band_counts = {snapshot.data.shape[0] for snapshot in ordered}
    if len(band_counts) != 1:
        raise ValueError("Все HistoricalSnapshot должны иметь одинаковое число каналов.")

    if all(_same_grid(ordered[0], snapshot) for snapshot in ordered[1:]):
        output_data = np.zeros_like(ordered[0].data)
        output_mask = np.zeros_like(ordered[0].observation_mask, dtype=bool)
        output_semantic = np.zeros_like(ordered[0].observation_mask, dtype=bool)
        stacked_source_files: List[str] = []
        for snapshot in ordered:
            _overlay_arrays(
                output_data,
                output_mask,
                output_semantic,
                snapshot.data,
                snapshot.observation_mask,
            )
            stacked_source_files.extend(snapshot.source_files)

        query_polygon_bounds = ordered[-1].metadata.get("query_polygon_bounds")
        if query_polygon_bounds is None:
            query_polygon_bounds = array_bounds(
                ordered[0].data.shape[1],
                ordered[0].data.shape[2],
                ordered[0].transform,
            )
        canvas_bounds = array_bounds(
            ordered[0].data.shape[1],
            ordered[0].data.shape[2],
            ordered[0].transform,
        )
        query_polygon = box(*query_polygon_bounds)
        stacked_snapshot = HistoricalSnapshot(
            data=output_data,
            transform=ordered[0].transform,
            crs=ordered[0].crs,
            timestamp=ordered[-1].timestamp,
            observation_mask=output_mask,
            nodata=ordered[-1].nodata,
            metadata={
                "query_polygon_bounds": tuple(query_polygon_bounds),
                "centered_canvas_bounds": tuple(canvas_bounds),
                "stacked_dates": [snapshot.timestamp.strftime("%Y-%m-%d") for snapshot in ordered],
                "stack_order": "oldest_to_newest",
            },
            source_files=list(dict.fromkeys(stacked_source_files)),
        )
        return stacked_snapshot, query_polygon

    query_polygon_bounds = None
    for snapshot in reversed(ordered):
        candidate_bounds = snapshot.metadata.get("query_polygon_bounds")
        if candidate_bounds is not None:
            query_polygon_bounds = tuple(candidate_bounds)
            break

    snapshot_bounds = []
    for snapshot in ordered:
        bounds = array_bounds(snapshot.data.shape[1], snapshot.data.shape[2], snapshot.transform)
        snapshot_bounds.append(bounds)

    min_left = min(bounds[0] for bounds in snapshot_bounds)
    min_bottom = min(bounds[1] for bounds in snapshot_bounds)
    max_right = max(bounds[2] for bounds in snapshot_bounds)
    max_top = max(bounds[3] for bounds in snapshot_bounds)

    if query_polygon_bounds is None:
        query_polygon_bounds = (min_left, min_bottom, max_right, max_top)

    poly_left, poly_bottom, poly_right, poly_top = query_polygon_bounds
    center_x = (poly_left + poly_right) / 2
    center_y = (poly_bottom + poly_top) / 2

    half_extent_x = max(
        center_x - min_left,
        max_right - center_x,
        center_x - poly_left,
        poly_right - center_x,
    )
    half_extent_y = max(
        center_y - min_bottom,
        max_top - center_y,
        center_y - poly_bottom,
        poly_top - center_y,
    )

    resolutions_x = [abs(snapshot.transform.a) for snapshot in ordered if snapshot.transform.a != 0]
    resolutions_y = [abs(snapshot.transform.e) for snapshot in ordered if snapshot.transform.e != 0]
    res_x = min(resolutions_x)
    res_y = min(resolutions_y)

    width = max(2, int(np.ceil((2 * half_extent_x) / res_x)))
    height = max(2, int(np.ceil((2 * half_extent_y) / res_y)))
    if width % 2 != 0:
        width += 1
    if height % 2 != 0:
        height += 1

    canvas_half_width = (width * res_x) / 2
    canvas_half_height = (height * res_y) / 2
    canvas_left = center_x - canvas_half_width
    canvas_top = center_y + canvas_half_height
    canvas_transform = from_origin(canvas_left, canvas_top, res_x, res_y)
    canvas_bounds = (
        canvas_left,
        center_y - canvas_half_height,
        center_x + canvas_half_width,
        canvas_top,
    )

    output_dtype = np.result_type(*[snapshot.data.dtype for snapshot in ordered])
    output_data = np.zeros((ordered[0].data.shape[0], height, width), dtype=output_dtype)
    output_mask = np.zeros((height, width), dtype=bool)
    output_semantic = np.zeros((height, width), dtype=bool)

    stacked_source_files: List[str] = []
    for snapshot in ordered:
        reprojected_data = np.zeros_like(output_data, dtype=output_dtype)
        reprojected_mask = np.zeros((height, width), dtype=np.uint8)

        for band_index in range(snapshot.data.shape[0]):
            reproject(
                source=snapshot.data[band_index],
                destination=reprojected_data[band_index],
                src_transform=snapshot.transform,
                src_crs=snapshot.crs,
                dst_transform=canvas_transform,
                dst_crs=snapshot.crs,
                resampling=Resampling.nearest,
            )

        reproject(
            source=snapshot.binary_mask.astype(np.uint8),
            destination=reprojected_mask,
            src_transform=snapshot.transform,
            src_crs=snapshot.crs,
            dst_transform=canvas_transform,
            dst_crs=snapshot.crs,
            resampling=Resampling.nearest,
        )

        _overlay_arrays(
            output_data,
            output_mask,
            output_semantic,
            reprojected_data,
            reprojected_mask > 0,
        )
        stacked_source_files.extend(snapshot.source_files)

    query_polygon = box(*query_polygon_bounds)
    stacked_snapshot = HistoricalSnapshot(
        data=output_data,
        transform=canvas_transform,
        crs=ordered[-1].crs,
        timestamp=ordered[-1].timestamp,
        observation_mask=output_mask,
        nodata=ordered[-1].nodata,
        metadata={
            "query_polygon_bounds": query_polygon_bounds,
            "centered_canvas_bounds": canvas_bounds,
            "stacked_dates": [snapshot.timestamp.strftime("%Y-%m-%d") for snapshot in ordered],
            "stack_order": "oldest_to_newest",
        },
        source_files=list(dict.fromkeys(stacked_source_files)),
    )
    return stacked_snapshot, query_polygon


def stack_historical_snapshots(
    snapshots: List[HistoricalSnapshot],
    output_prefix: str = "stacked_history",
) -> Tuple[str, str]:
    """
    Накладывает исторические слепки друг на друга в общем геопривязанном холсте.

    Более новые слепки рисуются поверх более старых. Возвращает пути к PNG и TIF.
    Если в metadata есть `query_polygon_bounds`, полигон фиксируется по центру общего холста.
    """
    stacked_snapshot, query_polygon = compose_stacked_historical_snapshot(snapshots)

    output_prefix = os.path.splitext(output_prefix)[0]
    output_dir = os.path.dirname(os.path.abspath(output_prefix))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    output_tif = f"{output_prefix}.tif"
    output_png = f"{output_prefix}.png"
    stacked_snapshot.save(output_tif)
    save_snapshot_png_with_polygon(
        stacked_snapshot,
        query_polygon,
        output_png,
        title=f"Stacked history up to {stacked_snapshot.timestamp:%Y-%m-%d}",
    )

    return output_png, output_tif


def polygon_buffer_bounds(
    polygon: Polygon,
    buffer_fraction: float = 0.2,
) -> Tuple[float, float, float, float]:
    """
    Прямоугольник обрезки: bbox полигона + полоса снаружи.

    Ширина полосы = buffer_fraction * max(ширина bbox, высота bbox) в единицах CRS.
    """
    if buffer_fraction < 0:
        raise ValueError("buffer_fraction не должен быть отрицательным.")
    minx, miny, maxx, maxy = polygon.bounds
    w = maxx - minx
    h = maxy - miny
    margin_x = buffer_fraction * max(w, 1e-9)
    margin_y = buffer_fraction * max(h, 1e-9)
    return (minx - margin_x, miny - margin_y, maxx + margin_x, maxy + margin_y)


def crop_historical_snapshot_to_bounds(
    snapshot: HistoricalSnapshot,
    bounds: Tuple[float, float, float, float],
) -> HistoricalSnapshot:
    """Обрезает слепок по географическим границам (left, bottom, right, top)."""
    left, bottom, right, top = bounds
    height, width = snapshot.data.shape[1], snapshot.data.shape[2]
    window = from_bounds(left, bottom, right, top, snapshot.transform)
    window = window.round_offsets(op="floor").round_lengths(op="ceil")
    window = window.intersection(Window(0, 0, width, height))
    if window.width <= 0 or window.height <= 0:
        raise ValueError("Окно обрезки не пересекается с растром слепка.")

    row_off = int(window.row_off)
    col_off = int(window.col_off)
    row_count = int(window.height)
    col_count = int(window.width)

    cropped_data = snapshot.data[:, row_off : row_off + row_count, col_off : col_off + col_count].copy()
    cropped_mask = snapshot.observation_mask[row_off : row_off + row_count, col_off : col_off + col_count].copy()

    new_transform = window_transform(window, snapshot.transform)
    meta = dict(snapshot.metadata)
    meta["crop_bounds"] = (left, bottom, right, top)
    return HistoricalSnapshot(
        data=cropped_data,
        transform=new_transform,
        crs=snapshot.crs,
        timestamp=snapshot.timestamp,
        observation_mask=cropped_mask,
        nodata=snapshot.nodata,
        metadata=meta,
        source_files=snapshot.source_files,
    )


def extract_class0_mask_snapshot(snapshot: HistoricalSnapshot) -> HistoricalSnapshot:
    """
    Маска пикселей класса 0 IceClass — RGB [1, 1, 1] («нет класса»).

    Один канал uint8: 255 там, где все три канала совпадают с EMPTY_CLASS_RGB, иначе 0.
    """
    if snapshot.data.shape[0] < 3:
        empty = np.zeros(snapshot.data.shape[1:], dtype=bool)
    else:
        rgb = snapshot.data[:3]
        empty = np.all(rgb == EMPTY_CLASS_RGB[:, np.newaxis, np.newaxis], axis=0)

    band = (empty.astype(np.uint8) * 255)[np.newaxis, ...]
    meta = dict(snapshot.metadata)
    meta["mask_kind"] = "iceclass_class0_rgb_1_1_1"
    return HistoricalSnapshot(
        data=band,
        transform=snapshot.transform,
        crs=snapshot.crs,
        timestamp=snapshot.timestamp,
        observation_mask=snapshot.observation_mask.copy(),
        nodata=0.0,
        metadata=meta,
        source_files=snapshot.source_files,
    )


def save_snapshot_png_with_polygon(
    snapshot: HistoricalSnapshot,
    polygon: Polygon,
    png_path: str,
    title: Optional[str] = None,
    rgb_bands: Tuple[int, int, int] = (1, 2, 3),
    max_preview_size: int = 2048,
):
    """Сохраняет PNG-превью слепка с красным контуром полигона (в CRS слепка)."""
    directory = os.path.dirname(os.path.abspath(png_path))
    if directory:
        os.makedirs(directory, exist_ok=True)

    height, width = snapshot.data.shape[1:]
    preview_step = max(1, int(np.ceil(max(height, width) / max_preview_size)))

    fig, ax = plt.subplots(figsize=(8, 8))
    if snapshot.data.shape[0] == 1:
        image = snapshot.data[0, ::preview_step, ::preview_step].astype(np.float32)
        image[~snapshot.observation_mask[::preview_step, ::preview_step]] = np.nan
        ax.imshow(image, cmap="viridis")
    else:
        band_indexes = [item - 1 for item in rgb_bands]
        rgb = np.moveaxis(
            snapshot.data[band_indexes, ::preview_step, ::preview_step],
            0,
            -1,
        ).astype(np.float32)
        max_value = float(np.nanmax(rgb)) if rgb.size else 0.0
        if max_value > 0:
            rgb /= max_value
        rgb[~snapshot.observation_mask[::preview_step, ::preview_step]] = 0
        ax.imshow(rgb)
    ax.set_title(title or f"HistoricalSnapshot {snapshot.timestamp:%Y-%m-%d}")
    ax.set_axis_off()

    polygon_x, polygon_y = polygon.exterior.xy
    pixel_points = [(~snapshot.transform) * (x, y) for x, y in zip(polygon_x, polygon_y)]
    pixel_x = [point[0] / preview_step for point in pixel_points]
    pixel_y = [point[1] / preview_step for point in pixel_points]
    ax.plot(pixel_x, pixel_y, color="red", linewidth=2)
    plt.tight_layout()
    ax.figure.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(ax.figure)
    return png_path


def save_mask_snapshot_png_with_polygon(
    snapshot: HistoricalSnapshot,
    polygon: Polygon,
    png_path: str,
    title: Optional[str] = None,
    max_preview_size: int = 2048,
):
    """PNG одноканальной маски (0/255) с контуром полигона."""
    directory = os.path.dirname(os.path.abspath(png_path))
    if directory:
        os.makedirs(directory, exist_ok=True)

    height, width = snapshot.data.shape[1:]
    preview_step = max(1, int(np.ceil(max(height, width) / max_preview_size)))

    fig, ax = plt.subplots(figsize=(8, 8))
    band = snapshot.data[0, ::preview_step, ::preview_step].astype(np.float32)
    ax.imshow(band, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title or "Class-0 mask")
    ax.set_axis_off()

    polygon_x, polygon_y = polygon.exterior.xy
    pixel_points = [(~snapshot.transform) * (x, y) for x, y in zip(polygon_x, polygon_y)]
    pixel_x = [point[0] / preview_step for point in pixel_points]
    pixel_y = [point[1] / preview_step for point in pixel_points]
    ax.plot(pixel_x, pixel_y, color="red", linewidth=2)
    plt.tight_layout()
    ax.figure.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return png_path


def save_tif_preview_with_polygon(
    tif_path: str,
    polygon: Polygon,
    png_path: str,
    title: Optional[str] = None,
    max_preview_size: int = 2048,
    is_mask: bool = False,
):
    """Сохраняет downsampled PNG-превью прямо из GeoTIFF, не загружая весь растр в память."""
    directory = os.path.dirname(os.path.abspath(png_path))
    if directory:
        os.makedirs(directory, exist_ok=True)

    with rasterio.open(tif_path) as src:
        src_width = src.width
        src_height = src.height
        src_transform = src.transform
        preview_width = min(src.width, max_preview_size)
        preview_height = min(src.height, max_preview_size)
        scale = max(src.width / max(preview_width, 1), src.height / max(preview_height, 1))
        if scale > 1:
            preview_width = max(1, int(np.ceil(src.width / scale)))
            preview_height = max(1, int(np.ceil(src.height / scale)))
        else:
            preview_width = src.width
            preview_height = src.height

        data = src.read(
            out_shape=(src.count, preview_height, preview_width),
            resampling=Resampling.nearest,
        )

    fig, ax = plt.subplots(figsize=(8, 8))
    if is_mask or data.shape[0] == 1:
        band = data[0].astype(np.float32)
        ax.imshow(band, cmap="gray", vmin=0, vmax=255 if is_mask else None)
    else:
        rgb = np.moveaxis(data[:3], 0, -1).astype(np.float32)
        max_value = float(np.nanmax(rgb)) if rgb.size else 0.0
        if max_value > 0:
            rgb /= max_value
        ax.imshow(rgb)
    ax.set_title(title or os.path.basename(tif_path))
    ax.set_axis_off()

    scale_x = src_width / preview_width
    scale_y = src_height / preview_height
    polygon_x, polygon_y = polygon.exterior.xy
    pixel_points = [(~src_transform) * (x, y) for x, y in zip(polygon_x, polygon_y)]
    pixel_x = [point[0] / scale_x for point in pixel_points]
    pixel_y = [point[1] / scale_y for point in pixel_points]
    ax.plot(pixel_x, pixel_y, color="red", linewidth=2)
    plt.tight_layout()
    ax.figure.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return png_path


class VizardDataManager:
    """Менеджер ленивого доступа к архиву IceClass GeoTIFF."""

    def __init__(self, data_dir: str, index_path: str = DEFAULT_INDEX_NAME):
        self.data_dir = os.path.abspath(data_dir)
        self.index_path = os.path.abspath(index_path)
        self.index_df = gpd.GeoDataFrame(geometry=[], crs=None)
        self.crs = None
        self.scan_errors: List[Dict[str, str]] = []
        self._build_or_load_index()

    def _parse_filename(self, filename: str) -> Dict:
        """Парсит устойчивое ядро имени Sentinel-1, игнорируя служебные суффиксы."""
        stem = os.path.splitext(os.path.basename(filename))[0]
        parts = stem.split("_")
        if len(parts) < 9:
            raise ValueError(f"Недостаточно частей в имени файла: {filename}")

        core = parts[:9]
        suffix = "_".join(parts[9:]) if len(parts) > 9 else ""

        start_time = datetime.strptime(core[4], FILENAME_TIME_FORMAT)
        stop_time = datetime.strptime(core[5], FILENAME_TIME_FORMAT)
        product_token = core[8]
        if product_token.upper().endswith(".SAFE"):
            product_id = product_token[:-5]
        else:
            product_id = product_token

        return {
            "mission": core[0],
            "acquisition_mode": core[1],
            "product_type": core[2],
            "product_level": core[3],
            "start_time": start_time,
            "stop_time": stop_time,
            "date": start_time.date().isoformat(),
            "orbit": core[6],
            "datatake_id": core[7],
            "product_token": product_token,
            "product_id": product_id,
            "suffix": suffix or None,
        }

    def _normalize_filepath(self, filepath: str) -> str:
        return os.path.normcase(os.path.abspath(filepath))

    def _iter_tiff_files(self) -> Iterable[str]:
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"Папка с данными не найдена: {self.data_dir}")

        for root, _, files in os.walk(self.data_dir):
            for filename in files:
                if filename.lower().endswith((".tif", ".tiff")):
                    yield self._normalize_filepath(os.path.join(root, filename))

    def _record_scan_error(self, filepath: str, error: Exception):
        self.scan_errors.append({"filepath": filepath, "error": str(error)})

    def _build_index_record(self, filepath: str) -> Dict:
        filepath = self._normalize_filepath(filepath)
        parsed = self._parse_filename(os.path.basename(filepath))

        with rasterio.open(filepath) as src:
            if src.crs is None:
                raise ValueError(f"Файл без CRS: {filepath}")

            if self.crs is None:
                self.crs = src.crs
            elif src.crs != self.crs:
                raise ValueError(
                    f"Файл {filepath} имеет CRS {src.crs}, отличный от базового CRS {self.crs}"
                )

            nodata = src.nodata
            dtypes = sorted(set(src.dtypes))

            parsed.update(
                {
                    "filepath": filepath,
                    "source_crs": src.crs.to_string(),
                    "band_count": src.count,
                    "width": src.width,
                    "height": src.height,
                    "resolution_x": float(src.res[0]),
                    "resolution_y": float(src.res[1]),
                    "dtype": dtypes[0] if len(dtypes) == 1 else ",".join(src.dtypes),
                    "nodata": float(nodata) if nodata is not None else np.nan,
                    "geometry": box(*src.bounds),
                }
            )

        return parsed

    def _normalize_index(self, index_df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        if index_df.empty:
            return gpd.GeoDataFrame(index_df, geometry="geometry", crs=self.crs)

        if "filepath" in index_df.columns:
            index_df["filepath"] = index_df["filepath"].astype(str).map(self._normalize_filepath)

        for column in INDEX_DATETIME_COLUMNS:
            if column in index_df.columns:
                index_df[column] = pd.to_datetime(index_df[column], errors="coerce")

        if "date" not in index_df.columns and "start_time" in index_df.columns:
            index_df["date"] = pd.to_datetime(index_df["start_time"], errors="coerce").dt.strftime("%Y-%m-%d")

        if "date" in index_df.columns:
            index_df["date"] = pd.to_datetime(index_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")

        numeric_columns = ("band_count", "width", "height", "resolution_x", "resolution_y", "nodata")
        for column in numeric_columns:
            if column in index_df.columns:
                index_df[column] = pd.to_numeric(index_df[column], errors="coerce")

        index_df = index_df.sort_values(
            [column for column in INDEX_SORT_COLUMNS if column in index_df.columns]
        ).reset_index(drop=True)
        if "filepath" in index_df.columns:
            index_df = index_df.drop_duplicates(subset="filepath", keep="last").reset_index(drop=True)
        return gpd.GeoDataFrame(index_df, geometry="geometry", crs=index_df.crs)

    def _index_has_required_schema(self, index_df: gpd.GeoDataFrame) -> bool:
        return all(column in index_df.columns for column in REQUIRED_INDEX_COLUMNS)

    def _persist_index(self):
        if self.index_df.empty:
            return

        directory = os.path.dirname(self.index_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        save_df = self.index_df.copy()
        save_df.to_file(self.index_path, driver="GPKG")

    def _load_index(self) -> bool:
        if not os.path.exists(self.index_path):
            return False

        loaded = gpd.read_file(self.index_path)
        if not self._index_has_required_schema(loaded):
            return False
        self.index_df = self._normalize_index(loaded)
        self.crs = self.index_df.crs
        if len(self.index_df) != len(loaded):
            self._persist_index()
        return not self.index_df.empty

    def _collect_index_records(self, filepaths: Iterable[str]) -> List[Dict]:
        records: List[Dict] = []
        for filepath in filepaths:
            try:
                records.append(self._build_index_record(filepath))
            except Exception as error:
                self._record_scan_error(filepath, error)
        return records

    def _build_or_load_index(self):
        """Создает индекс при первом запуске или поднимает его из GeoPackage."""
        if self._load_index():
            return
        self.rebuild_index()

    def rebuild_index(self):
        """Полностью пересобирает индекс по всем GeoTIFF в директории данных."""
        self.scan_errors = []
        self.crs = None
        records = self._collect_index_records(self._iter_tiff_files())
        self.index_df = self._normalize_index(gpd.GeoDataFrame(records, geometry="geometry", crs=self.crs))
        self._persist_index()
        return self.index_df

    def update_data(self) -> int:
        """Добавляет в индекс только новые файлы, если они появились в архиве."""
        if self.index_df.empty:
            self.rebuild_index()
            return len(self.index_df)

        known_paths = set(self.index_df["filepath"].astype(str))
        new_files = [filepath for filepath in self._iter_tiff_files() if filepath not in known_paths]
        if not new_files:
            return 0

        new_records = self._collect_index_records(new_files)
        if not new_records:
            return 0

        new_df = gpd.GeoDataFrame(new_records, geometry="geometry", crs=self.crs)
        merged = pd.concat([self.index_df, new_df], ignore_index=True)
        merged = merged.drop_duplicates(subset="filepath", keep="last")
        self.index_df = self._normalize_index(gpd.GeoDataFrame(merged, geometry="geometry", crs=self.crs))
        self._persist_index()
        return len(new_records)

    def list_dates(self) -> List[str]:
        """Возвращает отсортированный список доступных дат наблюдений."""
        if self.index_df.empty or "date" not in self.index_df.columns:
            return []
        return sorted(item for item in self.index_df["date"].dropna().unique().tolist())

    def _validate_polygon(self, polygon: BaseGeometry) -> BaseGeometry:
        if not isinstance(polygon, BaseGeometry):
            raise TypeError("polygon должен быть объектом shapely geometry.")
        if polygon.is_empty:
            raise ValueError("polygon не должен быть пустым.")
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.is_empty or not polygon.is_valid:
            raise ValueError("polygon не удалось привести к валидной геометрии.")
        if self.crs is None:
            raise ValueError("CRS индекса не определен. Сначала постройте индекс.")
        return polygon

    def query_metadata_by_polygon(self, polygon: Polygon) -> gpd.GeoDataFrame:
        """Возвращает метаданные файлов, пересекающих заданный полигон."""
        polygon = self._validate_polygon(polygon)
        if self.index_df.empty:
            return self.index_df.copy()

        candidates = self.index_df
        try:
            spatial_index = self.index_df.sindex
            if spatial_index is not None:
                candidate_idx = list(spatial_index.query(polygon, predicate="intersects"))
                candidates = self.index_df.iloc[candidate_idx].copy()
        except Exception:
            candidates = self.index_df.copy()

        if candidates.empty:
            return candidates

        filtered = candidates[candidates.geometry.intersects(polygon)].copy()
        return self._normalize_index(filtered)

    def _select_date_groups(
        self,
        matches: gpd.GeoDataFrame,
        history_depth: Optional[int] = None,
    ) -> List[Tuple[str, gpd.GeoDataFrame]]:
        if matches.empty:
            return []

        unique_dates = sorted(matches["date"].dropna().unique().tolist())
        if history_depth is not None:
            if history_depth <= 0:
                raise ValueError("history_depth должен быть положительным числом.")
            unique_dates = unique_dates[-history_depth:]

        groups: List[Tuple[str, gpd.GeoDataFrame]] = []
        for day in unique_dates:
            day_group = matches[matches["date"] == day].copy()
            day_group = day_group.sort_values(["start_time", "stop_time", "filepath"]).reset_index(drop=True)
            groups.append((day, day_group))
        return groups

    def _resolve_snapshot_nodata(self, group: gpd.GeoDataFrame):
        nodata_values = group["nodata"].dropna().unique().tolist() if "nodata" in group.columns else []
        return nodata_values[0] if len(nodata_values) == 1 else None

    def _build_snapshot_for_group(
        self,
        day: str,
        group: gpd.GeoDataFrame,
        polygon: Polygon,
        target_bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> HistoricalSnapshot:
        filepaths = group["filepath"].tolist()
        if not filepaths:
            raise ValueError("group не должен быть пустым.")

        with rasterio.open(filepaths[0]) as reference:
            res_x = abs(reference.transform.a)
            res_y = abs(reference.transform.e)
            output_dtype = np.dtype(reference.dtypes[0])
            snapshot_bounds = tuple(target_bounds or group.total_bounds.tolist())
            width = max(1, int(np.ceil((snapshot_bounds[2] - snapshot_bounds[0]) / res_x)))
            height = max(1, int(np.ceil((snapshot_bounds[3] - snapshot_bounds[1]) / res_y)))
            out_transform = from_origin(snapshot_bounds[0], snapshot_bounds[3], res_x, res_y)
            band_count = reference.count

        filled_data = np.zeros((band_count, height, width), dtype=output_dtype)
        observation_mask = np.zeros((height, width), dtype=bool)
        semantic_valid = np.zeros((height, width), dtype=bool)

        for filepath in filepaths:
            with rasterio.open(filepath) as dataset:
                dataset_bounds = (dataset.bounds.left, dataset.bounds.bottom, dataset.bounds.right, dataset.bounds.top)
                read_bounds = _intersect_bounds(dataset_bounds, snapshot_bounds)
                if read_bounds is None:
                    continue

                full_window = Window(0, 0, dataset.width, dataset.height)
                read_window = from_bounds(*read_bounds, transform=dataset.transform)
                read_window = read_window.intersection(full_window)
                read_window = read_window.round_offsets(op="floor").round_lengths(op="ceil")
                read_window = read_window.intersection(full_window)
                if read_window.width <= 0 or read_window.height <= 0:
                    continue

                dataset_data = dataset.read(window=read_window)
                dataset_mask = dataset.read_masks(window=read_window).all(axis=0) > 0
                read_transform = window_transform(read_window, dataset.transform)

                reprojected_data = np.zeros_like(filled_data, dtype=output_dtype)
                reprojected_mask = np.zeros((height, width), dtype=np.uint8)

                for band_index in range(dataset.count):
                    reproject(
                        source=dataset_data[band_index],
                        destination=reprojected_data[band_index],
                        src_transform=read_transform,
                        src_crs=dataset.crs,
                        dst_transform=out_transform,
                        dst_crs=dataset.crs,
                        resampling=Resampling.nearest,
                    )

                reproject(
                    source=dataset_mask.astype(np.uint8),
                    destination=reprojected_mask,
                    src_transform=read_transform,
                    src_crs=dataset.crs,
                    dst_transform=out_transform,
                    dst_crs=dataset.crs,
                    resampling=Resampling.nearest,
                )

                _overlay_arrays(
                    filled_data,
                    observation_mask,
                    semantic_valid,
                    reprojected_data,
                    reprojected_mask > 0,
                )

        return HistoricalSnapshot(
            data=filled_data,
            transform=out_transform,
            crs=self.crs,
            timestamp=day,
            observation_mask=observation_mask,
            nodata=self._resolve_snapshot_nodata(group),
            metadata={
                "date": day,
                "overlap_strategy": "last_semantic",
                "source_file_count": len(group),
                "query_polygon_bounds": tuple(polygon.bounds),
                "mosaic_bounds": tuple(group.total_bounds.tolist()),
                "snapshot_bounds": tuple(snapshot_bounds),
            },
            source_files=filepaths,
        )

    def get_history_by_polygon(
        self,
        polygon: Polygon,
        history_depth: Optional[int] = None,
        crop_bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> List[HistoricalSnapshot]:
        """
        Ищет все GeoTIFF, пересекающие полигон, и возвращает историю слепков.

        Если задан `crop_bounds`, то каждый дневной слепок строится сразу в этом
        окне, без сборки большой мозаики на все найденные фреймы.
        """
        matches = self.query_metadata_by_polygon(polygon)
        if matches.empty:
            return []

        history: List[HistoricalSnapshot] = []
        for day, group in self._select_date_groups(matches, history_depth=history_depth):
            history.append(self._build_snapshot_for_group(day, group, polygon, target_bounds=crop_bounds))
        return history

    def iter_model_chunks_from_geotiff(
        self,
        geotiff_path: str,
        patch_size: int = DEFAULT_MODEL_PATCH_SIZE_PX,
        stride: Optional[int] = None,
        pad_edges: bool = True,
    ) -> Iterator[ModelInputChunk]:
        """
        Нарезает GeoTIFF на квадратные патчи patch_size×patch_size без изменения значений пикселей.

        Чтение идёт окнами rasterio (без загрузки всего файла в память).
        """
        path = os.path.abspath(geotiff_path)
        if patch_size <= 0:
            raise ValueError("patch_size должен быть положительным.")
        step = stride if stride is not None else _default_model_chunk_stride(patch_size)

        with rasterio.open(path) as src:
            height, width = src.height, src.width
            crs = src.crs
            base_transform = src.transform
            pad_value = _resolve_chunk_pad_value(src.nodata, np.dtype(src.dtypes[0]))

            for row_off, col_off in _iter_model_chunk_anchors(height, width, patch_size, step, pad_edges):
                read_h = min(patch_size, height - row_off)
                read_w = min(patch_size, width - col_off)
                read_window = Window(col_off, row_off, read_w, read_h)
                tile = src.read(window=read_window)
                if pad_edges and (read_h < patch_size or read_w < patch_size):
                    tile = _crop_pad_multiband(tile, 0, 0, patch_size, pad_value)
                logic_window = Window(col_off, row_off, patch_size, patch_size)
                chunk_transform = window_transform(logic_window, base_transform)
                bounds = array_bounds(patch_size, patch_size, chunk_transform)
                yield ModelInputChunk(
                    data=tile,
                    col_off=col_off,
                    row_off=row_off,
                    transform=chunk_transform,
                    crs=crs,
                    bounds=bounds,
                    window=logic_window,
                )

    def iter_model_chunks_from_snapshot(
        self,
        snapshot: HistoricalSnapshot,
        patch_size: int = DEFAULT_MODEL_PATCH_SIZE_PX,
        stride: Optional[int] = None,
        pad_edges: bool = True,
    ) -> Iterator[ModelInputChunk]:
        """То же, что iter_model_chunks_from_geotiff, но для HistoricalSnapshot в памяти."""
        if patch_size <= 0:
            raise ValueError("patch_size должен быть положительным.")
        step = stride if stride is not None else _default_model_chunk_stride(patch_size)
        _, height, width = snapshot.data.shape
        pad_value = _resolve_chunk_pad_value(snapshot.nodata, snapshot.data.dtype)

        for row_off, col_off in _iter_model_chunk_anchors(height, width, patch_size, step, pad_edges):
            if pad_edges:
                tile = _crop_pad_multiband(snapshot.data, row_off, col_off, patch_size, pad_value)
            else:
                tile = snapshot.data[:, row_off : row_off + patch_size, col_off : col_off + patch_size]
            logic_window = Window(col_off, row_off, patch_size, patch_size)
            chunk_transform = window_transform(logic_window, snapshot.transform)
            bounds = array_bounds(patch_size, patch_size, chunk_transform)
            yield ModelInputChunk(
                data=tile,
                col_off=col_off,
                row_off=row_off,
                transform=chunk_transform,
                crs=snapshot.crs,
                bounds=bounds,
                window=logic_window,
            )

    def iter_model_input_chunks(
        self,
        source: Union[str, HistoricalSnapshot],
        patch_size: int = DEFAULT_MODEL_PATCH_SIZE_PX,
        stride: Optional[int] = None,
        pad_edges: bool = True,
    ) -> Iterator[ModelInputChunk]:
        """Единая точка входа: путь к GeoTIFF или HistoricalSnapshot."""
        if isinstance(source, HistoricalSnapshot):
            return self.iter_model_chunks_from_snapshot(source, patch_size, stride, pad_edges)
        if isinstance(source, str):
            return self.iter_model_chunks_from_geotiff(source, patch_size, stride, pad_edges)
        raise TypeError("source должен быть str (путь к GeoTIFF) или HistoricalSnapshot.")

    def export_model_chunks_to_folder(
        self,
        source: Union[str, HistoricalSnapshot],
        output_dir: str,
        patch_size: int = DEFAULT_MODEL_PATCH_SIZE_PX,
        stride: Optional[int] = None,
        pad_edges: bool = True,
        *,
        save_png: bool = True,
        name_prefix: str = "patch",
        nodata: Optional[float] = None,
    ) -> Dict[str, object]:
        """
        Нарезает растр на патчи и сохраняет GeoTIFF в ``output_dir/tif/``.

        При ``save_png=True`` дополнительно пишет RGB-предпросмотр в ``output_dir/png/``
        (для float-растров — поканальное растяжение на 0…255 только в PNG).

        ``nodata`` в GeoTIFF: из аргумента, иначе из исходного файла / слепка.
        """
        out_root = os.path.abspath(output_dir)
        tif_dir = os.path.join(out_root, "tif")
        png_dir = os.path.join(out_root, "png") if save_png else None
        os.makedirs(tif_dir, exist_ok=True)
        if save_png:
            os.makedirs(png_dir, exist_ok=True)

        eff_nodata: Optional[float] = nodata
        if eff_nodata is None:
            if isinstance(source, HistoricalSnapshot):
                nd = source.nodata
                if nd is not None and not (isinstance(nd, float) and np.isnan(nd)):
                    eff_nodata = float(nd)
            else:
                with rasterio.open(os.path.abspath(source)) as src:
                    nd = src.nodata
                    if nd is not None and not (isinstance(nd, float) and np.isnan(nd)):
                        eff_nodata = float(nd)

        tif_paths: List[str] = []
        png_paths: List[str] = []
        count = 0
        for chunk in self.iter_model_input_chunks(source, patch_size, stride, pad_edges):
            base = _model_chunk_base_name(name_prefix, chunk.row_off, chunk.col_off)
            tif_path = os.path.join(tif_dir, f"{base}.tif")
            _write_model_chunk_geotiff(chunk, tif_path, nodata=eff_nodata)
            tif_paths.append(tif_path)
            if save_png and png_dir is not None:
                png_path = os.path.join(png_dir, f"{base}.png")
                _write_model_chunk_png(chunk, png_path)
                png_paths.append(png_path)
            count += 1

        return {
            "output_dir": out_root,
            "tif_dir": tif_dir,
            "png_dir": png_dir,
            "chunk_count": count,
            "tif_paths": tif_paths,
            "png_paths": png_paths,
        }

    def export_polygon_cropped_stack_and_mask(
        self,
        polygon: Polygon,
        history_depth: Optional[int] = None,
        buffer_fraction: float = 0.2,
        output_prefix: str = "polygon_export",
        tile_size: int = 1024,
    ) -> Dict[str, object]:
        """
        Пайплайн: история по полигону → стек в окне bbox(полигон + buffer) → GeoTIFF + маска.

        Маска (*_class0_mask.tif): значение 255 для пикселей **внутри исходного полигона
        запроса**, если там класс 0 IceClass RGB [1,1,1] **или** нет наблюдения в стеке
        (дыра между снимками). Снаружи полигона — 0. Внутри полигона с осмысленным льдом — 0.

        Возвращает словарь с ключами stacked_cropped, class0_mask, paths, crop_bounds.
        """
        polygon = self._validate_polygon(polygon)
        crop_bounds = polygon_buffer_bounds(polygon, buffer_fraction=buffer_fraction)
        matches = self.query_metadata_by_polygon(polygon)
        date_groups = self._select_date_groups(matches, history_depth=history_depth)
        if not date_groups:
            raise ValueError("По полигону не найдено ни одной сцены для истории.")

        output_prefix = os.path.splitext(output_prefix)[0]
        output_dir = os.path.dirname(os.path.abspath(output_prefix))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        base = output_prefix
        paths = {
            "stack_cropped_tif": f"{base}_stack_cropped.tif",
            "stack_cropped_png": f"{base}_stack_cropped.png",
            "class0_mask_tif": f"{base}_class0_mask.tif",
            "class0_mask_png": f"{base}_class0_mask.png",
            "streaming_tiles_json": f"{base}_streaming_write_tiles.json",
        }

        ordered_filepaths: List[str] = []
        history_dates: List[str] = []
        for day, group in date_groups:
            history_dates.append(day)
            ordered_filepaths.extend(group["filepath"].tolist())

        with ExitStack() as stack:
            datasets = []
            for filepath in ordered_filepaths:
                dataset = stack.enter_context(rasterio.open(filepath))
                datasets.append((filepath, dataset))

            if not datasets:
                raise ValueError("После отбора истории не осталось файлов для сборки стека.")

            reference = datasets[0][1]
            width, height, out_transform = _grid_from_bounds(crop_bounds, reference)
            output_dtype = np.dtype(reference.dtypes[0])
            band_count = reference.count

            stack_meta = {
                "driver": "GTiff",
                "height": height,
                "width": width,
                "count": band_count,
                "dtype": output_dtype.name,
                "crs": self.crs,
                "transform": out_transform,
                "compress": "LZW",
                "tiled": True,
                "BIGTIFF": "IF_SAFER",
            }
            mask_meta = {
                "driver": "GTiff",
                "height": height,
                "width": width,
                "count": 1,
                "dtype": "uint8",
                "crs": self.crs,
                "transform": out_transform,
                "nodata": 0,
                "compress": "LZW",
                "tiled": True,
                "BIGTIFF": "IF_SAFER",
            }

            observation_pixels = 0
            with rasterio.open(paths["stack_cropped_tif"], "w", **stack_meta) as stack_dst, rasterio.open(
                paths["class0_mask_tif"], "w", **mask_meta
            ) as mask_dst:
                for window in _iter_windows(width, height, tile_size=tile_size):
                    tile_height = int(window.height)
                    tile_width = int(window.width)
                    tile_data = np.zeros((band_count, tile_height, tile_width), dtype=output_dtype)
                    tile_mask = np.zeros((tile_height, tile_width), dtype=bool)
                    tile_semantic = np.zeros((tile_height, tile_width), dtype=bool)
                    tile_transform = window_transform(window, out_transform)
                    tile_bounds = array_bounds(tile_height, tile_width, tile_transform)

                    for _, dataset in datasets:
                        dataset_bounds = (
                            dataset.bounds.left,
                            dataset.bounds.bottom,
                            dataset.bounds.right,
                            dataset.bounds.top,
                        )
                        read_bounds = _intersect_bounds(dataset_bounds, tile_bounds)
                        if read_bounds is None:
                            continue

                        full_window = Window(0, 0, dataset.width, dataset.height)
                        read_window = from_bounds(*read_bounds, transform=dataset.transform)
                        read_window = read_window.intersection(full_window)
                        read_window = read_window.round_offsets(op="floor").round_lengths(op="ceil")
                        read_window = read_window.intersection(full_window)
                        if read_window.width <= 0 or read_window.height <= 0:
                            continue

                        dataset_data = dataset.read(window=read_window)
                        dataset_mask = dataset.read_masks(window=read_window).all(axis=0) > 0
                        read_transform = window_transform(read_window, dataset.transform)

                        reprojected_data = np.zeros((band_count, tile_height, tile_width), dtype=output_dtype)
                        reprojected_mask = np.zeros((tile_height, tile_width), dtype=np.uint8)

                        for band_index in range(dataset.count):
                            reproject(
                                source=dataset_data[band_index],
                                destination=reprojected_data[band_index],
                                src_transform=read_transform,
                                src_crs=dataset.crs,
                                dst_transform=tile_transform,
                                dst_crs=dataset.crs,
                                resampling=Resampling.nearest,
                            )

                        reproject(
                            source=dataset_mask.astype(np.uint8),
                            destination=reprojected_mask,
                            src_transform=read_transform,
                            src_crs=dataset.crs,
                            dst_transform=tile_transform,
                            dst_crs=dataset.crs,
                            resampling=Resampling.nearest,
                        )

                        _overlay_arrays(
                            tile_data,
                            tile_mask,
                            tile_semantic,
                            reprojected_data,
                            reprojected_mask > 0,
                        )

                    inside_query = _pixels_inside_polygon_mask(
                        polygon, tile_height, tile_width, tile_transform
                    )
                    class0_tile = np.zeros((1, tile_height, tile_width), dtype=np.uint8)
                    empty_mask = _iceclass_empty_mask_rgb(tile_data, tile_mask)
                    # Внутри полигона: класс 0 [1,1,1] и «дыры» без наблюдения — всё в маску (255).
                    class0_tile[0, inside_query & (~tile_mask | empty_mask)] = 255

                    stack_dst.write(tile_data, window=window)
                    stack_dst.write_mask((tile_mask.astype(np.uint8) * 255), window=window)
                    mask_dst.write(class0_tile, window=window)
                    mask_dst.write_mask((inside_query.astype(np.uint8) * 255), window=window)
                    observation_pixels += int(tile_mask.sum())

                stack_dst.update_tags(
                    timestamp=pd.Timestamp(history_dates[-1]).isoformat(),
                    source_file_count=len(ordered_filepaths),
                    polygon_buffer_fraction=buffer_fraction,
                    history_dates=",".join(history_dates),
                    iceclass_empty_class_rgb="1,1,1",
                    streaming_tile_size_px=str(tile_size),
                    streaming_tiles_manifest=paths["streaming_tiles_json"],
                )
                mask_dst.update_tags(
                    timestamp=pd.Timestamp(history_dates[-1]).isoformat(),
                    source_file_count=len(ordered_filepaths),
                    polygon_buffer_fraction=buffer_fraction,
                    history_dates=",".join(history_dates),
                    mask_kind="iceclass_class0_and_nodata_inside_query_polygon",
                    streaming_tiles_manifest=paths["streaming_tiles_json"],
                )

        write_streaming_tile_manifest_json(paths["streaming_tiles_json"], width, height, tile_size)

        save_tif_preview_with_polygon(
            paths["stack_cropped_tif"],
            polygon,
            paths["stack_cropped_png"],
            title=f"Stack (cropped) до {history_dates[-1]}",
        )
        save_tif_preview_with_polygon(
            paths["class0_mask_tif"],
            polygon,
            paths["class0_mask_png"],
            title="Маска класса 0 [1,1,1]",
            is_mask=True,
        )

        return {
            "stacked_cropped": {
                "shape": (band_count, height, width),
                "observation_pixels": observation_pixels,
                "timestamp": history_dates[-1],
            },
            "class0_mask": {
                "shape": (1, height, width),
                "timestamp": history_dates[-1],
            },
            "history": history_dates,
            "paths": paths,
            "crop_bounds": crop_bounds,
            "streaming": {
                "tile_size_px": tile_size,
                "raster_shape_px": (height, width),
                "tiles_manifest": paths["streaming_tiles_json"],
                "model_chunks": (
                    "VizardDataManager.iter_model_chunks_from_geotiff(paths['stack_cropped_tif'], "
                    f"patch_size={DEFAULT_MODEL_PATCH_SIZE_PX})"
                ),
            },
        }


def _resolve_default_iceclass_dir() -> str:
    """Ищет папку архива IceClass относительно текущей рабочей директории."""
    candidates = (
        "Dataset_2025_IceClass",
        os.path.join("vizard_iceclass", "Dataset_2025_IceClass"),
    )
    for rel in candidates:
        abs_path = os.path.abspath(rel)
        if os.path.isdir(abs_path):
            return abs_path
    raise FileNotFoundError(
        "Не найдена папка с IceClass. Ожидается одна из: "
        + ", ".join(os.path.abspath(c) for c in candidates)
    )


def run_tests():
    """Небольшой smoke-test на реальных данных IceClass."""
    print("=== ЗАПУСК SMOKE-ТЕСТА ===")

    try:
        data_root = _resolve_default_iceclass_dir()
        print(f"[TEST 0] data_dir: {data_root}")
        manager = VizardDataManager(data_dir=data_root)
        print(f"[TEST 1] Индекс: OK ({len(manager.index_df)} файлов)")
    except Exception as error:
        print(f"[TEST 1] Индекс: FAILED ({error})")
        return

    if manager.index_df.empty:
        print("[TEST 2] Поиск: FAILED (индекс пуст)")
        return

    test_polygon = manager.index_df.iloc[0].geometry
    history = manager.get_history_by_polygon(test_polygon, history_depth=3)
    if not history:
        print("[TEST 2] Поиск/мозаика: FAILED (история не найдена)")
        return
    print(f"[TEST 2] Поиск/мозаика: OK ({len(history)} слоев)")

    try:
        latest = history[-1]
        output_dir = os.path.join("output_test")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"history_{latest.timestamp:%Y%m%d}.tif")
        latest.save(output_file)
        with rasterio.open(output_file) as src:
            print(f"[TEST 3] Сохранение: OK ({src.width}x{src.height}, bands={src.count})")
    except Exception as error:
        print(f"[TEST 3] Сохранение: FAILED ({error})")
        return

    if manager.scan_errors:
        print(f"[TEST 4] Предупреждение: часть файлов пропущена ({len(manager.scan_errors)})")
    else:
        print("[TEST 4] Пропущенные файлы: OK")

    print("=== SMOKE-ТЕСТ ЗАВЕРШЕН ===")


def run_model_chunk_tests(
    *,
    patch_size: int = DEFAULT_MODEL_PATCH_SIZE_PX,
    stride: Optional[int] = None,
    raster_height: Optional[int] = None,
    raster_width: Optional[int] = None,
    bands: int = 3,
    crs: str = "EPSG:32637",
    transform: Optional[object] = None,
    transform_origin_xy: Tuple[float, float] = (500000.0, 7000000.0),
    transform_resolution: Tuple[float, float] = (10.0, -10.0),
    rng_seed: int = 42,
    output_subdir: str = "model_chunk_smoke",
) -> bool:
    """
    Проверка нарезки: размеры патчей, число якорей, affine углового окна.

    По умолчанию растр чуть больше сетки из трёх якорей по каждой оси (как при stride из prepare.py).
    Исходный ``synthetic_raster.tif`` — детерминированные градиенты (удобно смотреть в превью);
    ``rng_seed`` задаёт небольшой сдвиг яркости по каналам.
    """
    if patch_size <= 0:
        raise ValueError("patch_size должен быть положительным.")
    step = stride if stride is not None else _default_model_chunk_stride(patch_size)
    height = raster_height if raster_height is not None else patch_size + 2 * step
    width = raster_width if raster_width is not None else patch_size + 2 * step
    base_transform = transform or from_origin(
        transform_origin_xy[0],
        transform_origin_xy[1],
        transform_resolution[0],
        transform_resolution[1],
    )
    data = _synthetic_chunk_test_raster(bands, height, width, seed=rng_seed)

    base = os.path.abspath(os.path.join("test_output3", output_subdir))
    os.makedirs(base, exist_ok=True)
    tif_path = os.path.join(base, "synthetic_raster.tif")
    meta = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": bands,
        "dtype": "uint8",
        "crs": crs,
        "transform": base_transform,
        "nodata": 0,
    }
    with rasterio.open(tif_path, "w", **meta) as dst:
        dst.write(data)

    # Методы нарезки не используют состояние менеджера; обходим __init__ без пустого индекса.
    manager = VizardDataManager.__new__(VizardDataManager)

    chunks = list(
        manager.iter_model_chunks_from_geotiff(
            tif_path, patch_size=patch_size, stride=stride, pad_edges=True
        )
    )
    expected_n = len(
        list(_iter_model_chunk_anchors(height, width, patch_size, step, pad_edges=True))
    )
    assert len(chunks) == expected_n, (len(chunks), expected_n)
    for ch in chunks:
        assert ch.data.shape == (bands, patch_size, patch_size)

    last = chunks[-1]
    expected_affine = window_transform(
        Window(last.col_off, last.row_off, patch_size, patch_size), base_transform
    )
    assert np.allclose(tuple(last.transform)[:6], tuple(expected_affine)[:6], atol=1e-9)

    snap = HistoricalSnapshot(
        data=data.copy(),
        transform=base_transform,
        crs=crs,
        timestamp="2020-01-01",
        observation_mask=np.ones((height, width), dtype=bool),
        nodata=0,
    )
    snap_chunks = list(
        manager.iter_model_input_chunks(snap, patch_size=patch_size, stride=stride, pad_edges=True)
    )
    assert len(snap_chunks) == expected_n
    assert np.array_equal(snap_chunks[-1].data, last.data)

    partial = list(
        manager.iter_model_chunks_from_geotiff(
            tif_path, patch_size=patch_size, stride=stride, pad_edges=False
        )
    )
    assert all(c.data.shape == (bands, patch_size, patch_size) for c in partial)

    export_root = os.path.join(base, "exported_with_png")
    with_png = manager.export_model_chunks_to_folder(
        tif_path,
        export_root,
        patch_size=patch_size,
        stride=stride,
        pad_edges=True,
        save_png=True,
        name_prefix="test",
    )
    assert with_png["chunk_count"] == expected_n
    assert len(with_png["tif_paths"]) == expected_n
    assert len(with_png["png_paths"]) == expected_n
    assert os.path.isfile(with_png["tif_paths"][0])
    assert os.path.isfile(with_png["png_paths"][0])

    export_np = os.path.join(base, "exported_tif_only")
    no_png = manager.export_model_chunks_to_folder(
        tif_path,
        export_np,
        patch_size=patch_size,
        stride=stride,
        pad_edges=True,
        save_png=False,
        name_prefix="test",
    )
    assert no_png["png_dir"] is None
    assert no_png["png_paths"] == []
    assert len(no_png["tif_paths"]) == expected_n

    print(
        f"[CHUNK TEST] OK: patch={patch_size}, stride={step}, растр {height}x{width}, "
        f"{len(chunks)} патчей, якорь ({last.row_off}, {last.col_off}); артефакты: {base}"
    )
    return True


def run_file_polygon_test(
    target_file: str,
    data_dir: Optional[str] = None,
    history_depth: int = 5,
    buffer_fraction: float = 0.2,
):
    """
    Загружает один GeoTIFF, строит полигон с тем же центром, в ~3 раза шире/выше сцены,
    и прогоняет export_polygon_cropped_stack_and_mask (стек + обрезка + маска класса 0).

    Если data_dir не задан, индекс строится по папке, в которой лежит target_file
    (ожидается корень архива IceClass, например Dataset_2025_IceClass).
    """
    print("=== ЗАПУСК FILE-POLYGON TEST ===")
    target_file = os.path.abspath(target_file)
    if data_dir is None:
        data_dir = os.path.dirname(target_file)
    else:
        data_dir = os.path.abspath(data_dir)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Папка с данными не найдена: {data_dir}")
    print(f"[TEST FP 0] data_dir (индекс архива): {data_dir}")

    with rasterio.open(target_file) as src:
        bounds = src.bounds
        crs = src.crs
        center_x = (bounds.left + bounds.right) / 2
        center_y = (bounds.bottom + bounds.top) / 2
        scene_width = bounds.right - bounds.left
        scene_height = bounds.top - bounds.bottom
        linear_scale = float(np.sqrt(3.0) / 2.0)
        polygon = box(
            center_x - scene_width * linear_scale,
            center_y - scene_height * linear_scale,
            center_x + scene_width * linear_scale,
            center_y + scene_height * linear_scale,
        )
        print(f"[TEST FP 1] Файл: {target_file}")
        print(f"[TEST FP 1] CRS: {crs}")
        print(f"[TEST FP 1] Исходные bounds: {bounds}")
        print(f"[TEST FP 1] Полигон ~x3 по площади относительно сцены, bounds: {polygon.bounds}")
        print(f"[TEST FP 1] Shape: {src.height}x{src.width}, bands={src.count}")

    manager = VizardDataManager(data_dir=data_dir)
    test_path = os.path.join("test_output3", "")
    os.makedirs(test_path, exist_ok=True)
    export_prefix = os.path.join(test_path, "polygon_pipeline_demo")

    result = manager.export_polygon_cropped_stack_and_mask(
        polygon,
        history_depth=history_depth,
        buffer_fraction=buffer_fraction,
        output_prefix=export_prefix,
    )
    history = result["history"]
    print(f"[TEST FP 2] Найдено исторических слоев: {len(history)}")
    if not history:
        print("[TEST FP 2] FAILED: история по полигону не найдена")
        return
    print(
        "[TEST FP 3] Даты истории:",
        history,
    )
    sc = result["stacked_cropped"]
    print(f"[TEST FP 4] Обрезанный стек: shape={sc['shape']}, crop_bounds={result['crop_bounds']}")
    print(f"[TEST FP 4] Пикселей observation: {sc['observation_pixels']}")
    print(f"[TEST FP 5] Файлы пайплайна:")
    for key, path in result["paths"].items():
        print(f"    {key}: {path}")
    print("=== FILE-POLYGON TEST ЗАВЕРШЕН ===")


if __name__ == "__main__":
    TIFS_PATH = 'Dataset_2025_IceClass/'
    run_file_polygon_test(TIFS_PATH + 'S1A_EW_GRDM_1SDH_20250218T031712_20250218T031812_057945_07267C_D886_fix_IceClass.tif')
    # run_model_chunk_tests()
