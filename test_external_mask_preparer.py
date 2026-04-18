from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from external_mask_preparer import ExternalMaskGeoTIFFPreparer


class ExternalMaskGeoTIFFPreparerTest(unittest.TestCase):
    def test_prepare_all_builds_model_ready_geotiff(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "images"
            mask_dir = root / "masks"
            output_dir = root / "output"
            image_dir.mkdir()
            mask_dir.mkdir()

            image_path = image_dir / "scene_001.tif"
            mask_path = mask_dir / "scene_001_class0_mask.tif"
            transform = from_origin(500000.0, 7000000.0, 40.0, 40.0)
            crs = "EPSG:3413"

            rgb = np.array(
                [
                    [
                        [0, 0, 250, 250],
                        [0, 0, 250, 250],
                        [0, 0, 250, 250],
                        [0, 0, 250, 250],
                    ],
                    [
                        [100, 100, 0, 0],
                        [100, 100, 0, 0],
                        [100, 100, 0, 0],
                        [100, 100, 0, 0],
                    ],
                    [
                        [255, 255, 255, 255],
                        [255, 255, 255, 255],
                        [255, 255, 255, 255],
                        [255, 255, 255, 255],
                    ],
                ],
                dtype=np.uint8,
            )
            hole_mask = np.zeros((4, 4), dtype=np.uint8)
            hole_mask[1, 1] = 255
            hole_mask[2, 2] = 255

            self._write_tif(image_path, rgb, transform, crs)
            self._write_tif(mask_path, hole_mask[None, ...], transform, crs)

            preparer = ExternalMaskGeoTIFFPreparer(
                image_dir=str(image_dir),
                mask_dir=str(mask_dir),
                output_dir=str(output_dir),
                save_drift_prev=True,
            )
            result = preparer.prepare_all()

            self.assertEqual(len(result.records), 1)
            self.assertEqual(result.unmatched_images, [])
            self.assertEqual(result.unused_masks, [])

            output_path = Path(result.records[0].output_path)
            self.assertTrue(output_path.exists())

            with rasterio.open(output_path) as src:
                data = src.read()
                self.assertEqual(src.count, 7)
                self.assertEqual(src.crs.to_string(), crs)
                self.assertEqual(src.transform, transform)
                self.assertEqual(src.descriptions, (
                    "source_band_1",
                    "source_band_2",
                    "source_band_3",
                    "masked_image",
                    "hole_mask",
                    "land_mask",
                    "drift_prev",
                ))
                self.assertEqual(src.tags()["external_mask_true_value"], "255")
                self.assertEqual(int(data[4, 1, 1]), 255)
                self.assertEqual(int(data[4, 2, 2]), 255)
                self.assertEqual(int(data[3, 1, 1]), 0)
                self.assertEqual(int(data[3, 2, 2]), 0)
                self.assertTrue(np.all(data[5] == 0))

    def _write_tif(self, path: Path, data: np.ndarray, transform, crs: str) -> None:
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=data.shape[1],
            width=data.shape[2],
            count=data.shape[0],
            dtype="uint8",
            crs=crs,
            transform=transform,
            compress="LZW",
        ) as dst:
            dst.write(data)


if __name__ == "__main__":
    unittest.main()
