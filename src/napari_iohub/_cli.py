"""Command-line entry points for napari-iohub widgets."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _make_annotator_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the cell-state annotator CLI."""
    parser = argparse.ArgumentParser(
        prog="napari-iohub-annotate",
        description=(
            "Launch napari with the cell-state annotator widget. "
            "Provide --images, --tracks, and --fov to auto-load a dataset, "
            "and --resume-csv to resume from a previous annotation file."
        ),
    )
    parser.add_argument(
        "-i",
        "--images",
        type=Path,
        help="Path to the images OME-Zarr dataset (HCS plate).",
    )
    parser.add_argument(
        "-t",
        "--tracks",
        type=Path,
        help="Path to the tracks OME-Zarr dataset (HCS plate with labels).",
    )
    parser.add_argument(
        "-f",
        "--fov",
        type=str,
        help="FOV path 'row/well/fov' (e.g. 'A/1/0').",
    )
    parser.add_argument(
        "-r",
        "--resume-csv",
        type=Path,
        help="Optional existing annotation CSV to resume from.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Folder where saved annotation CSVs will be written.",
    )
    parser.add_argument(
        "--project-2d",
        action="store_true",
        help=(
            "Project images to 2D around the FOV's in-focus Z "
            "(focus_slice.Phase3D.fov_statistics.z_focus_mean). "
            "Phase channels are sliced at the focus; fluorescence channels "
            "are max-projected over [focus-5, focus+10]. Warns if metadata "
            "is missing and falls back to full Z-stack."
        ),
    )
    parser.add_argument(
        "--no-expand-labels",
        action="store_true",
        help=(
            "Do not repeat the single-plane segmentation labels along Z "
            "to match the image Z-stack. Implicit when --project-2d is set."
        ),
    )
    parser.add_argument(
        "--no-load",
        action="store_true",
        help="Pre-fill the fields but do not trigger the load action.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def annotator_main(argv: list[str] | None = None) -> int:
    """Entry point: open napari with the cell-state annotator widget."""
    parser = _make_annotator_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.fov and not args.tracks:
        parser.error("--fov requires --tracks to be provided")

    import napari

    from napari_iohub._cell_state_annotator import CellStateAnnotatorWidget

    viewer = napari.Viewer()
    widget = CellStateAnnotatorWidget(viewer)
    viewer.window.add_dock_widget(widget, name="Cell state annotation")

    should_load = bool(
        args.images and args.tracks and args.fov and not args.no_load
    )
    widget.prefill(
        images=args.images,
        tracks=args.tracks,
        resume_csv=args.resume_csv,
        fov=args.fov,
        output=args.output,
        project_2d=args.project_2d,
        expand_labels=not args.no_expand_labels,
        load=should_load,
    )

    napari.run()
    return 0


if __name__ == "__main__":
    sys.exit(annotator_main())
