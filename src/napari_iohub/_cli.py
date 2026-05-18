"""Command-line entry points for napari-iohub widgets."""

from __future__ import annotations

import argparse
import getpass
import logging
import os
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path


def _make_annotator_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the cell-state annotator CLI."""
    parser = argparse.ArgumentParser(
        prog="napari-iohub-annotate",
        description=(
            "Launch napari with the cell-state annotator widget. "
            "Provide --images, --tracks, and --fov to auto-load a dataset, "
            "and --resume-csv to resume from a previous annotation file. "
            "Omitting --tracks enables centroid-only mode: no labels layer, "
            "no track_id on save (use napari-iohub-map-centroids to assign "
            "track IDs against a tracking.zarr later)."
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

    # --fov no longer requires --tracks: omitting tracks runs the annotator
    # in centroid-only mode (no labels, no track_id on save).
    if args.fov and not args.images:
        parser.error("--fov requires --images to be provided")

    import napari

    from napari_iohub._cell_state_annotator import CellStateAnnotatorWidget

    viewer = napari.Viewer()
    widget = CellStateAnnotatorWidget(viewer)
    viewer.window.add_dock_widget(widget, name="Cell state annotation")

    should_load = bool(args.images and args.fov and not args.no_load)
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


_STATE_COLUMNS = [
    "cell_division_state",
    "infection_state",
    "organelle_state",
    "cell_death_state",
]

_ULTRACK_INDEX_COLUMNS = [
    "fov_name",
    "track_id",
    "t",
    "id",
    "parent_track_id",
    "parent_id",
    "z",
    "y",
    "x",
]


def _make_map_centroids_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the centroid-mapping CLI."""
    parser = argparse.ArgumentParser(
        prog="napari-iohub-map-centroids",
        description=(
            "Map centroid annotations from a CSV onto a tracking.zarr by "
            "sampling a kernel around each (t, y, x) point in the "
            "segmentation labels and taking the mode of the non-zero "
            "labels as the track_id. Writes an ultrack-compatible CSV "
            "with full per-(track_id, t) expansion of the four state "
            "columns. Centroids that land entirely on background are "
            "dropped from the main output and listed in a sibling "
            "'<output_stem>_unmapped.csv'."
        ),
    )
    parser.add_argument(
        "-c",
        "--csv",
        type=Path,
        required=True,
        help="Input CSV (centroid-only or previously track-mapped).",
    )
    parser.add_argument(
        "-t",
        "--tracks",
        type=Path,
        required=True,
        help="Path to the tracking.zarr (HCS plate with segmentation labels).",
    )
    parser.add_argument(
        "-f",
        "--fov",
        type=str,
        required=True,
        help="FOV path 'row/well/fov' (e.g. 'A/1/000000').",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output CSV path for the mapped annotations.",
    )
    parser.add_argument(
        "--kernel",
        type=int,
        default=2,
        help=(
            "Kernel half-size in pixels (default 2, i.e. a 5x5 window). "
            "Mode is taken over the non-zero labels in the window."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def _mode_nonzero_label(labels_image, y: int, x: int, k: int) -> int:
    """Return the most-frequent non-zero label in a (2k+1, 2k+1) window around (y, x).

    Returns 0 if every pixel in the window is background (label == 0).
    """
    H, W = labels_image.shape[-2:]
    y0 = max(0, y - k)
    y1 = min(H, y + k + 1)
    x0 = max(0, x - k)
    x1 = min(W, x + k + 1)
    window = labels_image[y0:y1, x0:x1]
    # Filter out zeros, then take the most common label.
    flat = window.ravel()
    nonzero = flat[flat > 0]
    if nonzero.size == 0:
        return 0
    counts = Counter(int(v) for v in nonzero.tolist())
    return counts.most_common(1)[0][0]


def _expand_marked_events_to_ultrack(
    tracks_df,
    marked_events: dict[str, list[dict]],
    fov_name: str,
    annotator: str,
    annotation_date: str,
    annotation_version: str,
):
    """Reproduce the in-app per-(track_id, t) expansion used by the annotator.

    Returns a DataFrame whose rows are one per (track_id, t) combination
    present in ``tracks_df``, with the four state columns set according to
    the annotator's semantics:
        - mitosis: per-frame (only the marked frames)
        - infection: sticky-forward from first occurrence per track
        - remodel: per-frame
        - death: sticky-forward; once dead, other state columns become null
    """
    import pandas as pd

    all_annotations: list[dict] = []
    all_track_timepoints = tracks_df[["track_id", "t"]].drop_duplicates()

    for track_id in all_track_timepoints["track_id"].unique():
        track_timepoints = (
            all_track_timepoints[all_track_timepoints["track_id"] == track_id]["t"]
            .sort_values()
        )

        division = [e for e in marked_events["cell_division_state"] if e["track_id"] == track_id]
        mitosis_t = [e["t"] for e in division]

        infection = [e for e in marked_events["infection_state"] if e["track_id"] == track_id]
        first_infected_t = min((e["t"] for e in infection), default=None)

        organelle = [e for e in marked_events["organelle_state"] if e["track_id"] == track_id]
        remodel_t = [e["t"] for e in organelle]

        death = [e for e in marked_events["cell_death_state"] if e["track_id"] == track_id]
        first_death_t = min((e["t"] for e in death), default=None)

        for t in track_timepoints:
            is_dead = first_death_t is not None and t >= first_death_t
            if is_dead:
                cell_division_state = None
                infection_state = None
                organelle_state = None
                cell_death_state = "dead"
            else:
                cell_division_state = "mitosis" if t in mitosis_t else "interphase"
                if first_infected_t is None:
                    infection_state = None
                else:
                    infection_state = "infected" if t >= first_infected_t else "uninfected"
                organelle_state = "remodel" if t in remodel_t else "noremodel"
                cell_death_state = "alive" if first_death_t is not None else None

            all_annotations.append(
                {
                    "track_id": track_id,
                    "t": t,
                    "cell_division_state": cell_division_state,
                    "infection_state": infection_state,
                    "organelle_state": organelle_state,
                    "cell_death_state": cell_death_state,
                    "annotator": annotator,
                    "annotation_date": annotation_date,
                    "annotation_version": annotation_version,
                }
            )

    annotations_df = pd.DataFrame(all_annotations)
    tracks_df = tracks_df.copy()
    tracks_df["fov_name"] = fov_name
    merged = tracks_df.merge(annotations_df, on=["track_id", "t"], how="left")

    index_cols = [c for c in _ULTRACK_INDEX_COLUMNS if c in merged.columns]
    metadata_cols = ["annotator", "annotation_date", "annotation_version"]
    return merged[index_cols + _STATE_COLUMNS + metadata_cols]


def map_centroids_main(argv: list[str] | None = None) -> int:
    """Entry point: map centroid annotations onto a tracking.zarr.

    Workflow:
      1. Load the input CSV and restrict to ``--fov``.
      2. Open the matching FOV in ``--tracks``; locate its sibling tracks CSV.
      3. For each annotated centroid, sample a (2k+1)x(2k+1) window in the
         segmentation labels at that timepoint and take the mode of non-zero
         labels. That becomes the row's ``track_id``.
      4. Drop unmapped (background) rows into ``<output_stem>_unmapped.csv``.
      5. Expand the mapped events to the full ultrack per-(track_id, t)
         schema and write to ``--output``.
    """
    parser = _make_map_centroids_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger("napari_iohub.map_centroids")

    if not args.csv.exists():
        parser.error(f"Input CSV does not exist: {args.csv}")
    if not args.tracks.exists():
        parser.error(f"Tracks zarr does not exist: {args.tracks}")
    if args.kernel < 0:
        parser.error(f"--kernel must be >= 0 (got {args.kernel})")

    # Imports kept inside the function so `--help` is fast.
    import pandas as pd
    from iohub.ngff import open_ome_zarr

    df = pd.read_csv(args.csv)
    logger.info("Loaded %d rows from %s", len(df), args.csv)

    # Filter to the requested FOV (if the CSV has fov_name)
    if "fov_name" in df.columns:
        fov_rows = df[df["fov_name"] == args.fov]
        if fov_rows.empty:
            parser.error(
                f"No rows match --fov {args.fov!r} in {args.csv}. "
                f"Available FOVs: {sorted(df['fov_name'].dropna().unique().tolist())}"
            )
        df = fov_rows
        logger.info("Filtered to FOV %s: %d rows", args.fov, len(df))

    # Collapse to one row per (t, y, x, state-column). For old track-mapped
    # CSVs, the per-(track_id, t) expansion produces many redundant rows for
    # each state value (e.g. "infected" propagated forward for every t after
    # first_infected_t). We only want the *original click events*, which is:
    #   - mitosis / remodel: every row where state == mark value
    #   - infected / dead:   the first row per track_id where state == mark
    # If track_id is absent (centroid-only CSV), every row is already a click.
    marked_events: dict[str, list[dict]] = {col: [] for col in _STATE_COLUMNS}
    unmapped_rows: list[dict] = []

    # Open the tracks FOV once and cache per-t label arrays.
    plate = open_ome_zarr(str(args.tracks))
    try:
        tracks_fov = plate[args.fov]
    except KeyError:
        parser.error(f"FOV {args.fov!r} not found in {args.tracks}")
    tracks_fov_array = tracks_fov["0"]

    # Build mark events per state column with the right collapse rule
    state_mark_value = {col: val for _, _, col, val in _ANNOTATION_LAYERS_FOR_CLI}
    has_track_id = "track_id" in df.columns and df["track_id"].notna().any()

    for state_col, mark in state_mark_value.items():
        if state_col not in df.columns:
            continue
        annotated = df[df[state_col] == mark]
        if annotated.empty:
            continue

        if state_col in {"cell_division_state", "organelle_state"}:
            click_df = annotated  # per-frame: every annotated row is a click
        elif has_track_id:
            # First occurrence per old track_id is the click
            click_df = annotated.loc[annotated.groupby("track_id")["t"].idxmin()]
        else:
            click_df = annotated  # centroid-only: each row is a click

        for _, row in click_df.iterrows():
            t = int(row["t"])
            y = int(round(float(row["y"])))
            x = int(round(float(row["x"])))
            # Single-Z labels: index [t, 0, 0]
            labels_image = tracks_fov_array[t, 0, 0]
            label = _mode_nonzero_label(labels_image, y, x, args.kernel)
            if label > 0:
                marked_events[state_col].append({"track_id": int(label), "t": t})
            else:
                unmapped_rows.append(
                    {
                        "fov_name": args.fov,
                        "state_column": state_col,
                        "t": t,
                        "y": y,
                        "x": x,
                        "reason": (
                            f"no non-zero label in {2 * args.kernel + 1}x"
                            f"{2 * args.kernel + 1} kernel"
                        ),
                    }
                )

    n_mapped = sum(len(v) for v in marked_events.values())
    n_unmapped = len(unmapped_rows)
    logger.info(
        "Mapping summary: %d clicks mapped to a track, %d landed on background.",
        n_mapped,
        n_unmapped,
    )

    # Load the tracks CSV from the FOV directory to drive the expansion
    fov_dir = args.tracks / args.fov.strip("/")
    csv_candidates = sorted(fov_dir.glob("*.csv"))
    if not csv_candidates:
        parser.error(
            f"No tracks CSV (e.g. tracks_*.csv) found under {fov_dir}. "
            "The mapper needs one to expand annotations across the lineage."
        )
    if len(csv_candidates) > 1:
        logger.warning(
            "Multiple CSVs under %s; using %s.", fov_dir, csv_candidates[0]
        )
    tracks_csv_path = csv_candidates[0]
    logger.info("Loading tracks lineage from %s", tracks_csv_path)
    tracks_df = pd.read_csv(tracks_csv_path)

    try:
        annotator = os.getlogin()
    except OSError:
        annotator = getpass.getuser()
    annotation_date = datetime.now().isoformat()
    annotation_version = "1.0"

    mapped_df = _expand_marked_events_to_ultrack(
        tracks_df=tracks_df,
        marked_events=marked_events,
        fov_name=args.fov,
        annotator=annotator,
        annotation_date=annotation_date,
        annotation_version=annotation_version,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mapped_df.to_csv(args.output, index=False)
    logger.info("Wrote %d rows to %s", len(mapped_df), args.output)

    if unmapped_rows:
        unmapped_path = args.output.with_name(args.output.stem + "_unmapped.csv")
        pd.DataFrame(unmapped_rows).to_csv(unmapped_path, index=False)
        logger.info("Wrote %d unmapped centroids to %s", n_unmapped, unmapped_path)

    return 0


# Layer config duplicated here (small list) so the CLI doesn't need to import
# the Qt-heavy widget module. Kept in sync with ANNOTATION_LAYERS in
# _cell_state_annotator.
_ANNOTATION_LAYERS_FOR_CLI = [
    ("_mitosis_events", "blue", "cell_division_state", "mitosis"),
    ("_infected_events", "orange", "infection_state", "infected"),
    ("_remodel_events", "purple", "organelle_state", "remodel"),
    ("_death_events", "red", "cell_death_state", "dead"),
]


# Matches the trailing `_<row>_<col>_<fov>` segment of a per-FOV CSV filename,
# tolerating an optional `_centroids` suffix written by centroid-only saves.
# Row is one or more alphanumerics (covers `A`, `Control`, etc.), col is
# digits, fov is digits. Example matches:
#   2025_07_24_..._A_2_000000.csv          -> ("A", "2", "000000")
#   plate_C_2_000000_centroids.csv         -> ("C", "2", "000000")
_FOV_FILENAME_RE = re.compile(r"_([A-Za-z]+)_(\d+)_(\d+)(?:_centroids)?\.csv$")


def _parse_fov_from_filename(name: str) -> str | None:
    """Return ``row/col/fov`` parsed from a per-FOV CSV filename, or ``None``."""
    m = _FOV_FILENAME_RE.search(name)
    if not m:
        return None
    row, col, fov = m.groups()
    return f"{row}/{col}/{fov}"


def _make_combine_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the combine-annotations CLI."""
    parser = argparse.ArgumentParser(
        prog="napari-iohub-combine-annotations",
        description=(
            "Combine per-FOV annotation CSVs in a directory into one dataset-level "
            "CSV. Auto-detects whether the files are track-mapped (have track_id) "
            "or centroid-only (no track_id) and refuses to mix them. When --tracks "
            "is provided alongside track-mapped CSVs, missing lineage columns "
            "(id, parent_track_id, parent_id) are filled in from the matching "
            "tracks_<fov>.csv before concatenation. Per-FOV CSVs are never "
            "modified on disk."
        ),
    )
    parser.add_argument(
        "-d",
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing the per-FOV annotation CSVs.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output combined CSV path.",
    )
    parser.add_argument(
        "-t",
        "--tracks",
        type=Path,
        help=(
            "Optional tracks OME-Zarr. When provided alongside track-mapped "
            "input CSVs, missing id / parent_track_id / parent_id columns are "
            "filled in from each FOV's tracks_<fov>.csv."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def combine_annotations_main(argv: list[str] | None = None) -> int:
    """Entry point: combine per-FOV annotation CSVs into a single CSV.

    Workflow:
      1. Discover per-FOV CSVs in ``--input-dir`` (excluding ``*combined*.csv``
         and anything under ``backup/``).
      2. Auto-detect schema by checking whether ``track_id`` is present in
         every file. Mixed schemas are rejected.
      3. If ``--tracks`` is provided and the CSVs are track-mapped, fill in
         missing ``id`` / ``parent_track_id`` / ``parent_id`` columns by
         merging each per-FOV CSV with its matching ``tracks_<fov>.csv``
         (keyed on ``track_id, t, y, x``).
      4. Add a ``fov_name`` column if missing, parsed from the filename.
      5. Concatenate everything and write to ``--output``.
    """
    parser = _make_combine_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger("napari_iohub.combine_annotations")

    if not args.input_dir.is_dir():
        parser.error(f"--input-dir is not a directory: {args.input_dir}")
    if args.tracks is not None and not args.tracks.exists():
        parser.error(f"--tracks does not exist: {args.tracks}")

    import pandas as pd

    # Discover candidate files
    candidates = []
    for path in sorted(args.input_dir.glob("*.csv")):
        if "combined" in path.name:
            continue
        if path.parent.name == "backup":
            continue
        candidates.append(path)

    if not candidates:
        parser.error(
            f"No per-FOV CSVs found under {args.input_dir} "
            "(skipped *combined*.csv and backup/)."
        )
    logger.info("Found %d candidate per-FOV CSVs", len(candidates))

    # Load and classify each file
    loaded: list[tuple[Path, str | None, "pd.DataFrame"]] = []
    skipped: list[tuple[Path, str]] = []
    for path in candidates:
        fov_name = _parse_fov_from_filename(path.name)
        if fov_name is None:
            skipped.append((path, "unparseable FOV from filename"))
            continue
        try:
            df = pd.read_csv(path)
        except Exception as e:  # noqa: BLE001 - we want to keep going
            skipped.append((path, f"read error: {e}"))
            continue
        loaded.append((path, fov_name, df))

    if not loaded:
        parser.error(
            "No usable per-FOV CSVs after discovery. Skipped:\n  "
            + "\n  ".join(f"{p.name}: {reason}" for p, reason in skipped)
        )

    # Auto-detect schema: every file must consistently have track_id or not
    has_track_id_flags = [
        "track_id" in df.columns and df["track_id"].notna().any() for _, _, df in loaded
    ]
    if all(has_track_id_flags):
        schema = "track_mapped"
    elif not any(has_track_id_flags):
        schema = "centroid_only"
    else:
        mixed = [
            (p.name, "track-mapped" if h else "centroid-only")
            for (p, _, _), h in zip(loaded, has_track_id_flags)
        ]
        parser.error(
            "Mixed schemas in --input-dir (some files have track_id, others "
            "don't). Resolve by mapping the centroid-only files with "
            "napari-iohub-map-centroids first, or move them out of the "
            "directory. Mixed set:\n  "
            + "\n  ".join(f"{n}: {kind}" for n, kind in mixed)
        )
    logger.info("Schema detected: %s", schema)

    # Optional lineage-column fill-in for track-mapped CSVs
    if schema == "track_mapped" and args.tracks is not None:
        lineage_cols = ["id", "parent_track_id", "parent_id"]
        for i, (path, fov_name, df) in enumerate(loaded):
            missing = [c for c in lineage_cols if c not in df.columns]
            if not missing:
                continue
            row, col, fov = fov_name.split("/")
            tracks_csv = args.tracks / row / col / fov / f"tracks_{row}_{col}_{fov}.csv"
            if not tracks_csv.exists():
                logger.warning(
                    "No tracks_<fov>.csv at %s; leaving %d missing lineage "
                    "columns null for %s",
                    tracks_csv,
                    len(missing),
                    path.name,
                )
                continue
            tracks_df = pd.read_csv(tracks_csv)
            merge_keys = [k for k in ["track_id", "t", "y", "x"] if k in df.columns and k in tracks_df.columns]
            keep = merge_keys + [c for c in lineage_cols if c in tracks_df.columns]
            tracks_subset = tracks_df[keep].drop_duplicates(subset=merge_keys)
            filled = df.merge(tracks_subset, on=merge_keys, how="left")
            loaded[i] = (path, fov_name, filled)
            logger.info(
                "Filled %s in %s from %s",
                ",".join(missing),
                path.name,
                tracks_csv.name,
            )

    # Ensure fov_name column is present
    for i, (path, fov_name, df) in enumerate(loaded):
        if "fov_name" not in df.columns:
            df = df.copy()
            df["fov_name"] = fov_name
            loaded[i] = (path, fov_name, df)

    combined = pd.concat([df for _, _, df in loaded], ignore_index=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)

    logger.info(
        "Wrote %d rows (%d FOVs, schema=%s) to %s",
        len(combined),
        len({fov for _, fov, _ in loaded}),
        schema,
        args.output,
    )
    if skipped:
        logger.info("Skipped %d files:", len(skipped))
        for p, reason in skipped:
            logger.info("  %s: %s", p.name, reason)
    if schema == "track_mapped":
        n_missing_id = combined["id"].isna().sum() if "id" in combined.columns else 0
        if n_missing_id:
            logger.warning(
                "%d rows in the combined CSV have a NaN id column (missing "
                "lineage info). Pass --tracks to fill these from tracks_<fov>.csv.",
                n_missing_id,
            )

    return 0


if __name__ == "__main__":
    sys.exit(annotator_main())
