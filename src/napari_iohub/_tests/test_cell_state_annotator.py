"""Tests for cell state annotator widget."""

import pandas as pd
import pytest

from napari_iohub._cell_state_annotator import (
    ANNOTATION_LAYERS,
    ULTRACK_INDEX_COLUMNS,
    CellStateAnnotatorWidget,
)


def test_cell_state_annotator_widget_creation(make_napari_viewer):
    """Test that the widget can be instantiated."""
    viewer = make_napari_viewer()
    widget = CellStateAnnotatorWidget(viewer)
    assert widget is not None
    assert widget.viewer is viewer


def test_annotation_layers_constant():
    """Test that annotation layers are properly defined."""
    assert len(ANNOTATION_LAYERS) == 4
    layer_names = [layer[0] for layer in ANNOTATION_LAYERS]
    assert "_mitosis_events" in layer_names
    assert "_infected_events" in layer_names
    assert "_remodel_events" in layer_names
    assert "_death_events" in layer_names


def test_index_columns_constant():
    """Test that index columns match ultrack expectations."""
    expected = ["fov_name", "track_id", "t", "id", "parent_track_id", "parent_id", "z", "y", "x"]
    assert ULTRACK_INDEX_COLUMNS == expected


def test_state_expansion_logic():
    """Test the binary state expansion logic for annotations."""
    # Simulate tracks DataFrame
    tracks_df = pd.DataFrame(
        {
            "track_id": [1, 1, 1, 1, 2, 2, 2],
            "t": [0, 1, 2, 3, 0, 1, 2],
        }
    )

    # Simulate marked events
    marked_events = {
        "cell_division_state": [{"track_id": 1, "t": 1}],  # mitosis at t=1
        "infection_state": [{"track_id": 1, "t": 2}],  # infected from t=2
        "organelle_state": [],  # no remodel
        "cell_death_state": [{"track_id": 2, "t": 1}],  # death at t=1
    }

    # Expand annotations (logic extracted from widget)
    all_annotations = []
    all_track_timepoints = tracks_df[["track_id", "t"]].drop_duplicates()

    for track_id in all_track_timepoints["track_id"].unique():
        track_timepoints = all_track_timepoints[
            all_track_timepoints["track_id"] == track_id
        ]["t"].sort_values()

        division_events = [
            e
            for e in marked_events["cell_division_state"]
            if e["track_id"] == track_id
        ]
        mitosis_timepoints = [e["t"] for e in division_events]

        infection_events = [
            e for e in marked_events["infection_state"] if e["track_id"] == track_id
        ]
        first_infected_t = (
            min([e["t"] for e in infection_events]) if infection_events else None
        )

        organelle_events = [
            e for e in marked_events["organelle_state"] if e["track_id"] == track_id
        ]
        first_remodel_t = (
            min([e["t"] for e in organelle_events]) if organelle_events else None
        )

        death_events = [
            e for e in marked_events["cell_death_state"] if e["track_id"] == track_id
        ]
        first_death_t = min([e["t"] for e in death_events]) if death_events else None

        for t in track_timepoints:
            is_dead = first_death_t is not None and t >= first_death_t

            if is_dead:
                cell_division_state = None
                infection_state = None
                organelle_state = None
                cell_death_state = "dead"
            else:
                cell_division_state = (
                    "mitosis" if t in mitosis_timepoints else "interphase"
                )
                infection_state = (
                    "infected"
                    if (first_infected_t is not None and t >= first_infected_t)
                    else "uninfected"
                    if first_infected_t is not None
                    else None
                )
                organelle_state = (
                    "remodel"
                    if (first_remodel_t is not None and t >= first_remodel_t)
                    else "noremodel"
                )
                cell_death_state = "alive" if first_death_t is not None else None

            all_annotations.append(
                {
                    "track_id": track_id,
                    "t": t,
                    "cell_division_state": cell_division_state,
                    "infection_state": infection_state,
                    "organelle_state": organelle_state,
                    "cell_death_state": cell_death_state,
                }
            )

    df = pd.DataFrame(all_annotations)

    # Verify track 1 states
    track1 = df[df["track_id"] == 1]
    assert track1[track1["t"] == 0]["cell_division_state"].iloc[0] == "interphase"
    assert track1[track1["t"] == 1]["cell_division_state"].iloc[0] == "mitosis"
    assert track1[track1["t"] == 2]["cell_division_state"].iloc[0] == "interphase"

    # Infection propagates forward
    assert track1[track1["t"] == 0]["infection_state"].iloc[0] == "uninfected"
    assert track1[track1["t"] == 1]["infection_state"].iloc[0] == "uninfected"
    assert track1[track1["t"] == 2]["infection_state"].iloc[0] == "infected"
    assert track1[track1["t"] == 3]["infection_state"].iloc[0] == "infected"

    # Verify track 2 death propagation
    track2 = df[df["track_id"] == 2]
    assert track2[track2["t"] == 0]["cell_death_state"].iloc[0] == "alive"
    assert track2[track2["t"] == 1]["cell_death_state"].iloc[0] == "dead"
    assert track2[track2["t"] == 2]["cell_death_state"].iloc[0] == "dead"

    # Dead cells have None/NaN for other states
    assert pd.isna(track2[track2["t"] == 1]["cell_division_state"].iloc[0])
    assert pd.isna(track2[track2["t"] == 1]["infection_state"].iloc[0])


@pytest.mark.parametrize(
    "layer_name,expected_color",
    [
        ("_mitosis_events", "blue"),
        ("_infected_events", "orange"),
        ("_remodel_events", "purple"),
        ("_death_events", "red"),
    ],
)
def test_annotation_layer_colors(layer_name, expected_color):
    """Test that annotation layers have correct colors defined."""
    layer_config = next(
        (layer for layer in ANNOTATION_LAYERS if layer[0] == layer_name), None
    )
    assert layer_config is not None
    assert layer_config[1] == expected_color
