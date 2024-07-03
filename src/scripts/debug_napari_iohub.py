# This script can be modified to debug and test specific methods of the plugin

import napari
import time
from napari_iohub._widget import MainWidget
from pathlib import Path

zarr_store = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/2024_05_03_DENV_eFT226_Timecourse/0-convert/2024_05_03_DENV_eFT226_Timecourse_1.zarr"
)
metadata_file = Path("20240503_Plate Map Template.xlsx")


def main():
    viewer = napari.Viewer()
    napari_iohub = MainWidget(viewer)
    viewer.window.add_dock_widget(napari_iohub)
    # recorder.ui.qbutton_connect_to_mm.click()
    # recorder.calib_scheme = "5-State"

    # for repeat in range(REPEATS):
    #     for swing in SWINGS:
    #         print("Calibrating with swing = " + str(swing))
    #         recorder.swing = swing
    #         recorder.directory = SAVE_DIR
    #         recorder.run_calibration()
    #         time.sleep(100)


if __name__ == "__main__":
    main()
    input("Press Enter to close")
