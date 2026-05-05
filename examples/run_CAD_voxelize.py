# examples/run_blade.py
from cad_voxelize import main

main(
    "/Users/3az/ORNL Dropbox/Amir Ziabari/TIP/TCF/2022-2023/code/Mesh_voxelisation/extra_files/01 ZCS AMMT Zeiss DOE 07.stl",
    resol=0.05,
    padding=25,
    out_folder="/Users/3az/ORNL Dropbox/Amir Ziabari/FY24-25_onward/Research/2026/AMMTO/Data/Mesh_voxelize_output/",
)