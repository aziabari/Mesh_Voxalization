# examples/run_blade.py
from cad_voxelize import main

main(
    "/Users/3az/Library/CloudStorage/OneDrive-OakRidgeNationalLaboratory/2025-2026/Leidos/CUI_GAMMA-H Part.STL",
    resol=0.36,
    padding=25,
    out_folder="/Users/3az/Library/CloudStorage/OneDrive-OakRidgeNationalLaboratory/2025-2026/Leidos/",
    out_dtype="tiff", # "npy"
)