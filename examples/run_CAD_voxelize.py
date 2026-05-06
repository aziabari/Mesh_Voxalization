# examples/run_blade.py
from cad_voxelize import main

main(
    "FILENAME",
    resol=0.36,
    padding=25,
    out_folder="OUTFOLDER",
    out_dtype="tiff", # "npy"
)