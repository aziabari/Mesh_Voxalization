# examples/run_blade.py
from cad_voxelize import main

main(
    "/full/path/to/*.stl",
    resol=0.5,
    padding=25,
    out_folder="./tiffs/",
)