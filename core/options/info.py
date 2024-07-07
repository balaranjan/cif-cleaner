import click
import pandas as pd
import time
from core.utils import folder, prompt, intro
from cifkit import CifEnsemble


def get_cif_folder_info(cif_dir_path):
    intro.prompt_info_intro()

    # Keep track of data for .csv
    results = []

    # Keep track overall time
    overall_start_time = time.perf_counter()

    ensemble = CifEnsemble(cif_dir_path)

    # Ask user to calculate distance
    click.echo("\nQ. Do you want to compute minimum distance per file (slow)?")
    compute_dist = click.confirm("(Default: N)", default=False)

    # Process each cif object
    for i, cif in enumerate(ensemble.cifs, start=1):
        file_start_time = time.perf_counter()

        prompt.print_progress_current(
            i, cif.file_name, cif.supercell_atom_count, ensemble.file_count
        )
        min_distance = None
        if compute_dist:
            min_distance = round(cif.shortest_distance, 3)
        elapsed_time = time.perf_counter() - file_start_time

        data = {
            "Filename": cif.file_name_without_ext,
            "Formula": cif.formula,
            "Structure": cif.structure,
            "Tag": cif.tag,
            "Supercell atom count": cif.supercell_atom_count,
            "Site mixing type": cif.site_mixing_type,
            "Composition type": cif.composition_type,
            "Min distance (Å)": min_distance,
            "Processing time (s)": round(elapsed_time, 3),
        }
        results.append(data)

        prompt.print_finished_progress(
            cif.file_name, cif.supercell_atom_count, elapsed_time
        )

    # Save csv
    folder.save_to_csv_directory(cif_dir_path, pd.DataFrame(results), "info")

    # Total processing time
    total_elapsed_time = time.perf_counter() - overall_start_time
    print(f"Total processing time for all files: {total_elapsed_time:.2f} s")

    # Done message
    prompt.print_done_with_option("Info")
