import os
import click
import time
from core.utils import intro, prompt, object
from cifkit import CifEnsemble
from cifkit.utils import folder
import traceback
import multiprocessing as mp
from cifkit import Cif
import os


def move_files_based_on_coordination_number(
    cif_dir_path: str,
    is_interactive_mode=True,
    numbers: list[int] = None,
    option: int = None,
) -> None:
    intro.prompt_coordination_number_intro()
    ensemble = object.init_cif_ensemble(cif_dir_path)

    if is_interactive_mode:
        # Prompt for elements
        CN_input = click.prompt(
            "Q1. Enter the coordination number(s) to filter by,"
            " separated by a space (Ex: '12 16')",
            type=str,
        ).strip()

        # Split by space
        numbers = [number for number in CN_input.split() if number]

        # Convert to int
        numbers = [int(num) for num in numbers]

        # Ask user for the type of filter
        click.echo("\nQ2. Now choose your option:")
        click.echo("[1] Move files exactly matching the coordination numbers")
        click.echo("[2] Move files containing at least one of the coordination numbers")
        filter_choice = click.prompt("Enter your choice (1 or 2)", type=int)
    else:
        filter_choice = option

    filter_and_move_files(ensemble, filter_choice, cif_dir_path, numbers)


def CN_Num_worker(idx, cif_path, file_count, file_names_and_CNs):
    
    start_time = time.perf_counter()
    cif = Cif(cif_path, is_formatted=True)
    file_name = cif.file_name
    atom_count = cif.supercell_atom_count
    
    print(f"Processing {file_name} with {atom_count} ({idx}/{file_count})")
    try:
        # Compute CN values for each .cif
        CN_values = cif.CN_unique_values_by_min_dist_method
        file_names_and_CNs.append([file_name, CN_values])
    except:
        print(f"Error while processing {file_name}")
        print(traceback.format_exc())
    
    elasped_time = time.perf_counter() - start_time
    print(f"Processed {file_name} with {atom_count} atoms in {elasped_time:.2f}s")


def mp_aux(*args):
    for arg in args:
        CN_Num_worker(**arg)
        
        
def filter_and_move_files(
    ensemble: CifEnsemble,
    filter_choice: int,
    cif_dir_path: str,
    numbers: list[int],
) -> None:
    # Folder info

    numbers_str = "_".join(str(number) for number in numbers)
    overall_start_time = time.perf_counter()
    folder_name = os.path.basename(cif_dir_path)
    filtered_file_paths = set()
    
    # parallel
    mp_manager = mp.Manager() 
    file_names_and_CNs = mp_manager.list()
    tasks = []
    
    for i, cif in enumerate(ensemble.cifs, start=1):
        file_name = cif.file_name
        file_count = ensemble.file_count
        
        tasks.append({'idx': i,
            'cif_path': f"{cif_dir_path}{os.sep}{cif.file_name}", 
            'file_count': ensemble.file_count, 'file_names_and_CNs': file_names_and_CNs})
    
    print(f"Num tasks: {len(tasks)}")
    with mp.Pool(mp.cpu_count() - 2) as pool:
        pool.map(mp_aux, tasks)
        
    pool.close()
    pool.join()

    file_names_and_CNs = list(file_names_and_CNs)
    
    for file_name, CN_values in file_names_and_CNs:

        if filter_choice == 1:
            destination_path = os.path.join(
                cif_dir_path, f"{folder_name}_CN_exact_{numbers_str}"
            )
            # Check if the CN values are exactly the same
            if set(numbers) == CN_values:
                filtered_file_paths.add(f"{cif_dir_path}{os.sep}{file_name}")

        elif filter_choice == 2:
            destination_path = os.path.join(
                cif_dir_path, f"{folder_name}_CN_contain_{numbers_str}"
            )
            # Check if at least one of the CN values is present
            if any(num in CN_values for num in numbers):
                filtered_file_paths.add(f"{cif_dir_path}{os.sep}{file_name}")

    move_files_and_prompt(
        filtered_file_paths, destination_path, file_count, overall_start_time
    )


def move_files_and_prompt(
    filtered_file_paths: set[str],
    destination_path: str,
    file_count: int,
    overall_start_time: float,
) -> None:
    if filtered_file_paths:
        # Create folder and move files
        folder.move_files(destination_path, filtered_file_paths)

    overall_elapsed_time = time.perf_counter() - overall_start_time
    prompt.print_total_time(overall_elapsed_time, file_count)
    prompt.print_moved_files_summary(filtered_file_paths, file_count, destination_path)
    prompt.print_done_with_option("filter by coordination numbers")
