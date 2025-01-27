import time
import click
from os.path import join
from core.utils import prompt, intro, object
from cifkit import CifEnsemble
from core.utils.histogram import plot_distance_histogram
import traceback
import multiprocessing as mp
from cifkit import Cif
import os


def move_files_based_on_min_dist(cif_dir):
    intro.prompt_min_dist_intro()
    filter_files_by_min_dist(cif_dir)
    

def min_dist_worker(idx, cif_path, file_count, file_names_and_min_dists):
    start_time = time.perf_counter()
    
    cif = Cif(cif_path, is_formatted=True)
    
    file_name = cif.file_name
    atom_count = cif.supercell_atom_count

    # prompt.print_progress_current(idx, file_name, atom_count, file_count)
    print(f"Processing {file_name} with {atom_count} ({idx}/{file_count})")
    # Compute min distance
    try:
        min_dist = cif.shortest_distance
        file_names_and_min_dists.append([file_name, min_dist])
    except:
        print(f"Error while processing {file_name}")
        print(traceback.format_exc())
    
    elasped_time = time.perf_counter() - start_time
    # prompt.print_finished_progress(file_name, atom_count, elasped_time)
    print(f"Processed {file_name} with {atom_count} atoms in {elasped_time:.2f}s")
        
def mp_aux(*args):
    for arg in args:
        min_dist_worker(**arg)


def filter_files_by_min_dist(cif_dir_path, is_interactive_mode=True):
    """
    Filter files for files below the minimum distance threshold.
    """

    # Initialize the ensemble
    ensemble = object.init_cif_ensemble(cif_dir_path)
    
    # parallel
    num_cpu = 1
    if is_interactive_mode:
        # mp
        click.echo("\nSelect the number of core(s) for parallel/serial processing.")
        click.echo("[1] Serial process (uses one CPU core).")
        click.echo(f"[2] Parallel process with maximum ({mp.cpu_count()-2}) CPU cores (for 1000s of cifs).")
        click.echo(f"[3] Enter the number of CPU cores (<={mp.cpu_count()-2}) manually for parallel processing.")
        filter_choice = click.prompt("Enter your choice (1, 2, or 3)", type=int)
    
        if filter_choice == 2:
            num_cpu = mp.cpu_count()-2
        elif filter_choice == 3:
            num_cpu = click.prompt(f"Enter the number of CPU cores ({mp.cpu_count()-2})", type=int)
            num_cpu = min(num_cpu, mp.cpu_count()-2)

    mp_manager = mp.Manager() 
    file_names_and_min_dists = mp_manager.list()
    
    tasks = []     
    for idx, cif in enumerate(ensemble.cifs, start=1):
        tasks.append({'idx': idx,
            'cif_path': f"{cif_dir_path}{os.sep}{cif.file_name}", 
            'file_count': ensemble.file_count, 'file_names_and_min_dists': file_names_and_min_dists})

    with mp.Pool(num_cpu) as pool:
        pool.map(mp_aux, tasks)
        
    pool.close()
    pool.join()
    
    file_names_and_min_dists = list(file_names_and_min_dists)
    min_dists = [m[1] for m in file_names_and_min_dists]
    
    # Find files encountered error
    processed_file_names = [f[0] for f in file_names_and_min_dists]
    files_encountered_errors = [f"{cif_dir_path}{os.sep}{f.file_name}" for f in ensemble.cifs if f.file_name not in processed_file_names]
    if files_encountered_errors:
        ensemble.move_cif_files(files_encountered_errors, join(ensemble.dir_path, f"cifs_encountered_error"))

    # Folder to save the histogram
    plot_distance_histogram(cif_dir_path, min_dists, ensemble.file_count)

    if is_interactive_mode:    
        click.echo("Note: .cif with minimum distance out of the bounds will be relocated.")
        prompt_dist_threshold_min = "\nEnter the threashold low minimum distance (unit in Å)"
        dist_threshold_min = click.prompt(prompt_dist_threshold_min, type=float)
        
        prompt_dist_threshold_max = "\nEnter the threashold high minimum distance (unit in Å)"
        dist_threshold_max = click.prompt(prompt_dist_threshold_max, type=float)
    else:
        dist_threshold_min = 2.6  # For testing set to 2.6
        dist_threshold_max = 12.0
    
    # Filter files based on the minimum distance
    filtered_file_paths = [f"{cif_dir_path}{os.sep}{p[0]}" \
        for p in file_names_and_min_dists if not dist_threshold_min < p[1] < dist_threshold_max]
    destination_path = join(ensemble.dir_path, f"dist_between_{dist_threshold_min}_{dist_threshold_max}")

    # Move filtered files to a new directory
    if filtered_file_paths:
        ensemble.move_cif_files(filtered_file_paths, destination_path)

    prompt.print_moved_files_summary(
        filtered_file_paths, ensemble.file_count, destination_path
    )
    prompt.print_done_with_option("min_dist_below_{dist_threshold}")
