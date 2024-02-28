import click
import os
import shutil
import pandas as pd
from click import style
import preprocess.cif_parser as cif_parser
import util.folder as folder
import matplotlib.pyplot as plt
import textwrap
import preprocess.cif_parser_handler as cif_parser_handler
import preprocess.supercell_handler as supercell_handler

def print_intro_prompt():
    """Filters and moves CIF files based on the shortest atomic distance."""
    introductory_paragraph = textwrap.dedent("""\
    ===
    Welcome to the CIF Atomic Distance Filter Tool!

    This tool reads CIF files and calculates the shortest atomic distance for each file. 
    Once these distances are determined, it displays a histogram, allowing you to visually 
    understand the distribution of the shortest atomic distances for all processed CIF files.

    You will then be prompted to enter a distance threshold after you close the histogram.
    Based on this threshold, CIF files having the shortest atomic distance less than the given
    threshold will be moved to a new sub-directory.

    At the end, a comprehensive log will be saved in CSV format, capturing:
    1. File names of CIFs.
    2. Compound formula for each CIF.
    3. Shortest atomic distance computed.
    4. Whether the file was moved (filtered) based on the threshold.
    5. Number of atoms in each file's supercell.

    Additionally, you can optionally choose to skip files based on the number of unique atoms 
    present in the supercell.

    Let's get started!
    ===
    """)
    
    print(introductory_paragraph)



def move_files_save_csv(files_lst, skipped_indices, shortest_dist_list, loop_tags,
                        DISTANCE_THRESHOLD, filtered_folder, folder_info, result_df):
    # Now, use the computed shortest distances to move the files based on the provided threshold
    processed_files_count = 0
    for idx, file_path in enumerate(files_lst, start=1):
        # Skip indices above MAX_ATOMS_COUNT
        if idx in skipped_indices:
            continue

        shortest_dist = shortest_dist_list[processed_files_count]  # Retrieve the precomputed shortest distance based on processed files count
        processed_files_count += 1

        # Re-calculate the formula_string here before using it for the DataFrame
        result = cif_parser_handler.get_CIF_info(file_path, loop_tags)
        CIF_block, _, _, _, all_points, _, _ = result
        _, _, formula_string = cif_parser.extract_formula_and_atoms(CIF_block)

        # Initialize the "Filtered" flag
        filtered_flag = "No"

        # If the file meets the threshold criterion, update the flag
        if shortest_dist < DISTANCE_THRESHOLD:
            if not os.path.exists(filtered_folder):
                os.mkdir(filtered_folder)
            
            # Full path to where the file will be moved
            new_file_path = os.path.join(filtered_folder, os.path.basename(file_path))
            
            # If the file already exists in the destination, delete it
            if os.path.exists(new_file_path):
                os.remove(new_file_path)

            filtered_flag = "Yes"
            shutil.move(file_path, new_file_path)

        new_row = pd.DataFrame({
            "Entry": [CIF_block.name],
            "Compound": [formula_string],
            "Shortest distance": [shortest_dist],
            "Filtered": [filtered_flag],
            "Number of atoms": [len(all_points)]
        })

        result_df = pd.concat([result_df, new_row], ignore_index=True)

    folder.save_to_csv_directory(folder_info, result_df, "filter_dist_min_log")



def plot_histogram(distances, save_path, num_of_files):
    plt.figure(figsize=(10,6))
    plt.hist(distances, bins=50, color='blue', edgecolor='black')
    plt.title(f"Histogram of Shortest Distances of {num_of_files} files")
    plt.xlabel('Distance (Å)')
    plt.ylabel('Number of CIF Files')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig(save_path, dpi=300)


def move_files_based_on_min_dist(script_directory, isInteractiveMode=True):
    print_intro_prompt()
    shortest_dist_list = []
    skipped_indices = set()
    result_df = pd.DataFrame()
    MAX_ATOMS_COUNT = float('inf')
    DISTANCE_THRESHOLD = 1.0 # Set a default value of 1.0 Å
    folder_info, filtered_folder, files_lst, num_of_files, loop_tags = cif_parser_handler.get_folder_and_files_info(script_directory, isInteractiveMode)

    if isInteractiveMode:
        click.echo("\nQ. Do you want to skip any CIF files based on the number of unique atoms in the supercell? Any file above the number will be skipped.")
        skip_based_on_atoms = click.confirm('(Default: N)', default=False)
        
        if skip_based_on_atoms:
            click.echo("\nEnter the threshold for the maximum number of atoms in the supercell.")
            MAX_ATOMS_COUNT = click.prompt('Files with atoms exceeding this count will be skipped', type=int)
    
    # Process CIF files
    shortest_dist_list, skipped_indices = supercell_handler.get_shortest_dist_list_and_skipped_indices(files_lst, loop_tags, MAX_ATOMS_COUNT)
    
    # Create histogram directory and save
    plot_directory = os.path.join(folder_info, "plot")
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)

    histogram_save_path = os.path.join(folder_info, "plot", "histogram-min-dist.png")
    plot_histogram(shortest_dist_list, histogram_save_path, num_of_files)
    print("Histogram saved. Please check the 'plot' folder of the selected cif directory.")

    if isInteractiveMode:
        prompt_dist_threshold = '\nNow, please enter the threashold distance (unit in Å)'
        DISTANCE_THRESHOLD = click.prompt(prompt_dist_threshold, type=float)

    # Move CIF files with min distance below the threshold
    move_files_save_csv(files_lst, skipped_indices, shortest_dist_list, loop_tags,
                        DISTANCE_THRESHOLD, filtered_folder, folder_info, result_df)

 