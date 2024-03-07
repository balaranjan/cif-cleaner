import pytest
import os
import pytest
import preprocess.cif_parser as cif_parser
import preprocess.supercell as supercell
import filter.format as format
import util.folder as folder
import preprocess.cif_editor as cif_editor
import shutil
import tempfile


def get_is_identical_cif(file_path1, file_path2):
    with open(file_path1, 'r') as file1:
        content1 = file1.readlines()
    
    with open(file_path2, 'r') as file2:
        content2 = file2.readlines()
    
    # Check if both files have the same number of lines
    if len(content1) != len(content2):
        return False

    # Compare each line
    for i, (line1, line2) in enumerate(zip(content1, content2)):
        if line1 != line2:
            return False

    return True


def test_preprocess_cif_file_by_removing_author_loop():
    print("Hello worlod")

    # Create a temp folder
    cif_directory = "test/preprocess/cif_files/format_author"
    
    temp_dir = tempfile.mkdtemp()
    temp_cif_directory = os.path.join(
        temp_dir, os.path.basename(cif_directory)
    )
    shutil.copytree(cif_directory, temp_cif_directory)
    temp_cif_file_num = folder.get_cif_file_count_from_directory(
        temp_cif_directory
    )

    # Check the number of files
    assert temp_cif_file_num == 6
    
    # Select the file
    cif_file_path_list = folder.get_cif_file_path_list_from_directory(
        temp_cif_directory
    )    

    for file_path in cif_file_path_list:
        filename = os.path.basename(file_path)
        correct_file_path = (
           f"test/preprocess/cif_files/format_author/formatted/{filename}"
        )
        cif_editor.preprocess_cif_file_by_removing_author_loop(file_path)
        assert get_is_identical_cif(file_path, correct_file_path)

    # Clean up
    shutil.rmtree(temp_dir)


    
