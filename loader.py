import os
import numpy as np
import tifffile
from pathlib import Path

def extract_middle_slice_from_directories(base_dir, output_dir="middle_slices"):
    """
    Extract the middle slice from 100-slice TIFF files in the first 5 directories.
    
    Args:
        base_dir: Base directory containing subdirectories with TIFF files
        output_dir: Directory to save the extracted middle slices
    """
    base_path = Path(base_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Get all subdirectories
    subdirs = [d for d in base_path.iterdir() if d.is_dir()]
    subdirs.sort()  # Sort for consistent ordering
    
    # Process first 5 directories
    processed = 0
    for subdir in subdirs[:5]:
        print(f"Processing directory: {subdir.name}")
        
        tiff_files = list(subdir.glob("*100.tiff"))
        
        for tiff_file in tiff_files:
            try:
                # Open TIFF file to check number of slices
                with tifffile.TiffFile(tiff_file) as tif:
                    num_slices = len(tif.pages)
                    
                    if num_slices == 100:
                        print(f"  Found 100-slice TIFF: {tiff_file.name}")
                        
                        # Read the middle slice (index 49 for 0-indexed, slice 50)
                        middle_slice_idx = 49
                        middle_slice = tifffile.imread(tiff_file, key=middle_slice_idx)
                        
                        # Save the middle slice
                        output_filename = f"{subdir.name}_{tiff_file.stem}_middle_slice.tiff"
                        output_file = output_path / output_filename
                        tifffile.imwrite(output_file, middle_slice)
                        
                        print(f"  Saved middle slice to: {output_file}")
                        processed += 1
                        break  # Process only one TIFF file per directory
            except Exception as e:
                print(f"  Error processing {tiff_file.name}: {e}")
                continue
    
    print(f"\nProcessed {processed} TIFF files")

if __name__ == "__main__":
    base_directory = "G:/proj_MM"
    extract_middle_slice_from_directories(base_directory)
