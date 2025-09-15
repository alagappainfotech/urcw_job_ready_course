import os
import pandas as pd


def get_files_and_sizes(folder_path):
    """
    Scans a folder and all its subfolders to get a list of files and their sizes.

    Args:
        folder_path (str): The absolute or relative path to the folder.

    Returns:
        list: A list of dictionaries, where each dictionary contains the
              'File Path' and 'Size (Bytes)'.
    """
    file_data = []
    # os.walk() recursively goes through the directory tree
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            # Construct the full path to the file
            file_path = os.path.join(root, filename)
            try:
                # Get the size of the file in bytes
                file_size = os.path.getsize(file_path)
                file_data.append({'File Path': file_path, 'Size (Bytes)': file_size})
            except FileNotFoundError:
                print(f"File not found, skipping: {file_path}")
            except Exception as e:
                print(f"An error occurred with file {file_path}: {e}")

    return file_data


def convert_bytes(size_bytes):
    """Converts bytes to a more readable format (KB, MB, GB)."""
    if size_bytes == 0:
        return "0 B"
    power = 1024
    n = 0
    power_labels = {0: '', 1: 'KB', 2: 'MB', 3: 'GB', 4: 'TB'}
    while size_bytes >= power and n < len(power_labels) - 1:
        size_bytes /= power
        n += 1
    return f"{size_bytes:.2f} {power_labels[n]}"


# --- Main part of the script ---
if __name__ == "__main__":
    # 1. SPECIFY THE FOLDER YOU WANT TO SCAN
    # IMPORTANT: Replace this with the actual path to your folder.
    # Use raw string (r"...") for Windows paths to avoid issues with backslashes.
    # Example for Windows: target_folder = r"C:\Users\YourUser\Documents"
    # Example for macOS/Linux: target_folder = "/home/youruser/documents"
    target_folder = r"/Users/acharyathiyagarajan/Applications/urcw_job_ready_course/python_outputs"

    # 2. SPECIFY THE OUTPUT EXCEL FILENAME
    output_excel_file = "file_sizes_report_v0.xlsx"

    # Check if the folder exists
    if not os.path.isdir(target_folder):
        print(f"Error: The folder '{target_folder}' does not exist.")
    else:
        # Get the file data
        print(f"Scanning folder: '{target_folder}'...")
        all_files = get_files_and_sizes(target_folder)

        if not all_files:
            print("No files found in the specified folder.")
        else:
            # Create a pandas DataFrame from the list of files
            df = pd.DataFrame(all_files)

            # (Optional) Add a column with human-readable sizes
            df['Size (Readable)'] = df['Size (Bytes)'].apply(convert_bytes)

            # Save the DataFrame to an Excel file
            # The 'index=False' argument prevents pandas from writing row indices to the file
            df.to_excel(output_excel_file, index=False, engine='openpyxl')

            print(f"\n✅ Success! Report saved as '{output_excel_file}'")
            print(f"Total files found: {len(df)}")