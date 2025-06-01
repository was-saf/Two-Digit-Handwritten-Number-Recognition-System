import os
import shutil

def sort_images_by_label(source_dir):
    for filename in os.listdir(source_dir):
        if filename.endswith(".png"):
            try:
                # Extract label prefix (e.g., '72' from '72_1234.png')
                label = filename.split("_")[0]
                label_folder = os.path.join(source_dir, label)
                os.makedirs(label_folder, exist_ok=True)

                # Move file to label-specific folder
                src = os.path.join(source_dir, filename)
                dst = os.path.join(label_folder, filename)
                shutil.move(src, dst)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Call it on your folders
sort_images_by_label("images/train")
sort_images_by_label("images/test")

print("✅ Images sorted into class folders.")
