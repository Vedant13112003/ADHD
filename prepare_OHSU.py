import os, shutil, re

source_dir = "/Users/vedantshirapure/Desktop/University/Capestone Project /Diagnosing-ADHD-With-ConvLSTM/OHSU/"
target_dir = "/Users/vedantshirapure/Desktop/University/Capestone Project /Diagnosing-ADHD-With-ConvLSTM/model_data"

os.makedirs(target_dir, exist_ok=True)

for sub in os.listdir(source_dir):
    func_path = os.path.join(source_dir, sub, "ses-1", "func")
    if not os.path.exists(func_path):
        continue
    for file in os.listdir(func_path):
        if "bold.nii.gz" in file:
            subject_id = re.findall(r'\d+', sub)[0]
            new_name = f"OHSU_snwmrda{subject_id}_session_1_rest_1.nii.gz"
            shutil.copy(os.path.join(func_path, file), os.path.join(target_dir, new_name))
            print(f"Copied {file} to {new_name}")
