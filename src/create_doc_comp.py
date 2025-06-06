import os
from collections import defaultdict
from PIL import Image
import matplotlib.pyplot as plt

#path to folder
folder = "/home/danielbugelnig/AAU/6.Semester/AI4EO/project/data/indices"
output_folder = os.path.join(folder, "paired_plots")
os.makedirs(output_folder, exist_ok=True)

def extract_components(filename):
    name = filename.split(".")[0]
    parts = name.split("_")
    if len(parts) >= 3:
        platform = parts[0]
        date = parts[1]
        content = "_".join(parts[2:])  # all after index 2
        return platform, date, content
    return None, None, None

groups = defaultdict(list)
for file in os.listdir(folder):
    if file.lower().endswith(".png"):
        platform, date, content = extract_components(file)
        if platform and content:
            key = f"{platform}_{content}"
            groups[key].append((date, file))

# store pairs
for key, entries in groups.items():
    if len(entries) >= 2:
        entries.sort()  # sort after date
        (date1, file1), (date2, file2) = entries[:2]
        paths = [os.path.join(folder, f) for f in (file1, file2)]
        imgs = [Image.open(p) for p in paths]

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        for ax, img, path in zip(axes, imgs, paths):
            ax.imshow(img)
            ax.set_title(os.path.basename(path))
            ax.axis("off")

        plt.tight_layout()
        save_name = f"paired_{key}.png"
        plt.savefig(os.path.join(output_folder, save_name), bbox_inches='tight')
        plt.close()
