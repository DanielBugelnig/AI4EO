import os
from PIL import Image, ImageDraw, ImageFont
import math

def create_image_collage_with_titles(folder_path, output_path, thumb_size=(200, 200), padding=10, font_path=None, font_size=20):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    if not image_files:
        print("No images found in the folder.")
        return

    # Load font
    try:
        font = ImageFont.truetype(font_path if font_path else "arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # Load images and resize
    images = []
    titles = []
    for fname in image_files:
        img_path = os.path.join(folder_path, fname)
        img = Image.open(img_path).convert("RGB")
        img.thumbnail(thumb_size)
        images.append(img)
        titles.append(os.path.splitext(fname)[0])

    # Determine grid size
    cols = int(math.sqrt(len(images)))
    rows = math.ceil(len(images) / cols)
    thumb_w, thumb_h = thumb_size
    title_height = font_size + 5

    total_w = cols * thumb_w + (cols + 1) * padding
    total_h = rows * (thumb_h + title_height) + (rows + 1) * padding

    collage = Image.new("RGB", (total_w, total_h), color="white")
    draw = ImageDraw.Draw(collage)

    for i, (img, title) in enumerate(zip(images, titles)):
        col = i % cols
        row = i // cols
        x = padding + col * (thumb_w + padding)
        y = padding + row * (thumb_h + title_height + padding)

        draw.text((x, y), title, fill="black", font=font)
        collage.paste(img, (x, y + title_height))

    collage.save(output_path)
    print(f"Saved collage to {output_path}")

input_folder = "/home/danielbugelnig/AAU/6.Semester/AI4EO/project/data/indices/indices_s1_0823"
output_file = "/home/danielbugelnig/AAU/6.Semester/AI4EO/project/data/indices/indices_s1_0823/total_image.jpg"
create_image_collage_with_titles(input_folder, output_file, padding=2, font_size=20)