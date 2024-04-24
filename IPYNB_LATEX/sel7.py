dataset_dir = "C:/Users/jerem/Downloads/TENSORFLOW/DatSetK"

num_classes_to_show = 5
num_images_per_class = 1
class_directories = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
selected_classes = random.sample(class_directories, min(num_classes_to_show, len(class_directories)))
fig, axes = plt.subplots(num_classes_to_show, num_images_per_class, figsize=(15, 3*num_classes_to_show))

for i, selected_class in enumerate(selected_classes):
    class_dir = os.path.join(dataset_dir, selected_class)
    image_files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith((".jpg", ".png"))]
    selected_image_files = random.sample(image_files, min(num_images_per_class, len(image_files)))

    for j, image_path in enumerate(selected_image_files):
        image = plt.imread(image_path)
        axes[i].imshow(image)
        axes[i].axis("off")

plt.show()