import os
import shutil

def create_second_class():
    """Create a second class by copying some images from the existing Cat class"""
    # Create Dog class directory
    dog_dir = os.path.join("data", "raw", "Dog")
    os.makedirs(dog_dir, exist_ok=True)
    
    # Copy some images from Cat class to Dog class (as example data)
    cat_dir = os.path.join("data", "raw", "Cat")
    if os.path.exists(cat_dir):
        cat_images = [f for f in os.listdir(cat_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        # Copy first 100 images as example
        for i, img in enumerate(cat_images[:100]):
            src = os.path.join(cat_dir, img)
            dst = os.path.join(dog_dir, f"dog_example_{i}.jpg")
            shutil.copy2(src, dst)
        print(f"Created Dog class with {min(100, len(cat_images))} example images")
    else:
        print("Cat directory not found")

if __name__ == "__main__":
    create_second_class()