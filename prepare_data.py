import os
import shutil
import random

def split_data():
    """Split raw data into train/test sets"""
    raw_path = os.path.join("data", "raw")
    processed_path = os.path.join("data", "processed")
    train_path = os.path.join(processed_path, "train")
    test_path = os.path.join(processed_path, "test")
    
    # Clear existing processed data
    if os.path.exists(train_path):
        shutil.rmtree(train_path)
    if os.path.exists(test_path):
        shutil.rmtree(test_path)
    
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    
    # Get classes
    classes = [d for d in os.listdir(raw_path) 
               if os.path.isdir(os.path.join(raw_path, d))]
    
    print(f"Found classes: {classes}")
    
    # Create class directories
    for class_name in classes:
        os.makedirs(os.path.join(train_path, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_path, class_name), exist_ok=True)
    
    # Split data for each class
    random.seed(42)
    test_size = 0.2
    
    total_images = 0
    for class_name in classes:
        class_raw_path = os.path.join(raw_path, class_name)
        if not os.path.exists(class_raw_path):
            continue
            
        # Get all images
        images = [f for f in os.listdir(class_raw_path) 
                 if os.path.isfile(os.path.join(class_raw_path, f)) and 
                 f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if len(images) == 0:
            print(f"Warning: No images found in class '{class_name}'")
            continue
            
        total_images += len(images)
        
        # Shuffle and split
        random.shuffle(images)
        split_idx = int(len(images) * (1 - test_size))
        # Ensure at least one image in each set if possible
        if len(images) > 1 and split_idx == len(images):
            split_idx = len(images) - 1
        elif split_idx == 0 and len(images) > 1:
            split_idx = 1
            
        train_images = images[:split_idx] if split_idx > 0 else images[:1]
        test_images = images[split_idx:] if split_idx < len(images) else images[1:]
        
        print(f"Class '{class_name}': {len(train_images)} train, {len(test_images)} test")
        
        # Copy to respective directories
        for img in train_images:
            src = os.path.join(class_raw_path, img)
            dst = os.path.join(train_path, class_name, img)
            shutil.copy2(src, dst)
            
        for img in test_images:
            src = os.path.join(class_raw_path, img)
            dst = os.path.join(test_path, class_name, img)
            shutil.copy2(src, dst)
    
    if total_images == 0:
        raise ValueError("No images found in any of the selected classes")
        
    print(f"Data split completed. Total images processed: {total_images}")
    return classes

if __name__ == "__main__":
    classes = split_data()
    print(f"Successfully prepared data for classes: {classes}")