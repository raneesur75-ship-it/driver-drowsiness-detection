'''"""
Data Augmentation Script for Driver Drowsiness Detection
Uses Stable Diffusion with ControlNet to generate realistic face variations
while preserving background structure.
"""

import os
import shutil
import torch
import numpy as np
import cv2
import pandas as pd
from PIL import Image, ImageFilter, ImageEnhance
from diffusers import StableDiffusionImg2ImgPipeline, ControlNetModel
from controlnet_aux import CannyDetector
from skimage.metrics import structural_similarity as ssim
from tqdm.auto import tqdm


# Configuration
BASE_DIR = "./data/raw"  # Change this to your dataset path
OUTPUT_DIR = "./data/augmented"
CLASSES = ["Yawn", "No_Yawn", "Open_Eyes", "Closed_Eyes"]
NUM_AUGMENTATIONS = 5  # New images per original

# Model settings - balanced for quality vs diversity
STRENGTH = 0.18          # Low = subtle changes, keeps identity
NUM_STEPS = 60           # Higher = better quality, slower
GUIDANCE_SCALE = 12.0    # How closely to follow prompt
CONTROL_SCALE = 1.0      # Edge preservation strength
IMG_SIZE = (768, 768)    # Processing resolution

# Prompts tailored for each class
PROMPTS = {
    "Yawn": "high resolution dslr photo, driver yawning, tired face, natural skin texture, detailed face, sharp focus, 8k raw photo",
    "No_Yawn": "high resolution dslr photo, alert driver, neutral expression, clear eyes, natural skin texture, detailed face, sharp focus, 8k raw photo",
    "Open_Eyes": "high resolution dslr photo, driver open eyes, alert look, natural skin texture, detailed face, sharp focus, 8k raw photo",
    "Closed_Eyes": "high resolution dslr photo, tired driver closed eyes, sleepy face, natural skin texture, detailed face, sharp focus, 8k raw photo"
}

NEGATIVE_PROMPT = "blurry, deformed face, melted features, warped eyes, bad anatomy, extra limbs, cartoon, low quality, plastic skin, doll face, background change"


def calculate_similarity(original, generated):
    """Calculate SSIM similarity between original and generated image."""
    gen_resized = generated.resize(original.size)
    org_gray = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2GRAY)
    gen_gray = cv2.cvtColor(np.array(gen_resized), cv2.COLOR_RGB2GRAY)
    score, _ = ssim(org_gray, gen_gray, full=True)
    return round(score * 100, 2)


def setup_directories():
    """Create output directory, clean if exists."""
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
        print(f"Cleaned existing output: {OUTPUT_DIR}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for class_name in CLASSES:
        os.makedirs(os.path.join(OUTPUT_DIR, f"Aug_{class_name}"), exist_ok=True)


def load_models():
    """Load Stable Diffusion and ControlNet models."""
    print("Loading models... (this may take a few minutes)")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny", 
        torch_dtype=torch.float16
    )
    
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "dreamlike-art/dreamlike-photoreal-2.0",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(device)
    
    canny = CannyDetector()
    
    print("Models loaded successfully!")
    return pipe, canny


def augment_image(pipe, canny, image_path, class_name, output_folder):
    """Generate augmented variations of a single image."""
    results = []
    
    # Load and resize image
    original = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    
    # Create edge map for structure preservation
    control_image = canny(original)
    
    prompt = PROMPTS[class_name]
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    for i in range(NUM_AUGMENTATIONS):
        # Generate variation
        generated = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            image=original,
            control_image=control_image,
            controlnet_conditioning_scale=CONTROL_SCALE,
            strength=STRENGTH,
            num_inference_steps=NUM_STEPS,
            guidance_scale=GUIDANCE_SCALE
        ).images[0]
        
        # Post-process: sharpen and enhance contrast
        generated = generated.filter(ImageFilter.SHARPEN)
        generated = ImageEnhance.Contrast(generated).enhance(1.12)
        
        # Calculate similarity score
        similarity = calculate_similarity(original, generated)
        
        # Save result
        save_name = f"{class_name}_real_{base_name}_aug_{i}.jpg"
        save_path = os.path.join(output_folder, save_name)
        generated.save(save_path)
        
        results.append({
            "class": class_name,
            "file": save_name,
            "similarity": similarity
        })
        
        print(f"  Generated: {save_name} (Similarity: {similarity:.1f}%)")
    
    return results


def print_summary(results_df):
    """Display quality summary statistics."""
    if results_df.empty:
        print("No results to display.")
        return
    
    print("\\n" + "="*60)
    print("AUGMENTATION SUMMARY")
    print("="*60)
    
    summary = results_df.groupby("class")["similarity"].agg([
        ("count", "count"),
        ("mean", "mean"),
        ("min", "min"),
        ("max", "max")
    ]).round(2)
    
    print(summary)
    print(f"\\nOverall average similarity: {results_df['similarity'].mean():.2f}%")
    print(f"Total images generated: {len(results_df)}")


def main():
    """Main execution function."""
    print("Driver Drowsiness Detection - Data Augmentation")
    print("="*60)
    
    # Setup
    setup_directories()
    pipe, canny = load_models()
    
    all_results = []
    
    # Process each class
    for class_name in CLASSES:
        input_folder = os.path.join(BASE_DIR, class_name)
        output_folder = os.path.join(OUTPUT_DIR, f"Aug_{class_name}")
        
        if not os.path.exists(input_folder):
            print(f"\\nWarning: Folder not found - {input_folder}")
            continue
        
        # Get image files
        images = [f for f in os.listdir(input_folder) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not images:
            print(f"\\nNo images found in {input_folder}")
            continue
        
        print(f"\\nProcessing {class_name}: {len(images)} images")
        print("-"*60)
        
        # Augment each image
        for img_file in tqdm(images, desc=class_name):
            img_path = os.path.join(input_folder, img_file)
            results = augment_image(pipe, canny, img_path, class_name, output_folder)
            all_results.extend(results)
    
    # Summary
    if all_results:
        results_df = pd.DataFrame(all_results)
        print_summary(results_df)
        
        # Save CSV log
        csv_path = os.path.join(OUTPUT_DIR, "augmentation_log.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"\\nLog saved: {csv_path}")
    
    print("\\n" + "="*60)
    print(f"Done! Augmented images saved to: {OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
'''
