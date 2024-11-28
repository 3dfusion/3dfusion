from diffusers import DiffusionPipeline
import torch
import numpy as np
from PIL import Image
import os

def setup_multiview_pipeline():
    """Initialize multi-view diffusion pipeline."""
    mv_pipeline = DiffusionPipeline.from_pretrained(
        "dylanebert/multi-view-diffusion",
        custom_pipeline="dylanebert/multi-view-diffusion",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to("cuda")
    return mv_pipeline

def generate_front_back_views(image_path, output_dir="output_views"):
    """
    Generate front and back views of the input image.
    
    Args:
        image_path (str): Path to the input image
        output_dir (str): Directory to save generated views
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Setup multi-view pipeline
    mv_pipeline = setup_multiview_pipeline()

    # Load and preprocess input image
    input_image = Image.open(image_path).convert("RGB")
    input_array = np.array(input_image).astype("float32") / 255.0

    # Generate front and back views
    views = []
    view_names = ["front", "back"]
    elevations = [0, 180]  # Front and back views

    for name, elevation in zip(view_names, elevations):
        # Generate view with specified elevation
        outputs = mv_pipeline(
            image=input_array,
            guidance_scale=5,
            num_inference_steps=30,
            elevation=elevation,
        )

        # Convert numpy array to PIL Image
        if len(outputs) > 0:
            view_img = Image.fromarray((outputs[0] * 255).astype("uint8"))
            
            # Generate output filename
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            view_path = os.path.join(output_dir, f"{base_name}_{name}_view.png")
            
            # Save the view image
            view_img.save(view_path)
            print(f"Saved {name} view to: {view_path}")
            
            views.append(view_img)
        else:
            print(f"No {name} view generated")

    return views

# Example usage
if __name__ == "__main__":
    input_image_path = "path/to/your/input_image.png"  # Replace with your image path
    generate_front_back_views(input_image_path)
