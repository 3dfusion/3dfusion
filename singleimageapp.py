import gradio as gr
import os
import torch
import numpy as np
import open3d as o3d
from PIL import Image
from source.combine import PointCloudCombiner
from source.depth import DepthEstimator

# Initialize estimators and combiners
depth_estimator = DepthEstimator()
point_cloud_combiner = PointCloudCombiner()

def process_image(input_image):
    """
    Main workflow for processing an input image:
    1. Generate depth maps for the front and back views.
    2. Create point clouds from depth maps.
    3. Combine the front and back point clouds into a single point cloud.
    
    Args:
        input_image (PIL.Image): Input image provided by the user.
    
    Returns:
        tuple: Paths to saved depth maps and combined point cloud.
    """
    output_dir = "gradio_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Preprocess input image
    front_input, front_processed = depth_estimator.preprocess_image(input_image)
    back_input, back_processed = depth_estimator.preprocess_image(input_image)  # Assuming the back view is the same image for simplicity
    
    # Estimate depth maps
    front_depth = depth_estimator.estimate_depth(front_input)
    back_depth = depth_estimator.estimate_depth(back_input)
    
    # Save depth maps
    front_depth_path = os.path.join(output_dir, "front_depth.png")
    back_depth_path = os.path.join(output_dir, "back_depth.png")
    depth_estimator.save_depth_map(front_depth, front_depth_path)
    depth_estimator.save_depth_map(back_depth, back_depth_path)
    
    # Create point clouds
    front_point_cloud = point_cloud_combiner.create_point_cloud(front_processed, front_depth)
    back_point_cloud = point_cloud_combiner.create_point_cloud(back_processed, back_depth)
    
    # Combine point clouds
    combined_point_cloud = point_cloud_combiner.combine_point_clouds(front_point_cloud, back_point_cloud)
    combined_point_cloud_path = os.path.join(output_dir, "combined_pointcloud.ply")
    o3d.io.write_point_cloud(combined_point_cloud_path, combined_point_cloud)
    
    return front_depth_path, back_depth_path, combined_point_cloud_path

# Gradio Interface
def gradio_pipeline(input_image):
    front_depth_path, back_depth_path, combined_point_cloud_path = process_image(input_image)
    
    return front_depth_path, back_depth_path, combined_point_cloud_path, combined_point_cloud_path

interface = gr.Interface(
    fn=gradio_pipeline,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=[
        gr.Image(label="Front Depth Map"),
        gr.Image(label="Back Depth Map"),
        gr.File(label="Download Combined Point Cloud (PLY File)"),
        gr.Model3D(label="Interactive 3D Point Cloud Viewer")  # Add interactive 3D visualization
    ],
    title="3D Model Generator with Visualization",
    description="Upload an image to generate depth maps, point clouds, and a combined 3D point cloud with interactive visualization."
)

if __name__ == "__main__":
    interface.launch()
