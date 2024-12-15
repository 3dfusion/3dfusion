
import os
from source.multiview import generate_front_back_views
from source.depth import DepthEstimator
from source.combine import PointCloudCombiner
from PIL import Image
import open3d as o3d

def main():
    """
    Main workflow for multi-view point cloud generation:
    1. Generate front and back view images
    2. Estimate depth maps
    3. Create point clouds
    4. Combine point clouds
    5. Visualize and save results
    """
    # Input image path - replace with your actual image path
    input_image_path = r"C:\Users\navee\OneDrive\Desktop\New folder\repos\New folder\3d fusion\images\images (1).jpg"
    
    # Output directories and files
    output_views_dir = "output_views"
    depth_output_dir = "depth_maps"
    point_cloud_output_dir = "point_clouds"
    
    # Create output directories
    os.makedirs(output_views_dir, exist_ok=True)
    os.makedirs(depth_output_dir, exist_ok=True)
    os.makedirs(point_cloud_output_dir, exist_ok=True)

    try:
        # Step 1: Generate multi-view images
        print("1. Generating front and back view images...")
        views = generate_front_back_views(input_image_path)
        
        # Step 2: Initialize depth estimator and point cloud combiner
        depth_estimator = DepthEstimator()
        point_cloud_combiner = PointCloudCombiner()
        
        # Step 3: Process front and back views
        view_types = ["front", "back"]
        point_clouds = []
        
        for view_type in view_types:
            # Find the corresponding view image
            view_image_path = os.path.join(output_views_dir, 
                                           f"{os.path.splitext(os.path.basename(input_image_path))[0]}_{view_type}_view.png")
            
            # Open and preprocess image
            view_image = Image.open(view_image_path).convert("RGB")
            view_input, view_processed = depth_estimator.preprocess_image(view_image)
            
            # Estimate depth
            view_depth = depth_estimator.estimate_depth(view_input)
            
            # Save depth map
            depth_output_path = os.path.join(depth_output_dir, f"{view_type}_depth.png")
            depth_estimator.save_depth_map(view_depth, depth_output_path)
            
            # Create point cloud
            point_cloud = point_cloud_combiner.create_point_cloud(view_processed, view_depth)
            point_clouds.append(point_cloud)
            
            # Save individual point clouds
            point_cloud_path = os.path.join(point_cloud_output_dir, f"{view_type}_pointcloud.ply")
            o3d.io.write_point_cloud(point_cloud_path, point_cloud)
            print(f"Saved {view_type} point cloud to: {point_cloud_path}")
        
        # Step 4: Combine point clouds
        print("Combining point clouds...")
        combined_point_cloud = point_cloud_combiner.combine_point_clouds(point_clouds[0], point_clouds[1])
        
        # Save combined point cloud
        combined_point_cloud_path = os.path.join(point_cloud_output_dir, "combined_pointcloud.ply")
        o3d.io.write_point_cloud(combined_point_cloud_path, combined_point_cloud)
        print(f"Saved combined point cloud to: {combined_point_cloud_path}")
        
        # Step 5: Visualize combined point cloud
        print("Visualizing combined point cloud...")
        o3d.visualization.draw_geometries([combined_point_cloud])
    
    except Exception as e:
        print(f"An error occurred during processing: {e}")
        import traceback
        traceback.print_exc()

def visualize_point_cloud(point_cloud_path):
    """
    Utility function to visualize a saved point cloud
    
    Args:
        point_cloud_path (str): Path to the .ply point cloud file
    """
    try:
        point_cloud = o3d.io.read_point_cloud(point_cloud_path)
        o3d.visualization.draw_geometries([point_cloud])
    except Exception as e:
        print(f"Error visualizing point cloud: {e}")

if __name__ == "__main__":
    main()
    
    # Uncomment to visualize a specific point cloud after main workflow
    # visualize_point_cloud("path/to/your/pointcloud.ply")
