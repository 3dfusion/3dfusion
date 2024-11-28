import os
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from PIL import Image
from skimage.transform import resize
from transformers import DPTForDepthEstimation, DPTImageProcessor

class AdvancedPointCloudCombiner:
    def __init__(self, model_name="Intel/dpt-large"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.model = DPTForDepthEstimation.from_pretrained(model_name).to(self.device)
        self.processor = DPTImageProcessor.from_pretrained(model_name)

    def preprocess_image(self, image, target_height=384):
        width, height = image.size
        aspect_ratio = width / height
        new_width = int(target_height * aspect_ratio)
        new_size = (new_width, target_height)
        image = image.resize(new_size, Image.LANCZOS)
        
        inputs = self.processor(images=image, return_tensors="pt")
        input_image = inputs['pixel_values'].to(self.device)
        return input_image, image

    def estimate_depth(self, input_image):
        with torch.no_grad():
            outputs = self.model(input_image)
            predicted_depth = outputs.predicted_depth
            
            predicted_depth = predicted_depth.squeeze().cpu().numpy()
            depth_map = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())
            
            return depth_map

    def create_point_cloud(self, rgb_image, depth_map):
        try:
            rgb_array = np.array(rgb_image)
            
            depth_map_resized = resize(depth_map, (rgb_array.shape[0], rgb_array.shape[1]), 
                                       anti_aliasing=True, preserve_range=True)
            
            depth_normalized = (depth_map_resized * 1000).astype(np.uint16)
            
            rgb_o3d = o3d.geometry.Image(rgb_array)
            depth_o3d = o3d.geometry.Image(depth_normalized)
            
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb_o3d, depth_o3d, 
                depth_scale=1000.0,
                convert_rgb_to_intensity=False
            )
            
            height, width = rgb_array.shape[:2]
            camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width=width, 
                height=height,
                fx=width * 0.8,
                fy=height * 0.8,
                cx=width / 2,
                cy=height / 2
            )
            
            point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)
            
            point_cloud.estimate_normals()
            point_cloud.orient_normals_consistent_tangent_plane(100)
            
            # Remove statistical outliers
            point_cloud, _ = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            
            return point_cloud
        
        except Exception as e:
            print(f"Error creating point cloud: {e}")
            raise

    def combine_point_clouds(self, point_cloud_front, point_cloud_back):
        """
        Combine two point clouds with a simplified alignment approach
        """
        try:
            # Downsample point clouds to reduce computational complexity
            voxel_size = 0.01
            front_down = point_cloud_front.voxel_down_sample(voxel_size=voxel_size)
            back_down = point_cloud_back.voxel_down_sample(voxel_size=voxel_size)

            # Estimate normals
            front_down.estimate_normals()
            back_down.estimate_normals()

            # Create a search tree for FPFH feature computation
            front_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                front_down, 
                o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
            )
            back_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                back_down, 
                o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
            )

            # RANSAC-based initial alignment
            result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                front_down, back_down, 
                front_fpfh, back_fpfh,
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )

            # Refine alignment with ICP
            final_transform = o3d.pipelines.registration.registration_icp(
                front_down, back_down, 0.05, result_ransac.transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )

            # Transform back point cloud
            back_transformed = copy.deepcopy(point_cloud_back)
            back_transformed.transform(final_transform.transformation)

            # Combine point clouds
            combined_cloud = front_down + back_transformed

            return combined_cloud

        except Exception as e:
            print(f"Error combining point clouds: {e}")
            # Fallback: simple concatenation if advanced registration fails
            combined_cloud = point_cloud_front + point_cloud_back
            return combined_cloud

    def process_images(self, front_image_path, back_image_path, 
                       output_front_depth='front_depth.png', 
                       output_back_depth='back_depth.png', 
                       output_combined_point_cloud='combined_pointcloud.ply'):
        try:
            # Process front image
            front_image = Image.open(front_image_path).convert("RGB")
            front_input, front_processed = self.preprocess_image(front_image)
            front_depth = self.estimate_depth(front_input)
            
            # Save front depth map
            plt.figure(figsize=(10, 10))
            plt.imshow(front_depth, cmap='viridis')
            plt.axis('off')
            plt.colorbar()
            plt.savefig(output_front_depth, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            # Create front point cloud
            front_point_cloud = self.create_point_cloud(front_processed, front_depth)

            # Process back image
            back_image = Image.open(back_image_path).convert("RGB")
            back_input, back_processed = self.preprocess_image(back_image)
            back_depth = self.estimate_depth(back_input)
            
            # Save back depth map
            plt.figure(figsize=(10, 10))
            plt.imshow(back_depth, cmap='viridis')
            plt.axis('off')
            plt.colorbar()
            plt.savefig(output_back_depth, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            # Create back point cloud
            back_point_cloud = self.create_point_cloud(back_processed, back_depth)

            # Combine point clouds
            combined_point_cloud = self.combine_point_clouds(front_point_cloud, back_point_cloud)

            # Save combined point cloud
            o3d.io.write_point_cloud(output_combined_point_cloud, combined_point_cloud)

            # Visualize combined point cloud
            o3d.visualization.draw_geometries([combined_point_cloud])

            print(f"Depth maps saved: {output_front_depth}, {output_back_depth}")
            print(f"Combined point cloud saved: {output_combined_point_cloud}")

        except Exception as e:
            print(f"Error processing images: {e}")

def main():
    # Specify directory containing front and back images
    image_directory = r"output_views"
    
    # Find front and back images
    front_image = None
    back_image = None
    for filename in os.listdir(image_directory):
        if "front" in filename.lower():
            front_image = os.path.join(image_directory, filename)
        elif "back" in filename.lower():
            back_image = os.path.join(image_directory, filename)

    if not front_image or not back_image:
        print("Could not find front and back images in the directory.")
        return

    # Initialize point cloud combiner
    combiner = AdvancedPointCloudCombiner()

    # Process images and combine point clouds
    combiner.process_images(front_image, back_image)

if __name__ == "__main__":
    main()