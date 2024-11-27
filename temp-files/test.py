import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import open3d as o3d
from PIL import Image
import cv2
from diffusers import DiffusionPipeline
from transformers import GLPNImageProcessor, GLPNForDepthEstimation, DPTForDepthEstimation, DPTImageProcessor
from skimage.transform import resize
from scipy.ndimage import median_filter

class AdvancedMultiViewReconstructor:
    def __init__(self, max_memory_gb=4):
        """
        Initialize models for multi-view depth estimation and 3D reconstruction
        
        Args:
            max_memory_gb (float): Maximum GPU memory to use
        """
        # Set device and memory management
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_memory_gb = max_memory_gb
        print(f"Using device: {self.device}")

        # Clear initial GPU memory
        self.clear_gpu_memory()

        # Initialize Multi-View Diffusion Pipeline
        try:
            print("Loading Multi-View Diffusion Pipeline...")
            self.mv_pipeline = DiffusionPipeline.from_pretrained(
                "dylanebert/multi-view-diffusion",
                custom_pipeline="dylanebert/multi-view-diffusion",
                torch_dtype=torch.float32,
                trust_remote_code=True,
            ).to(self.device)
        except Exception as e:
            print(f"Error loading Multi-View Diffusion Pipeline: {e}")
            self.clear_gpu_memory()
            raise

        # Initialize Depth Estimation Models
        try:
            print("Loading Depth Estimation Models...")
            # GLPN Model for depth estimation
            self.glpn_feature_extractor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-nyu")
            self.glpn_depth_model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu").to(self.device)

            # DPT Model for additional depth estimation
            self.dpt_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to(self.device)
            self.dpt_processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
        except Exception as e:
            print(f"Error loading depth estimation models: {e}")
            self.clear_gpu_memory()
            raise

    def clear_gpu_memory(self):
        """
        Clear GPU memory and perform garbage collection
        """
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    def check_gpu_memory(self):
        """
        Check and log GPU memory usage
        """
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            current_memory = torch.cuda.memory_allocated(0) / (1024**3)
            print(f"GPU Memory - Total: {total_memory:.2f} GB, Current Usage: {current_memory:.2f} GB")

    def generate_multi_view_images(self, input_image_path):
        """
        Generate multiple views of the input image (front, right, left, back)
        
        Args:
            input_image_path (str): Path to the input image
        
        Returns:
            dict: Generated view images with predefined names
        """
        try:
            # Load and preprocess input image
            input_image = Image.open(input_image_path).convert("RGB")
            
            # Resize image to reduce memory usage
            input_image = input_image.resize((512, 512))
            input_array = np.array(input_image).astype("float32") / 255.0

            # Predefined view names
            view_names = ["front", "right", "left", "back"]
            
            # Generate multiple views with different parameters for varied perspectives
            views_params = [
                {"elevation": 0, "guidance_scale": 5},     # front view
                {"elevation": 30, "guidance_scale": 6},    # right view
                {"elevation": -30, "guidance_scale": 7},   # left view
                {"elevation": 180, "guidance_scale": 8}    # back view
            ]

            # Generate views
            generated_views = {}
            for name, params in zip(view_names, views_params):
                view_outputs = self.mv_pipeline(
                    image=input_array,
                    guidance_scale=params["guidance_scale"],
                    num_inference_steps=20,
                    elevation=params["elevation"],
                )
                
                # Take first generated view for each perspective
                if view_outputs and len(view_outputs) > 0:
                    view_array = view_outputs[0]
                    generated_view = Image.fromarray((view_array * 255).astype("uint8")).resize((512, 512))
                    generated_views[name] = generated_view

            self.clear_gpu_memory()
            return generated_views

        except Exception as e:
            print(f"Error generating multi-view images: {e}")
            self.clear_gpu_memory()
            raise

    def generate_multi_view_images_with_count(self, input_image_path, num_views=2):
        """
        Generate multiple views of the input image using Multi-View Diffusion
        
        Args:
            input_image_path (str): Path to the input image
            num_views (int): Number of views to generate
        
        Returns:
            list: Generated view images
        """
        try:
            # Load and preprocess input image
            input_image = Image.open(input_image_path).convert("RGB")
            
            # Resize image to reduce memory usage
            input_image = input_image.resize((512, 512))
            input_array = np.array(input_image).astype("float32") / 255.0

            # Generate multiple views
            outputs = self.mv_pipeline(
                image=input_array,
                guidance_scale=5,
                num_inference_steps=20,
                elevation=0,
            )

            # Convert outputs to PIL Images
            generated_views = [
                Image.fromarray((view_array * 255).astype("uint8")).resize((512, 512))
                for view_array in outputs[:num_views]
            ]

            self.clear_gpu_memory()
            return generated_views

        except Exception as e:
            print(f"Error generating multi-view images: {e}")
            self.clear_gpu_memory()
            raise

    def estimate_depth_glpn(self, image):
        """
        Estimate depth using GLPN model
        
        Args:
            image (PIL.Image): Input image
        
        Returns:
            np.ndarray: Depth map
        """
        try:
            # Prepare inputs
            inputs = self.glpn_feature_extractor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.glpn_depth_model(**inputs)
                predicted_depth = outputs.predicted_depth
            
            # Interpolate depth to match image size
            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=image.size[::-1],
                mode="bicubic",
                align_corners=False,
            )
            
            depth_map = prediction.squeeze().cpu().numpy()
            
            # Normalize depth map
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
            
            self.clear_gpu_memory()
            return depth_map

        except Exception as e:
            print(f"GLPN depth estimation error: {e}")
            self.clear_gpu_memory()
            raise

    def estimate_depth_dpt(self, image):
        """
        Estimate depth using DPT model
        
        Args:
            image (PIL.Image): Input image
        
        Returns:
            np.ndarray: Depth map
        """
        try:
            # Preprocess image
            inputs = self.dpt_processor(images=image, return_tensors="pt")
            input_image = inputs['pixel_values'].to(self.device)
            
            # Estimate depth
            with torch.no_grad():
                outputs = self.dpt_model(input_image)
                depth_prediction = outputs.predicted_depth
                
            # Convert to numpy and normalize
            depth_np = depth_prediction.squeeze().cpu().numpy()
            depth_np = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())
            
            self.clear_gpu_memory()
            return depth_np

        except Exception as e:
            print(f"DPT depth estimation error: {e}")
            self.clear_gpu_memory()
            raise

    def create_point_cloud(self, rgb_image, depth_map, view_name='default'):
        """
        Create 3D point cloud from RGB image and depth map
        
        Args:
            rgb_image (PIL.Image): RGB image
            depth_map (np.ndarray): Depth map
            view_name (str): Name of the view for logging
        
        Returns:
            o3d.geometry.PointCloud: Generated point cloud
        """
        try:
            # Convert PIL Image to numpy array
            rgb_array = np.array(rgb_image)
            
            # Ensure depth map is the same size as RGB image
            depth_map_resized = resize(depth_map, (rgb_array.shape[0], rgb_array.shape[1]), 
                                       anti_aliasing=True, preserve_range=True)
            
            # Normalize depth map
            depth_normalized = (depth_map_resized * 1000).astype(np.uint16)
            
            # Convert images to Open3D format
            rgb_o3d = o3d.geometry.Image(rgb_array)
            depth_o3d = o3d.geometry.Image(depth_normalized)
            
            # Create RGBD image
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb_o3d, depth_o3d, 
                depth_scale=1000.0,  # Convert to millimeters
                convert_rgb_to_intensity=False
            )
            
            # Camera intrinsics
            height, width = rgb_array.shape[:2]
            camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width=width, 
                height=height,
                fx=width * 0.8,  # Focal length in x direction
                fy=height * 0.8,  # Focal length in y direction
                cx=width / 2,     # Principal point x-coordinate
                cy=height / 2     # Principal point y-coordinate
            )
            
            # Create point cloud
            point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)
            
            # Enhance point cloud quality
            point_cloud.estimate_normals()
            point_cloud.orient_normals_consistent_tangent_plane(100)
            
            print(f"Point cloud created for {view_name} view")
            return point_cloud

        except Exception as e:
            print(f"Error creating point cloud for {view_name} view: {e}")
            raise

    def save_depth_visualization(self, depth_map, output_path):
        """
        Save depth map as a color-coded visualization
        
        Args:
            depth_map (np.ndarray): Depth map to visualize
            output_path (str): Path to save visualization
        """
        plt.figure(figsize=(10, 10))
        plt.imshow(depth_map, cmap='plasma')
        plt.axis('off')
        plt.colorbar(label='Depth')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

def reconstruct_3d(self, input_image_path, output_dir='3d_reconstruction_output', num_views=2, multi_perspective=False):
        """
        Perform complete 3D reconstruction process
        
        Args:
            input_image_path (str): Path to input image
            output_dir (str): Directory to save outputs
            num_views (int): Number of views to generate (if not multi_perspective)
            multi_perspective (bool): Use predefined multi-view perspectives
        """
        try:
            # Clear initial GPU memory
            self.clear_gpu_memory()
            self.check_gpu_memory()

            # Create output directories
            os.makedirs(output_dir, exist_ok=True)
            views_dir = os.path.join(output_dir, 'views')
            depth_dir = os.path.join(output_dir, 'depth_maps')
            point_cloud_dir = os.path.join(output_dir, 'point_clouds')
            os.makedirs(views_dir, exist_ok=True)
            os.makedirs(depth_dir, exist_ok=True)
            os.makedirs(point_cloud_dir, exist_ok=True)

            # Generate multi-view images
            if multi_perspective:
                generated_views = self.generate_multi_view_images(input_image_path)
            else:
                generated_views_list = self.generate_multi_view_images_with_count(input_image_path, num_views)
                generated_views = {f'view_{i}': view for i, view in enumerate(generated_views_list)}
            
            # List to store point clouds
            point_clouds = []
            
            # Process each generated view
            for view_name, view in generated_views.items():
                # Save view image
                view_path = os.path.join(views_dir, f'{view_name}.png')
                view.save(view_path)
                
                # Estimate depth using both models
                depth_glpn = self.estimate_depth_glpn(view)
                depth_dpt = self.estimate_depth_dpt(view)
                
                # Save depth maps
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.title('GLPN Depth Map')
                plt.imshow(depth_glpn, cmap='plasma')
                plt.axis('off')
                
                plt.subplot(1, 2, 2)
                plt.title('DPT Depth Map')
                plt.imshow(depth_dpt, cmap='viridis')
                plt.axis('off')
                
                depth_path = os.path.join(depth_dir, f'{view_name}_depth_comparison.png')
                plt.savefig(depth_path)
                plt.close()
                
                # Create point clouds
                point_cloud_glpn = self.create_point_cloud(view, depth_glpn, f'{view_name}_GLPN')
                point_cloud_dpt = self.create_point_cloud(view, depth_dpt, f'{view_name}_DPT')
                
                # Save point clouds
                glpn_ply_path = os.path.join(point_cloud_dir, f'{view_name}_GLPN.ply')
                dpt_ply_path = os.path.join(point_cloud_dir, f'{view_name}_DPT.ply')
                
                o3d.io.write_point_cloud(glpn_ply_path, point_cloud_glpn)
                o3d.io.write_point_cloud(dpt_ply_path, point_cloud_dpt)
                
                point_clouds.extend([point_cloud_glpn, point_cloud_dpt])
            
            # Optional: Combine point clouds if multiple views
            if len(point_clouds) > 1:
                combined_point_cloud = o3d.geometry.PointCloud()
                for pc in point_clouds:
                    combined_point_cloud += pc
                
                # Downsample and clean combined point cloud
                combined_point_cloud = combined_point_cloud.voxel_down_sample(voxel_size=0.05)
                combined_point_cloud.remove_statistical_outliers(nb_neighbors=20, std_ratio=2.0)
                
                # Save combined point cloud
                combined_ply_path = os.path.join(output_dir, 'combined_point_cloud.ply')
                o3d.io.write_point_cloud(combined_ply_path, combined_point_cloud)
                print(f"Combined point cloud saved to {combined_ply_path}")
            
            print("3D reconstruction completed successfully!")
            return point_clouds

        except Exception as e:
            print(f"Error during 3D reconstruction: {e}")
            self.clear_gpu_memory()
            raise

def main():
    """
    Example usage of AdvancedMultiViewReconstructor
    """
    reconstructor = AdvancedMultiViewReconstructor(max_memory_gb=8)
    input_image_path = '/home/kali1/Pictures/falcon.jpeg'
    
    # Reconstruct with default settings (2 views)
    reconstructor.reconstruct_3d(input_image_path)
    
    # Reconstruct with multi-perspective views
    reconstructor.reconstruct_3d(input_image_path, multi_perspective=True)

if __name__ == "__main__":
    main()