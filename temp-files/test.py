import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import open3d as o3d
from PIL import Image, ImageOps
import cv2
import random
from transformers import DPTForDepthEstimation, DPTImageProcessor, GLPNImageProcessor, GLPNForDepthEstimation
from diffusers import DiffusionPipeline
from scipy.ndimage import median_filter
from skimage.transform import resize
import gc

class Enhanced3DReconstructor:
    def __init__(self, max_memory_gb=4):
        # Memory and device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_memory_gb = max_memory_gb
        print(f"Using device: {self.device}")
        
        # Configure CUDA memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            memory_fraction = min(0.8, self.max_memory_gb / total_memory)
            torch.cuda.set_per_process_memory_fraction(memory_fraction)
        
        # Load models
        self.load_models()

    def load_models(self):
        try:
            print("Loading models...")
            
            # DPT model for high-quality depth estimation
            self.dpt_model = DPTForDepthEstimation.from_pretrained(
                "Intel/dpt-large",
                torch_dtype=torch.float16
            ).to(self.device)
            self.dpt_processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
            
            # GLPN model for additional depth perspective
            self.glpn_processor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-nyu")
            self.glpn_model = GLPNForDepthEstimation.from_pretrained(
                "vinvino02/glpn-nyu",
                torch_dtype=torch.float16
            ).to(self.device)
            
            # Multi-view diffusion model
            self.mv_pipeline = DiffusionPipeline.from_pretrained(
                "dylanebert/multi-view-diffusion",
                custom_pipeline="dylanebert/multi-view-diffusion",
                torch_dtype=torch.float16,
                trust_remote_code=True
            ).to(self.device)
            
            print("All models loaded successfully")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            raise

    def clear_gpu_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    def generate_synthetic_views(self, original_image, num_views=3):
        synthetic_views = [original_image]
        
        # Geometric transformations
        transforms = [
            lambda img: img.rotate(random.uniform(-20, 20)),
            lambda img: self._perspective_transform(img, 20),
            lambda img: ImageOps.autocontrast(img)
        ]
        
        for _ in range(num_views - 1):
            transform = random.choice(transforms)
            new_view = transform(original_image.copy())
            synthetic_views.append(new_view)
        
        return synthetic_views

    def generate_multi_views(self, image_path):
        try:
            input_image = Image.open(image_path).convert("RGB")
            input_image = input_image.resize((512, 512))
            input_array = np.array(input_image).astype(np.float32) / 255.0
            
            view_configs = [
                {"name": "front", "elevation": 0, "guidance_scale": 5},
                {"name": "right", "elevation": 30, "guidance_scale": 6},
                {"name": "left", "elevation": -30, "guidance_scale": 7},
                {"name": "back", "elevation": 180, "guidance_scale": 8}
            ]
            
            generated_views = {}
            for config in view_configs:
                outputs = self.mv_pipeline(
                    image=input_array,
                    guidance_scale=config["guidance_scale"],
                    num_inference_steps=10,
                    elevation=config["elevation"]
                )
                
                if outputs and len(outputs) > 0:
                    view_array = outputs[0]
                    generated_view = Image.fromarray((view_array * 255).astype("uint8")).resize((512, 512))
                    generated_views[config["name"]] = generated_view
            
            self.clear_gpu_memory()
            return generated_views
        
        except Exception as e:
            print(f"Error generating multi views: {e}")
            self.clear_gpu_memory()
            raise

    def estimate_depth(self, image, use_dpt=True):
        try:
            if use_dpt:
                # DPT depth estimation
                inputs = self.dpt_processor(images=image, return_tensors="pt")
                input_image = inputs['pixel_values'].to(self.device)
                
                with torch.no_grad():
                    outputs = self.dpt_model(input_image)
                    depth_map = outputs.predicted_depth.squeeze().cpu().numpy()
            else:
                # GLPN depth estimation
                inputs = self.glpn_processor(images=image, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.glpn_model(**inputs)
                    depth_map = outputs.predicted_depth.squeeze().cpu().numpy()
            
            # Normalize depth map
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
            return depth_map
        
        except Exception as e:
            print(f"Error in depth estimation: {e}")
            raise

    def create_point_cloud(self, rgb_image, depth_map, view_name='default'):
        try:
            rgb_array = np.array(rgb_image)
            depth_map_resized = resize(depth_map, (rgb_array.shape[0], rgb_array.shape[1]), 
                                     anti_aliasing=True, preserve_range=True)
            
            # Convert to Open3D format
            rgb_o3d = o3d.geometry.Image(rgb_array)
            depth_o3d = o3d.geometry.Image((depth_map_resized * 1000).astype(np.uint16))
            
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb_o3d, depth_o3d,
                depth_scale=1000.0,
                convert_rgb_to_intensity=False
            )
            
            # Camera parameters
            height, width = rgb_array.shape[:2]
            camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width=width,
                height=height,
                fx=width * 0.8,
                fy=height * 0.8,
                cx=width / 2,
                cy=height / 2
            )
            
            # Create and enhance point cloud
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)
            pcd.estimate_normals()
            pcd.orient_normals_consistent_tangent_plane(100)
            
            return pcd
        
        except Exception as e:
            print(f"Error creating point cloud for {view_name}: {e}")
            raise

    def process_image(self, image_path, output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Create subdirectories
            views_dir = os.path.join(output_dir, 'views')
            depth_dir = os.path.join(output_dir, 'depth_maps')
            cloud_dir = os.path.join(output_dir, 'point_clouds')
            for dir_path in [views_dir, depth_dir, cloud_dir]:
                os.makedirs(dir_path, exist_ok=True)
            
            # Generate views
            original_image = Image.open(image_path).convert("RGB")
            synthetic_views = self.generate_synthetic_views(original_image)
            multi_views = self.generate_multi_views(image_path)
            
            # Combine all views
            all_views = {f'synthetic_{i}': view for i, view in enumerate(synthetic_views)}
            all_views.update(multi_views)
            
            point_clouds = []
            
            # Process each view
            for view_name, view in all_views.items():
                # Save view
                view.save(os.path.join(views_dir, f'{view_name}.png'))
                
                # Generate and save depth maps
                depth_dpt = self.estimate_depth(view, use_dpt=True)
                depth_glpn = self.estimate_depth(view, use_dpt=False)
                
                plt.imsave(os.path.join(depth_dir, f'{view_name}_dpt.png'), depth_dpt, cmap='plasma')
                plt.imsave(os.path.join(depth_dir, f'{view_name}_glpn.png'), depth_glpn, cmap='plasma')
                
                # Create and save point clouds
                pcd_dpt = self.create_point_cloud(view, depth_dpt, f'{view_name}_dpt')
                pcd_glpn = self.create_point_cloud(view, depth_glpn, f'{view_name}_glpn')
                
                o3d.io.write_point_cloud(os.path.join(cloud_dir, f'{view_name}_dpt.ply'), pcd_dpt)
                o3d.io.write_point_cloud(os.path.join(cloud_dir, f'{view_name}_glpn.ply'), pcd_glpn)
                
                point_clouds.extend([pcd_dpt, pcd_glpn])
            
            # Combine all point clouds
            if point_clouds:
                combined_cloud = point_clouds[0]
                for pc in point_clouds[1:]:
                    combined_cloud += pc
                
                # Save combined point cloud
                o3d.io.write_point_cloud(os.path.join(cloud_dir, 'combined_model.ply'), combined_cloud)
            
            print(f"Processing complete. Results saved in {output_dir}")
            
        except Exception as e:
            print(f"Error processing image: {e}")
            self.clear_gpu_memory()
            raise

def main():
    # Initialize with conservative memory limit for Acer Predator Helios Neo 16
    reconstructor = Enhanced3DReconstructor(max_memory_gb=4)
    
    # Set your input image path and output directory
    image_path = "/home/kali1/Pictures/falcom.jpeg"
    output_dir = "/home/kali1/Pictures"
    
    try:
        reconstructor.process_image(image_path, output_dir)
    except Exception as e:
        print(f"Error in main execution: {e}")
    finally:
        reconstructor.clear_gpu_memory()

if __name__ == "__main__":
    main()