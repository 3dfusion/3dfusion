import os
import copy
import torch
import numpy as np
import open3d as o3d
from PIL import Image
from skimage.transform import resize

class PointCloudCombiner:
    def __init__(self):
        pass

    def create_point_cloud(self, rgb_image, depth_map):
        """
        Create point cloud from RGB image and depth map
        
        Args:
            rgb_image (PIL.Image): RGB image
            depth_map (numpy.ndarray): Depth map
        
        Returns:
            o3d.geometry.PointCloud: Generated point cloud
        """
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
        Combine two point clouds using registration techniques
        
        Args:
            point_cloud_front (o3d.geometry.PointCloud): Front view point cloud
            point_cloud_back (o3d.geometry.PointCloud): Back view point cloud
        
        Returns:
            o3d.geometry.PointCloud: Combined point cloud
        """
        try:
            voxel_size = 0.01
            front_down = point_cloud_front.voxel_down_sample(voxel_size=voxel_size)
            back_down = point_cloud_back.voxel_down_sample(voxel_size=voxel_size)

            front_down.estimate_normals()
            back_down.estimate_normals()

            front_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                front_down, 
                o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
            )
            back_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                back_down, 
                o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
            )

            initial_alignment = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                front_down, back_down, 
                front_fpfh, back_fpfh,
                mutual_filter=True,
                max_correspondence_distance=voxel_size * 1.5,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                ransac_n=4,
                checkers=[
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 1.5)
                ],
                max_iteration=100000
            )

            icp_result = o3d.pipelines.registration.registration_icp(
                back_down, front_down, 
                max_correspondence_distance=voxel_size * 1.5,
                init=initial_alignment.transformation,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
            )

            back_transformed = copy.deepcopy(point_cloud_back)
            back_transformed.transform(icp_result.transformation)

            combined_cloud = point_cloud_front + back_transformed
            combined_cloud.remove_statistical_outliers(nb_neighbors=20, std_ratio=2.0)

            return combined_cloud

        except Exception as e:
            print(f"Point cloud registration error: {e}")
            return point_cloud_front + point_cloud_back

# Example usage
def main():
    from depth import DepthEstimator
    import os
    from PIL import Image
    import open3d as o3d

    # Initialize estimators and combiners
    depth_estimator = DepthEstimator()
    point_cloud_combiner = PointCloudCombiner()

    # Find input images
    input_dir = "output_views"
    front_image_path = None
    back_image_path = None
    for filename in os.listdir(input_dir):
        if "front_view" in filename:
            front_image_path = os.path.join(input_dir, filename)
        elif "back_view" in filename:
            back_image_path = os.path.join(input_dir, filename)

    if not front_image_path or not back_image_path:
        print("Could not find front and back images.")
        return

    # Preprocess images
    front_image = Image.open(front_image_path).convert("RGB")
    back_image = Image.open(back_image_path).convert("RGB")

    # Estimate depths
    front_input, front_processed = depth_estimator.preprocess_image(front_image)
    back_input, back_processed = depth_estimator.preprocess_image(back_image)

    front_depth = depth_estimator.estimate_depth(front_input)
    back_depth = depth_estimator.estimate_depth(back_input)

    # Create point clouds
    front_point_cloud = point_cloud_combiner.create_point_cloud(front_processed, front_depth)
    back_point_cloud = point_cloud_combiner.create_point_cloud(back_processed, back_depth)

    # Combine point clouds
    combined_point_cloud = point_cloud_combiner.combine_point_clouds(front_point_cloud, back_point_cloud)

    # Save combined point cloud
    o3d.io.write_point_cloud("combined_pointcloud.ply", combined_point_cloud)
    print("Combined point cloud saved.")

if __name__ == "__main__":
    main()