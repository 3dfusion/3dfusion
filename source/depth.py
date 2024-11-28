import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.transform import resize
from transformers import DPTForDepthEstimation, DPTImageProcessor

class DepthEstimator:
    def __init__(self, model_name="Intel/dpt-large"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.model = DPTForDepthEstimation.from_pretrained(model_name).to(self.device)
        self.processor = DPTImageProcessor.from_pretrained(model_name)

    def preprocess_image(self, image, target_height=384):
        """
        Preprocess image for depth estimation
        
        Args:
            image (PIL.Image): Input image
            target_height (int): Target height for resizing
        
        Returns:
            tuple: Processed tensor and resized PIL image
        """
        width, height = image.size
        aspect_ratio = width / height
        new_width = int(target_height * aspect_ratio)
        new_size = (new_width, target_height)
        image = image.resize(new_size, Image.LANCZOS)
        
        inputs = self.processor(images=image, return_tensors="pt")
        input_image = inputs['pixel_values'].to(self.device)
        return input_image, image

    def estimate_depth(self, input_image):
        """
        Estimate depth from input image tensor
        
        Args:
            input_image (torch.Tensor): Preprocessed image tensor
        
        Returns:
            numpy.ndarray: Normalized depth map
        """
        with torch.no_grad():
            outputs = self.model(input_image)
            predicted_depth = outputs.predicted_depth
            
            predicted_depth = predicted_depth.squeeze().cpu().numpy()
            depth_map = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())
            
            return depth_map

    def save_depth_map(self, depth_map, output_path):
        """
        Save depth map as an image
        
        Args:
            depth_map (numpy.ndarray): Depth map to save
            output_path (str): Path to save depth map image
        """
        plt.figure(figsize=(10, 10))
        plt.imshow(depth_map, cmap='viridis')
        plt.axis('off')
        plt.colorbar()
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Depth map saved to: {output_path}")

# Example usage
def main():
    from multiview import generate_front_back_views
    
    # Generate front and back views first
    input_image_path = "path/to/your/input_image.png"  # Replace with your image path
    generate_front_back_views(input_image_path)
    
    # Now estimate depths for those views
    depth_estimator = DepthEstimator()
    
    # Process front view
    front_image_path = f"output_views/{os.path.splitext(os.path.basename(input_image_path))[0]}_front_view.png"
    front_image = Image.open(front_image_path).convert("RGB")
    front_input, front_processed = depth_estimator.preprocess_image(front_image)
    front_depth = depth_estimator.estimate_depth(front_input)
    depth_estimator.save_depth_map(front_depth, 'front_depth.png')
    
    # Process back view
    back_image_path = f"output_views/{os.path.splitext(os.path.basename(input_image_path))[0]}_back_view.png"
    back_image = Image.open(back_image_path).convert("RGB")
    back_input, back_processed = depth_estimator.preprocess_image(back_image)
    back_depth = depth_estimator.estimate_depth(back_input)
    depth_estimator.save_depth_map(back_depth, 'back_depth.png')

if __name__ == "__main__":
    main()
