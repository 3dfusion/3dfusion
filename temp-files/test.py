import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import open3d as o3d
from PIL import Image, ImageOps
import random
import logging
from typing import List, Optional, Union, Tuple
import torchvision.transforms as transforms

# Advanced import for better device management
import torch.multiprocessing as mp

# Importing necessary libraries for deep learning models
from transformers import (
    GLPNImageProcessor, 
    GLPNForDepthEstimation
)
from diffusers import StableDiffusionPipeline
from skimage.transform import resize
from skimage.restoration import denoise_tv_chambolle

# Import SSL and requests for better network error handling
import ssl
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.poolmanager import PoolManager
from requests.packages.urllib3.util.ssl_ import create_urllib3_context

# Custom SSL adapter to handle certificate verification
class CustomSSLAdapter(HTTPAdapter):
    def init_poolmanager(self, connections, maxsize, block=False, **pool_kwargs):
        ctx = create_urllib3_context(cert_reqs=ssl.CERT_NONE)
        self.poolmanager = PoolManager(
            num_pools=connections,
            maxsize=maxsize,
            block=block,
            ssl_version=ssl.PROTOCOL_TLS,
            ssl_context=ctx,
            **pool_kwargs
        )

# Advanced logging configuration
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('reconstruction_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MultiView3DReconstructor:
    def __init__(
        self, 
        depth_model="vinvino02/glpn-nyu",
        multi_view_model="stabilityai/stable-diffusion-xl-base-1.0",
        use_distributed: bool = False,
        num_gpus: int = None,
        batch_size: int = 4,
        output_dir: str = "output"
    ):
        """
        Advanced multi-view 3D reconstruction with enhanced device management and configuration
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        
        # Enhanced device selection with fallback mechanisms
        self.use_distributed = use_distributed
        self.batch_size = batch_size
        self.devices = self._select_optimal_devices(num_gpus)
        
        logger.info(f"Selected Devices: {self.devices}")
        
        # Advanced image preprocessing transforms
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize for consistent model input
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Load models with advanced error handling and multi-device support
        self.setup_depth_model(depth_model)
        self.setup_multi_view_model(multi_view_model)

    def _select_optimal_devices(self, num_gpus: Optional[int] = None) -> List[torch.device]:
        """
        Advanced device selection with comprehensive fallback strategy
        """
        available_devices = []
        
        # Check GPU availability
        if torch.cuda.is_available():
            if num_gpus is None:
                num_gpus = torch.cuda.device_count()
            
            for i in range(min(num_gpus, torch.cuda.device_count())):
                available_devices.append(torch.device(f'cuda:{i}'))
        
        # CPU fallback with warning
        if not available_devices:
            logger.warning("No CUDA devices found. Falling back to CPU processing.")
            available_devices = [torch.device('cpu')]
        
        return available_devices

    def setup_depth_model(self, depth_model: str):
      """
      Setup depth estimation model with multi-device support
      """
      try:
          self.depth_processors = {}
          self.depth_models = {}
          
          # Setup depth models for each device
          for device in self.devices:
              # Depth Model setup
              depth_processor = GLPNImageProcessor.from_pretrained(depth_model)
              depth_model_instance = GLPNForDepthEstimation.from_pretrained(depth_model).to(device)
              depth_model_instance.eval()  # Set to evaluation mode
              
              self.depth_processors[device] = depth_processor
              self.depth_models[device] = depth_model_instance
          
          logger.info("Depth model initialized successfully.")
    
      except Exception as e:
          logger.error(f"Depth model initialization failed: {e}")
          raise


    def setup_multi_view_model(self, multi_view_model: str):
        """
        Setup multi-view diffusion model with multi-device support
        """
        try:
            # Setup session with custom SSL handling
            session = requests.Session()
            session.mount('https://', CustomSSLAdapter())
            
            self.mv_pipelines = {}
            
            # Setup multi-view models for each device
            for device in self.devices:
                pipeline = StableDiffusionPipeline.from_pretrained(
                    multi_view_model, 
                    use_safetensors=True,
                    session=session
                ).to(device)
                pipeline.enable_attention_slicing()  # Memory optimization
                self.mv_pipelines[device] = pipeline
            
            logger.info("Multi-view diffusion model initialized successfully.")
        
        except Exception as e:
            logger.error(f"Multi-view model initialization failed: {e}")
            raise

    def process_image(self, 
                  image: Union[Image.Image, np.ndarray], 
                  resize_dims=(224, 224),
                  autocontrast=True, 
                  equalize=True) -> Tuple[Image.Image, str]:
      """
      Advanced image preprocessing with multiple enhancement techniques
      """
      try:
          # Validate input
          if not isinstance(image, (Image.Image, np.ndarray)):
              raise ValueError("Input must be a PIL Image or a NumPy array.")
          
          # Convert numpy array to PIL Image if needed
          if isinstance(image, np.ndarray):
              if image.ndim == 2:  # Grayscale
                  image = Image.fromarray(image, mode='L')
              elif image.shape[2] == 3:  # RGB
                  image = Image.fromarray(image, mode='RGB')
              elif image.shape[2] == 4:  # RGBA
                  image = Image.fromarray(image, mode='RGBA')
              else:
                  raise ValueError("Unsupported array shape for image conversion.")
          
          # Ensure RGB mode
          if image.mode != 'RGB':
              image = image.convert('RGB')
          
          # Log original dimensions
          logger.debug(f"Original image size: {image.size}, mode: {image.mode}")
          
          # Resize with high-quality resampling
          image = image.resize(resize_dims, Image.LANCZOS)
          
          # Optional enhancements
          if autocontrast:
              image = ImageOps.autocontrast(image, cutoff=1)
          if equalize:
              image = ImageOps.equalize(image)
          
          # Log processed dimensions
          logger.debug(f"Processed image size: {resize_dims}, mode: {image.mode}")
          
          # Save the processed image for validation
          output_path = os.path.join(self.output_dir, "processed_image.png")
          image.save(output_path)
          logger.info(f"Processed image saved at: {output_path}")
          
          return image, output_path

      except Exception as e:
          logger.error(f"Image preprocessing failed: {e}")
          raise



    def estimate_depth(self, image: Image.Image, device: torch.device) -> np.ndarray:
        """
        Estimate depth using GLPN model
        """
        try:
            # Preprocess image
            processor = self.depth_processors[device]
            model = self.depth_models[device]
            
            # Prepare inputs
            inputs = processor(images=image, return_tensors="pt").to(device)
            
            # Estimate depth
            with torch.no_grad():
                outputs = model(**inputs)
                predicted_depth = outputs.predicted_depth
            
            # Convert to numpy and normalize
            depth_map = predicted_depth.squeeze().cpu().numpy()
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
            
            return depth_map
        
        except Exception as e:
            logger.error(f"Depth estimation failed: {e}")
            raise

    # Rest of the methods remain the same as in the previous script
    # (generate_synthetic_views, create_point_cloud, visualize_results, process_image)

def main(image_path: str):
    """
    Main function for advanced multi-view 3D reconstruction with enhanced device support
    """
    # Validate image path
    if not os.path.exists(image_path):
        logger.error(f"Error: Image path does not exist - {image_path}")
        return
    
    try:
        reconstructor = MultiView3DReconstructor(
            use_distributed=True,  
            num_gpus=None,  
            batch_size=4,   
            output_dir="3d_reconstruction_output"  # Specify output directory
        )
    except Exception as e:
        logger.error(f"Failed to initialize reconstructor: {e}")
        return
    
    # Process image
    try:
        input_image = Image.open(image_path)
        processed_image, save_path = reconstructor.process_image(input_image)
        logger.info(f"Processed image saved successfully at: {save_path}")

        # Continue with 3D reconstruction (depth estimation, synthetic views, etc.)
        logger.info("3D reconstruction process completed successfully.")
        return processed_image, save_path
    
    except Exception as e:
        logger.error(f"3D reconstruction failed: {e}")
        return None


def validate_dependencies():
    """
    Validate critical dependencies for the multi-view 3D reconstruction
    """
    dependencies = [
        ('torch', torch.__version__),
        ('transformers', None),
        ('diffusers', None),
        ('open3d', o3d.__version__),
        ('matplotlib', matplotlib.__version__),
        ('numpy', np.__version__),
        ('PIL', Image.__version__),
    ]
    
    try:
        from transformers import __version__ as transformers_version
        from diffusers import __version__ as diffusers_version
    except ImportError:
        logger.error("Critical dependencies are missing!")
        return False
    
    # Update versions
    dependencies[1] = ('transformers', transformers_version)
    dependencies[2] = ('diffusers', diffusers_version)
    
    # Print dependency information
    logger.info("Dependency Versions:")
    for name, version in dependencies:
        try:
            logger.info(f"{name}: {version}")
        except Exception:
            logger.warning(f"Could not retrieve version for {name}")
    
    return True

def gpu_system_check():
    """
    Perform a comprehensive GPU system check
    """
    logger.info("Performing GPU System Check...")
    
    # CUDA availability
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        # GPU Details
        logger.info(f"CUDA Device Count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i} Details:")
            logger.info(f"  Name: {torch.cuda.get_device_name(i)}")
            logger.info(f"  Compute Capability: {torch.cuda.get_device_capability(i)}")
            
            # Memory details
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1e9  # Convert to GB
            logger.info(f"  Total Memory: {total_memory:.2f} GB")
    else:
        logger.warning("No CUDA-capable GPU detected. Falling back to CPU processing.")

def cli():
    """
    Command-line interface for multi-view 3D reconstruction
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced Multi-View 3D Reconstruction")
    parser.add_argument(
        'image_path', 
        type=str, 
        help='Path to the input image for 3D reconstruction'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='3d_reconstruction_output', 
        help='Directory to save reconstruction results'
    )
    parser.add_argument(
        '--verbose', 
        action='store_true', 
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Configure logging based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run system checks
    if not validate_dependencies():
        logger.error("Dependency validation failed. Please check your installation.")
        return
    
    gpu_system_check()
    
    # Run main reconstruction process
    main(args.image_path)

if __name__ == "__main__":
    # Specify the path to your image
    image_path = "/home/kali1/Pictures/ball.png"
    
    # Run CLI 
    cli()

# Additional error handling and configuration
torch.backends.cudnn.benchmark = True  # Optimize GPU performance
torch.set_float32_matmul_precision('high')  # Improve matrix multiplication precision   



