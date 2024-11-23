
def create_image_grid(images):
    """Utility function to create a 2x2 grid of images."""
    images = [Image.fromarray((img * 255).astype("uint8")) for img in images]

    width, height = images[0].size
    grid_img = Image.new("RGB", (2 * width, 2 * height))

    grid_img.paste(images[0], (0, 0))
    grid_img.paste(images[1], (width, 0))
    grid_img.paste(images[2], (0, height))
    grid_img.paste(images[3], (width, height))

    return grid_img

def image_to_mv(image_path):
    """Function to generate multi-view images from an input image."""
    # Load and preprocess the input image
    input_image = Image.open(image_path).convert("RGB")
    input_image = np.array(input_image).astype("float32") / 255.0

    # Generate multi-view images
    outputs = image_pipeline(
        image=input_image,
        guidance_scale=5,
        num_inference_steps=30,
        elevation=0,
    )

    # `outputs` is a numpy array directly; pass it to `create_image_grid`
    grid_img = create_image_grid(outputs)
    return grid_img

# Example Usage
input_image_path = "/content/3D dragon without background.png"  # Replace with the path to your input image

# Generate and save the output
output_grid = image_to_mv(input_image_path)
output_grid.save("multi_view_output.jpg")
output_grid.show()
