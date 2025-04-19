import math
import random
from PIL import Image, ImageDraw

def generate_speckle_pattern(randomness=100, speckle_size=5, spacing=10, dpi=300, output_filename="speckle_pattern.png"):
    """
    Generates a speckle pattern for an A4 paper size (210 x 297 mm) at the given DPI,
    with adjustable randomness, speckle size, and grid spacing.

    Parameters:
      randomness (int): 0 (organized grid) to 100 (completely random positions).
      speckle_size (int): Diameter of each speckle, in pixels.
      spacing (int): The center-to-center distance between speckles in the grid (in pixels).
      dpi (int): Printer DPI, used to compute the image size.
      output_filename (str): Filename for the output image.
    """
    # A4 dimensions in inches (210x297 mm = 8.27x11.69 inches)
    A4_width_in = 210 / 25.4
    A4_height_in = 297 / 25.4

    # Convert to pixels using the provided DPI
    width_px = int(dpi * A4_width_in)
    height_px = int(dpi * A4_height_in)

    # Create a white background image
    img = Image.new("L", (width_px, height_px), color=255)
    draw = ImageDraw.Draw(img)

    # Maximum displacement allowed is half the spacing multiplied by (randomness/100)
    max_disp = (spacing / 2.0) * (randomness / 100.0)

    # Loop over the grid.
    # Start at spacing/2 to center the speckle within each grid cell.
    y = spacing / 2.0
    while y < height_px:
        x = spacing / 2.0
        while x < width_px:
            # Compute random perturbations; if randomness is 0, no displacement is applied.
            dx = random.uniform(-max_disp, max_disp)
            dy = random.uniform(-max_disp, max_disp)
            x_pos = x + dx
            y_pos = y + dy

            # Calculate the bounding box for a filled circle (speckle)
            radius = speckle_size / 2.0
            left = x_pos - radius
            top = y_pos - radius
            right = x_pos + radius
            bottom = y_pos + radius

            draw.ellipse([left, top, right, bottom], fill=0)
            x += spacing
        y += spacing

    # Save the generated speckle pattern image
    img.save(output_filename)
    print(f"Speckle pattern saved to {output_filename}")

# Example usage:
if __name__ == "__main__":
    # randomness: 0 = organized grid, 100 = completely random
    # speckle_size: diameter of each speckle in pixels, e.g., 5 pixels
    # spacing: grid spacing in pixels, e.g., 10 pixels between speckles
    # dpi: desired printer DPI for A4 paper, e.g., 300
    generate_speckle_pattern(randomness=70, speckle_size=1, spacing=3, dpi=600, output_filename="speckle_pattern.png")
