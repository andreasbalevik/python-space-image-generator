import numpy as np
import matplotlib.pyplot as plt
import os
import random
import shutil
import cv2
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Generate nebula-like images.")
parser.add_argument("-num", "--number", type=int, default=1, help="Number of images to generate")
args = parser.parse_args()
num_images = args.number  # Use the number from the command-line argument

# Feature toggles
bayer_filter_enabled = True  # Enable Bayer filter effect
leica_filter_enabled = True  # Enable Leica color processing effect
blur_filter_enabled = False  # Enable blur filter effect
display_images_enabled = False  # Enable displaying images
center_planet_enabled = True  # Enable centering the planet

# Configurable variables
image_dimensions = (400, 400)
height, width = image_dimensions
star_count_range = (500, 1500)
output_folder = 'generated-images'

# Nebula color configuration
nebula_color_factors = {
    "red_factor": (0.1, 1.0),
    "green_factor": (0.1, 1.0),
    "blue_factor": (0.1, 1.0)
}

# General configuration for celestial bodies
celestial_bodies = {
    "planet": {
        "probability": 0.99,
        "color_range": (0, 255),
        "size_range": (height // 20, height // 2),
        "outline_thickness": 5,
        "element_probability": {"land": 0.9, "water": 0.1},
        "element_amount_range": {"land": (10, 30), "water": (100, 200)},
        "water_color_range": (100, 200)
    },
    "moon": {
        "probability": 0.33,
        "color_range": (0, 255),
        "size_ratio": 0.3,
        "min_size": 5,
        "outline_thickness": 5,
        "element_probability": {"land": 0.7, "water": 0.1},
        "element_amount_range": {"land": (10, 30), "water": (100, 200)},
        "water_color_range": (100, 200)  # Added water_color_range for moon
    },
    "asteroid": {
        "probability": 0.66,  # Probability of asteroids appearing around a planet
        "color_range": (100, 200),  # Color range of asteroids
        "size_range": (1, 20),  # Size range of asteroids
        "count_range": (1, 7),  # Range of the number of asteroids
        "distance_from_planet_range": (2, 5.1)  # Distance range from the planet (as a multiple of the planet's radius)
    }
}

distant_planets_config = {
    "probability": 0.50,  # Chance to add a distant planet
    "color": (255, 255, 255),  # Bright white color for distant planets
    "size_range": (2, 4),  # Size range (slightly bigger than stars)
    "count_range": (1,5)  # Number of distant planets to add
}

atmosphere_config = {
    "probability": 0.05,
    "color_range": {"white": [255, 255, 255], "silver": [192, 192, 192]},  # White to Silver color range
    "thickness_range": (3, 7)  # Thickness range of the atmosphere
}

def apply_black_and_white_filter(image):
    # Use the luminosity method to convert to grayscale
    grayscale_image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    # Stack the grayscale values to create a three-channel image
    return np.stack((grayscale_image,) * 3, axis=-1)

def add_distant_planets(img, width, height):
    if np.random.rand() < distant_planets_config["probability"]:
        num_planets = np.random.randint(*distant_planets_config["count_range"])
        for _ in range(num_planets):
            planet_size = np.random.randint(*distant_planets_config["size_range"])
            planet_x = np.random.randint(0, width)
            planet_y = np.random.randint(0, height)
            cv2.circle(img, (planet_x, planet_y), planet_size, distant_planets_config["color"], -1)


def add_atmosphere(img, center, radius, width, height):
    if np.random.rand() < atmosphere_config["probability"]:
        atmosphere_thickness = np.random.randint(*atmosphere_config["thickness_range"])

        # Interpolate between white and silver colors
        t = np.random.rand()
        atmosphere_color = (1 - t) * np.array(atmosphere_config["color_range"]["white"]) + t * np.array(atmosphere_config["color_range"]["silver"])

        for y in range(-radius - atmosphere_thickness, radius + atmosphere_thickness):
            for x in range(-radius - atmosphere_thickness, radius + atmosphere_thickness):
                dist_squared = x**2 + y**2
                if radius**2 < dist_squared <= (radius + atmosphere_thickness)**2:
                    coord_y = center[1] + y
                    coord_x = center[0] + x
                    if 0 <= coord_y < height and 0 <= coord_x < width:
                        distance_from_edge = np.sqrt(dist_squared) - radius
                        opacity_factor = 1 - distance_from_edge / atmosphere_thickness
                        img[coord_y, coord_x] = np.clip(opacity_factor * atmosphere_color + (1 - opacity_factor) * img[coord_y, coord_x], 0, 255)

def create_asteroids(img, planet_center, planet_radius, elements, width, height, light_source_angle):
    asteroid_config = elements["asteroid"]
    if np.random.rand() < asteroid_config["probability"]:
        num_asteroids = np.random.randint(*asteroid_config["count_range"])
        for _ in range(num_asteroids):
            asteroid_size = np.random.randint(*asteroid_config["size_range"])
            asteroid_color = np.random.randint(*asteroid_config["color_range"], size=3)

            # Each asteroid gets its own distance factor
            distance_factor = np.random.uniform(*asteroid_config["distance_from_planet_range"])
            angle = np.random.uniform(0, 2 * np.pi)
            distance_from_planet = planet_radius * distance_factor
            asteroid_center = (
                int(planet_center[0] + distance_from_planet * np.cos(angle)),
                int(planet_center[1] + distance_from_planet * np.sin(angle))
            )

            # Draw the asteroid
            if 0 <= asteroid_center[0] < width and 0 <= asteroid_center[1] < height:
                draw_circle_with_outline(img, asteroid_center, asteroid_size, asteroid_color, asteroid_color, 1, width, height)
                
            # Add texture to asteroids
            for dy in range(-asteroid_size, asteroid_size):
                for dx in range(-asteroid_size, asteroid_size):
                    if dx**2 + dy**2 <= asteroid_size**2:
                        asteroid_pixel = (
                            int(asteroid_center[0] + dx),
                            int(asteroid_center[1] + dy)
                        )
                        if 0 <= asteroid_pixel[0] < width and 0 <= asteroid_pixel[1] < height:
                            texture_color_variation = np.random.randint(-20, 20, size=3)
                            texture_color = np.clip(asteroid_color + texture_color_variation, 0, 255)
                            img[asteroid_pixel[1], asteroid_pixel[0]] = texture_color

            # Add shadow to asteroids
            add_asteroid_shadow(img, asteroid_center, asteroid_size, width, height, light_source_angle)

def draw_circle_with_outline(img, center, radius, color, outline_color, outline_thickness, width, height):
    for y in range(-radius, radius):
        for x in range(-radius, radius):
            dist_squared = x**2 + y**2
            if dist_squared <= radius**2:
                coord_y = center[1] + y
                coord_x = center[0] + x
                if 0 <= coord_y < height and 0 <= coord_x < width:
                    # Check if pixel is within the outline boundary
                    if radius - outline_thickness <= np.sqrt(dist_squared) <= radius:
                        img[coord_y, coord_x] = outline_color
                    else:
                        # Calculate distance from the edge
                        edge_distance = radius - np.sqrt(dist_squared)
                        # Create a gradient effect for the edge
                        edge_factor = edge_distance / radius
                        pixel_color = (color * edge_factor + outline_color * (1 - edge_factor)).astype(int)
                        img[coord_y, coord_x] = pixel_color

def add_asteroid_shadow(img, center, radius, width, height, light_source_angle):
    shadow_intensity = 0.6  # Adjust this for darker/lighter shadows
    for y in range(-radius, radius):
        for x in range(-radius, radius):
            if x**2 + y**2 <= radius**2:
                coord_y = center[1] + y
                coord_x = center[0] + x
                if 0 <= coord_y < height and 0 <= coord_x < width:
                    angle_to_light = np.arctan2(y, x) + light_source_angle
                    distance_from_center = np.sqrt(x**2 + y**2)
                    angle_intensity = (np.cos(angle_to_light) + 1) / 2
                    distance_intensity = 1 - np.clip(distance_from_center / radius, 0, 1)
                    shadow_intensity_factor = angle_intensity * distance_intensity * shadow_intensity
                    img[coord_y, coord_x] = np.clip(img[coord_y, coord_x] * shadow_intensity_factor, 0, 255).astype(np.uint8)


def generate_surface_noise(size):
    """Generate layered noise for planetary surface texturing.

    The previous implementation assumed the intermediate noise arrays would
    always have a positive size. When the celestial body radius was small and
    the frequency grew larger than the dimensions, the calculated shape became
    ``(0, 0)`` which caused ``cv2.resize`` to raise an assertion error. To avoid
    this we clamp the intermediate dimensions to at least 1x1 before resizing.
    """

    noise = np.zeros((size * 2, size * 2), dtype=np.float32)
    frequency = np.random.uniform(2.0, 5.0)
    for _ in range(4):
        small_dim = max(1, int((size * 2) / frequency))
        small_noise = np.random.rand(small_dim, small_dim).astype(np.float32)
        small_noise = cv2.resize(small_noise, (size * 2, size * 2))
        noise += small_noise / frequency
        frequency *= 2

    if noise.max() > noise.min():
        noise = (noise - noise.min()) / (noise.max() - noise.min())
    return noise

def create_celestial_body(img, body_type, center, size, elements, width, height):
    body_config = elements[body_type]
    base_color = np.random.randint(*body_config["color_range"], size=3)
    outline_color = base_color
    outline_thickness = body_config["outline_thickness"]

    draw_circle_with_outline(img, center, size, base_color, outline_color, outline_thickness, width, height)

    surface_noise = generate_surface_noise(size)

    terrain_types = {
        "land": {"color_variation": (-20, 20), "probability": 0.6},
        "mountain": {"color_variation": (-30, 30), "probability": 0.15},
        "crater": {"color_variation": (-40, -20), "probability": 0.05},
        "desert": {"color_variation": (10, 30), "probability": 0.1},
        "ice": {"color_variation": (-10, 10), "probability": 0.05},
        "forest": {"color_variation": (-20, 20), "probability": 0.05},
        "volcanic": {"color_variation": (-20, 20), "probability": 0.05},
        "river": {"color_variation": (-10, 10), "probability": 0.05}
    }

    patch_size = 1  # Size of each terrain patch

    for y in range(-size + outline_thickness, size - outline_thickness, patch_size):
        for x in range(-size + outline_thickness, size - outline_thickness, patch_size):
            if x**2 + y**2 <= (size - outline_thickness)**2:
                # Randomly select terrain type
                terrain_type = random.choices(
                    list(terrain_types.keys()),
                    weights=[terrain_types[t]["probability"] for t in terrain_types],
                    k=1)[0]
                
                # Calculate color variation based on base color
                color_variation_range = terrain_types[terrain_type]["color_variation"]
                color_variation = np.random.randint(*color_variation_range, size=3)
                patch_color = np.clip(base_color + color_variation, 0, 255)

                # Draw the terrain patch
                for dy in range(patch_size):
                    for dx in range(patch_size):
                        coord_y = center[1] + y + dy
                        coord_x = center[0] + x + dx
                        if 0 <= coord_y < height and 0 <= coord_x < width and (dx**2 + dy**2 <= (size - outline_thickness)**2):
                            noise_val = surface_noise[y + size + dy, x + size + dx]
                            varied_color = np.clip(patch_color * (0.7 + 0.3 * noise_val), 0, 255)
                            img[coord_y, coord_x] = varied_color

    # Add glow effect
    glow_color = np.array([255, 255, 180])  # Soft yellow-white glow
    glow_intensity = 0.05  # Adjust this value as needed
   
    add_glow(img, center, size, glow_color, glow_intensity, width, height)
    add_atmosphere(img, center, size, width, height)

def generate_nebula_like_image(dimensions, star_count_range, elements, color_factors):
    width, height = dimensions
    base_noise = np.random.standard_normal([height, width, 3])
    base_noise = (base_noise - base_noise.min()) / (base_noise.max() - base_noise.min())

    # Randomize the color factors for each image
    red_factor = np.random.uniform(0.2, 1.0)
    green_factor = np.random.uniform(0.2, 1.0)
    blue_factor = np.random.uniform(0.2, 1.0)

    nebula_colors = base_noise * np.array([red_factor, green_factor, blue_factor]) * 255
    nebula_colors = nebula_colors.astype(np.uint8)
    
    # Add distant planets to the image
    add_distant_planets(nebula_colors, width, height)
    
    for _ in range(np.random.randint(*star_count_range)):
        x, y = np.random.randint(0, width), np.random.randint(0, height)
        nebula_colors[y, x] = [255, 255, 255]

    planet_center = None
    planet_radius = None
    
    max_planet_radius = celestial_bodies["planet"]["size_range"][1]

    for body_type in elements:
        if np.random.rand() < elements[body_type]["probability"]:
            if body_type == "planet":
                planet_radius = np.random.randint(*celestial_bodies[body_type]["size_range"])
                if center_planet_enabled:
                    planet_center = (width // 2, height // 2)
                else:
                    planet_center = generate_planet_center(width, height, planet_radius, max_planet_radius)
                light_source_angle = closest_corner_angle(planet_center, width, height)
                create_celestial_body(nebula_colors, body_type, planet_center, planet_radius, celestial_bodies, width, height)
                add_shadow(nebula_colors, planet_center, planet_radius, width, height, light_source_angle)
                create_asteroids(nebula_colors, planet_center, planet_radius, elements, width, height, light_source_angle)
            elif body_type == "moon" and planet_center is not None:
                # Adjusted moon position calculation for more variability
                distance_factor = np.random.uniform(1.5, 3)  # Increase this range for more variability
                angle = np.random.uniform(0, 2 * np.pi)
                moon_distance = planet_radius * distance_factor
                moon_center = (
                    np.clip(int(planet_center[0] + moon_distance * np.cos(angle)), 0, width),
                    np.clip(int(planet_center[1] + moon_distance * np.sin(angle)), 0, height)
                )
                moon_radius = max(int(planet_radius * elements[body_type]["size_ratio"]), elements[body_type]["min_size"])
                create_celestial_body(nebula_colors, body_type, moon_center, moon_radius, elements, width, height)
                add_shadow(nebula_colors, moon_center, moon_radius, width, height, light_source_angle)
        
        # Apply vignette effect centered on the planet
        if planet_center is not None:
            apply_vignette(nebula_colors, planet_center, width, height)
    
    image = nebula_colors
    
    if bayer_filter_enabled:        
        image = apply_bayer_filter(nebula_colors)
        
    if leica_filter_enabled:
        image = apply_leica_color_processing(image)
        
    if blur_filter_enabled:
        image = cv2.GaussianBlur(image, (5, 5), 0)
            
    return image

def apply_vignette(img, center, width, height):
    max_distance = np.sqrt(width**2 + height**2) / 2
    vignette_intensity = np.random.uniform(0.25, 0.65)  # Random density

    for y in range(height):
        for x in range(width):
            distance = np.sqrt((center[0] - x)**2 + (center[1] - y)**2)
            vignette_factor = 1 - (distance / max_distance) * vignette_intensity
            # Correctly apply the vignette effect
            img[y, x] = np.clip(img[y, x] * vignette_factor, 0, 255).astype(np.uint8)

def apply_color_boost(img, intensity=1.1):
    # Convert image to float to prevent clipping during operations
    img_float = img.astype(np.float32)
    # Boost the color by increasing the saturation
    img_hsv = cv2.cvtColor(img_float, cv2.COLOR_RGB2HSV)
    img_hsv[:, :, 1] *= intensity
    img_hsv = np.clip(img_hsv, 0, 255)
    img_boosted = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return img_boosted

def add_shadow(img, center, radius, width, height, light_source_angle):
    light_dir = np.array([np.cos(light_source_angle), np.sin(light_source_angle)])
    for y in range(-radius, radius):
        for x in range(-radius, radius):
            if x**2 + y**2 <= radius**2:
                coord_y = center[1] + y
                coord_x = center[0] + x
                if 0 <= coord_y < height and 0 <= coord_x < width:
                    normal = np.array([x, y]) / radius
                    intensity = np.dot(normal, light_dir)
                    shade = 0.5 + 0.5 * intensity
                    shade += np.random.uniform(-0.05, 0.05)
                    shade = np.clip(shade, 0, 1)
                    img[coord_y, coord_x] = np.clip(img[coord_y, coord_x] * shade, 0, 255)

def closest_corner_angle(center, width, height):
    # Define the corners of the image
    corners = [(0, 0), (width, 0), (0, height), (width, height)]

    # Find the corner closest to the planet center
    closest_corner = min(corners, key=lambda corner: np.hypot(center[0] - corner[0], center[1] - corner[1]))

    # Calculate the angle to the closest corner
    angle = np.arctan2(closest_corner[1] - center[1], closest_corner[0] - center[0])
    return angle

def add_glow(img, center, radius, glow_color, intensity, width, height):
    extended_radius = radius * 2  # Extend the glow to twice the radius
    for y in range(-extended_radius, extended_radius):
        for x in range(-extended_radius, extended_radius):
            dist_squared = x**2 + y**2
            if dist_squared <= extended_radius**2:
                coord_y = center[1] + y
                coord_x = center[0] + x
                if 0 <= coord_y < height and 0 <= coord_x < width:
                    distance = np.sqrt(dist_squared)
                    if distance <= radius:
                        continue
                    # Adjust glow factor for the extended range
                    glow_factor = intensity * (1 - (distance - radius) / radius)
                    img[coord_y, coord_x] = np.clip(img[coord_y, coord_x] + glow_color * glow_factor, 0, 255)
                    
def calculate_exceedance(radius, max_radius, max_exceedance=20):
    """
    Calculate exceedance percentage based on the planet's radius.
    :param radius: Radius of the planet.
    :param max_radius: Maximum possible radius of a planet.
    :param max_exceedance: Maximum exceedance percentage.
    :return: Exceedance percentage.
    """
    exceedance_percentage = (radius / max_radius) * max_exceedance
    return min(exceedance_percentage, max_exceedance)

def generate_planet_center(width, height, radius, max_radius):
    exceedance_percentage = calculate_exceedance(radius, max_radius)
    exceedance_width = int(width * exceedance_percentage / 100)
    exceedance_height = int(height * exceedance_percentage / 100)

    x_center = np.random.randint(-exceedance_width, width + exceedance_width)
    y_center = np.random.randint(-exceedance_height, height + exceedance_height)

    return (x_center, y_center)

def save_image(image, folder=output_folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    sequence_number = ''.join([str(random.randint(0, 9)) for _ in range(6)])
    file_name = f"transmission_{sequence_number}.png"
    path = os.path.join(folder, file_name)
    
    # Save the image using OpenCV
    cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))  # Convert to BGR format as OpenCV uses BGR

    print(f"Image saved as {path}")

def apply_bayer_filter(image):
    height, width, _ = image.shape
    for y in range(height):
        for x in range(width):
            if (x % 2 == 0) and (y % 2 == 0):  # Green pixel
                image[y, x, [0, 2]] = 0  # Zero out red and blue
            elif (x % 2 == 1) and (y % 2 == 0):  # Red pixel
                image[y, x, [1, 2]] = 0  # Zero out green and blue
            elif (x % 2 == 0) and (y % 2 == 1):  # Blue pixel
                image[y, x, [0, 1]] = 0  # Zero out red and green
            # Else, another green pixel (as green pixels are more frequent)
    
    # Apply a color enhancement to mimic the deep and rich colors
    enhanced_image = np.clip(image * 1.5, 0, 255).astype(np.uint8)

    return enhanced_image

def apply_leica_color_processing(image):
    # Convert image to a floating point type for manipulation
    processed_image = image.astype(np.float32)

    # Increase saturation and contrast
    saturation_factor = 1.6  # Increase this to boost colors more
    contrast_factor = 8.1    # Increase this to make the image more contrasty

    # Convert to HSV to adjust saturation
    hsv_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2HSV)
    hsv_image[:, :, 1] *= saturation_factor  # Increase saturation
    hsv_image = np.clip(hsv_image, 0, 255)

    processed_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

    # Adjust contrast
    mean_intensity = np.mean(processed_image)
    processed_image = contrast_factor * (processed_image - mean_intensity) + mean_intensity
    processed_image = np.clip(processed_image, 0, 255)

    # Apply a warm color tone by adjusting the red and green channels
    warm_factor = 1.05  # Increase this for a warmer effect
    processed_image[:, :, 0] *= warm_factor  # Red channel
    processed_image[:, :, 1] *= warm_factor  # Green channel
    processed_image = np.clip(processed_image, 0, 255)

    return processed_image.astype(np.uint8)

def simple_demosaic(image):
    return cv2.GaussianBlur(image, (1, 1), 0)  # Increased blur to simulate a plastic lens effect

def clear_folder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

def generate_and_save_images(num_images):
    for _ in range(num_images):
        img = generate_nebula_like_image(image_dimensions, star_count_range, celestial_bodies, nebula_color_factors)
        plt.imshow(img, interpolation='none')
        
        if display_images_enabled:
            plt.show()  # Uncomment this if you want to display each image
        save_image(img)

clear_folder(output_folder)
generate_and_save_images(num_images)
