import numpy as np
from PIL import Image
from stl import mesh
import argparse
import os

def read_tiff_heightmap(filepath):
    """Reads TIFF file and returns height data as numpy array."""
    try:
        image = Image.open(filepath)
        heightmap = np.array(image)
        return heightmap[:, :, 0] if len(heightmap.shape) > 2 else heightmap
    except Exception as e:
        print(f"Error reading TIFF file: {e}")
        return None

def generate_faces(width, height, offset=0):
    """Generates triangle faces for a grid."""
    faces = []
    for y in range(height - 1):
        for x in range(width - 1):
            v0, v1, v2, v3 = (
                offset + y * width + x,
                offset + y * width + (x + 1),
                offset + (y + 1) * width + x,
                offset + (y + 1) * width + (x + 1),
            )
            faces.extend([[v0, v1, v2], [v1, v3, v2]])
    return faces

def heightmap_to_mesh(heightmap, scale_x, scale_y, scale_z, min_height):
    """Converts heightmap to 3D mesh with walls and bottom."""
    height, width = heightmap.shape


    # Generate vertices
    vertices = [
        [(x * scale_x, y * scale_y, heightmap[y, x] * scale_z) for x in range(width)]
        for y in range(height)
    ]
    vertices += [
        [(x * scale_x, y * scale_y, min_height) for x in range(width)]
        for y in range(height)
    ]
    vertices = np.array(vertices).reshape(-1, 3)

    # Generate faces
    faces = generate_faces(width, height)
    faces += generate_faces(width, height, offset=width * height)  # Bottom faces
    faces += generate_wall_faces(width, height, width * height)

    # Create STL mesh
    stl_mesh = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            stl_mesh.vectors[i][j] = vertices[face[j]]
    return stl_mesh

def generate_wall_faces(width, height, bottom_offset):
    """Generates wall faces for the mesh."""
    faces = []
    for y in range(height - 1):  # Left and right walls
        faces.extend([
            [y * width, bottom_offset + y * width, (y + 1) * width],
            [bottom_offset + y * width, bottom_offset + (y + 1) * width, (y + 1) * width],
            [y * width + (width - 1), (y + 1) * width + (width - 1), bottom_offset + y * width + (width - 1)],
            [bottom_offset + y * width + (width - 1), bottom_offset + (y + 1) * width + (width - 1), (y + 1) * width + (width - 1)],
        ])
    for x in range(width - 1):  # Front and back walls
        faces.extend([
            [x, x + 1, bottom_offset + x],
            [bottom_offset + x, x + 1, bottom_offset + x + 1],
            [(height - 1) * width + x, bottom_offset + (height - 1) * width + x, (height - 1) * width + x + 1],
            [bottom_offset + (height - 1) * width + x, bottom_offset + (height - 1) * width + x + 1, (height - 1) * width + x + 1],
        ])
    return faces

def subdivide_heightmap(heightmap):
    """Subdivides heightmap into 4 quadrants."""
    mid_h, mid_w = heightmap.shape[0] // 2, heightmap.shape[1] // 2
    return {
        'top_left': heightmap[:mid_h, :mid_w],
        'top_right': heightmap[:mid_h, mid_w:],
        'bottom_left': heightmap[mid_h:, :mid_w],
        'bottom_right': heightmap[mid_h:, mid_w:]
    }

def preprocess_heightmap(heightmap):
    """
    Preprocesses the heightmap to smooth out errors.
    If a pixel significantly differs from its 8 neighbors, replace it with the average of the neighbors.
    """
    height, width = heightmap.shape
    processed = heightmap.copy()
    number_of_fixed_pixels = 0
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            # Extract the 8 neighbors
            neighbors = [
                heightmap[y - 1, x - 1], heightmap[y - 1, x], heightmap[y - 1, x + 1],
                heightmap[y, x - 1],                     heightmap[y, x + 1],
                heightmap[y + 1, x - 1], heightmap[y + 1, x], heightmap[y + 1, x + 1]
            ]
            center = heightmap[y, x]
            min_neighbors = np.min(neighbors)
            max_neighbors = np.max(neighbors)

            # abs difference from the average of neighbors
            diff = abs(max_neighbors - min_neighbors)
            if diff > 2.5:
                continue

            avg_neighbors = np.mean(neighbors)

            # Replace the center pixel if it significantly differs from the neighbors
            #if abs(center - avg_neighbors) > 2:
            processed[y, x] = avg_neighbors
            number_of_fixed_pixels += 1

    if number_of_fixed_pixels > 0:
        print(f"Preprocessed {number_of_fixed_pixels} pixels to smooth out errors.")

    return processed

def convert_tiff_to_stl(input_path, output_path, scale_x, scale_y, scale_z, wall_height):
    """Main function for converting TIFF to 4 STL files."""
    heightmap = read_tiff_heightmap(input_path)
    if heightmap is None:
        return False

    print("Preprocessing heightmap to smooth out errors...")
    heightmap = preprocess_heightmap(heightmap)
    print("Preprocessing complete.")

    min_height = np.min(heightmap) * scale_z - wall_height
    min_height = np.floor(min_height)
    print(f"Minimum height to: {min_height}")

    quadrants = subdivide_heightmap(heightmap)
    for name, quad in quadrants.items():
        quad_output = f"{os.path.splitext(output_path)[0]}_{name}.stl"
        print(f"Creating {name} quadrant...")
        mesh = heightmap_to_mesh(quad, scale_x, scale_y, scale_z, min_height)
        mesh.save(quad_output)
        print(f"Saved {quad_output}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Converts TIFF heightmaps to 4 STL files.')
    parser.add_argument('input', help='Path to TIFF input file')
    parser.add_argument('output', help='Base path for STL output files (e.g., output.stl)')
    parser.add_argument('--scale-x', type=float, default=1.0, help='X-axis scaling (default: 1.0)')
    parser.add_argument('--scale-y', type=float, default=1.0, help='Y-axis scaling (default: 1.0)')
    parser.add_argument('--scale-z', type=float, default=1.0, help='Z-axis scaling for height (default: 1.0)')
    parser.add_argument('--wall-height', type=float, default=6.0, help='Height of walls below terrain (default: 10.0)')
    args = parser.parse_args()

    if not convert_tiff_to_stl(args.input, args.output, args.scale_x, args.scale_y, args.scale_z, args.wall_height):
        sys.exit(1)

if __name__ == "__main__":
    main()
