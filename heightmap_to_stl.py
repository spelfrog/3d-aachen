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

def create_text_pattern(text, width, height, char_width=8, char_height=10):
    """Creates a 2D pattern for text on the bottom surface."""
    pattern = np.zeros((height, width))

    # Simple character patterns (very basic)
    char_patterns = {
        'A': [[0,1,1,0], [1,0,0,1], [1,1,1,1], [1,0,0,1]],
        'B': [[1,1,1,0], [1,0,0,1], [1,1,1,0], [1,0,0,1], [1,1,1,0]],
        'C': [[0,1,1,1], [1,0,0,0], [1,0,0,0], [0,1,1,1]],
        'D': [[1,1,1,0], [1,0,0,1], [1,0,0,1], [1,1,1,0]],
        'E': [[1,1,1,1], [1,0,0,0], [1,1,1,0], [1,0,0,0], [1,1,1,1]],
        'F': [[1,1,1,1], [1,0,0,0], [1,1,1,0], [1,0,0,0]],
        'G': [[0,1,1,1], [1,0,0,0], [1,0,1,1], [0,1,1,1]],
        'H': [[1,0,0,1], [1,0,0,1], [1,1,1,1], [1,0,0,1]],
        'I': [[1,1,1], [0,1,0], [0,1,0], [1,1,1]],
        'L': [[1,0,0,0], [1,0,0,0], [1,0,0,0], [1,1,1,1]],
        'M': [[1,0,0,1], [1,1,1,1], [1,0,0,1], [1,0,0,1]],
        'N': [[1,0,0,1], [1,1,0,1], [1,0,1,1], [1,0,0,1]],
        'O': [[0,1,1,0], [1,0,0,1], [1,0,0,1], [0,1,1,0]],
        'P': [[1,1,1,0], [1,0,0,1], [1,1,1,0], [1,0,0,0]],
        'R': [[1,1,1,0], [1,0,0,1], [1,1,1,0], [1,0,0,1]],
        'S': [[0,1,1,1], [1,0,0,0], [0,1,1,0], [0,0,0,1], [1,1,1,0]],
        'T': [[1,1,1], [0,1,0], [0,1,0], [0,1,0]],
        'U': [[1,0,0,1], [1,0,0,1], [1,0,0,1], [0,1,1,0]],
        'X': [[1,0,0,1], [0,1,1,0], [0,1,1,0], [1,0,0,1]],
        '0': [[0,1,1,0], [1,0,0,1], [1,0,0,1], [0,1,1,0]],
        '1': [[0,1,0], [1,1,0], [0,1,0], [1,1,1]],
        '2': [[1,1,1,0], [0,0,0,1], [0,1,1,0], [1,1,1,1]],
        '3': [[1,1,1,0], [0,0,0,1], [0,1,1,0], [0,0,0,1], [1,1,1,0]],
        '4': [[1,0,0,1], [1,0,0,1], [1,1,1,1], [0,0,0,1]],
        '5': [[1,1,1,1], [1,0,0,0], [1,1,1,0], [0,0,0,1], [1,1,1,0]],
        '_': [[0,0,0,0], [0,0,0,0], [0,0,0,0], [1,1,1,1]],
        '.': [[0,0], [0,0], [0,0], [1,1]]
    }

    # Place characters
    x_start = 10
    y_start = 10

    for i, char in enumerate(text.upper()):
        if char == ' ':
            x_start += char_width // 2
            continue

        if char in char_patterns:
            char_pattern = char_patterns[char]
            for row_idx, row in enumerate(char_pattern):
                for col_idx, pixel in enumerate(row):
                    y_pos = y_start + row_idx
                    x_pos = x_start + col_idx
                    if 0 <= y_pos < height and 0 <= x_pos < width and pixel:
                        pattern[y_pos, x_pos] = 1

        x_start += char_width
        if x_start >= width - char_width:
            break

    return pattern

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

def heightmap_to_mesh(heightmap, filename, quadrant_name):
    """Converts heightmap to 3D mesh with walls, bottom, and embossed text."""
    height, width = heightmap.shape

    # Calculate base height (rounded)
    base_height = round(np.min(heightmap))
    print(f"Base height: {base_height}")

    # Generate terrain vertices
    vertices = []
    for y in range(height):
        for x in range(width):
            vertices.append([x, y, heightmap[y, x]])

    # Generate bottom vertices with text embossing
    base_filename = os.path.splitext(os.path.basename(filename))[0]
    label_text = f"{width}x{height}m_{base_filename}_{quadrant_name}"
    text_pattern = create_text_pattern(label_text, width, height)

    text_height = 2.0  # Height of embossed text
    for y in range(height):
        for x in range(width):
            z_offset = text_pattern[y, x] * text_height
            vertices.append([x, y, base_height - 5 + z_offset])  # Bottom at base_height - 5

    vertices = np.array(vertices)
    terrain_vertex_count = height * width

    # Generate faces
    faces = []

    # Terrain faces (top surface)
    faces.extend(generate_faces(width, height, 0))

    # Bottom faces (inverted to face downward)
    bottom_faces = generate_faces(width, height, terrain_vertex_count)
    for face in bottom_faces:
        faces.append([face[2], face[1], face[0]])  # Invert face normals

    # Generate walls
    wall_thickness = 5

    # Front wall (y=0)
    for x in range(width - 1):
        v1 = x  # terrain front
        v2 = x + 1  # terrain front
        v3 = terrain_vertex_count + x  # bottom front
        v4 = terrain_vertex_count + x + 1  # bottom front
        faces.extend([[v1, v3, v2], [v2, v3, v4]])

    # Back wall (y=height-1)
    for x in range(width - 1):
        v1 = (height - 1) * width + x  # terrain back
        v2 = (height - 1) * width + x + 1  # terrain back
        v3 = terrain_vertex_count + (height - 1) * width + x  # bottom back
        v4 = terrain_vertex_count + (height - 1) * width + x + 1  # bottom back
        faces.extend([[v1, v2, v3], [v2, v4, v3]])

    # Left wall (x=0)
    for y in range(height - 1):
        v1 = y * width  # terrain left
        v2 = (y + 1) * width  # terrain left
        v3 = terrain_vertex_count + y * width  # bottom left
        v4 = terrain_vertex_count + (y + 1) * width  # bottom left
        faces.extend([[v1, v2, v3], [v2, v4, v3]])

    # Right wall (x=width-1)
    for y in range(height - 1):
        v1 = y * width + (width - 1)  # terrain right
        v2 = (y + 1) * width + (width - 1)  # terrain right
        v3 = terrain_vertex_count + y * width + (width - 1)  # bottom right
        v4 = terrain_vertex_count + (y + 1) * width + (width - 1)  # bottom right
        faces.extend([[v1, v3, v2], [v2, v3, v4]])

    # Create STL mesh
    stl_mesh = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            stl_mesh.vectors[i][j] = vertices[face[j]]

    return stl_mesh

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

def convert_tiff_to_stl(input_path, output_path):
    """Main function for converting TIFF to 4 STL files."""
    heightmap = read_tiff_heightmap(input_path)
    if heightmap is None:
        return False

    print("Preprocessing heightmap to smooth out errors...")
    heightmap = preprocess_heightmap(heightmap)
    print("Preprocessing complete.")

    quadrants = subdivide_heightmap(heightmap)
    for name, quad in quadrants.items():
        quad_output = f"{os.path.splitext(output_path)[0]}_{name}.stl"
        print(f"Creating {name} quadrant...")
        mesh_obj = heightmap_to_mesh(quad, input_path, name)
        mesh_obj.save(quad_output)
        print(f"Saved {quad_output}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Converts TIFF heightmaps to 4 STL files.')
    parser.add_argument('input', help='Path to TIFF input file')
    parser.add_argument('output', help='Base path for STL output files (e.g., output.stl)')
    args = parser.parse_args()

    if not convert_tiff_to_stl(args.input, args.output):
        import sys
        sys.exit(1)

if __name__ == "__main__":
    main()
