import numpy as np
from PIL import Image
from stl import mesh
import argparse
import os
import requests
import json

from pathlib import Path
from font import char_patterns

def get_cache_dir():
    """Get or create cache directory."""
    cache_dir = Path.home() / ".cache" / "aachen-3d"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

def download_file_index():
    """Download and cache the file index from OpenGeoData NRW."""
    cache_dir = get_cache_dir()
    index_cache = cache_dir / "file_index.json"

    # Check if cached index exists and is recent (less than 24 hours old)
    if index_cache.exists():
            with open(index_cache, 'r') as f:
                return json.load(f)

    print("Downloading file index from OpenGeoData NRW...")
    url = "https://www.opengeodata.nrw.de/produkte/geobasis/hm/dom1_tiff/dom1_tiff/index.json"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Cache the index
        with open(index_cache, 'w') as f:
            json.dump(data, f)

        print("File index downloaded and cached")
        return data
    except requests.RequestException as e:
        print(f"Error downloading file index: {e}")
        return None

def find_file_by_coordinates(tile_east, tile_north, year=2022):
    """Find the filename for given coordinates."""


    filename = f"dom1_32_{tile_east}_{tile_north}_1_nw_{year}.tif"
    return filename

def download_tiff_file(filename):
    """Download and cache a TIFF file."""
    cache_dir = get_cache_dir()
    cached_file = cache_dir / filename

    # Return cached file if it exists
    if cached_file.exists():
        print(f"Using cached file: {filename}")
        return str(cached_file)

    print(f"Downloading {filename}...")
    base_url = "https://www.opengeodata.nrw.de/produkte/geobasis/hm/dom1_tiff/dom1_tiff/"
    url = base_url + filename

    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        # Save to cache
        with open(cached_file, 'wb') as f:
            f.write(response.content)

        print(f"Downloaded and cached: {filename}")
        return str(cached_file)

    except requests.RequestException as e:
        print(f"Error downloading {filename}: {e}")
        return None

def verify_file_exists(filename):
    """Verify that a file exists in the online index."""
    index_data = download_file_index()
    if not index_data:
        return False

    # Look for the file in the datasets
    for dataset in index_data.get("datasets", []):
        if dataset.get("name") == "dom1_kacheln":
            for file_info in dataset.get("files", []):
                if file_info.get("name") == filename:
                    return True

    return False

def parse_filename_info(filename):
    """
    Parses filename format: dom1_32_294_5628_1_nw_2022.tif
    Returns extracted information as dictionary.
    """
    base_filename = os.path.splitext(os.path.basename(filename))[0]
    parts = base_filename.split('_')

    if len(parts) >= 6:
        return {
            'dataset': parts[0],  # dom1
            'zone': parts[1],     # 32
            'easting': parts[2],  # 294 (in km)
            'northing': parts[3], # 5628 (in km)
            'resolution': parts[4], # 1
            'quadrant': parts[5],   # nw
            'year': parts[6] if len(parts) > 6 else 'unknown'  # 2022
        }
    else:
        # Fallback for non-standard filenames
        return {
            'dataset': 'unknown',
            'zone': 'unknown',
            'easting': 'unknown',
            'northing': 'unknown',
            'resolution': 'unknown',
            'quadrant': 'unknown',
            'year': 'unknown'
        }

def format_coordinates(info):
    """Format coordinate information for display."""
    try:
        easting_km = int(info['easting'])
        northing_km = int(info['northing'])
        return f"E{easting_km}km N{northing_km}km"
    except (ValueError, TypeError):
        return f"E{info['easting']} N{info['northing']}"

def read_tiff_heightmap(filepath):
    """Reads TIFF file and returns height data as numpy array."""
    try:
        image = Image.open(filepath)
        heightmap = np.asarray(image)
        return heightmap[:, :, 0] if len(heightmap.shape) > 2 else heightmap
    except Exception as e:
        print(f"Error reading TIFF file: {e}")
        return None

def create_text_pattern(text_lines, width, height, char_width=16, line_spacing=24):
    """Creates a 2D pattern for multi-line text on the bottom surface."""
    pattern = np.zeros((height, width))

    # Higher resolution character patterns (8x12 pixels per character)

    # Calculate text block dimensions for centering
    max_line_length = max(len(line) for line in text_lines)
    text_block_width = max_line_length * char_width
    text_block_height = len(text_lines) * line_spacing

    # Center the text block
    x_start = max(10, (width - text_block_width) // 2)
    y_start = max(10, (height - text_block_height) // 2)

    for line_idx, line in enumerate(text_lines):
        current_y = y_start + line_idx * line_spacing
        # Left align text within the text block
        current_x = x_start

        for i, char in enumerate(line.upper()):
            if char in char_patterns:
                char_pattern = char_patterns[char]
                for row_idx, row in enumerate(char_pattern):
                    for col_idx, pixel in enumerate(row):
                        y_pos = current_y + row_idx
                        x_pos = current_x + col_idx
                        if 0 <= y_pos < height and 0 <= x_pos < width and pixel:
                            pattern[y_pos, x_pos] = 1

            current_x += char_width
            if current_x >= width - char_width:
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

def heightmap_to_mesh(heightmap, filename, quadrant_name, base_height_override=None):
    """Converts heightmap to 3D mesh with walls, bottom, and embossed text."""
    height, width = heightmap.shape

    # Calculate base height (rounded) or use override
    if base_height_override is not None:
        base_height = base_height_override
        print(f"Using override base height: {base_height}")
    else:
        base_height = round(np.min(heightmap))
        print(f"Calculated base height: {base_height}")

    # Generate terrain vertices
    vertices = []
    for y in range(height):
        for x in range(width):
            vertices.append([x, y, heightmap[y, x]])

    # Parse filename information
    file_info = parse_filename_info(filename)
    coordinates = format_coordinates(file_info)

    # Create multi-line text with parsed information
    text_lines = [
        f"Dataset: {file_info['dataset'].upper()}",
        f"Zone: ETRS89 UTM {file_info['zone']}",
        f"Coords: {coordinates}",
        f"Year: {file_info['year']}",
        f"Quadrant: {quadrant_name}",
        f"Scale: {width}x{height}m",
        f"Base Height: {base_height}m"
    ]

    text_pattern = create_text_pattern(text_lines, width, height)

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

def convert_tiff_to_stl_from_coords(tile_east, tile_north, year=2022, base_height_override=None):
    """Convert TIFF to STL using coordinates instead of filename."""
    # Find the appropriate filename
    filename = find_file_by_coordinates(tile_east, tile_north, year)

    # Verify file exists online
    if not verify_file_exists(filename):
        print(f"Error: File {filename} not found in the online database")
        return False

    # Download the file
    local_file_path = download_tiff_file(filename)
    if not local_file_path:
        return False

    # Determine output path automatically
    base_name = os.path.splitext(os.path.basename(local_file_path))[0]
    output_path = os.path.join(os.getcwd(), base_name + ".stl")

    # Convert using the existing function
    return convert_tiff_to_stl(local_file_path, output_path, base_height_override)

def convert_tiff_to_stl(input_path, output_path, base_height_override=None):
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
        mesh_obj = heightmap_to_mesh(quad, input_path, name, base_height_override)
        mesh_obj.save(quad_output)
        print(f"Saved {quad_output}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Converts TIFF heightmaps to 4 STL files.')

    # Create subparsers for different input methods
    subparsers = parser.add_subparsers(dest='mode', help='Input mode')

    # Coordinate-based input
    coord_parser = subparsers.add_parser('coords', help='Download using coordinates')
    coord_parser.add_argument('tile_east', type=int, help='east tile coordinate (ETRS89 UTM)')
    coord_parser.add_argument('tile_north', type=int, help='north tile coordinate (ETRS89 UTM)')
    coord_parser.add_argument('--year', type=int, default=2022, help='Year (default: 2022)')
    coord_parser.add_argument('--base-height', type=float, help='Override base height (meters)')

    # File-based input (legacy)
    file_parser = subparsers.add_parser('file', help='Use local file')
    file_parser.add_argument('input', help='Path to TIFF input file')
    file_parser.add_argument('--base-height', type=float, help='Override base height (meters)')

    args = parser.parse_args()

    if args.mode == 'coords':
        success = convert_tiff_to_stl_from_coords(
            args.tile_east, args.tile_north,
            args.year, getattr(args, 'base_height', None)
        )
    elif args.mode == 'file':
        # Determine output path automatically
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        output_path = os.path.join(os.getcwd(), base_name + ".stl")
        success = convert_tiff_to_stl(args.input, output_path, getattr(args, 'base_height', None))
    else:
        parser.print_help()
        return

    if not success:
        import sys
        sys.exit(1)

if __name__ == "__main__":
    main()
