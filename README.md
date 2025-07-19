# TIFF to STL Converter

This program converts TIFF heightmaps to 3D STL files. The input is automatically subdivided into 4 quadrants, each saved as a separate STL file with walls, bottom, and embossed text labels.

## Features

- Converts TIFF heightmaps to 4 separate STL files (quadrants)
- Preprocesses heightmaps to smooth out pixel errors
- Creates complete 3D models with:
  - Terrain surface from heightmap
  - Walls around the perimeter
  - Bottom surface with embossed text showing:
    - Scale dimensions (e.g., 500x500m)
    - Original filename
    - Quadrant section (top_left, top_right, bottom_left, bottom_right)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python heightmap_to_stl.py input.tiff output.stl
```

### Example

```bash
python heightmap_to_stl.py heightmap.tiff terrain.stl
```

This will create 4 files:
- terrain_top_left.stl
- terrain_top_right.stl  
- terrain_bottom_left.stl
- terrain_bottom_right.stl

Each file contains a complete 3D model with walls, bottom, and embossed text identification suitable for 3D printing.
import numpy as np
from PIL import Image
from stl import mesh
import argparse
import sys

def read_tiff_heightmap(filepath):
    """Reads TIFF file and returns height data as numpy array."""
    try:
        image = Image.open(filepath)
        # Convert to numpy array and normalize if necessary
        heightmap = np.array(image)
        
        # If RGB/RGBA, take only one channel
        if len(heightmap.shape) > 2:
            heightmap = heightmap[:, :, 0]
            
        return heightmap
    except Exception as e:
        print(f"Error reading TIFF file: {e}")
        return None

def heightmap_to_mesh(heightmap, scale_x=1.0, scale_y=1.0, scale_z=1.0):
    """Converts heightmap to 3D mesh."""
    height, width = heightmap.shape
    
    # Create vertices for each pixel
    vertices = []
    faces = []
    
    # Generate vertices
    for y in range(height):
        for x in range(width):
            # Pixel coordinates to world coordinates
            world_x = x * scale_x
            world_y = y * scale_y
            world_z = heightmap[y, x] * scale_z
            vertices.append([world_x, world_y, world_z])
    
    vertices = np.array(vertices)
    
    # Generate triangle faces
    for y in range(height - 1):
        for x in range(width - 1):
            # Vertex indices for current quad
            v0 = y * width + x
            v1 = y * width + (x + 1)
            v2 = (y + 1) * width + x
            v3 = (y + 1) * width + (x + 1)
            
            # Two triangles per quad
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])
    
    faces = np.array(faces)
    
    # Create STL mesh
    stl_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            stl_mesh.vectors[i][j] = vertices[face[j], :]
    
    return stl_mesh

def convert_tiff_to_stl(input_path, output_path, scale_x=1.0, scale_y=1.0, scale_z=1.0):
    """Main function for converting TIFF to STL."""
    print(f"Reading TIFF file: {input_path}")
    heightmap = read_tiff_heightmap(input_path)
    
    if heightmap is None:
        return False
    
    print(f"Heightmap read: {heightmap.shape[1]}x{heightmap.shape[0]} pixels")
    print(f"Height range: {heightmap.min()} - {heightmap.max()}")
    
    print("Creating 3D mesh...")
    stl_mesh = heightmap_to_mesh(heightmap, scale_x, scale_y, scale_z)
    
    print(f"Saving STL file: {output_path}")
    stl_mesh.save(output_path)
    
    print(f"Conversion complete! Mesh has {len(stl_mesh.vectors)} triangles.")
    return True

def main():
    parser = argparse.ArgumentParser(description='Converts TIFF heightmaps to STL files')
    parser.add_argument('input', help='Path to TIFF input file')
    parser.add_argument('output', help='Path to STL output file')
    parser.add_argument('--scale-x', type=float, default=1.0, help='X-axis scaling (default: 1.0)')
    parser.add_argument('--scale-y', type=float, default=1.0, help='Y-axis scaling (default: 1.0)')
    parser.add_argument('--scale-z', type=float, default=1.0, help='Z-axis scaling for height (default: 1.0)')
    
    args = parser.parse_args()
    
    success = convert_tiff_to_stl(
        args.input, 
        args.output, 
        args.scale_x, 
        args.scale_y, 
        args.scale_z
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
