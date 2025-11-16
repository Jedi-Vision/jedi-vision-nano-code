# Jedi Spatial Audio

A C++ project for spatial audio processing using Steam Audio, designed to spatialize audio sources based on object detection data from computer vision systems (e.g., YOLO).

## Features

- **3D Spatial Audio**: Uses Steam Audio's binaural rendering to create immersive 3D audio experiences
- **Object Tracking**: Processes object detection data (2D position + depth) and converts to 3D spatial audio
- **Multi-Source Support**: Handles up to 10 simultaneous audio sources with independent spatialization
- **Smooth Interpolation**: Interpolates object positions between video frames for continuous audio playback
- **Custom HRTF**: Supports custom SOFA HRTF files for personalized spatial audio

## Requirements

- CMake 4.1 or higher
- C++ compiler with C++17 support
- [vcpkg](https://vcpkg.io/) for dependency management
- Steam Audio SDK (installed via vcpkg)

## Dependencies

The project uses the following dependencies (managed via vcpkg):

- **steam-audio**: Steam Audio SDK for spatial audio processing
- **fmt**: Fast formatting library
- **nlohmann-json**: JSON parsing library

## Building

1. **Install vcpkg** (if not already installed):
   ```bash
   git clone https://github.com/Microsoft/vcpkg.git
   cd vcpkg
   ./bootstrap-vcpkg.sh  # On macOS/Linux
   ```

2. **Configure CMake with vcpkg**:
   ```bash
   cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=[path to vcpkg]/scripts/buildsystems/vcpkg.cmake
   ```

3. **Build the project**:
   ```bash
   cmake --build build
   ```

The executables will be in the `build/` directory:
- `build/spatial-audio-demo` - Basic spatial audio demo
- `build/object-spatial-audio` - Object tracking spatial audio

## Usage

### Spatial Audio Demo

Basic demo that spatializes a single audio source moving in 3D space:

```bash
./build/spatial-audio-demo [OPTIONS]
```

Options:
- `--hrtf <default|custom>` - Use default or custom HRTF (default: custom)
- `--export-trajectory, -e` - Export trajectory data to CSV file
- `--help, -h` - Show help message

Input: `sample_music.wav` (must be in project root)
Output: `output_binaural.wav`

### Object Spatial Audio

Processes object detection data and generates spatialized audio for multiple objects:

```bash
./build/object-spatial-audio [INPUT_JSON] [OUTPUT_WAV] [OPTIONS]
```

Options:
- `--hrtf <default|custom>` - Use default or custom HRTF (default: custom)
- `--help, -h` - Show help message

**Input JSON Format** (JSONL - one JSON object per line):
```json
{"frame": 0, "timestamp_ms": 0.0, "objects": [{"id": 1, "x": 0.5, "y": 0.5, "depth": 2.0}]}
{"frame": 1, "timestamp_ms": 33.33, "objects": [{"id": 1, "x": 0.52, "y": 0.51, "depth": 2.0}, {"id": 2, "x": 0.3, "y": 0.7, "depth": 3.0}]}
```

Fields:
- `frame`: Frame number (integer)
- `timestamp_ms`: Timestamp in milliseconds (float)
- `objects`: Array of detected objects
  - `id`: Unique object identifier (integer)
  - `x`: Normalized X position [0, 1] (float)
  - `y`: Normalized Y position [0, 1] (float)
  - `depth`: Depth in meters (float)
  - `confidence`: Optional confidence score (float)
  - `class`: Optional class name (string)

**Audio File Mapping**:
- Object ID 1 uses `sample_music.wav`
- Object ID 2 uses `sample_music_2.wav`

**Example**:
```bash
./build/object-spatial-audio sample_detections.json output.wav
```

## Project Structure

```
jedi-spatial-audio/
├── CMakeLists.txt              # CMake build configuration
├── vcpkg.json                  # vcpkg dependencies
├── spatial-audio-demo.cpp      # Basic spatial audio demo
├── object-spatial-audio.cpp    # Object tracking spatial audio
├── sample_detections.json      # Sample object detection data
├── D2_HRIR_SOFA/               # HRTF SOFA files
│   ├── D2_44K_16bit_256tap_FIR_SOFA.sofa
│   ├── D2_48K_24bit_256tap_FIR_SOFA.sofa
│   └── D2_96K_24bit_512tap_FIR_SOFA.sofa
└── build/                      # Build directory (generated)
```

## Technical Details

### Coordinate System

The system uses Steam Audio's right-handed coordinate system:
- **+X**: Right
- **+Y**: Up
- **+Z**: Backward (opposite of forward)

Object detection coordinates (2D + depth) are converted to 3D positions using:
- Camera FOV: 60° horizontal, 45° vertical (configurable)
- Normalized coordinates [0, 1] where (0, 0) is top-left

### Audio Processing

- **Sample Rate**: 44.1 kHz
- **Frame Size**: 1024 samples
- **Video Frame Rate**: 30 FPS
- **Max Simultaneous Objects**: 10

### Spatialization Features

- **Binaural Rendering**: Uses HRTF (Head-Related Transfer Function) for 3D audio
- **Distance Attenuation**: Inverse distance model for realistic volume falloff
- **Smooth Interpolation**: Linear interpolation between detection frames
- **Fade In/Out**: Objects fade in when appearing and fade out when disappearing

## Integration with Object Detection

This project is designed to work with object detection pipelines (e.g., YOLO). The typical workflow:

1. **Object Detection**: Run YOLO or similar on video frames (30 FPS)
2. **Depth Estimation**: Get depth for each detected object
3. **Export Data**: Convert detection results to JSON format
4. **Spatial Audio**: Process JSON with `object-spatial-audio` to generate spatialized audio

## License

[Add your license here]

## Acknowledgments

- [Steam Audio](https://valvesoftware.github.io/steam-audio/) by Valve Software
- [vcpkg](https://vcpkg.io/) by Microsoft

