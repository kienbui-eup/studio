# ComfyUI-Lovis-Node

Custom nodes for text and audio processing in ComfyUI.

**Part of**: [ComfyUI-Workflow-Sora2Alike-Full-loop-video](https://github.com/lovis93/ComfyUI-Workflow-Sora2Alike-Full-loop-video)

---

## Nodes

### 1. Line Count Node
Counts the number of lines in a text input.

### 2. Text to Single Line Node  
Converts multi-line text to single line text by replacing line breaks with customizable separators (perfect for converting poetry or multi-line text to prose format).

### 3. Audio Duration Node
Analyzes audio data from other nodes to extract duration information in seconds or frame count based on FPS parameter.

## Features

### Line Count Node:
- Count total lines in text input
- Option to include or exclude empty lines
- Returns both line count (as integer) and summary text
- Clean, simple interface

### Text to Single Line Node:
- Convert multi-line text to single line
- Customizable separator (default: ". ")
- Option to trim whitespace from lines
- Option to skip empty lines
- Perfect for converting poetry to prose format

### Audio Duration Node:
- Extract duration from audio data in seconds
- Calculate frame count based on customizable FPS parameter
- Accepts audio input from other ComfyUI nodes
- Compatible with ComfyUI audio format (waveform + sample rate)
- Automatic tensor/numpy conversion handling
- Returns duration, frame count, and summary information

## Installation

1. Clone this repository into your ComfyUI custom nodes directory:
   ```bash
   cd /path/to/your/ComfyUI/custom_nodes/
   git clone https://github.com/yourusername/comfyui-node-count-line.git
   ```

2. Install dependencies (optional for Audio Duration Node fallback):
   ```bash
   cd comfyui-node-count-line
   pip install numpy  # Required for audio processing
   pip install librosa  # Optional: for fallback audio processing
   ```

3. Restart ComfyUI to load the new nodes.

## Usage

### Line Count Node
1. In ComfyUI, look for the "Line Count" node in the `text/utility` category
2. Connect a text input to the node
3. Configure the "include_empty_lines" parameter

### Text to Single Line Node
1. In ComfyUI, look for the "Text to Single Line" node in the `text/utility` category
2. Connect a text input to the node
3. Configure the parameters:
   - **separator**: What to put between lines (default: ". ")
   - **trim_lines**: Remove whitespace from each line
   - **skip_empty_lines**: Skip empty lines in the output

### Audio Duration Node
1. In ComfyUI, look for the "Audio Duration" node in the `audio/utility` category
2. Connect an audio output from another node to the audio input
3. Configure the parameters:
   - **fps**: Frames per second for frame count calculation (default: 30.0)
   - **output_format**: Choose between "duration_seconds", "frame_count", or "both"
4. The node outputs:
   - Duration in seconds (FLOAT)
   - Frame count at specified FPS (INT)  
   - Summary information (STRING)

## Examples

### Text to Single Line Node Example:

**Input:**
```
Une orange sur la table  
Ta robe sur le tapis  
Et toi dans mon lit  
Doux présent du présent  
Fraîcheur de la nuit  
Chaleur de ma vie.
```

**Output** (with default settings):
```
Une orange sur la table. Ta robe sur le tapis. Et toi dans mon lit. Doux présent du présent. Fraîcheur de la nuit. Chaleur de ma vie.
```

### Line Count Node Example:

**Input text:**
```
Hello world
This is a test

Another line
```

**With include_empty_lines = True:**
- line_count: `4`
- summary: `"Total lines: 4 (including empty lines)"`

**With include_empty_lines = False:**
- line_count: `3`
- summary: `"Non-empty lines: 3 (total: 4, empty: 1)"`

### Audio Duration Node Example:

**Input audio file:** `music.mp3` (120 seconds long)
**FPS parameter:** `24.0`
**Output format:** `"both"`

**Outputs:**
- duration_seconds: `120.0`
- frame_count: `2880` (120 seconds × 24 fps)
- info: `"Duration: 120.000s | Frames: 2880 at 24.0 FPS"`

## Node Parameters

### Line Count Node:
- **text** (STRING, multiline): The input text to count lines for
- **include_empty_lines** (BOOLEAN): Whether to include empty lines in the count

### Text to Single Line Node:
- **text** (STRING, multiline): The input multi-line text
- **separator** (STRING): The separator to use between lines (default: ". ")
- **trim_lines** (BOOLEAN): Whether to trim whitespace from each line (default: True)
- **skip_empty_lines** (BOOLEAN): Whether to skip empty lines (default: True)

### Audio Duration Node:
- **audio_path** (STRING): Path to the audio file to analyze
- **fps** (FLOAT): Frames per second for frame count calculation (default: 30.0, range: 0.1-120.0)
- **output_format** (DROPDOWN): Output format preference - "duration_seconds", "frame_count", or "both" (default: "both")

## Outputs

### Line Count Node:
- **line_count** (INT): The number of lines as an integer
- **summary** (STRING): A descriptive text summary of the count

### Text to Single Line Node:
- **flattened_text** (STRING): The converted single-line text

### Audio Duration Node:
- **duration_seconds** (FLOAT): The audio duration in seconds
- **frame_count** (INT): The number of frames at the specified FPS
- **info** (STRING): Summary information about the analysis

## License

This project is open source and available under the MIT License.
