import numpy as np
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False


class AudioDurationNode:
    """
    A ComfyUI node that analyzes audio data to get duration information.
    Can output duration in seconds or frame count based on FPS parameter.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "fps": ("FLOAT", {
                    "default": 30.0,
                    "min": 0.1,
                    "max": 120.0,
                    "step": 0.1
                }),
                "output_format": (["duration_seconds", "frame_count", "both"], {
                    "default": "both"
                }),
            }
        }
    
    RETURN_TYPES = ("FLOAT", "INT", "STRING")
    RETURN_NAMES = ("duration_seconds", "frame_count", "info")
    FUNCTION = "analyze_audio_duration"
    CATEGORY = "audio/utility"
    
    def analyze_audio_duration(self, audio, fps, output_format):
        """
        Analyze audio data to get duration and frame count information.
        
        Args:
            audio: Audio data from ComfyUI (dict with 'waveform' and 'sample_rate')
            fps (float): Frames per second for frame count calculation
            output_format (str): Output format preference
            
        Returns:
            tuple: (duration_seconds, frame_count, info_string)
        """
        if audio is None:
            return (0.0, 0, "No audio data provided")
        
        try:
            duration_seconds = self._get_audio_duration_from_data(audio)
            
            if duration_seconds is None:
                return (0.0, 0, "Failed to process audio data")
            
            # Calculate frame count based on FPS
            frame_count = int(duration_seconds * fps)
            
            # Create info string based on output format
            if output_format == "duration_seconds":
                info = f"Duration: {duration_seconds:.3f} seconds"
            elif output_format == "frame_count":
                info = f"Frame count: {frame_count} frames at {fps} FPS"
            else:  # both
                info = f"Duration: {duration_seconds:.3f}s | Frames: {frame_count} at {fps} FPS"
            
            return (duration_seconds, frame_count, info)
            
        except Exception as e:
            error_msg = f"Error analyzing audio: {str(e)}"
            return (0.0, 0, error_msg)
    
    def _get_audio_duration_from_data(self, audio):
        """
        Get audio duration from ComfyUI audio data.
        
        Args:
            audio: ComfyUI audio data (dict with 'waveform' and 'sample_rate' keys)
            
        Returns:
            float: Duration in seconds, or None if failed
        """
        try:
            # ComfyUI audio format: {"waveform": tensor, "sample_rate": int}
            if isinstance(audio, dict) and "waveform" in audio and "sample_rate" in audio:
                waveform = audio["waveform"]
                sample_rate = audio["sample_rate"]
                
                # Convert tensor to numpy if needed
                if hasattr(waveform, 'numpy'):
                    waveform = waveform.numpy()
                elif hasattr(waveform, 'cpu'):
                    waveform = waveform.cpu().numpy()
                
                # Calculate duration: number of samples / sample rate
                if len(waveform.shape) > 1:
                    # Multi-channel audio: use the number of samples in the last dimension
                    num_samples = waveform.shape[-1]
                else:
                    # Mono audio
                    num_samples = len(waveform)
                
                duration = num_samples / sample_rate
                return duration
                
            # Fallback: try to process as numpy array with librosa
            elif LIBROSA_AVAILABLE and hasattr(audio, 'shape'):
                # Assume it's a numpy array or tensor with audio data
                if hasattr(audio, 'numpy'):
                    audio_data = audio.numpy()
                elif hasattr(audio, 'cpu'):
                    audio_data = audio.cpu().numpy()
                else:
                    audio_data = audio
                
                # Use librosa to get duration (assumes 22050 Hz if no sample rate provided)
                duration = librosa.get_duration(y=audio_data, sr=22050)
                return duration
                
            else:
                print(f"Unsupported audio format: {type(audio)}")
                return None
                
        except Exception as e:
            print(f"Error processing audio data: {e}")
            return None
    
    @classmethod
    def IS_CHANGED(cls, audio, fps, output_format):
        """
        Check if the node needs to be re-executed.
        For audio data inputs, we'll always re-execute when inputs change.
        """
        # Return a hash of the inputs to trigger re-execution when they change
        return hash((str(audio), fps, output_format))
