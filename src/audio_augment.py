import torch
import numpy as np
import librosa
from typing import Tuple, List
import random

def augment_audio(
    audio: torch.Tensor,
    sample_rate: int = 24000,
    possible_pitch_shifts: List[int] = [-1, 0, 1],
    speed_change_range: Tuple[float, float] = (0.9, 1.1)
) -> torch.Tensor:
    """
    Augment audio by adjusting speed and pitch using librosa, working with PyTorch tensors.
    
    Args:
        audio: Input audio tensor (channels, samples) or (samples,)
        sample_rate: Audio sample rate in Hz
        possible_pitch_shifts: List of possible pitch shifts in semitones
        speed_change_range: Range for random speed change (min, max)
    
    Returns:
        Augmented audio tensor with the same number of channels as input
    """
    # Handle input tensor shape and convert to numpy
    original_shape = audio.shape
    is_one_dim = len(original_shape) == 1
    
    if is_one_dim:
        audio = audio.unsqueeze(0)
    
    # Remember original device
    device = audio.device
    
    # Convert to numpy array for librosa processing
    audio_np = audio.cpu().numpy()
    
    # Process each channel separately
    augmented_channels = []
    for channel in audio_np:
        # Step 1: Change speed without affecting pitch
        speed_change = 1#random.uniform(*speed_change_range)
        
        # Use librosa's time_stretch for changing speed
        speed_changed = librosa.effects.time_stretch(
            y=channel.astype(np.float32),
            rate=speed_change
        )
        
        # Step 2: Apply pitch shift
        pitch_shift = random.choice(possible_pitch_shifts)
        
        # Use librosa's pitch_shift for changing pitch
        augmented_channel = librosa.effects.pitch_shift(
            y=speed_changed,
            sr=sample_rate,
            n_steps=pitch_shift
        )
        
        # Normalize audio to prevent clipping
        augmented_channel = librosa.util.normalize(augmented_channel)
        augmented_channels.append(augmented_channel)
    
    # Find the maximum length among augmented channels
    max_length = max(len(channel) for channel in augmented_channels)
    
    # Pad shorter channels to match the longest one
    padded_channels = []
    for channel in augmented_channels:
        if len(channel) < max_length:
            padding = np.zeros(max_length - len(channel))
            channel = np.concatenate([channel, padding])
        padded_channels.append(channel)
    
    # Stack channels back together
    augmented_audio = np.stack(padded_channels)
    
    # Convert back to torch tensor
    augmented_tensor = torch.from_numpy(augmented_audio).to(device)
    
    # Ensure float32 dtype
    augmented_tensor = augmented_tensor.to(torch.float32)
    
    # If input was single channel and one-dimensional, remove the extra dimension
    if is_one_dim:
        augmented_tensor = augmented_tensor.squeeze(0)
    
    return augmented_tensor

if __name__ == "__main__":
    import torchaudio
    
    # Load audio file
    waveform, sample_rate = torchaudio.load(
        "/home/k4/Python/F5-TTS-Fork/ckpts/russian_dataset_ft_translit_pinyin/samples/step_56500_ref.wav"
    )
    print(f"Loaded audio: {waveform.shape} at {sample_rate}Hz")
    
    # Apply augmentation
    try:
        augmented_audio = augment_audio(
            audio=waveform,
            sample_rate=sample_rate
        )
        
        print(f"Augmented audio shape: {augmented_audio.shape}")
        
        # Ensure the audio has the correct channel layout
        if augmented_audio.dim() == 1:
            augmented_audio = augmented_audio.unsqueeze(0)
            
        # Save the augmented audio 
        torchaudio.save(
            "/home/k4/Python/F5-TTS-Fork/out.mp3",  
            augmented_audio,
            sample_rate,
        )
        
        print("Successfully saved augmented audio to out.wav")
        
    except Exception as e:
        print(f"Error processing audio: {str(e)}")