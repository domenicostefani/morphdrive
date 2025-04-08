import numpy as np
from pedalboard import Pedalboard, Plugin, io, load_plugin
import os
from glob import glob

def process_with_swankyamp(input_path, output_path, sample_rate=48000):
    """
    Process an audio file with SwankyAmp VST3 plugin and save the result.
    
    Args:
        input_path (str): Path to the input audio file
        output_path (str): Path to save the processed audio file
        sample_rate (int): Sample rate to use for processing
    """
    # Read the input audio file
    # audio, sample_rate = io.AudioFile(input_path).read(
    #     sample_rate=sample_rate,
    #     num_channels=1  # Use the file's native channel count
    # )

    with io.AudioFile(input_path) as f:
        audio = f.read(f.frames)
        read_sample_rate = f.samplerate
        assert read_sample_rate == sample_rate, f"Sample rate mismatch: {read_sample_rate} != {sample_rate}"

    
    plugin = load_plugin(r'C:\Program Files\Common Files\VST3\SwankyAmp.vst3')

    # Create a Pedalboard with SwankyAmp VST3 plugin
    board = Pedalboard([
        plugin,
    ])
    
    # Configure SwankyAmp parameters if needed
    swankyamp = board[0]
    # print('type(swankyamp.parameters)', type(swankyamp.parameters))
    # print('swankyamp.parameters', swankyamp.parameters)
    # print('\n'.join([str(k) +"\t\t"+ str(v) for k,v in swankyamp.parameters.items()]))
    
    # Example parameters (adjust these according to SwankyAmp's available parameters)
    # These are just examples - the actual parameter names will depend on the plugin
    try:
        # swankyamp.gain = 0.0        # Adjust gain level
        # swankyamp.output = 0.8      # Adjust output level
        # swankyamp.tone = 0.6        # Adjust tone
        board[0].cabonoff = True  # Enable cab
        board[0].powerampdrive = 0.0  # Enable cab
        board[0].tshigh = -0.5
        # Add more parameter adjustments as needed
    except AttributeError as e:
        print(f"Note: Could not set some parameters: {e}")
        print("Available parameters may differ. Use print(swankyamp.parameters) to see available options.")
    
    # Process the audio with the pedalboard
    effected = board(audio, sample_rate=sample_rate)
    


    # Write the processed audio to the output file
    io.AudioFile(output_path, 'w', sample_rate, effected.shape[0]).write(effected)
    
    print(f"Processing complete. Output saved to {output_path}")
    
    # Uncomment to print available parameters
    # print("Available parameters:")
    # print(swankyamp.parameters)

if __name__ == "__main__":
    # Process the audio file
    os.makedirs('amp', exist_ok=True)
    input_files = glob(r"*.wav")

    for input_file in input_files:
        output_file = os.path.join('amp', 'amp_'+os.path.splitext(os.path.basename(input_file))[0]+'.mp3')
        process_with_swankyamp(input_file, output_file)
        # exit()