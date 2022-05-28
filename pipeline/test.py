
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from pipeline import Pipeline
from LipSync.inference import LipSyncing
import torch

if __name__ == "__main__":
    # pipeline = Pipeline()
    # input_folder, cloned_audio_path, video_name, output_folder = pipeline.generate_spanish_to_english_speech('test_4.mp4')
    # del pipeline
    #################################################################### Lip-Syncing #############################################################################
    print('Before Loading LipSync' ,torch.cuda.memory_allocated('cuda'))
    lip_syncing = LipSyncing()
    print('After Loading LipSync' ,torch.cuda.memory_allocated('cuda'))

    input_folder = os.path.join(os.path.dirname(__file__), "..\\demo\\input")
    video_name = 'test_4.mp4'
    output_folder = os.path.join(os.path.dirname(__file__), "..\\demo\\output")    
    cloned_audio_path = output_folder+ '\\'+ 'generated_output_41.wav'
    # lip_syncing_progress = tqdm(range(1), desc="Lip-syncing", disable=False)
    lip_syncing.generate_video(input_folder+'\\'+video_name, cloned_audio_path, output_folder +'\\'+video_name)
    # lip_syncing_progress.update(1)
    # lip_syncing_progress.close()
    