import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..\\.."))

from modules.synthesizer.synthesizer import Synthesizer


if __name__ == "__main__":
    # Synthesizer.__preprocess(dataset_root, out_dir)
    synthesizer = Synthesizer()
    synthesizer.preprocess(sys.argv[1], sys.argv[2])
    # synthesizer.train()
    # synthesizer.start_training(sys.argv[2],'../../models')
    print(sys.argv)