from modules.synthesizer.synthesizer import Synthesizer
import sys


if __name__ == "__main__":
    # Synthesizer.__preprocess(dataset_root, out_dir)
    synthesizer = Synthesizer()
    synthesizer.preprocess(sys.argv[1], sys.argv[2])
    synthesizer.train()
    synthesizer.start_training('./output','../models')
    print(sys.argv)