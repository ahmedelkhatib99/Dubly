import sys, os
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from encoder import Encoder

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(device)
    encoder.preprocess_dataset()
    encoder.start_training()

if __name__ == '__main__':
    main()