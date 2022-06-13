import sys, os
import getopt
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from encoder import Encoder

def print_help_message():
    print("Please use the -p to preprocess and -t to train. If you include both it will preprocess first then train")

def main(argv):
    try:
        opt_tuples, _ = getopt.getopt(argv, "hpt")
        opts = [opt[0] for opt in opt_tuples]
        if (len(opts) == 0) or ("-h" in opts):
            print_help_message()
        else:
            device = torch.device("cpu")
            encoder = Encoder(device)
            if "-p" in opts:
                encoder.preprocess_dataset()
            if "-t" in opts:
                encoder.start_training()
    except getopt.GetoptError:
        print_help_message()

if __name__ == '__main__':
    main(sys.argv[1:])