import argparse
from .noisemap import NoiseMap

def main():
    parser = argparse.ArgumentParser(description="Compute noise map from a NIfTI image.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the input NIfTI image.")
    args = parser.parse_args()

    mapper = NoiseMap(args.input)
    mapper.estimate()
    mapper.save_maps()

if __name__ == "__main__":
    main()
