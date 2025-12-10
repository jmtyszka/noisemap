import argparse
from .noisemap import NoiseMap

def main():
    parser = argparse.ArgumentParser(description="Compute noise map from a NIfTI image.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Noisy 3D scalar magnitude NIfTI image")
    parser.add_argument("-o", "--outdir", type=str, default=None, help="Output directory for noise maps")
    parser.add_argument("-m", "--method", type=str, default='homomorphic', help="Estimation method")
    args = parser.parse_args()

    mapper = NoiseMap(args.input)
    mapper.estimate(method=args.method)
    mapper.save_maps(out_dir=args.outdir)

if __name__ == "__main__":
    main()
