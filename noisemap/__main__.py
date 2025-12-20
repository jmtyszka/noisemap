import argparse
from .noisemap import NoiseMap

def main():
    parser = argparse.ArgumentParser(description="Compute noise map from a NIfTI image.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Noisy 3D scalar magnitude NIfTI image")
    parser.add_argument("-o", "--outdir", type=str, default=None, help="Output directory for noise maps")
    parser.add_argument("-m", "--method", type=str, default='anlm', help="Estimation method")
    args = parser.parse_args()

    # Splash text
    print("\nNoisemap: Rician MRI Noise Map Estimation")
    print("-" * 40)
    print(f"Input image: {args.input}")
    print(f"Estimation method: {args.method}\n")

    mapper = NoiseMap(args.input, method=args.method, out_dir=args.outdir)
    mapper.estimate()
    mapper.save_maps()

if __name__ == "__main__":
    main()
