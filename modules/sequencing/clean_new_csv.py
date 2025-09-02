import argparse
import os
import pandas as pd


def clean_csv(input_file: str, output_file: str) -> None:
    """
    Clean the CSV file by applying the following steps:
    1) Remove rows where 'ssim' == -1
    2) Keep only rows with 0.9 <= scale <= 1.1
    3) Add a 'score' column defined as: ssim * num_inliers (if 'num_inliers' exists)
    """

    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)

    print(f"Original data shape: {df.shape}")
    print(f"Original data columns: {list(df.columns)}")

    required_columns = ['ssim', 'scale']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Warning: Missing columns: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        return

    # 1) Remove invalid SSIM rows
    before = len(df)
    df = df[df['ssim'] != -1]
    after = len(df)
    print(f"After removing ssim=-1: {before} -> {after} rows (removed {before - after} rows)")

    # 2) Keep near-unity scale rows
    before = len(df)
    df = df[(df['scale'] >= 0.9) & (df['scale'] <= 1.1)]
    after = len(df)
    print(f"After filtering scale (0.9-1.1): {before} -> {after} rows (removed {before - after} rows)")

    # 3) Score = ssim * num_inliers
    if 'num_inliers' in df.columns:
        df['score'] = df['ssim'] * df['num_inliers']
        print("Added 'score' column: ssim * num_inliers")
        print(f"Score range: {df['score'].min():.4f} - {df['score'].max():.4f}")
    else:
        print("Warning: 'num_inliers' column not found, skipping score calculation")

    # Save
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")
    print(f"Final data shape: {df.shape}")

    # Summary stats
    print("\nData statistics:")
    print(f"SSIM range: {df['ssim'].min():.4f} - {df['ssim'].max():.4f}")
    print(f"Scale range: {df['scale'].min():.4f} - {df['scale'].max():.4f}")
    if 'num_inliers' in df.columns:
        print(f"Num_inliers range: {df['num_inliers'].min()} - {df['num_inliers'].max()}")
        print(f"Score range: {df['score'].min():.4f} - {df['score'].max():.4f}")
    if 'fixed' in df.columns:
        print(f"Number of unique fixed sections: {df['fixed'].nunique()}")
    if 'moving' in df.columns:
        print(f"Number of unique moving sections: {df['moving'].nunique()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean pairwise alignment CSV")
    parser.add_argument('input_file', help='Input CSV file path')
    parser.add_argument('-o', '--output', help='Output CSV path (default: results/sequencing/cleaned_csv/<input>_cleaned.csv)')
    args = parser.parse_args()

    # Default output: <project_root>/results/sequencing/cleaned_csv
    if args.output:
        output_path = args.output
    else:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        cleaned_dir = os.path.join(project_root, 'results', 'sequencing', 'cleaned_csv')
        os.makedirs(cleaned_dir, exist_ok=True)
        base = os.path.basename(args.input_file).rsplit('.', 1)[0]
        output_path = os.path.join(cleaned_dir, f"{base}_cleaned.csv")

    clean_csv(args.input_file, output_path)

