# python download_oras5.py \
#     --years 2018 2019 \
#     --variables sea_surface_temperature sea_surface_salinity

import os
import argparse
import cdsapi
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Download ORAS5 ocean reanalysis data from CDS API")

    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        required=True,
        help="List of years to download, e.g., --years 2010 2011 2012"
    )

    parser.add_argument(
        "--variables",
        nargs="+",
        type=str,
        required=False,
        default=[
            "sea_surface_salinity",
            "sea_surface_temperature"
        ],
        help="Variables to download (default includes major SST and salinity)"
    )

    parser.add_argument(
        "--months",
        nargs="+",
        type=str,
        default=[f"{i:02d}" for i in range(1, 13)],
        help="List of months (01–12), e.g., --months 06 07 08"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./ORAS5/single_level",
        help="Directory to save the downloaded files"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    years = args.years
    variables = args.variables
    months = args.months
    save_root = args.output_dir
    os.makedirs(save_root, exist_ok=True)

    dataset = "reanalysis-oras5"
    client = cdsapi.Client(timeout=600)

    total = len(years)
    pbar = tqdm(total=total, desc="Downloading ORAS5 Data", unit="year")

    for year in years:
        product_type = "consolidated" if year < 2015 else "operational"

        save_dir = os.path.join(save_root, "single_level")
        os.makedirs(save_dir, exist_ok=True)

        target = os.path.join(save_dir, f"{year}.zip")
        tmp_target = target + ".part"

        if os.path.exists(target) and os.path.getsize(target) > 0:
            pbar.update(1)
            print(f"File {target} already exists. Skipping download.")
            continue

        request = {
            "product_type": [product_type],
            "vertical_resolution": "single_level",
            "variable": variables,
            "year": [str(year)],
            "month": months,
        }

        try:
            client.retrieve(dataset, request, tmp_target)
            os.replace(tmp_target, target)
            print(f"Downloaded ORAS5 data for {year} to {target}")
        except Exception as e:
            print(f"[FAIL] ORAS5 {year}: {e}")
            if os.path.exists(tmp_target):
                try:
                    os.remove(tmp_target)
                except Exception:
                    pass
        finally:
            pbar.update(1)
            print("============================================================")

    pbar.close()


if __name__ == "__main__":
    main()