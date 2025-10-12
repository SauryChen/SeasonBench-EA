# python download_oras5_pressure.py \
#     --years 2018 2019 \
#     --variables potential_temperature

import os
import argparse
import cdsapi
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Download ORAS5 reanalysis data from CDS API")

    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        required=True,
        help="List of years to download, e.g., --years 1990 1991 1992"
    )

    parser.add_argument(
        "--variables",
        nargs="+",
        type=str,
        required=False,
        default=["potential_temperature", "salinity"],
        help="List of variables to download (default: potential_temperature salinity)"
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
        default="./ORAS5/pressure_levels",
        help="Directory to save the downloaded files"
    )

    return parser.parse_args()


def download_oras5_data(year, variables, months, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    dataset = "reanalysis-oras5"
    product_type = "consolidated" if year < 2015 else "operational"

    request = {
        "product_type": [product_type],
        "vertical_resolution": "all_levels",
        "variable": variables,
        "year": [str(year)],
        "month": months,
        "grid": [1, 1],
    }

    target = os.path.join(save_dir, f"{year}.zip")
    tmp_target = target + ".part"

    if os.path.exists(target) and os.path.getsize(target) > 0:
        print(f"File already exists: {target}")
        return

    try:
        print(f"Downloading ORAS5 data for {year} ...")
        client = cdsapi.Client(timeout=600)
        client.retrieve(dataset, request, tmp_target)
        os.replace(tmp_target, target)
        print(f"Downloaded {target}")
    except Exception as e:
        print(f"[FAIL] {year}: {e}")
        if os.path.exists(tmp_target):
            try:
                os.remove(tmp_target)
            except Exception:
                pass


def main():
    args = parse_args()

    years = args.years
    variables = args.variables
    months = args.months
    save_root = args.output_dir
    os.makedirs(save_root, exist_ok=True)

    pbar = tqdm(total=len(years), desc="Downloading ORAS5 Pressure-Level Data", unit="year")

    for year in years:
        download_oras5_data(year, variables, months, save_root)
        pbar.update(1)

    pbar.close()


if __name__ == "__main__":
    main()