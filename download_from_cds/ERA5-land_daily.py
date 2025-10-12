# python download_era5land.py --years 1990 1991 1992 --variables volumetric_soil_water_layer_1 soil_temperature_level_1 --months 01 02 03

import os
import argparse
import cdsapi
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Download ERA5-Land daily (6 hour frequency) data via CDS API")

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
        required=True,
        help="List of variables to download, e.g., --variables volumetric_soil_water_layer_1 soil_temperature_level_1"
    )

    parser.add_argument(
        "--months",
        nargs="+",
        type=str,
        required=False,
        default=[f"{i:02d}" for i in range(1, 13)],
        help="List of months (01–12), e.g., --months 06 07 08"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./ERA5-land-daily",
        help="Directory to save the downloaded files"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    client = cdsapi.Client()

    years = args.years
    variables = args.variables
    months = args.months
    save_root = args.output_dir

    total = len(years) * len(variables) * len(months)
    pbar = tqdm(total=total, desc="Downloading ERA5-Land Data", unit="file")

    dataset = "derived-era5-land-daily-statistics"

    for year in years:
        for variable in variables:
            save_dir = os.path.join(save_root, str(year), variable)
            os.makedirs(save_dir, exist_ok=True)

            for month in months:
                target = os.path.join(save_dir, f"{variable}_{year}{month}.nc")
                tmp_target = target + ".part"

                if os.path.exists(target) and os.path.getsize(target) > 0:
                    pbar.update(1)
                    print(f"File {target} already exists. Skipping download.")
                    continue

                request = {
                    "variable": [variable],
                    "year": str(year),
                    "month": month,
                    "day": [f"{d:02d}" for d in range(1, 32)],
                    "daily_statistic": "daily_mean",
                    "time_zone": "utc+00:00",
                    "frequency": "6_hourly",
                }

                try:
                    client.retrieve(dataset, request, tmp_target)
                    os.replace(tmp_target, target)
                    print(f"Downloaded {target}")
                except Exception as e:
                    print(f"[FAIL] {variable} {year}-{month}: {e}")
                    if os.path.exists(tmp_target):
                        try:
                            os.remove(tmp_target)
                        except Exception:
                            pass
                finally:
                    pbar.update(1)

    pbar.close()


if __name__ == "__main__":
    main()
