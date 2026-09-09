import io
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests


# ============================================================
# CONFIG
# ============================================================

START_DATE = date(2016, 1, 1)
END_DATE = date(2026, 8, 20)

BASE_URL = "https://nsearchives.nseindia.com/content/cm"

OUTPUT_DIR = Path("data/raw/prices")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Referer": "https://www.nseindia.com/all-reports",
}


# ============================================================
# SESSION
# ============================================================

def create_session():

    session = requests.Session()
    session.headers.update(HEADERS)

    return session


# ============================================================
# DOWNLOAD ONE DAY
# ============================================================

def download_one_day(session, current_date):

    date_str = current_date.strftime("%Y%m%d")

    output_file = OUTPUT_DIR / f"{date_str}.parquet"

    # Resume support
    if output_file.exists():

        print(
            f"[SKIP] {date_str} already exists"
        )

        return "skipped"


    filename = (
        f"BhavCopy_NSE_CM_0_0_0_"
        f"{date_str}_F_0000.csv.zip"
    )

    url = f"{BASE_URL}/{filename}"


    for attempt in range(1, 4):

        try:

            print(
                f"[DOWNLOAD] {date_str} "
                f"(attempt {attempt}/3)"
            )

            response = session.get(
                url,
                timeout=30,
            )


            # ------------------------------------------------
            # SUCCESS
            # ------------------------------------------------

            if response.status_code == 200:

                # Validate ZIP before reading
                if not zipfile.is_zipfile(
                    io.BytesIO(response.content)
                ):

                    print(
                        f"[ERROR] {date_str}: "
                        "response is not a valid ZIP"
                    )

                    continue


                with zipfile.ZipFile(
                    io.BytesIO(response.content)
                ) as z:

                    csv_files = [
                        name
                        for name in z.namelist()
                        if name.lower().endswith(".csv")
                    ]


                    if not csv_files:

                        print(
                            f"[ERROR] {date_str}: "
                            "no CSV inside ZIP"
                        )

                        return "failed"


                    csv_name = csv_files[0]


                    with z.open(csv_name) as f:

                        df = pd.read_csv(f)


                # ------------------------------------------------
                # FILTER EQUITY
                # ------------------------------------------------

                required_columns = {
                    "Sgmt",
                    "FinInstrmTp",
                    "SctySrs",
                }

                missing = (
                    required_columns
                    - set(df.columns)
                )

                if missing:

                    print(
                        f"[ERROR] {date_str}: "
                        f"missing columns {missing}"
                    )

                    return "failed"


                df = df[
                    (df["Sgmt"] == "CM") &
                    (df["FinInstrmTp"] == "STK") &
                    (df["SctySrs"] == "EQ")
                ].copy()


                # ------------------------------------------------
                # NORMALIZE
                # ------------------------------------------------

                rename_map = {
                    "TradDt": "date",
                    "TckrSymb": "symbol",
                    "ISIN": "isin",
                    "OpnPric": "open",
                    "HghPric": "high",
                    "LwPric": "low",
                    "ClsPric": "close",
                    "PrvsClsgPric": "prev_close",
                    "TtlTradgVol": "volume",
                    "TtlTrfVal": "traded_value",
                }

                df = df.rename(
                    columns=rename_map
                )


                required_output = [
                    "date",
                    "symbol",
                    "isin",
                    "open",
                    "high",
                    "low",
                    "close",
                    "prev_close",
                    "volume",
                    "traded_value",
                ]


                # Make sure all columns exist
                missing_output = [
                    col
                    for col in required_output
                    if col not in df.columns
                ]

                if missing_output:

                    print(
                        f"[ERROR] {date_str}: "
                        f"missing {missing_output}"
                    )

                    return "failed"


                df = df[required_output]


                # ------------------------------------------------
                # DATA TYPES
                # ------------------------------------------------

                df["date"] = pd.to_datetime(
                    df["date"],
                    errors="coerce",
                )

                numeric_columns = [
                    "open",
                    "high",
                    "low",
                    "close",
                    "prev_close",
                    "volume",
                    "traded_value",
                ]

                for col in numeric_columns:

                    df[col] = pd.to_numeric(
                        df[col],
                        errors="coerce",
                    )


                # Remove invalid symbols
                df = df.dropna(
                    subset=[
                        "date",
                        "symbol",
                    ]
                )


                # ------------------------------------------------
                # SAVE
                # ------------------------------------------------

                df.to_parquet(
                    output_file,
                    index=False,
                    compression="snappy",
                )


                print(
                    f"[SAVED] {date_str} | "
                    f"{len(df):,} stocks"
                )

                return "downloaded"


            # ------------------------------------------------
            # NOT FOUND
            # ------------------------------------------------

            elif response.status_code == 404:

                print(
                    f"[NOT FOUND] {date_str}"
                )

                return "not_found"


            # ------------------------------------------------
            # RATE LIMIT / SERVER ERROR
            # ------------------------------------------------

            else:

                print(
                    f"[HTTP {response.status_code}] "
                    f"{date_str}"
                )


        except Exception as e:

            print(
                f"[ERROR] {date_str}: {e}"
            )


        time.sleep(
            2 * attempt
        )


    return "failed"


# ============================================================
# DOWNLOAD RANGE
# ============================================================

def download_range(
    start_date,
    end_date,
):

    session = create_session()

    current_date = start_date

    stats = {
        "downloaded": 0,
        "skipped": 0,
        "not_found": 0,
        "failed": 0,
    }


    total_days = (
        end_date - start_date
    ).days + 1

    day_number = 0


    while current_date <= end_date:

        day_number += 1


        # Skip weekends
        if current_date.weekday() >= 5:

            current_date += timedelta(
                days=1
            )

            continue


        print()
        print(
            f"Progress: "
            f"{day_number}/{total_days}"
        )


        result = download_one_day(
            session,
            current_date,
        )


        stats[result] += 1


        # Don't hammer NSE
        time.sleep(0.7)


        current_date += timedelta(
            days=1
        )


    print()
    print("=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)

    for key, value in stats.items():

        print(
            f"{key:12}: {value:,}"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    download_range(
        START_DATE,
        END_DATE,
    )
