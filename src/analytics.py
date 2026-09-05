"""
Analytics module for the EV Business Research Agent.

Contains:
    - Market analytics for India's electric passenger-car market
    - Competitor analytics for Tata Motors, Mahindra, and Hyundai

The functions operate on processed CSV datasets and return
structured Python dictionaries suitable for use by an AI agent.
"""

from pathlib import Path
from typing import Optional

import pandas as pd


# ============================================================
# DATASET PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MARKET_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "india_ev_market.csv"
)

COMPETITOR_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "competitor_ev_sales.csv"
)


# ============================================================
# LOAD DATASETS
# ============================================================

def load_market_data() -> pd.DataFrame:
    """Load India's EV market dataset."""

    if not MARKET_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Market dataset not found at:\n{MARKET_DATA_PATH}"
        )

    return pd.read_csv(MARKET_DATA_PATH)


def load_competitor_data() -> pd.DataFrame:
    """Load competitor EV sales dataset."""

    if not COMPETITOR_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Competitor dataset not found at:\n{COMPETITOR_DATA_PATH}"
        )

    return pd.read_csv(COMPETITOR_DATA_PATH)


# ============================================================
# MARKET ANALYTICS
# ============================================================

def market_analytics(
    metric: str,
    year: Optional[int] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None
):
    """
    Analyze India's electric passenger-car market.

    Supported metrics:
        - ev_registrations
        - total_cars
        - ev_penetration
        - yoy_growth
        - cagr
        - summary

    Parameters
    ----------
    metric : str
        Requested market metric.

    year : int, optional
        Required for yearly metrics.

    start_year : int, optional
        Starting year for CAGR or summary.

    end_year : int, optional
        Ending year for CAGR or summary.
    """

    df = load_market_data()

    # --------------------------------------------------------
    # Yearly metrics
    # --------------------------------------------------------

    if metric in [
        "ev_registrations",
        "total_cars",
        "ev_penetration",
        "yoy_growth"
    ]:

        if year is None:
            return {
                "status": "error",
                "error": (
                    f"Metric '{metric}' requires a specific year. "
                    "Please provide the 'year' parameter."
                ),
                "required_parameter": "year"
            }

        row = df[df["year"] == year]

        if row.empty:
            return {
                "status": "error",
                "error": (
                    f"No market data available for FY{year}."
                )
            }

        row = row.iloc[0]

        if metric == "ev_registrations":

            return {
                "status": "success",
                "year": year,
                "ev_registrations": int(
                    row["ev_registrations"]
                )
            }

        elif metric == "total_cars":

            return {
                "status": "success",
                "year": year,
                "total_car_registrations": int(
                    row["total_car_registrations"]
                )
            }

        elif metric == "ev_penetration":

            return {
                "status": "success",
                "year": year,
                "ev_penetration_pct": round(
                    float(row["ev_penetration_pct"]),
                    2
                )
            }

        elif metric == "yoy_growth":

            growth = row["yoy_growth_pct"]

            return {
                "status": "success",
                "year": year,
                "ev_yoy_growth_pct": (
                    None
                    if pd.isna(growth)
                    else round(float(growth), 2)
                )
            }

    # --------------------------------------------------------
    # CAGR
    # --------------------------------------------------------

    elif metric == "cagr":

        if start_year is None or end_year is None:

            return {
                "status": "error",
                "error": (
                    "CAGR requires both 'start_year' "
                    "and 'end_year'."
                ),
                "required_parameters": [
                    "start_year",
                    "end_year"
                ]
            }

        if end_year <= start_year:

            return {
                "status": "error",
                "error": (
                    "end_year must be greater than start_year."
                )
            }

        start_row = df[df["year"] == start_year]
        end_row = df[df["year"] == end_year]

        if start_row.empty or end_row.empty:

            return {
                "status": "error",
                "error": (
                    "Start or end year is not available "
                    "in the market dataset."
                )
            }

        start_value = float(
            start_row.iloc[0]["ev_registrations"]
        )

        end_value = float(
            end_row.iloc[0]["ev_registrations"]
        )

        periods = end_year - start_year

        if start_value <= 0:

            return {
                "status": "error",
                "error": (
                    "Starting EV registration value "
                    "must be greater than zero."
                )
            }

        cagr = (
            (end_value / start_value)
            ** (1 / periods) - 1
        ) * 100

        return {
            "status": "success",
            "start_year": start_year,
            "end_year": end_year,
            "start_ev_registrations": int(start_value),
            "end_ev_registrations": int(end_value),
            "cagr_pct": round(cagr, 2)
        }

    # --------------------------------------------------------
    # Market summary
    # --------------------------------------------------------

    elif metric == "summary":

        if start_year is None:
            start_year = int(df["year"].min())

        if end_year is None:
            end_year = int(df["year"].max())

        if end_year < start_year:

            return {
                "status": "error",
                "error": (
                    "end_year must be greater than "
                    "or equal to start_year."
                )
            }

        period_df = df[
            (df["year"] >= start_year) &
            (df["year"] <= end_year)
        ]

        if period_df.empty:

            return {
                "status": "error",
                "error": (
                    "No market data available "
                    "for the specified period."
                )
            }

        return {
            "status": "success",
            "start_year": start_year,
            "end_year": end_year,
            "data": period_df[
                [
                    "year",
                    "ev_registrations",
                    "total_car_registrations",
                    "yoy_growth_pct",
                    "ev_penetration_pct"
                ]
            ].round(2).to_dict(
                orient="records"
            )
        }

    # --------------------------------------------------------
    # Invalid metric
    # --------------------------------------------------------

    else:

        return {
            "status": "error",
            "error": (
                "Unsupported metric. Choose from: "
                "ev_registrations, total_cars, "
                "ev_penetration, yoy_growth, "
                "cagr, summary."
            )
        }


# ============================================================
# COMPETITOR ANALYTICS
# ============================================================

def competitor_analytics(
    metric: str,
    company: Optional[str] = None,
    year: Optional[int] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None
):
    """
    Analyze reported EV sales for Tata Motors, Mahindra,
    and Hyundai.

    Supported metrics:
        - ev_sales
        - yoy_growth
        - compare
        - trend
        - summary

    Important:
        Missing reported data is not treated as zero.
        Metric definitions and data-quality information
        are preserved.
    """

    df = load_competitor_data()

    # --------------------------------------------------------
    # Normalize company name
    # --------------------------------------------------------

    if company is not None:

        company_clean = company.strip().lower()

        company_map = {
            "tata": "Tata Motors",
            "tata motors": "Tata Motors",
            "mahindra": "Mahindra",
            "hyundai": "Hyundai"
        }

        if company_clean not in company_map:

            return {
                "status": "error",
                "error": (
                    "Unknown company. Choose from "
                    "Tata Motors, Mahindra, or Hyundai."
                )
            }

        company = company_map[company_clean]

    # --------------------------------------------------------
    # Individual EV sales
    # --------------------------------------------------------

    if metric == "ev_sales":

        if company is None or year is None:

            return {
                "status": "error",
                "error": (
                    "EV sales requires both 'company' "
                    "and 'year'."
                ),
                "required_parameters": [
                    "company",
                    "year"
                ]
            }

        rows = df[
            (df["company"] == company) &
            (df["year"] == year)
        ]

        if rows.empty:

            return {
                "status": "error",
                "error": (
                    f"No reported EV sales data available "
                    f"for {company} in FY{year}."
                )
            }

        row = rows.iloc[0]

        return {
            "status": "success",
            "company": company,
            "year": year,
            "ev_sales": int(row["ev_sales"]),
            "metric_definition": row["metric_definition"],
            "data_quality": row["data_quality"],
            "source": row["source"],
            "source_url": row["source_url"]
        }

    # --------------------------------------------------------
    # YoY growth
    # --------------------------------------------------------

    elif metric == "yoy_growth":

        if company is None or year is None:

            return {
                "status": "error",
                "error": (
                    "YoY growth requires both 'company' "
                    "and 'year'."
                ),
                "required_parameters": [
                    "company",
                    "year"
                ]
            }

        rows = df[
            (df["company"] == company) &
            (df["year"] == year)
        ]

        if rows.empty:

            return {
                "status": "error",
                "error": (
                    f"No reported data available "
                    f"for {company} in FY{year}."
                )
            }

        row = rows.iloc[0]

        growth = row["yoy_growth_pct"]

        return {
            "status": "success",
            "company": company,
            "year": year,
            "yoy_growth_pct": (
                None
                if pd.isna(growth)
                else round(float(growth), 2)
            ),
            "ev_sales": int(row["ev_sales"]),
            "data_quality": row["data_quality"],
            "source": row["source"],
            "source_url": row["source_url"]
        }

    # --------------------------------------------------------
    # Competitor comparison
    # --------------------------------------------------------

    elif metric == "compare":

        if year is None:

            return {
                "status": "error",
                "error": (
                    "Competitor comparison requires "
                    "'year'."
                ),
                "required_parameter": "year"
            }

        comparison = df[
            df["year"] == year
        ].copy()

        if comparison.empty:

            return {
                "status": "error",
                "error": (
                    f"No competitor data available "
                    f"for FY{year}."
                )
            }

        comparison = comparison.sort_values(
            "ev_sales",
            ascending=False
        )

        return {
            "status": "success",
            "year": year,
            "data": comparison[
                [
                    "company",
                    "year",
                    "ev_sales",
                    "metric_definition",
                    "data_quality"
                ]
            ].to_dict(
                orient="records"
            )
        }

    # --------------------------------------------------------
    # Company trend
    # --------------------------------------------------------

    elif metric == "trend":

        if company is None:

            return {
                "status": "error",
                "error": (
                    "Competitor trend requires "
                    "'company'."
                ),
                "required_parameter": "company"
            }

        if start_year is None:
            start_year = int(df["year"].min())

        if end_year is None:
            end_year = int(df["year"].max())

        if end_year < start_year:

            return {
                "status": "error",
                "error": (
                    "end_year must be greater than "
                    "or equal to start_year."
                )
            }

        trend = df[
            (df["company"] == company) &
            (df["year"] >= start_year) &
            (df["year"] <= end_year)
        ].sort_values("year")

        if trend.empty:

            return {
                "status": "error",
                "error": (
                    f"No data available for {company} "
                    f"between FY{start_year} and FY{end_year}."
                )
            }

        return {
            "status": "success",
            "company": company,
            "start_year": start_year,
            "end_year": end_year,
            "data": trend[
                [
                    "company",
                    "year",
                    "ev_sales",
                    "yoy_growth_pct",
                    "metric_definition",
                    "data_quality"
                ]
            ].round(2).to_dict(
                orient="records"
            )
        }

    # --------------------------------------------------------
    # Latest available summary
    # --------------------------------------------------------

    elif metric == "summary":

        summary_rows = []

        for company_name in df["company"].unique():

            company_df = df[
                df["company"] == company_name
            ].sort_values("year")

            if not company_df.empty:

                row = company_df.iloc[-1]

                summary_rows.append({
                    "company": company_name,
                    "latest_year": int(row["year"]),
                    "latest_ev_sales": int(
                        row["ev_sales"]
                    ),
                    "metric_definition": (
                        row["metric_definition"]
                    ),
                    "data_quality": row["data_quality"],
                    "source": row["source"],
                    "source_url": row["source_url"]
                })

        return {
            "status": "success",
            "data": summary_rows
        }

    # --------------------------------------------------------
    # Invalid metric
    # --------------------------------------------------------

    else:

        return {
            "status": "error",
            "error": (
                "Unsupported metric. Choose from: "
                "ev_sales, yoy_growth, compare, "
                "trend, summary."
            )
        }