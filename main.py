from query import Query
import polars as pl
import polars_ols as pls  # noqa: F401
from great_tables import GT
import os
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("results", exist_ok=True)

query = Query()


def compute_total_fund_performance(start_date: str, end_date: str) -> pl.DataFrame:
    benchmark_df = pl.from_pandas(query.get_benchmark_df(start_date, end_date)).select(
        "caldt", "return", "div_return", "xs_return", "xs_div_return"
    )

    fund_df = (
        pl.from_pandas(query.get_fund_df(start_date=start_date, end_date=end_date))
        .join(benchmark_df, on="caldt", how="left", suffix=("_bmk"))
        .with_columns(
            pl.col("xs_div_return", "xs_return_bmk", "div_return").fill_null(0)
        )
    )

    performance = (
        fund_df
        # Cummulative features
        .with_columns(
            pl.col("return").add(1).cum_prod().sub(1).alias("total_return"),
            pl.col("div_return").add(1).cum_prod().sub(1).alias("total_return_bmk"),
            pl.col("xs_return").sub("xs_div_return").alias("res_return"),
        )
        # Aggregation
        .select(
            # Total return
            pl.col("total_return").last(),
            # Total return
            pl.col("total_return_bmk").last(),
            # Volatility
            pl.col("return").std().mul(pl.lit(252).sqrt()).alias("volatility"),
            # Dividends
            pl.col("dividends").sum(),
            # Alpha + Beta
            pl.col("xs_return").least_squares.ols(
                pl.col("xs_div_return"), mode="coefficients", add_intercept=True
            ),
            # Tracking error
            pl.col("res_return").std().mul(pl.lit(252).sqrt()).alias("tracking_error"),
            # Expected return
            pl.col("xs_return").mean().mul(pl.lit(252)).alias("expected_return"),
        )
        # Clean up Alpha + Beta
        .unnest("coefficients")
        .rename({"xs_div_return": "beta", "const": "alpha"})
        # .with_columns(
        #     pl.col("alpha").mul(pl.lit(252)),
        # )
        .with_columns(
            pl.col("total_return")
            .sub(pl.col("total_return_bmk").mul(pl.col("beta")))
            .alias("alphas")
        )
        # Ratios
        .with_columns(
            pl.col("expected_return")
            .truediv(pl.col("volatility"))
            .alias("sharpe_ratio"),
            pl.col("alpha")
            .truediv(pl.col("tracking_error"))
            .alias("information_ratio"),
        )
        # Output
        .select(
            "total_return",
            "volatility",
            "dividends",
            "alpha",
            "beta",
            "tracking_error",
            "sharpe_ratio",
            "information_ratio",
        )
    )

    return performance


def compute_fund_performance(fund: str, start_date: str, end_date: str) -> pl.DataFrame:
    benchmark_df = pl.from_pandas(query.get_benchmark_df(start_date, end_date)).select(
        "caldt", "return", "div_return", "xs_return", "xs_div_return"
    )

    portfolio_df = (
        pl.from_pandas(
            query.get_portfolio_df(fund=fund, start_date=start_date, end_date=end_date)
        )
        .join(benchmark_df, on="caldt", how="left", suffix=("_bmk"))
        .with_columns(
            pl.col("xs_div_return", "xs_return_bmk", "div_return").fill_null(0)
        )
    )

    performance = (
        portfolio_df
        # Cummulative features
        .with_columns(
            pl.col("return").add(1).cum_prod().sub(1).alias("total_return"),
            pl.col("return_bmk").add(1).cum_prod().sub(1).alias("total_return_bmk"),
            pl.col("xs_return").sub("xs_div_return").alias("res_return"),
        )
        # Aggregation
        .select(
            # Total return
            pl.col("total_return").last(),
            # Total benchmark return
            pl.col("total_return_bmk").last(),
            # Volatility
            pl.col("return").std().mul(pl.lit(252).sqrt()).alias("volatility"),
            # Dividends
            pl.col("dividends").sum(),
            # Alpha + Beta
            pl.col("xs_return").least_squares.ols(
                pl.col("xs_return_bmk"), mode="coefficients", add_intercept=True
            ),
            # Tracking error
            pl.col("res_return").std().mul(pl.lit(252).sqrt()).alias("tracking_error"),
            # Expected return
            pl.col("xs_return").mean().mul(pl.lit(252)).alias("expected_return"),
        )
        # Clean up Alpha + Beta
        .unnest("coefficients")
        .rename({"xs_return_bmk": "beta", "const": "alpha"})
        # .with_columns(
        #     pl.col("alpha").mul(pl.lit(252)),
        # )
        .with_columns(
            pl.col("total_return")
            .sub(pl.col("beta").mul(pl.col("total_return_bmk")))
            .alias("alpha")
        )
        # Ratios
        .with_columns(
            pl.col("expected_return")
            .truediv(pl.col("volatility"))
            .alias("sharpe_ratio"),
            pl.col("alpha")
            .truediv(pl.col("tracking_error"))
            .alias("information_ratio"),
        )
        # Output
        .select(
            "total_return",
            "volatility",
            "dividends",
            "alpha",
            "beta",
            "tracking_error",
            "sharpe_ratio",
            "information_ratio",
            "total_return_bmk",
        )
    )

    return performance


def create_overall_fund_performance(start_date: str, end_date: str) -> None:
    funds = ["grad", "undergrad", "brigham_capital", "quant"]

    data = []

    total_performance = compute_total_fund_performance(start_date, end_date).to_dicts()[
        0
    ]
    data.append({**total_performance, "fund": "all"})

    for fund in funds:
        fund_performance = compute_fund_performance(
            fund, start_date, end_date
        ).to_dicts()[0]
        data.append({**fund_performance, "fund": fund})

    results = pl.DataFrame(data)
    results = (
        results.select(
            "fund",
            "total_return",
            "volatility",
            # "dividends",
            "alpha",
            "beta",
            "tracking_error",
            "sharpe_ratio",
            "information_ratio",
        )
        .with_columns(pl.col("fund").str.replace("_", " ").str.to_titlecase())
        .sort("fund")
    )

    results = results.rename(
        {col: col.replace("_", " ").title() for col in results.columns}
    )

    (
        GT(results)
        .tab_header(
            title="Overall Fund Performance",
            subtitle=f"From {start_date} to {end_date}",
        )
        .tab_source_note(
            source_note="Volatility and tracking error are annualized. All other values are period totals."
        )
        .fmt_percent(["Total Return", "Volatility", "Tracking Error", "Alpha"])
        .fmt_number(["Beta", "Sharpe Ratio", "Information Ratio"])
        .opt_stylize(style=3, color="gray")
        .save(f"results/table_all_funds_{end_date}.png", scale=2)
    )


def compute_fund_holdings_performance(
    fund: str, start_date: str, end_date: str
) -> pl.DataFrame:
    current_tickers = query.get_current_tickers(fund)
    benchmark_df = pl.from_pandas(query.get_benchmark_df(start_date, end_date)).select(
        "caldt", "return", "div_return", "xs_return", "xs_div_return"
    )

    holdings_df = (
        pl.from_pandas(
            query.get_all_holdings_df(
                fund=fund, start_date=start_date, end_date=end_date
            )
        )
        .join(benchmark_df, on="caldt", how="left", suffix=("_bmk"))
        .with_columns(pl.col("xs_div_return_bmk", "xs_div_return").fill_null(0))
    )

    performance = (
        holdings_df.filter(pl.col("ticker").is_in(current_tickers))
        .sort(["ticker", "caldt"])
        # Cummulative features
        .with_columns(
            pl.col("div_return")
            .add(1)
            .cum_prod()
            .sub(1)
            .over("ticker")
            .alias("total_return"),
            pl.col("div_return_bmk")
            .add(1)
            .cum_prod()
            .sub(1)
            .over("ticker")
            .alias("total_return_bmk"),
        )
        # Aggregation
        .group_by("ticker")
        .agg(
            # Total return
            pl.col("total_return").last(),
            # Total benchmark return
            pl.col("total_return_bmk").last(),
            # Volatility
            pl.col("div_return").std().mul(pl.lit(252).sqrt()).alias("volatility"),
            # Dividends
            pl.col("dividends").sum(),
            # Alpha + Beta
            pl.col("xs_div_return").least_squares.ols(
                pl.col("xs_div_return_bmk"), mode="coefficients", add_intercept=True
            ),
            # Weight
            pl.col("weight").last(),
        )
        # Clean up Alpha + Beta
        .unnest("coefficients")
        .rename({"xs_div_return_bmk": "beta", "const": "alpha"})
        .with_columns(
            pl.col("total_return")
            .sub(pl.col("total_return_bmk").mul(pl.col("beta")))
            .alias("alpha")
        )
        # Output
        .select(
            "ticker",
            "weight",
            "total_return",
            "volatility",
            # "dividends",
            "alpha",
            "beta",
        )
        .sort("ticker")
    )

    return performance


def create_holdings_performance(start_date: str, end_date: str) -> None:
    funds = ["grad", "undergrad", "brigham_capital", "quant"]
    for fund in funds:
        holdings_performance = compute_fund_holdings_performance(
            fund, start_date, end_date
        )
        holdings_performance = holdings_performance.rename(
            {col: col.replace("_", " ").title() for col in holdings_performance.columns}
        )
        (
            GT(holdings_performance)
            .tab_header(
                title=f"{fund.replace('_', ' ').title()} Holdings Performance",
                subtitle=f"From {start_date} to {end_date}",
            )
            .tab_source_note(source_note="Weight is the ending weight of the period.")
            .tab_source_note(
                source_note="Volatility is annualized. All other values are period totals."
            )
            .fmt_percent(["Total Return", "Volatility", "Alpha", "Weight"])
            .fmt_number(["Beta"])
            .opt_stylize(style=3, color="gray")
            .save(f"results/table_holdings_{fund}_{end_date}.png", scale=2)
        )


def create_total_fund_chart(start_date: str, end_date: str) -> None:
    benchmark_df = pl.from_pandas(query.get_benchmark_df(start_date, end_date)).select(
        "caldt", "return", "div_return", "xs_return", "xs_div_return"
    )

    fund_df = (
        pl.from_pandas(query.get_fund_df(start_date=start_date, end_date=end_date))
        .join(benchmark_df, on="caldt", how="left", suffix=("_bmk"))
        .with_columns(pl.col("div_return").fill_null(0))
    )

    cummulative_returns = fund_df.sort("caldt").with_columns(
        pl.col("return").add(1).cum_prod().sub(1).mul(100).alias("cummulative_return"),
        pl.col("div_return")
        .add(1)
        .cum_prod()
        .sub(1)
        .mul(100)
        .alias("cummulative_return_bmk"),
    )

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        cummulative_returns.to_pandas(),
        x="caldt",
        y="cummulative_return",
        label="Portfolio",
        color="royalblue",
        linewidth=2
    )
    sns.lineplot(
        cummulative_returns.to_pandas(),
        x="caldt",
        y="cummulative_return_bmk",
        label="Benchmark",
        color="lightcoral",
        linestyle="dashed",
        linewidth=2
    )
    plt.title("Overall Fund Performance")
    plt.xlabel(None)
    plt.ylabel("Cummulative Compounding Returns (%)")
    plt.savefig("results/line_chart_total_fund.png", dpi=300)

def create_fund_charts(start_date: str, end_date: str) -> None:
    benchmark_df = pl.from_pandas(query.get_benchmark_df(start_date, end_date)).select(
        "caldt", "return", "div_return", "xs_return", "xs_div_return"
    )
    funds = ['grad', 'undergrad', 'quant', 'brigham_capital']
    for fund in funds:

        fund_df = (
            pl.from_pandas(query.get_portfolio_df(fund=fund, start_date=start_date, end_date=end_date))
            .join(benchmark_df, on="caldt", how="left", suffix=("_bmk"))
            .with_columns(pl.col("div_return").fill_null(0))
        )

        cummulative_returns = fund_df.sort("caldt").with_columns(
            pl.col("return").add(1).cum_prod().sub(1).mul(100).alias("cummulative_return"),
            pl.col("div_return")
            .add(1)
            .cum_prod()
            .sub(1)
            .mul(100)
            .alias("cummulative_return_bmk"),
        )

        plt.figure(figsize=(10, 6))
        sns.lineplot(
            cummulative_returns.to_pandas(),
            x="caldt",
            y="cummulative_return",
            label=f"{fund.replace('_', ' ').title()}",
            color="royalblue",
            linewidth=2
        )
        sns.lineplot(
            cummulative_returns.to_pandas(),
            x="caldt",
            y="cummulative_return_bmk",
            label="Benchmark",
            color="lightcoral",
            linestyle="dashed",
            linewidth=2
        )
        plt.title(f"{fund.replace('_', ' ').title()} Performance")
        plt.xlabel(None)
        plt.ylabel("Cummulative Compounding Returns (%)")
        plt.savefig(f"results/line_chart_{fund}_{end_date}.png", dpi=300)

def create_combined_funds_chart(start_date: str, end_date: str) -> None:
    funds = sorted(['grad', 'undergrad', 'quant', 'brigham_capital'])
    colors = sns.color_palette('mako', 4)

    plt.figure(figsize=(10, 6))
    for fund, color in zip(funds, colors):

        fund_df = (
            pl.from_pandas(query.get_portfolio_df(fund=fund, start_date=start_date, end_date=end_date))
        )

        cummulative_returns = fund_df.sort("caldt").with_columns(
            pl.col("return").add(1).cum_prod().sub(1).mul(100).alias("cummulative_return"),
        )

        sns.lineplot(
            cummulative_returns.to_pandas(),
            x="caldt",
            y="cummulative_return",
            label=f"{fund.replace('_', ' ').title()}",
            color=color,
            linewidth=2,
        )

    plt.title("All Funds Performance")
    plt.xlabel(None)
    plt.ylabel("Cummulative Compounding Returns (%)")

    plt.savefig(f"results/line_chart_combined_{end_date}.png", dpi=300)    

if __name__ == "__main__":
    start_date = "2024-05-01"
    end_date = "2025-05-01"

    create_overall_fund_performance(start_date, end_date)
    create_holdings_performance(start_date, end_date)
    # create_total_fund_chart(start_date, end_date)
    # create_fund_charts(start_date, end_date)
    # create_combined_funds_chart(start_date, end_date)