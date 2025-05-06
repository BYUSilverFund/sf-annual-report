from query import Query
import polars as pl
import polars_ols as pls  # noqa: F401

query = Query()

start_date = "2024-04-01"
end_date = "2025-04-01"

benchmark_df = pl.from_pandas(query.get_benchmark_df(start_date, end_date)).select(
    "caldt", "return", "div_return", "xs_return", "xs_div_return"
)

fund_df = (
    pl.from_pandas(query.get_fund_df(start_date=start_date, end_date=end_date))
    .join(benchmark_df, on="caldt", how="left", suffix=("_bmk"))
    .with_columns(pl.col("xs_div_return", "xs_return_bmk").fill_null(0))
)

performance = (
    fund_df
    # Cummulative features
    .with_columns(
        pl.col("return").add(1).cum_prod().sub(1).mul(100).alias("total_return"),
        pl.col("xs_return").sub("xs_div_return").alias("res_return"),
    )
    # Aggregation
    .select(
        # Total return
        pl.col("total_return").last(),
        # Volatility
        pl.col("return").mul(100).std().mul(pl.lit(252).sqrt()).alias("volatility"),
        # Dividends
        pl.col("dividends").sum(),
        # Alpha + Beta
        pl.col("xs_return").least_squares.ols(
            pl.col("xs_div_return"), mode="coefficients", add_intercept=True
        ),
        # Tracking error
        pl.col("res_return")
        .mul(100)
        .std()
        .mul(pl.lit(252).sqrt())
        .alias("tracking_error"),
        # Expected return
        pl.col("xs_return").mean().mul(pl.lit(252)).mul(100).alias("expected_return"),
    )
    # Clean up Alpha + Beta
    .unnest("coefficients")
    .rename({"xs_div_return": "beta", "const": "alpha"})
    .with_columns(
        pl.col("alpha").mul(pl.lit(252)).mul(100),
    )
    # Ratios
    .with_columns(
        pl.col("expected_return").truediv(pl.col("volatility")).alias("sharpe_ratio"),
        pl.col("alpha").truediv(pl.col("tracking_error")).alias("information_ratio"),
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

print("Fund Performance")
print(performance)
