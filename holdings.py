from query import Query
import polars as pl
import polars_ols as pls  # noqa: F401

query = Query()

start_date = "2024-04-01"
end_date = "2025-04-01"
fund = 'grad'

benchmark_df = pl.from_pandas(query.get_benchmark_df(start_date, end_date)).select(
    "caldt", "return", "div_return", "xs_return", "xs_div_return"
)

holdings_df = (
    pl.from_pandas(query.get_all_holdings_df(fund=fund, start_date=start_date, end_date=end_date))
    .join(benchmark_df, on="caldt", how="left", suffix=("_bmk"))
    .with_columns(pl.col("xs_div_return", "xs_return_bmk").fill_null(0))
)

performance = (
    holdings_df
    # Cummulative features
    .with_columns(
        pl.col("return").add(1).cum_prod().sub(1).mul(100).over('ticker').alias("total_return"),
    )
    # Aggregation
    .group_by('ticker')
    .agg(
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
        # Weight
        pl.col('weight').last()
    )
    # Clean up Alpha + Beta
    .unnest("coefficients")
    .rename({"xs_div_return": "beta", "const": "alpha"})
    .with_columns(
        pl.col("alpha").mul(pl.lit(252)).mul(100),
    )
    # Output
    .select(
        "ticker",
        "weight",
        "total_return",
        "volatility",
        "dividends",
        "alpha",
        "beta",
    )
    .sort('ticker')
)

print(f"{fund.replace('_', ' ').title()} Fund Holdings Performance")
print(performance)
