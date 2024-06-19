import yfinance as yf
import pandas as pd
from pypfopt import risk_models, expected_returns, EfficientFrontier, DiscreteAllocation, objective_functions, EfficientSemivariance, EfficientCVaR, HRPOpt
from yahoo_fin import stock_info as si
import logging

# Get an instance of a logger
logger = logging.getLogger(__name__)

def download_prices(tickers, start_date="2000-01-01", end_date="2023-01-01"):
    prices = pd.DataFrame()
    errors = {}
    for ticker in tickers:
        try:
            # Download data for each ticker
            data = yf.download(ticker, start=start_date, end=end_date)
            if data.empty:
                raise ValueError(f"No data found for {ticker}")            
            # Use the adjusted close price
            prices[ticker] = data['Adj Close']
        except Exception as e:
            print(f"Failed to download data for {ticker}: {e}")
            errors[ticker] = f"Error: {e}"  # Store error message in the dictionary

    
    # Reindex to ensure consistent date range across all tickers
    all_dates = pd.date_range(start=start_date, end=end_date, freq='B')  # 'B' frequency is for business days
    prices = prices.reindex(all_dates)
    
    # Forward-fill and back-fill any missing data
    prices = prices.ffill().bfill()
    valid_prices = prices.dropna(axis=1)
    valid_tickers = valid_prices.columns.tolist()


    return valid_prices,valid_tickers, errors
def get_expected_returns(prices):
    """Compute expected returns using CAPM."""
    mu = expected_returns.capm_return(prices)
    return mu
def optimize__volatility(prices):
    try:
        mu = expected_returns.mean_historical_return(prices)
        S = risk_models.sample_cov(prices)
        ef = EfficientFrontier(mu, S)
        weights = ef.min_volatility()
        performance = ef.portfolio_performance(verbose=True)
        return weights, performance
    except Exception as e:
        logger.error("Error in optimize_min_volatility: %s", str(e))
        raise e

def efficient__cvar(prices, target_cvar):
    try:
        mu = expected_returns.mean_historical_return(prices)
        S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
        ef = EfficientFrontier(mu, S)
        ef.add_objective(lambda w: ef.portfolio_performance()[2] <= target_cvar)
        weights = ef.efficient_risk(target_cvar)
        performance = ef.portfolio_performance(verbose=True)
        return weights, performance
    except Exception as e:
        logger.error("Error in efficient_cvar: %s", str(e))
        raise e

def optimize_min_volatility(prices):
    """Construct a long/short portfolio to minimize variance."""
    S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
    ef = EfficientFrontier(None, S, weight_bounds=(None, None))
    ef.min_volatility()
    weights = ef.clean_weights()
    print('------performance for min_volatility_optimized_portfolio-----')
    ef.portfolio_performance(verbose=True)
    portfolio_performance = pd.DataFrame(ef.portfolio_performance(risk_free_rate=0), 
                                     index = ["Expected annual return", "Annual volatility", "Sharpe Ratio"],
                                     columns = ["MVO"])
    performance_data = portfolio_performance.to_dict()

    return  weights, performance_data

def perform_discrete_allocation(weights, prices, total_portfolio_value=1000, short_ratio=0.3):
    """Perform discrete allocation based on optimized weights."""
    latest_prices = prices.iloc[-1]
    da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=total_portfolio_value, short_ratio=short_ratio)
    alloc, leftover = da.lp_portfolio()
    allocations = {}
    for asset, shares in alloc.items():
        action = "buy" if shares > 0 else "sell"
        shares = abs(shares)  # Convert negative shares to positive for verbal output
        if asset in allocations:
            allocations[asset] = f"{action} {shares} shares of {asset}"
        else:
            allocations[asset] = f"{action} {shares} shares of {asset}"
#    print(f"leftover: {leftover}")
    print(allocations)
    return allocations, leftover

def max_sharpe_with_sector_constraints(prices):
    """Maximize Sharpe ratio with sector constraints."""
    mu = expected_returns.capm_return(prices)
    S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
    
    ef = EfficientFrontier(mu, S)


    ef.max_sharpe()
    weights = ef.clean_weights()
    print('-----performance for Maximum Sharperatio optimised portfolios-----')
    print("")
    ef.portfolio_performance(verbose=True)
    portfolio_performance = pd.DataFrame(ef.portfolio_performance(risk_free_rate=0), 
                                     index = ["Expected annual return", "Annual volatility", "Sharpe Ratio"],
                                     columns = ["MVO"])
    performance_data = portfolio_performance.to_dict()
    return weights, performance_data

def maximize_return_given_risk(prices, target_volatility):
    """Maximize return for a given risk, with L2 regularization."""
    mu = expected_returns.capm_return(prices)
    S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
    
    ef = EfficientFrontier(mu, S)
    ef.add_objective(objective_functions.L2_reg, gamma=0.1)  # gamma is the tuning parameter
    ef.efficient_risk(target_volatility)
    weights = ef.clean_weights()
    print("-----Performance for Maximised return for a given risk Optimised Portfolio-----")

    ef.portfolio_performance(verbose=True)
    portfolio_performance = pd.DataFrame(ef.portfolio_performance(risk_free_rate=0), 
                                     index = ["Expected annual return", "Annual volatility", "Sharpe Ratio"],
                                     columns = ["MVO"])
    performance_data = portfolio_performance.to_dict()
    return weights, performance_data

def minimize_risk_given_return(prices, target_return, market_neutral=True):
    """Minimize risk for a given return, market-neutral."""
    mu = expected_returns.capm_return(prices)
    S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
    
    ef = EfficientFrontier(mu, S, weight_bounds=(None, None))
    ef.add_objective(objective_functions.L2_reg)
    ef.efficient_return(target_return=target_return, market_neutral=market_neutral)
    weights = ef.clean_weights()
    print('------Performance for minimised risk for given return optimised Portfolio-----')
    ef.portfolio_performance(verbose=True)
    portfolio_performance = pd.DataFrame(ef.portfolio_performance(risk_free_rate=0), 
                                     index = ["Expected annual return", "Annual volatility", "Sharpe Ratio"],
                                     columns = ["MVO"])
    performance_data = portfolio_performance.to_dict()
    return weights, performance_data

def efficient_semivariance(prices, mu, benchmark=0, target_return=None):
    """Efficient semi-variance optimization."""
    semicov = risk_models.semicovariance(prices, benchmark=benchmark)
    returns = expected_returns.returns_from_prices(prices).dropna()
    
    es = EfficientSemivariance(mu, returns)
    if target_return:
        es.efficient_return(target_return)
    else:
        es.min_semivariance()
    weights = es.clean_weights()
    print('-----performance for efficient semivarience optimised portfolios------')
    es.portfolio_performance(verbose=True)
    portfolio_performance = pd.DataFrame(es.portfolio_performance(risk_free_rate=0), 
                                     index = ["Expected annual return", "Annual semi-deviation", "Sortino Ratio"],
                                     columns = ["MVO"])
    performance_data = portfolio_performance.to_dict()
    return weights,performance_data

def efficient_cvar(prices, mu, target_cvar):
    """Efficient CVaR optimization."""
    returns = expected_returns.returns_from_prices(prices).dropna()
    
    ec = EfficientCVaR(mu, returns)
    ec.efficient_risk(target_cvar=target_cvar)
    weights = ec.clean_weights()
    print('------performance for Efficient CVaR optimization-------')
    ec.portfolio_performance(verbose=True)
    portfolio_performance = pd.DataFrame(ec.portfolio_performance(), 
                                     index = ["Expected annual return", "Conditional Value at Risk"],
                                     columns = ["MVO"])
    performance_data = portfolio_performance.to_dict()
    return weights, performance_data

def optimize_hrp(prices):
    """
    Optimize portfolio using Hierarchical Risk Parity (HRP).
    
    Args:
        prices (DataFrame): DataFrame containing historical prices of assets.
    
    Returns:
        weights (dict): Dictionary containing the optimized weights for assets.
    """
    # Compute expected returns
    rets = expected_returns.returns_from_prices(prices)
    
    # Optimize using HRP
    hrp = HRPOpt(rets)
    hrp.optimize()
    weights = hrp.clean_weights()
    portfolio_performance = pd.DataFrame(hrp.portfolio_performance(risk_free_rate=0), 
                                     index = ["Expected annual return", "Annual volatility", "Sharpe Ratio"],
                                     columns = ["MVO"])
    performance_data = portfolio_performance.to_dict()
    
    return weights, performance_data


# Download historical prices
#prices = download_prices(tickers)

# Compute expected returns
#mu = get_expected_returns(prices)

# Optimize for minimum volatility
#weights = optimize_min_volatility(prices)
#alloc = perform_discrete_allocation(weights, prices)

#print(weights)
#print(performance)
# Perform discrete allocation


