from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
import pandas as pd
import numpy as np
import yfinance as yf
from itertools import combinations
from riskprofile.models import RiskProfile





def fetch_data(tickers: List[str], years_simulated: int):
    current_date = datetime.datetime.today().strftime('%Y-%m-%d')
    start_date = datetime.datetime.today() - datetime.timedelta(days=years_simulated * 365)
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = current_date

    tickers_str = ' '.join(tickers)
    data = yf.download(tickers=tickers_str, start=start_date, end=end_date, progress=False)['Adj Close']
    data.reset_index(inplace=True)

    risk_free_rate = yf.Ticker("^TNX").history(period="5y")['Close']
    risk_free_rate = risk_free_rate.rename('risk_free_rate')
    if isinstance(risk_free_rate.index, pd.DatetimeIndex) and risk_free_rate.index.tzinfo is not None:
        risk_free_rate.index = risk_free_rate.index.tz_localize(None)

    data = pd.merge(data, risk_free_rate, left_on='Date', right_index=True, how='left')
    data['risk_free_rate'] = data['risk_free_rate'].ffill()
    data = data.dropna(axis=1, how='all')

    return data

def portfolio_metrics(dataframe, columns_to_use, years_simulated=5, set_size=3, var_confidence=0.05, cvar_confidence=0.05, trading_days_per_year=252, user=None):
    # Initialize results DataFrame
    results_list = []

    # Check if SPY exists for Portfolio Beta calculation
    include_beta = 'SPY' in dataframe.columns

    # Generate all unique combinations
    all_combinations = list(combinations(columns_to_use, set_size))

    # Filter DataFrame based on years simulated
    latest_date = dataframe['Date'].max()
    earliest_date = pd.Timestamp(latest_date) - pd.DateOffset(years=years_simulated)
    filtered_df = dataframe[dataframe['Date'] >= earliest_date]

    for selected_columns in all_combinations:
        returns = filtered_df[list(selected_columns)].pct_change(fill_method=None).dropna()
        risk_free_rate = filtered_df['risk_free_rate'].loc[returns.index].mean() / 100  # Convert to decimal
        weights = np.array([1./set_size] * set_size)

        # Adjust weights based on user's risk tolerance and investment goals
        if user:
            if user.risk_tolerance == 'Aggressive':
                weights *= 1.2
            elif user.risk_tolerance == 'Conservative':
                weights *= 0.8
            
            if user.investment_goals == 'Short-Term Gains':
                # Emphasize total return for short-term gains
                metrics_score = total_return
            elif user.investment_goals == 'Long-Term Growth':
                # Emphasize CAGR for long-term growth
                metrics_score = cagr

        # Sharpe Ratio
        expected_return = np.sum(returns.mean() * weights) * trading_days_per_year
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * trading_days_per_year, weights)))
        sharpe_ratio = (expected_return - risk_free_rate) / portfolio_volatility

        # VaR and CVaR
        portfolio_return_series = (returns * weights).sum(axis=1)
        var_value = portfolio_return_series.quantile(var_confidence)
        cvar_value = portfolio_return_series[portfolio_return_series <= var_value].mean()

        # Maximum Drawdown
        cumulative_returns = (1 + portfolio_return_series).cumprod()
        max_value = cumulative_returns.cummax()
        drawdowns = cumulative_returns / max_value - 1
        max_drawdown = drawdowns.min()

        # Sortino Ratio
        negative_returns = portfolio_return_series[portfolio_return_series < 0]
        downside_deviation = negative_returns.std() * np.sqrt(trading_days_per_year)
        sortino_ratio = (expected_return - risk_free_rate) / downside_deviation

        # Portfolio Beta
        if include_beta:
            spy_returns = filtered_df['SPY'].diff() / filtered_df['SPY'].shift(1)
            aligned_data = pd.concat([portfolio_return_series, spy_returns], axis=1).dropna()
            portfolio_beta = aligned_data.iloc[:, 0].cov(aligned_data.iloc[:, 1]) / aligned_data.iloc[:, 1].var()

        # Total Return and CAGR
        total_return = cumulative_returns.iloc[-1] - 1  # subtract 1 to convert to percentage
        cagr = (cumulative_returns.iloc[-1] ** (1 / years_simulated)) - 1  # again, subtract 1 for percentage

        # Append to results
        metrics = {f'stock_{i+1}': stock for i, stock in enumerate(selected_columns)}
        metrics['Sharpe_Ratio'] = sharpe_ratio
        metrics['VaR'] = var_value
        metrics['CVaR'] = cvar_value
        metrics['Max_Drawdown'] = max_drawdown
        metrics['Sortino_Ratio'] = sortino_ratio
        metrics['Total_Return'] = total_return
        metrics['CAGR'] = cagr
        if include_beta:
            metrics['Portfolio_Beta'] = portfolio_beta

        results_list.append(metrics)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results_list)

    return results_df
@login_required
def portfolio_analysis(request):
    try:
        risk_profile = RiskProfile.objects.get(user=request.user)
    except RiskProfile.DoesNotExist:
        # Handle the case where the user does not have a RiskProfile
        risk_category = "No risk profile found"
    else:
        # Access the category attribute of the RiskProfile
        risk_category = risk_profile.category
        tickers =[]
        years_simulated = 2
        set_size = 3
        risk_tolerance = risk_category
        data =  fetch_data(tickers, years_simulated)
            columns_to_use = data.columns.tolist()
        columns_to_use.remove('Date')
        if 'risk_free_rate' in columns_to_use:
            columns_to_use.remove('risk_free_rate')

        results_df = portfolio_metrics(data, columns_to_use, set_size=set_size, years_simulated=years_simulated)

        # Apply filters based on risk tolerance
        if risk_tolerance == "Higher":
            # Prioritize portfolios with higher Total_Return and Sharpe_Ratio
            results_df = (
                results_df.sort_values(by='Sharpe_Ratio',ascending=False)
                        .sort_values(by='Sortino_Ratio', ascending=False)
                        .sort_values(by='Total_Return', ascending=False)
            )
        elif risk_tolerance == "Lower":
            # Prioritize portfolios with lower VaR, Max_Drawdown, and higher Total_Return
            results_df = (
                results_df.sort_values(by='VaR', ascending=True)
                        .sort_values(by='Total_Return', ascending=False)
                        
            )

        sorted_results = results_df.head(3)
        sorted_results_dict = sorted_results.to_dict(orient='records')
        context = sorted_results_dict


# Create your views here.
