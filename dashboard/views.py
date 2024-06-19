import csv
import json
from django.urls import reverse

import random
import requests
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.conf import settings
from .models import Portfolio, StockHolding
from riskprofile.models import RiskProfile
from riskprofile.views import risk_profile
import yfinance as yf
import datetime
# AlphaVantage API
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
import subprocess as sp
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from typing import List
from itertools import combinations
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for Matplotlib
import math
from datetime import datetime, timedelta
import yfinance as yf
import os
import json

from .optimizer import download_prices,optimize__volatility,efficient__cvar, get_expected_returns, optimize_min_volatility, perform_discrete_allocation, max_sharpe_with_sector_constraints, maximize_return_given_risk, minimize_risk_given_return, efficient_cvar

import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging

# Get an instance of a logger
logger = logging.getLogger(__name__)

def get_alphavantage_key():
  alphavantage_keys = [
    settings.ALPHAVANTAGE_KEY1,
    settings.ALPHAVANTAGE_KEY2,
    settings.ALPHAVANTAGE_KEY3,

  ]
  return random.choice(alphavantage_keys)



@login_required
def dashboard(request):
  if RiskProfile.objects.filter(user=request.user).exists():
    try:
      portfolio = Portfolio.objects.get(user=request.user)
    except:
      portfolio = Portfolio.objects.create(user=request.user)
    portfolio.update_investment()
    holding_companies = StockHolding.objects.filter(portfolio=portfolio)
    holdings = []
    sectors = [[], []]
    sector_wise_investment = {}
    stocks = [[], []]
    for c in holding_companies:
      company_symbol = c.company_symbol
      company_name = c.company_name
      number_shares = c.number_of_shares
      investment_amount = c.investment_amount
      average_cost = investment_amount / number_shares
      holdings.append({
        'CompanySymbol': company_symbol,
        'CompanyName': company_name,
        'NumberShares': number_shares,
        'InvestmentAmount': investment_amount,
        'AverageCost': average_cost,
      })
      stocks[0].append(round((investment_amount / portfolio.total_investment) * 100, 2))
      stocks[1].append(company_symbol)
      if c.sector in sector_wise_investment:
        sector_wise_investment[c.sector] += investment_amount
      else:
        sector_wise_investment[c.sector] = investment_amount
    for sec in sector_wise_investment.keys():
      sectors[0].append(round((sector_wise_investment[sec] / portfolio.total_investment) * 100, 2))
      sectors[1].append(sec)

    # Adding
    news = fetch_news()
    ###

    context = {
      'holdings': holdings,
      'totalInvestment': portfolio.total_investment,
      'stocks': stocks,
      'sectors': sectors,
      'news': news
    }

    return render(request, 'dashboard/dashboard.html', context)
  else:
    return redirect(risk_profile)


def get_portfolio_insights(request):
    try:
        portfolio = Portfolio.objects.get(user=request.user)
        holding_companies = StockHolding.objects.filter(portfolio=portfolio)
        
        portfolio_beta = 0
        portfolio_pe = 0
        tickers = [holding.company_symbol for holding in holding_companies]
        for c in holding_companies:
            stock_data = yf.Ticker(c.company_symbol)
            
            # Get beta and PE ratio from yfinance
            try:
                beta = "{:.2f}".format(stock_data.info['beta'])
            except KeyError:
                beta = 0  # Set a default value if beta is not available
            try:
                pe_ratio = "{:.2f}".format(stock_data.info['trailingPE'])
            except KeyError:
                pe_ratio = 0  # Set a default value if PE ratio is not available
            
            # Update portfolio beta and PE ratio weighted by investment amount
            portfolio_beta += float(beta) * (c.investment_amount / portfolio.total_investment)
            portfolio_pe += float(pe_ratio) * (c.investment_amount / portfolio.total_investment)
        
        # Fetch user's risk profile
        risk_profile = RiskProfile.objects.get(user=request.user).category
        
        return JsonResponse({
            "PortfolioBeta": portfolio_beta,
            "PortfolioPE": portfolio_pe,
            "RiskProfile": risk_profile,
            "CurrentStockHoldings": tickers
        })
    except Exception as e:
        return JsonResponse({"Error": str(e)})

def update_values(request):
    try:
        portfolio = Portfolio.objects.get(user=request.user)
        current_value = 0
        unrealized_pnl = 0
        growth = 0
        holding_companies = StockHolding.objects.filter(portfolio=portfolio)
        stockdata = {}
        for c in holding_companies:
            # Fetching stock data using yfinance
            stock_data = yf.Ticker(c.company_symbol)
            last_trading_price = stock_data.history(period="1d")['Close'].iloc[-1]  # Get last trading price
            pnl = (last_trading_price * c.number_of_shares) - c.investment_amount
            net_change = pnl / c.investment_amount
            stockdata[c.company_symbol] = {
                'LastTradingPrice': last_trading_price,
                'PNL': pnl,
                'NetChange': net_change * 100
            }
            current_value += (last_trading_price * c.number_of_shares)
            unrealized_pnl += pnl
        if portfolio.total_investment != 0:
            growth = unrealized_pnl / portfolio.total_investment
        return JsonResponse({
            "StockData": stockdata, 
            "CurrentValue": current_value,
            "UnrealizedPNL": unrealized_pnl,
            "Growth": growth * 100
        })
    except Exception as e:
        return JsonResponse({"Error": str(e)})


def get_financials(request):
    try:
        symbol = request.GET.get('symbol')
        stock_data = yf.Ticker(symbol)
        data = yf.download(symbol, period="1y")

        #Get the 52-week high from the 'High' column
        data_inf = yf.download(symbol, period="1y").info
        fifty_two_week_low = data['High'].min()
        fifty_two_week_high = data['High'].max()

        # Format Beta if it's not None
        beta = stock_data.info.get('beta')
        formatted_beta = "{:.2f}".format(beta) if beta is not None else "N/A"

        financials = {
            "52WeekHigh": fifty_two_week_high,
            "52WeekLow": fifty_two_week_low,
            "Beta": formatted_beta,
            "BookValue": stock_data.info.get('bookValue'),
            "EBITDA": stock_data.info.get('ebitda'),
            "EVToEBITDA": stock_data.info.get('enterpriseToEbitda'),
            "OperatingMarginTTM": stock_data.info.get('operatingMargins'),
            "PERatio": stock_data.info.get('forwardPE'),
            "PriceToBookRatio": stock_data.info.get('priceToBook'),
            "ProfitMargin": stock_data.info.get('profitMargins'),
            "ReturnOnAssetsTTM": stock_data.info.get('returnOnAssets'),
            "ReturnOnEquityTTM": stock_data.info.get('returnOnEquity'),
            "Sector": "N/A"  # Yahoo Finance doesn't provide sector information directly
        }
        
        return JsonResponse({ "financials": financials })
    except Exception as e:
        return JsonResponse({"Error": str(e)})


def add_holding(request):
    if request.method == "POST":
        try:
            portfolio = Portfolio.objects.get(user=request.user)
            holding_companies = StockHolding.objects.filter(portfolio=portfolio)
            
            # Retrieve ticker from the company input
            company_symbol = request.POST['company']

            # You may need to extract the ticker from the input if it contains both ticker and company name
            # Example: ticker = company_symbol.split('(')[1].split(')')[0]

            # Other form fields
            date = request.POST['date']
            number_stocks = int(request.POST['number-stocks'])

            # Fetching stock data using yfinance
            stock_data = yf.Ticker(company_symbol)
            history = stock_data.history(period="1d")  # Fetch daily historical data

            if history.empty:
                raise ValueError("Error: No data available for the specified symbol")

            # Extracting the latest closing price
            buy_price = history['Close'].iloc[-1]

            found = False
            for c in holding_companies:
                if c.company_symbol == company_symbol:
                    c.buying_value.append([buy_price, number_stocks])
                    c.save()
                    found = True

            if not found:
                c = StockHolding.objects.create(
                    portfolio=portfolio, 
                    company_symbol=company_symbol,
                    number_of_shares=number_stocks
                )
                c.buying_value.append([buy_price, number_stocks])
                c.save()

            return JsonResponse({"status": "Success"})
        except ValueError as e:
            print("Value Error:", e)
            return JsonResponse({"status": "Error", "message": str(e)})
        except Exception as e:
            print("Error:", e)
            return JsonResponse({"status": "Error", "message": str(e)})



def send_company_list(request):
    search_term = request.GET.get('q', None)
    if search_term:
        search_term = search_term.lower()  # Convert search term to lowercase for case-insensitive search

    rows = []
    with open('yahoo-listed.csv', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)  # Skip header row
        for row in csv_reader:
            ticker = row[0]
            company_name = row[1]
            if not search_term or search_term in ticker.lower() or search_term in company_name.lower():  # Filter based on search term
                rows.append({'id': ticker, 'text': f'{company_name} ({ticker})'})  # Add matched rows to the list with ticker included in the text

    return JsonResponse({"data": rows})

def fetch_news():
  query_params = {
    "country": "us",
    "category": "business",
    "sortBy": "top",
    "apiKey": settings.NEWSAPI_KEY
  }
  main_url = "https://newsapi.org/v2/top-headlines"
  # fetching data in json format
  res = requests.get(main_url, params=query_params)
  open_bbc_page = res.json()
  # getting all articles in a string article
  article = open_bbc_page["articles"]
  results = []
  for ar in article:
    results.append([ar["title"], ar["description"], ar["url"]])
  # Make news as 2 at a time to show on dashboard
  news = zip(results[::2], results[1::2])
  if len(results) % 2:
    news.append((results[-1], None))
  return news


@csrf_exempt
def prediction(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            if not data or 'ticker' not in data:
                return JsonResponse({'error': 'Please provide a valid ticker symbol'}, status=400)

            quote = data['ticker']
            forecast_out = data['predictionperiod']

            def get_historical(quote):
                end = datetime.now()
                start = datetime(end.year-3,end.month,end.day)
                data = yf.download(quote, start=start, end=end)
                df = pd.DataFrame(data=data)
                df.to_csv(f'{quote}.csv')
                if df.empty:
                    ts = TimeSeries(key='N6A6QT6IBFJOPJ70', output_format='pandas')
                    data, meta_data = ts.get_daily_adjusted(symbol='NSE:'+quote, outputsize='full')
                    data = data.head(503).iloc[::-1]
                    data = data.reset_index()
                    df = pd.DataFrame({
                        'Date': data['date'],
                        'Open': data['1. open'],
                        'High': data['2. high'],
                        'Low': data['3. low'],
                        'Close': data['4. close'],
                        'Adj Close': data['5. adjusted close'],
                        'Volume': data['6. volume']
                    })
                    df.to_csv(f'{quote}.csv', index=False)
                return df

            def ARIMA_ALGO(df, forecast_out):
                    def parser(x):
                        return datetime.strptime(x, '%Y-%m-%d')

                    def arima_model(train, test):
                        history = [x for x in train]
                        predictions = list()
                        for t in range(len(test)):
                            model = ARIMA(history, order=(6, 1, 0))
                            model_fit = model.fit()
                            output = model_fit.forecast()
                            yhat = output[0]
                            predictions.append(yhat)
                            obs = test[t]
                            history.append(obs)
                        return predictions

                    data = df.reset_index()
                    data['Price'] = data['Close']
                    Quantity_date = data[['Price', 'Date']]
                    Quantity_date.index = Quantity_date['Date'].map(lambda x: parser(x))
                    Quantity_date['Price'] = Quantity_date['Price'].map(lambda x: float(x))
                    Quantity_date = Quantity_date.fillna(Quantity_date.bfill())
                    Quantity_date = Quantity_date.drop(['Date'], axis=1)

                    quantity = Quantity_date.values
                    size = int(len(quantity) * 0.80)
                    train, test = quantity[0:size], quantity[size:len(quantity)]
                    predictions = arima_model(train, test)
                    
                    arima_pred = predictions[-2]
                    error_arima = math.sqrt(mean_squared_error(test, predictions))


                    return arima_pred, error_arima

            def LSTM_ALGO(df):
                dataset_train = df.iloc[0:int(0.8*len(df)), :]
                dataset_test = df.iloc[int(0.8*len(df)):, :]
                training_set = df.iloc[:, 4:5].values
                from sklearn.preprocessing import MinMaxScaler
                sc = MinMaxScaler(feature_range=(0, 1))
                training_set_scaled = sc.fit_transform(training_set)
                X_train, y_train = [], []
                for i in range(7, len(training_set_scaled)):
                    X_train.append(training_set_scaled[i-7:i, 0])
                    y_train.append(training_set_scaled[i, 0])
                X_train = np.array(X_train)
                y_train = np.array(y_train)
                X_forecast = np.array(X_train[-1, 1:])
                X_forecast = np.append(X_forecast, y_train[-1])
                X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
                X_forecast = np.reshape(X_forecast, (1, X_forecast.shape[0], 1))
                from keras.models import Sequential
                from keras.layers import Dense, Dropout, LSTM
                regressor = Sequential()
                regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
                regressor.add(Dropout(0.1))
                regressor.add(LSTM(units=50, return_sequences=True))
                regressor.add(Dropout(0.1))
                regressor.add(LSTM(units=50, return_sequences=True))
                regressor.add(Dropout(0.1))
                regressor.add(LSTM(units=50))
                regressor.add(Dropout(0.1))
                regressor.add(Dense(units=1))
                regressor.compile(optimizer='adam', loss='mean_squared_error')
                regressor.fit(X_train, y_train, epochs=25, batch_size=32)
                real_stock_price = dataset_test.iloc[:, 4:5].values
                dataset_total = pd.concat((dataset_train['Close'], dataset_test['Close']), axis=0)
                testing_set = dataset_total[len(dataset_total)-len(dataset_test)-7:].values
                testing_set = testing_set.reshape(-1, 1)
                testing_set = sc.transform(testing_set)
                X_test = []
                for i in range(7, len(testing_set)):
                    X_test.append(testing_set[i-7:i, 0])
                X_test = np.array(X_test)
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
                predicted_stock_price = regressor.predict(X_test)
                predicted_stock_price = sc.inverse_transform(predicted_stock_price)
     
                error_lstm = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
                forecasted_stock_price = regressor.predict(X_forecast)
                forecasted_stock_price = sc.inverse_transform(forecasted_stock_price)
                lstm_pred = forecasted_stock_price[0, 0]
                return lstm_pred, error_lstm
    
            def LIN_REG_ALGO(df, n):
                forecast_out = int(n)
                df['Close after n days'] = df['Close'].shift(-forecast_out)
                df_new = df[['Close', 'Close after n days']]
                y = np.array(df_new.iloc[:-forecast_out, -1])
                y = np.reshape(y, (-1, 1))
                X = np.array(df_new.iloc[:-forecast_out, 0:-1])
                X_to_be_forecasted = np.array(df_new.iloc[-forecast_out:, 0:-1])
                X_train = X[0:int(0.8*len(df)), :]
                X_test = X[int(0.8*len(df)):, :]
                y_train = y[0:int(0.8*len(df)), :]
                y_test = y[int(0.8*len(df)):, :]
                from sklearn.preprocessing import StandardScaler
                sc = StandardScaler()
                X_train = sc.fit_transform(X_train)
                X_test = sc.transform(X_test)
                X_to_be_forecasted = sc.transform(X_to_be_forecasted)
                from sklearn.linear_model import LinearRegression
                clf = LinearRegression(n_jobs=-1)
                clf.fit(X_train, y_train)
                y_test_pred = clf.predict(X_test)
                y_test_pred = y_test_pred * (1.04)
                error_lr = math.sqrt(mean_squared_error(y_test, y_test_pred))
                forecast_set = clf.predict(X_to_be_forecasted)
                forecast_set = forecast_set * (1.04)
                mean = forecast_set.mean()
                lr_pred = forecast_set[0, 0]
                return df, lr_pred, forecast_set, mean, error_lr
            

            def predict_price(quote):
                try:
                    get_historical(quote)
                except:
                    return {"error": "Stock symbol not found."}
                else:
                    df = pd.read_csv(''+quote+'.csv')
                    df = df.dropna()
                    today_stock=df.iloc[-1:]
                    last_7_days = df.tail(30)
                    actual_data = last_7_days['Close'].tolist()
                    actual_dates = last_7_days['Date'].tolist()

                    last_date = datetime.strptime(actual_dates[-1], '%Y-%m-%d')
                    forecast_dates = [(last_date + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(forecast_out)]

                    code_list=[]
                    for i in range(0,len(df)):
                        code_list.append(quote)
                    df2=pd.DataFrame(code_list,columns=['Code'])
                    df2 = pd.concat([df2, df], axis=1)
                    df=df2

                    arima_pred, error_arima = ARIMA_ALGO(df,forecast_out)
                    lstm_pred, error_lstm = LSTM_ALGO(df)
                    arima_pred =  round(arima_pred, 2)
                    lstm_pred = round(lstm_pred, 2)
                    df, lr_pred, forecast_set, mean, error_lr = LIN_REG_ALGO(df,forecast_out)
                    final_comp_pred = (arima_pred + lstm_pred + lr_pred) / 3
                    final_comp_pred = round(final_comp_pred, 2)
                    #print("arima"+ arima_pred)

                    return final_comp_pred, arima_pred, lstm_pred, error_lr, forecast_set, mean, today_stock, actual_data, actual_dates,forecast_dates

            final_comp_pred, arima_pred, lstm_pred, error_lr, forecast_set, mean, today_stock, actual_data,actual_dates,forecast_dates = predict_price(quote)

            response = {
                'prediction': final_comp_pred,
                'arima_pred': str(arima_pred),
                'lstm_pred': str(lstm_pred),
                'lr_error': error_lr,
                'forecast_set': forecast_set.tolist(),
                'forecast_mean': mean,
                'today_stock': today_stock.to_dict(orient='records'),
                'actual_data': actual_data,
                'actual_dates': actual_dates,  # Include the actual dates
                'forecast_dates': forecast_dates
            }
            return JsonResponse(response)

        except Exception as e:
            logger.log(e)
            return JsonResponse({'error': str(e)}, status=500)

    else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)


@csrf_exempt
@login_required
def optimize_portfolio(request):
    if request.method == 'GET':
        try:
            # Fetch the user's portfolio
            portfolio = Portfolio.objects.get(user=request.user)
            logger.debug("Fetched portfolio for user %s", request.user)
            
            holdings = StockHolding.objects.filter(portfolio=portfolio)
            total_portfolio_value = portfolio.total_investment

            # Fetch the user's risk profile
            risk_profile_obj = RiskProfile.objects.get(user=request.user)
            risk_profile = risk_profile_obj.category
            print(risk_profile)  # Adjust based on your actual RiskProfile model
            logger.debug("Fetched risk profile for user %s: %s", request.user, risk_profile)

            # Prepare tickers, sector mapper, and quantities
            tickers = [holding.company_symbol for holding in holdings]
            #tickers_str = ','.join(tickers)
            #sector_mapper = {holding.company_symbol: holding.sector for holding in holdings}
            quantities = [holding.number_of_shares for holding in holdings]
            logger.debug("Tickers: %s", tickers)

            logger.debug("Quantities: %s", quantities)

            response = {}
            print(tickers)
            # Fetching historical prices
            prices, valid_tickers, errors = download_prices(tickers)
            logger.debug("Downloaded prices: %s", prices)
            logger.debug("Errors: %s", errors)

            if errors:
                response = {
                    "error": "Failed to download prices for some tickers",
                    "errors": errors
                }
            else:
                valid_prices = prices[valid_tickers]
                mu  = get_expected_returns(valid_prices)
                # Choose optimization function based on risk profile
                if risk_profile == 'Conservative':
                    logger.debug("Optimizing for Conservative profile")
                    result = optimize_min_volatility(valid_prices)
                elif risk_profile == 'Balanced':
                    logger.debug("Optimizing for Balanced profile")
                    result = max_sharpe_with_sector_constraints(valid_prices)
                elif risk_profile == 'Assertive':
                    logger.debug("Optimizing for Assertive profile")
                    result = result = maximize_return_given_risk(valid_prices, target_volatility=0.25)
                elif risk_profile == 'Aggressive':
                    logger.debug("Optimizing for Aggressive profile")
                    result = efficient_cvar(valid_prices, target_cvar=0.2)
                else:
                    logger.error("Invalid risk profile for user %s", request.user)
                    return JsonResponse({"error": "Invalid risk profile"}, status=400)

                logger.debug("Optimization result: %s", result)

                # Unpack result safely
                if isinstance(result, tuple) and len(result) == 2:
                    weights, performance_data = result
                else:
                    raise ValueError("Unexpected result format from optimization function")

                logger.debug("Weights: %s", weights)
                logger.debug("Performance Data: %s", performance_data)

                # Perform discrete allocation
                allocations, leftover = perform_discrete_allocation(weights, valid_prices, total_portfolio_value=total_portfolio_value)
                logger.debug("Allocations: %s", allocations)
                logger.debug("Leftover: %s", leftover)

                response = {
                    "weights": weights,
                    "allocations": allocations,
                    "leftover": leftover,
                    "performance": performance_data
                }

            return JsonResponse(response)

        except Portfolio.DoesNotExist:
            logger.error("Portfolio not found for user %s", request.user)
            return JsonResponse({"error": "Portfolio not found"}, status=404)
        except RiskProfile.DoesNotExist:
            logger.error("Risk profile not found for user %s", request.user)
            return JsonResponse({"error": "Risk profile not found"}, status=404)
        except Exception as e:
            logger.error("Error optimizing portfolio for user %s: %s", request.user, str(e))
            return JsonResponse({"error": str(e)}, status=500)

    else:
        return JsonResponse({"error": "Method not allowed"}, status=405)