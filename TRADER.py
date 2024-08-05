
from python_bitvavo_api.bitvavo import Bitvavo
import json
import pandas as pd
import numpy as np
import time
import ta
from telegram import Bot
import asyncio
from datetime import datetime, timedelta
import os
import random

api_keys= json.loads(os.getenv('APIKEYS'))
api_key = api_keys['API_KEY']
api_secret = api_keys['API_SECRET']
token = Bot(token=api_keys['token'])
chat_id = api_keys["chat_id"]

base_url = 'https://api.kraken.com'
endpoint = '/0/public/OHLC'

class apibot():
    def __init__(self, file_path, markets):
        self._markets = markets
        self._file_path = file_path


    async def send_telegram_message(self, message):
      try:
          await token.send_message(chat_id=chat_id, text=message, read_timeout=20)
      except TimeoutError:
        print("Failed to send message due to timeout.")


    def load_data(self, file_path):
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                return data
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print('Error loading json data: {e}')
                return []


    def update_file(self, file_path, order):
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    print(data)

                    if not isinstance(data, list):
                        data = []

            except json.JSONDecodeError:
                data = []
        else:
            data = []


        if order['type'] == "Sold":
            for i in data:
                if i['order'] == order['order']:
                    i.update(order)

        elif order['type'] == 'Stoploss':
            for i in data:
                if i['order'] == order['order']:
                    i.update(order)

        else:
            data.append(order)

        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)



    def get_data(self, market):
        url = base_url + endpoint
        params = {
            'pair': {market},  # Het handelspaar, bijvoorbeeld Bitcoin tegen USD
            'interval': 60, 
            'since': int(time.time()) - 2592000
        }

        response = requests.get(url, params=params)
        data = response.json()
        data = data['result'][market]
        data = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'])
        data['market'] = market
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
        data = data.set_index('timestamp')
        data[['open', 'high', 'low', 'close', 'vwap', 'volume', 'count']] = \
            data[['open', 'high', 'low', 'close', 'vwap', 'volume', 'count']].apply(pd.to_numeric)
        data = data.sort_index()
        return data

    def generate_signals(self, df):
      for col in df.columns:
        if df[col].isnull().any():
          print('Nog niet genoeg data')
          break

        else:
          # Golden Cross / Death Cross
          df['Golden_Cross'] = np.where((df['SMA_50'] > df['SMA_200']) & (df['SMA_50'].shift(1) <= df['SMA_200'].shift(1)), True, False)
          df['Death_Cross'] = np.where((df['SMA_50'] < df['SMA_200']) & (df['SMA_50'].shift(1) >= df['SMA_200'].shift(1)), True, False)

          df['up_trend'] = np.where(df['SMA_20'] > df['SMA_50'], True, False) #Op SOLANA sma20 over 50 anders evt #200, 50
          df['down_trend'] = np.where(df['SMA_50'] > df['SMA_20'], True, False) #50, 200
          df['SMA20_Crossover'] = np.where(df['SMA_20'] > df['SMA_50'], True, False)
          df['SMA50_Crossover'] = np.where(df['SMA_20'] < df['SMA_50'], True, False)

          # RSI Overbought / Oversold
          df['RSI_Overbought'] = np.where(df['RSI'] >= 60, True, False)
          df['RSI_Oversold'] = np.where(df['RSI'] <= 35, True, False)
          df['RSI_Overbought_MACD'] = np.where(df['RSI'] >= 65, True, False)
          df['RSI_Oversold_MACD'] = np.where(df['RSI'] <= 45, True, False)

          # MACD Crossovers
          df['MACD_Bullish'] = np.where((df['MACD'] > df['MACD_signal']) & (df['MACD'].shift(1) <= df['MACD_signal'].shift(1)), True, False)
          df['MACD_Bearish'] = np.where((df['MACD'] < df['MACD_signal']) & (df['MACD'].shift(1) >= df['MACD_signal'].shift(1)), True, False)
            
          df['Bullish'] = np.where(df['SMA_20'] > df['SMA_200'], True, False)
          df['Bearish'] = np.where(df['SMA_200'] > df['SMA_20'], True, False)
      
          
          # Bollinger Bands Cross
          df['Bollinger_Breakout_High'] = np.where((df['close'] > df['Bollinger_High']), True, False)
          df['Bollinger_Breakout_Low'] = np.where((df['close'] < df['Bollinger_Low']), True, False)


          return df

    def add_indicators(self, df):
        # Beweeglijke gemiddelden

        df['SMA_50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['SMA_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['SMA_200'] = ta.trend.sma_indicator(df['close'], window=200)
        df['EMA_50'] = ta.trend.ema_indicator(df['close'], window=50)
        df['EMA_200'] = ta.trend.ema_indicator(df['close'], window=200)

        # Relatieve sterkte-index (RSI)
        df['RSI'] = ta.momentum.rsi(df['close'], window=14)

        # Moving Average Convergence Divergence (MACD)
        df['MACD'] = ta.trend.macd(df['close'])
        df['MACD_signal'] = ta.trend.macd_signal(df['close'])

        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['Bollinger_High'] = bollinger.bollinger_hband()
        df['Bollinger_Low'] = bollinger.bollinger_lband()

        # Commodity Channel Index (CCI)
        df['CCI'] = ta.trend.cci(df['high'], df['low'], df['close'], window=20)

        # Stochastische Oscillator
        stoch = ta.momentum.stoch(df['high'], df['low'], df['close'], window=14, smooth_window=3)
        df['Stoch_K'] = stoch
        df['Stoch_D'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'], window=14, smooth_window=3)
        df['SMA_20_above_SMA_200'] = df[['SMA_20', 'SMA_200']].apply(lambda row: row['SMA_20'] > row['SMA_200'], axis=1)
        df['SMA_above'] = df['SMA_20_above_SMA_200'].rolling(window=48).sum() == 12
        
        df['SMA_200_above_SMA_20'] = df[['SMA_20', 'SMA_200']].apply(lambda row: row['SMA_20'] < row['SMA_200'], axis=1)
        df['SMA_below'] = df['SMA_200_above_SMA_20'].rolling(window=48).sum() == 12

        # On Balance Volume (OBV)
        df['OBV'] = ta.volume.on_balance_volume(df['close'], df['volume'])

        # Ichimoku Cloud
        ichimoku = ta.trend.IchimokuIndicator(df['high'], df['low'], window1=9, window2=26, window3=52)
        df['Ichimoku_Conversion'] = ichimoku.ichimoku_conversion_line()
        df['Ichimoku_Base'] = ichimoku.ichimoku_base_line()
        df['Ichimoku_A'] = ichimoku.ichimoku_a()
        df['Ichimoku_B'] = ichimoku.ichimoku_b()

        return df

    # Functie om signalen te controleren
    async def check_signals(self, df):
        last_index = df.index[-1]
        last_row = df.iloc[-1]

        print("Laatste data:")
        print(last_row, last_index)
        print('--------------------------------------------------------')

        
        order_number = random.randint(1000, 9999)
        
        #Going long
        if df.loc[last_index, ['Bullish']].all():
            indicators_buy = df.loc[last_index, ['SMA20_Crossover', 'SMA_above', 'Bollinger_Breakout_Low']]
            indicators_sell = df.loc[last_index, ['RSI_Overbought']]
            if indicators_buy.all():
                buy_message = f"Koop:\n Positie: Long {last_row['market']} {last_row['close']}"
                buy_order = {'type': 'Bought', 'strategy': 'Long', 'symbol': last_row['market'],
                                                    'time': str(last_index.to_pydatetime()),
                                                    'closing_price': float(last_row['close']),
                                                    'order': order_number, 'strategy': 'RSI_Oversold, up_trend'}

      
            print(buy_order)
            self.update_file(self._file_path, buy_order)
            await self.send_telegram_message(buy_message)

        
        #take profit / Stop loss
        if self.load_data(self._file_path) is not None:
            for i in self.load_data(self._file_path):
                stop_limit = float(i['closing_price']) * 0.96
                if i['type'] == 'Bought' and i['symbol'] == last_row['market'] and \
                        float(last_row['close']) <= float(i['closing_price']) * 0.97 >= stop_limit and i['strategy'] == 'Long':
                                   
                    percentage_loss = (float(i['closing_price']) - float(last_row['close'])) * 100 / float(i['closing_price'])
                    percentage_loss = format(percentage_loss, ".2f")

                    stoploss_message = f"Stoploss:\n Positie: Long {last_row['market']} prijs: {last_row['close']}\n" \
                                       f"percentage loss: {percentage_loss}"

                    stoploss_order = {'type': 'Stoploss', "symbol": last_row['market'], 'order': i['order'],
                                                           'time': str(last_index.to_pydatetime()),
                                                           'closing_price': float(last_row['close']),
                                                           'aankoopprijs': float(i['closing_price']),
                                                           'aankoopdatum': str(i['time']),
                                                           'percentage_loss': percentage_loss}
        
                    print(stoploss_order)
                    self.update_file(self._file_path, stoploss_order)
                    await self.send_telegram_message(stoploss_message)

                elif indicators_sell.all():                                                               
                    if i['type'] == 'Bought' and i['symbol'] == last_row['market'] and \
                            float(last_row['close']) >= float(i['closing_price']) * 1.10 and \
                            i['strategy'] == 'Long':
                                
                        percentage = (float(last_row['close']) - float(i['closing_price'])) / float(i['closing_price']) * 100
                        percentage = format(percentage, ".2f")
                        sell_order = {'type': 'Sold', 'symbol': last_row['market'],
                                                           'order': i['order'],
                                                           'time': str(last_index.to_pydatetime()),
                                                           'closing_price': float(last_row['close']),
                                                           'aankoopprijs': float(i['closing_price']),
                                                           'aankoopdatum': str(i['time']),
                                                           'percentage_gain': percentage}

                        sell_message = f"Verkoop:\n {last_row['market']} prijs: {last_row['close']} " \
                                       f"aankoopkoers: {float(i['closing_price'])}\n " \
                                       f"percentage gained: {percentage}"

                        print(sell_order)
                        self.update_file(self._file_path, sell_order)
                        await self.send_telegram_message(sell_message)

        

        #Going short
        if df.loc[last_index, ['Bearish']].all():
            indicators_buy = df.loc[last_index, ['SMA50_Crossover', 'SMA_below', 'Bollinger_Breakout_High']]
            indicators_sell = df.loc[last_index, ['Bollinger_Breakout_Low']]
            if indicators_buy.all():
                buy_message = f"Koop:\n Positie: Short {last_row['market']} {last_row['close']}"
                buy_order = {'type': 'Bought', 'strategy': 'Short', 'symbol': last_row['market'],
                                                    'time': str(last_index.to_pydatetime()),
                                                    'closing_price': float(last_row['close']),
                                                    'order': order_number, 'strategy': 'RSI_Oversold, up_trend'}

      
                print(buy_order)
                self.update_file(self._file_path, buy_order)
                await self.send_telegram_message(buy_message)

        
        #take profit / Stop loss
        if self.load_data(self._file_path) is not None:
            for i in self.load_data(self._file_path):
                stop_limit = float(i['closing_price']) * 1.06
                if i['type'] == 'Bought' and i['symbol'] == last_row['market'] and \
                        float(last_row['close']) >= float(i['closing_price']) * 1.04 <= stop_limit and i['strategy'] == 'Short':
                                    
                    percentage_loss = (float(last_row['close']) - float(i['closing_price'])) * 100 / float(i['closing_price'])
                    percentage_loss = format(percentage_loss, ".2f")

                    stoploss_message = f"Stoploss:\n Positie: Short {last_row['market']} prijs: {last_row['close']}\n" \
                                       f"percentage loss: {percentage_loss}"

                    stoploss_order = {'type': 'Stoploss', "symbol": last_row['market'], 'order': i['order'],
                                                           'time': str(last_index.to_pydatetime()),
                                                           'closing_price': float(last_row['close']),
                                                           'aankoopprijs': float(i['closing_price']),
                                                           'aankoopdatum': str(i['time']),
                                                           'percentage_loss': percentage_loss}
        
                    print(stoploss_order)
                    self.update_file(self._file_path, stoploss_order)
                    await self.send_telegram_message(stoploss_message)

                elif indicators_sell.all():                                                               
                    if i['type'] == 'Bought' and i['symbol'] == last_row['market'] and \
                            float(last_row['close']) >= float(i['closing_price']) * 1.10 and \
                            i['strategy'] == 'Short':
                                
                        percentage = (float(i['closing_price']) - float(last_row['close'])) / float(i['closing_price']) * 100
                        percentage = format(percentage, ".2f")
                        sell_order = {'type': 'Sold', 'symbol': last_row['market'],
                                                           'order': i['order'],
                                                           'time': str(last_index.to_pydatetime()),
                                                           'closing_price': float(last_row['close']),
                                                           'aankoopprijs': float(i['closing_price']),
                                                           'aankoopdatum': str(i['time']),
                                                           'percentage_gain': percentage}

                        sell_message = f"Verkoop:\n {last_row['market']} prijs: {last_row['close']} " \
                                       f"aankoopkoers: {float(i['closing_price'])}\n " \
                                       f"percentage gained: {percentage}"

                        print(sell_order)
                        self.update_file(self._file_path, sell_order)
                        await self.send_telegram_message(sell_message)

        else:
            print('Geen verkoopsignalen gevonden')


    async def main(self, bot):
        # start_time = datetime.now()
        # end_time = start_time + timedelta(hours=15)

        # while datetime.now() < end_time:
        for i in self._markets:
            df = bot.get_data(market=i) 
            df = bot.add_indicators(df)
            data_complete = bot.generate_signals(df)
            await bot.check_signals(data_complete)

            # time.sleep(5)

if __name__ == '__main__':
    # file_path = 'CryptoOrders.json'
    file_path = os.getenv('FILE_PATH')
    bot = apibot(file_path=file_path, markets=['SOLEUR', 'ADAEUR', 'AVAXEUR', 'ICPEUR',
                                               'COTIEUR', 'JUPEUR', 'MANTAEUR', 'CFXEUR'])
    asyncio.run(bot.main(bot))



