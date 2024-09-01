
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
import requests

api_keys= json.loads(os.getenv('API_KEYS'))
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
        self._portfolio = "Portfolio\n"


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
                    
        elif order['type'] == 'Bought':
            if 'last_update' in order.keys():
                for i in data:
                    if order['order'] == i['order']:
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

          # RSI Overbought / Oversold
          df['RSI_Overbought'] = np.where(df['RSI'] >= 60, True, False)
          df['RSI_Oversold'] = np.where(df['RSI'] <= 35, True, False)
            
          # MACD Crossovers
          df['MACD_Bullish'] = np.where((df['MACD'] > df['MACD_signal']) & (df['MACD'].shift(1) <= df['MACD_signal'].shift(1)), True, False)
          df['MACD_Bearish'] = np.where((df['MACD'] < df['MACD_signal']) & (df['MACD'].shift(1) >= df['MACD_signal'].shift(1)), True, False)
      
          # Bollinger Bands Cross
          df['Bollinger_Breakout_High'] = np.where((df['close'] > df['Bollinger_High']), True, False)
          df['Bollinger_Breakout_Low'] = np.where((df['close'] < df['Bollinger_Low']), True, False)

          df['EMA_above'] = (df['EMA_8_above_EMA_13'] &
                             df['EMA_13_above_EMA_21'] &
                             df['EMA_21_above_EMA_55']).rolling(window=20).sum() == 20

          df['EMA_below'] = (~df['EMA_8_above_EMA_13'] &
                             ~df['EMA_13_above_EMA_21'] &
                             ~df['EMA_21_above_EMA_55']).rolling(window=20).sum() == 20

          return df

    def add_indicators(self, df):
        # Beweeglijke gemiddelden

        df['SMA_50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['SMA_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['SMA_200'] = ta.trend.sma_indicator(df['close'], window=200)

        # Relatieve sterkte-index (RSI)
        df['RSI'] = ta.momentum.rsi(df['close'], window=14)

        # Moving Average Convergence Divergence (MACD)
        df['MACD'] = ta.trend.macd(df['close'])
        df['MACD_signal'] = ta.trend.macd_signal(df['close'])

        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['Bollinger_High'] = bollinger.bollinger_hband()
        df['Bollinger_Low'] = bollinger.bollinger_lband()

        df['EMA_8'] = ta.trend.ema_indicator(df['close'], window=8)
        df['EMA_13'] = ta.trend.ema_indicator(df['close'], window=13)
        df['EMA_21'] = ta.trend.ema_indicator(df['close'], window=21)
        df['EMA_55'] = ta.trend.ema_indicator(df['close'], window=55)

        df['EMA_8_above_EMA_13'] = df['EMA_8'] > df['EMA_13']
        df['EMA_13_above_EMA_21'] = df['EMA_13'] > df['EMA_21']
        df['EMA_21_above_EMA_55'] = df['EMA_21'] > df['EMA_55']

        df['EMA_above'] = (df['EMA_8_above_EMA_13'] &
                           df['EMA_13_above_EMA_21'] &
                           df['EMA_21_above_EMA_55']).rolling(window=20).sum() == 20

        df['EMA_below'] = (~df['EMA_8_above_EMA_13'] &
                           ~df['EMA_13_above_EMA_21'] &
                           ~df['EMA_21_above_EMA_55']).rolling(window=20).sum() == 20


        df['volume_MA'] = df['volume'].rolling(window=20).mean()
        df['Bullish'] = (df['EMA_8'] > df['EMA_13']) & (df['EMA_13'] > df['EMA_21']) & (df['EMA_21'] > df['EMA_55'])
        df['Bearish'] = (df['EMA_8'] < df['EMA_13']) & (df['EMA_13'] < df['EMA_21']) & (df['EMA_21'] < df['EMA_55'])
        df['Buy Signal Long'] = df['EMA_above']
        df['Buy Signal Short'] = df['EMA_below']

        return df

    # Functie om signalen te controleren
    async def check_signals(self, df):
        last_index = df.index[-1]
        last_row = df.iloc[-1]
        
        print("Laatste data:")
        print(last_row, last_index)
        print('--------------------------------------------------------')
 
        #Going long
        indicators_buy_long = df.loc[last_index, ['Buy Signal Long']]

        #Going short
        indicators_buy_short = df.loc[last_index, ['Buy Signal Short', 'RSI_Oversold']]

        
        #take profit / Stop loss
        if self.load_data(self._file_path) is not None:
            for i in self.load_data(self._file_path):
                if i['type'] == 'Bought' and i['symbol'] == last_row['market'] and \
                        float(last_row['close']) <= float(i['price_bought']) * 0.95 and i['strategy'] == 'Long':
                            
                    percentage_loss = (float(i['price_bought']) - float(last_row['close'])) * 100 / float(i['price_bought'])
                    percentage_loss = round(percentage, 2)

                    stoploss_message = f"Stoploss:\n Positie: Long\n Market:{last_row['market']} Prijs: {last_row['close']}\n" \
                                       f"percentage loss: {percentage_loss}"

                    stoploss_order = {'type': 'Stoploss', "symbol": last_row['market'], 'order': i['order'],
                                                           'date_stoploss': str(last_index.to_pydatetime()),
                                                           'closing_price': float(last_row['close']),
                                                           'price_bought': float(i['price_bought']),
                                                           'date_bought': str(i['date_bought']),
                                                           'percentage_loss': percentage_loss}
        
                    print(stoploss_order)
                    self.update_file(self._file_path, stoploss_order)
                    await self.send_telegram_message(stoploss_message)
                                                                        
                elif i['type'] == 'Bought' and i['symbol'] == last_row['market'] and i['strategy'] == 'Long':
                    if float(last_row['close']) >= float(i['price_bought']) * 1.05 and i['strategy'] == 'Long' and df['Buy Signal Long'].all() == False:
         
                       percentage = (float(last_row['close']) - float(i['price_bought'])) / float(i['price_bought']) * 100
                       percentage = round(percentage, 2)
                       sell_order = {'type': 'Sold', 'symbol': last_row['market'],
                                                         'order': i['order'],
                                                         'date_sold': str(last_index.to_pydatetime()),
                                                         'closing_price': float(last_row['close']),
                                                         'price_bought': float(i['price_bought']),
                                                         'date_bought': str(i['date_bought']),
                                                         'percentage_gained': percentage}

                       sell_message = f"Verkoop:\n {last_row['market']} prijs: {last_row['close']} " \
                                      f"aankoopkoers: {i['price_bought']}\n " \
                                      f"percentage gained: {percentage}"

                       print(sell_order)
                       self.update_file(self._file_path, sell_order)
                       await self.send_telegram_message(sell_message)

                    
                    else:
                        percentage = (float(last_row['close']) - float(i['price_bought'])) / float(i['price_bought']) * 100
                        percentage = round(percentage, 2)
                        update_order = {'type': 'Bought', 'symbol': last_row['market'],
                                                       'order': i['order'],
                                                       'last_update': str(last_index.to_pydatetime()),
                                                       'closing_price': float(last_row['close']),
                                                       'price_bought': float(i['price_bought']),
                                                       'date_bought': str(i['date_bought']),
                                                       'percentage_gained': percentage}

                        update_message = f"Update:\n {last_row['market']} prijs: {last_row['close']} " \
                                   f"aankoopkoers: {i['price_bought']}\n " \
                                   f"percentage gained: {percentage}"

                        self.update_file(self._file_path, update_order)     

                
                if i['type'] == 'Bought' and i['symbol'] == last_row['market'] and \
                        float(last_row['close']) >= float(i['price_bought']) * 1.05 and i['strategy'] == 'Short':
                                    
                    percentage_loss = (float(last_row['close']) - float(i['price_bought'])) * 100 / float(i['price_bought'])
                    percentage_loss = round(percentage, 2)

                    stoploss_message = f"Stoploss:\n Positie: Short\n Market: {last_row['market']} Prijs: {last_row['close']}\n" \
                                       f"percentage loss: {percentage_loss}"

                    stoploss_order = {'type': 'Stoploss', "symbol": last_row['market'], 'order': i['order'],
                                                           'date_stoploss': str(last_index.to_pydatetime()),
                                                           'closing_price': float(last_row['close']),
                                                           'price_bought': float(i['price_bought']),
                                                           'date_bought': str(i['date_bought']),
                                                           'percentage_loss': percentage_loss}
        
                    print(stoploss_order)
                    self.update_file(self._file_path, stoploss_order)
                    await self.send_telegram_message(stoploss_message)

                                                                             
                elif i['type'] == 'Bought' and i['symbol'] == last_row['market'] and i['strategy'] == 'Short':
                    if float(last_row['close']) <= float(i['price_bought']) * 0.95 and \
                        i['strategy'] == 'Short' and df['Buy Signal Short'].all() == False:
                            
                        percentage = (float(i['price_bought']) - float(last_row['close'])) / float(i['price_bought']) * 100
                        percentage = format(percentage, ".2f")
                        sell_order = {'type': 'Sold', 'symbol': last_row['market'],
                                                       'order': i['order'],
                                                       'date_sold': str(last_index.to_pydatetime()),
                                                       'price_sold': float(last_row['close']),
                                                       'price_bought': float(i['price_bought']),
                                                       'date_bought': str(i['date_bought']),
                                                       'percentage_gained': percentage}

                        sell_message = f"Verkocht:\n Market: {last_row['market']} Prijs: {last_row['close']} " \
                                   f"aankoopkoers: {i['price_bought']}\n " \
                                   f"percentage gained: {percentage}"

                        print(sell_order)
                        self.update_file(self._file_path, sell_order)
                        await self.send_telegram_message(sell_message)

                    
                    else:
                        percentage = (float(i['price_bought']) - float(last_row['close'])) / float(i['price_bought']) * 100
                        percentage = format(percentage, ".2f")
                        update_order = {'type': 'Bought', 'symbol': last_row['market'],
                                                       'order': i['order'],
                                                       'last_update': str(last_index.to_pydatetime()),
                                                       'closing_price': float(last_row['close']),
                                                       'price_bought': float(i['price_bought']),
                                                       'date_bought': str(i['date_bought']),
                                                       'percentage_gain': percentage}

                        update_message = f"Update:\n {last_row['market']} prijs: {last_row['close']} " \
                                   f"aankoopkoers: {i['price_bought']}\n " \
                                   f"percentage gained: {percentage}"
                
                        self.update_file(self._file_path, update_order)
                        
                    

        #Going long
        if indicators_buy_long.all():
            order_number = random.randint(1000, 9999)
            buy_message = f"Koop:\n Positie: Long\n Market: {last_row['market']} Prijs: {last_row['close']}"
            buy_order = {'type': 'Bought', 'strategy': 'Long', 'symbol': last_row['market'],
                                                'date_bought': str(last_index.to_pydatetime()),
                                                'price_bought': float(last_row['close']),
                                                'order': order_number}

  
            print(buy_order)
            self.update_file(self._file_path, buy_order)
            await self.send_telegram_message(buy_message)

        else:
            print('Geen long koopsignalen gevonden')

        #Going short
        if indicators_buy_short.all():
            order_number = random.randint(1000, 9999)
            buy_message = f"Koop:\n Positie: Short\n Market: {last_row['market']} Prijs: {last_row['close']}"
            buy_order = {'type': 'Bought', 'strategy': 'Short', 'symbol': last_row['market'],
                                                'time': str(last_index.to_pydatetime()),
                                                'price_bought': float(last_row['close']),
                                                'order': order_number}

      
            print(buy_order)
            self.update_file(self._file_path, buy_order)
            await self.send_telegram_message(buy_message)
            
        else:
            print('Geen short signalen gevonden')
   
    
    async def main(self, bot):
        for i in self._markets:
            df = bot.get_data(market=i) 
            df = bot.add_indicators(df)
            data_complete = bot.generate_signals(df)
            await bot.check_signals(data_complete)

if __name__ == '__main__':
    # file_path = 'CryptoOrders.json'
    file_path = os.getenv('FILE_PATH')
    bot = apibot(file_path=file_path, markets=['ARBEUR', 'INJEUR', 'SOLEUR', 'ADAEUR', 'MNTEUR', 'STXEUR'])
    asyncio.run(bot.main(bot))



