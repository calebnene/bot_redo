import pandas as pd 
import utils 
import instrument
import numpy as np
import warnings
pd.set_option('display.max_rows', None)
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

def is_trade(row):
    if (row.mid_c > row.frac_top) and (row.mid_c > row.MA_200):
        return 1 #buy 
    if (row.mid_c < row.frac_bottom) and (row.mid_c < row.MA_200):
        return -1 #sell
    return 0
def sl(row):
    if row.IS_TRADE == 1:
        return row['mid_c']-row['mid_l']
    if row.IS_TRADE == -1:
        return row['mid_h']-row['mid_c']
def tp(row):
    if row.IS_TRADE == 1:
        return row['SL'] * 1.5

    if row.IS_TRADE == -1:
        return row['SL'] * 1.5
#checking the gains 
def gain(row, df_ma):
    if row.IS_TRADE == 1:
        entry_price = row['mid_c']
        tp_price = entry_price + row['TP']
        sl_price = entry_price - row['SL']

        for i in range(row.name+1, len(df_ma)):
            next_row = df_ma.iloc[i]
            # Check if TP is hit
            if next_row['mid_h'] >= tp_price:
                return row['TP']  # Gain equals TP
            
            # Check if SL is hit
            if next_row['mid_l'] <= sl_price:
                return -row['SL']  # Loss equals SL
    elif row.IS_TRADE == -1:  # Sell trade
        entry_price = row['mid_c']
        tp_price = entry_price - row['TP']
        sl_price = entry_price + row['SL']
        # Check next rows (candles) after the current row
        for i in range(row.name + 1, len(df_ma)):
            next_row = df_ma.iloc[i]
            # Check if TP is hit
            if next_row['mid_l'] <= tp_price:
                return row['TP']  # Gain equals TP
            # Check if SL is hit
            if next_row['mid_h'] >= sl_price:
                return -row['SL']  # Loss equals SL
    return 0  # No gain/loss if neither TP nor SL is hit

def get_ma_col(ma):
    return f'MA_{ma}'

def evaluate_pair(i_pair, ma, price_data):
    price_data = price_data[['time', 'mid_c', 'mid_h', 'mid_l',get_ma_col(ma)]].copy()
    price_data['frac_top_bool'] = np.where(
                                            price_data['mid_h'] == price_data['mid_h'].rolling(5, center=True).max(), True, False
                                            )
    price_data['frac_top'] = np.where(
                                price_data['mid_h'] == price_data['mid_h'].rolling(5, center=True).max(), price_data['mid_h'], None
                            )
    #filling the subsequent rows with the previous fractal high till the next fractal high 
    price_data['frac_top'] = price_data['frac_top'].ffill()

    price_data['frac_bottom_bool'] = np.where(
                                        price_data['mid_l'] == price_data['mid_l'].rolling(5, center=True).min(), True, False
                                        )
    price_data['frac_bottom'] = np.where(
                                    price_data['mid_l'] == price_data['mid_l'].rolling(5, center=True).min(), price_data['mid_l'], None
                                )
    price_data['frac_bottom'] = price_data['frac_bottom'].ffill()
    price_data['IS_TRADE'] = price_data.apply(is_trade, axis=1)
    price_data['SL'] = price_data.apply(sl, axis = 1)
    price_data['TP'] = price_data.apply(tp, axis=1)

    df_trades = price_data[price_data.IS_TRADE !=0].copy()
    df_trades['GAIN'] = (df_trades.apply(lambda row: gain(row, df_trades), axis=1)) / i_pair.pipLocation
    df_trades['time'] = pd.to_datetime(df_trades['time'])

    #print(f'{i_pair.name} trades: {df_trades.shape[0]} gain: {df_trades.GAIN.sum():.0f}')

    return df_trades['GAIN'].sum(), price_data

def get_price_data(pairname, granularity):
    df = pd.read_pickle(utils.get_his_data_filename(pairname, granularity))
    non_cols = ['time', 'volume']
    mod_cols = [x for x in df.columns if x not in non_cols]
    df[mod_cols] = df[mod_cols].apply(pd.to_numeric)
    return df[['time', 'mid_c', 'mid_h', 'mid_l']]

def process_data(ma, price_data):

    price_data[get_ma_col(ma)] = price_data.mid_c.rolling(window=ma).mean()


    return price_data 

def run():
    results = []
    trades_dict = {}
    pairnames = ['CAD_CHF', 'CAD_JPY', 'CHF_JPY', 'EUR_CAD', 'EUR_CHF', 'EUR_GBP', 'EUR_JPY', 'EUR_NZD', 'EUR_USD', 'GBP_CAD', 'GBP_CHF', 'GBP_JPY',
                 'GBP_NZD', 'GBP_USD', 'NZD_CAD', 'NZD_JPY', 'NZD_USD', 'NZD_CHF', 'USD_CAD', 'USD_CHF', 'USD_JPY']
    granularities = ['M5', 'M15', 'M30', 'H1', 'H4']
    ma = 200

    for pairname in pairnames:
        i_pair = instrument.Instrument.get_instruments_dict()[pairname]

        for granularity in granularities:
            print(f'Processing {pairname} with timeframe {granularity}')

            price_data = get_price_data(pairname, granularity)
            price_data = process_data(ma, price_data)

            total_gain, df_trades = evaluate_pair(i_pair, ma, price_data.copy())
            results.append((pairname, granularity, total_gain))
            trades_dict[(pairname, granularity)] = df_trades.dropna()  # Append df_trades to trades

    results_df = pd.DataFrame(results, columns=['pair', 'timeframe', 'total_gain'])
    print(results_df)

    for key, trades_df in trades_dict.items():
        pairname, granularity = key
        print(f'\nTrades for {pairname} {granularity}')
        print(trades_df)
        
        #save to CSV files
        trades_df.to_pickle(f'all_trades\{pairname}_{granularity}_trades.pkl')

    return results_df, trades_dict



if __name__ == "__main__":
    run()