def calculate_imbalance_features(df):
    # Calculate and add imbalance feature 1 (imb_s1)
    df['imb_s1'] = df.eval('(bid_size - ask_size) / (bid_size + ask_size)')  

    # Calculate and add imbalance feature 2 (imb_s2)
    df['imb_s2'] = df.eval('(imbalance_size - matched_size) / (matched_size + imbalance_size)') 

    return df