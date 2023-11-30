import akshare as ak
#个股详细信息的操作`
def zhifu_info(zhifudaima_code):
    aka=ak.stock_individual_info_em(symbol=zhifudaima_code)
    # 使用 akshare 获取主营业务信息
    stock_zyjs_df = ak.stock_zyjs_ths(symbol=zhifudaima_code)
    主营业务 = stock_zyjs_df["主营业务"].values[0] if not stock_zyjs_df.empty else "未知"
    总市值 = aka["value"][0]
    总市值=str(round(总市值/100000000,3))+"亿元"
    流通市值 = aka["value"][1]
    流通市值 = str(round(流通市值 / 100000000,3)) + "亿元"
    行业=aka["value"][2]
    股票代码=aka["value"][4]
    股票简称=aka["value"][5]
    六十日涨跌幅1=ak.stock_zh_a_spot_em().loc[ak.stock_zh_a_spot_em()['代码'] == zhifudaima_code]["60日涨跌幅"]
    六十日涨跌幅=六十日涨跌幅1.iloc[0].item()
       #K线图数据
    df = ak.stock_zh_a_hist(symbol=zhifudaima_code , period="daily").tail(30)
    date = [date.strftime('%Y-%m-%d') for date in df["日期"]]
    open_price = df["开盘"].tolist()
    close_price = df["收盘"].tolist()
    low_price = df["最低"].tolist()
    high_price = df["最高"].tolist()
    change_rate = df["换手率"].tolist()
    up_down_rate = df["涨跌幅"].tolist()
    k_data = [date, open_price, close_price, low_price, high_price, change_rate, up_down_rate]
    # "main_business": 主营业务,
    # "business": 行业,
    # "code": 股票代码,
    # "name": 股票简称,
    # "flow_value": 流通市值,
    # "all_value": 总市值,
    # "updown_60day": 六十日涨跌幅,
    # "k_line": k_data
    data=[股票代码,股票简称,主营业务,行业,流通市值,总市值,六十日涨跌幅,k_data]
    return data
# stock_set=zhifudaima_info=zhifu_info("300364")
# print(stock_set)
