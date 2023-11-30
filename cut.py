import akshare as ak
import jieba
from kline import StockKLinePlotter
class StockInfoExtractor:
    def __init__(self):
        # 获取 A 股股票基本信息
        self.stock_info_df = ak.stock_info_a_code_name()
        # 提取所有的股票名称
        self.stock_names = self.stock_info_df["name"].tolist()
        # 将所有股票名称添加到 jieba 的自定义词典中
        for stock_name in self.stock_names:
            jieba.add_word(stock_name)
    def show(self):
        return self.stock_names
    def get_stock_details(self, text):
        # 使用 jieba 进行分词
        words = jieba.lcut(text)
        matched_stocks = [stock_name for stock_name in self.stock_names if stock_name in words]
        stock_details = []
        # img_set=StockKLinePlotter()
        for stock_name in matched_stocks:
            code = self.stock_info_df[self.stock_info_df["name"] == stock_name]["code"].values[0]
            # 使用 akshare 获取主营业务信息
            stock_zyjs_df = ak.stock_zyjs_ths(symbol=code)
            #行业
            industry=ak.stock_individual_info_em(symbol=code)["value"][2]
            main_business = stock_zyjs_df["主营业务"].values[0] if not stock_zyjs_df.empty else "未知"

            # img = img_set.get_k_line_plot_base64(code)
            stock_details.append((stock_name, code, industry,main_business))
            stock_detail=[]
        for stock in stock_details:
            if stock[1].startswith("30") or stock[1].startswith("60") or stock[1].startswith("00"):
                stock_detail.append(list(stock))
        return stock_detail

    def k_line(self,data):
        # K线图  基于pyecharts画法
        for d in data:
            df = ak.stock_zh_a_hist(symbol= d[1], period="daily").tail(30)
            date5 = [date.strftime('%Y-%m-%d') for date in df["日期"]]
            open_price5 = df["开盘"].tolist()
            close_price5 = df["收盘"].tolist()
            low_price5 = df["最低"].tolist()
            high_price5 = df["最高"].tolist()
            change_rate5 = df["换手率"].tolist()
            up_down_rate5 = df["涨跌幅"].tolist()
            # for i in len(date5):
            k_data = [date5, open_price5, close_price5, low_price5, high_price5, change_rate5, up_down_rate5]
            d.append(k_data)
            # k_data.append(k_)
        return data


# ####示例：提取文本中的股票名称并查询详细信息
# text = "最近，齐鲁华信和长信科技的股价都有所上涨。慧博云通"
# stock_details = StockInfoExtractor()
# data=stock_details.get_stock_details(text)
# print(data)
# a=stock_details.k_line(data)
# print(a)


