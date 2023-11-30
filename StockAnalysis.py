import adata
from datetime import datetime, timedelta
import pandas as pd

class LongTouAnalysis:
    def __init__(self, lianban_code="883958", zhangting_code="883900"):
        self.lianban_code = lianban_code
        self.zhangting_code = zhangting_code

    @staticmethod
    def calculate_ma_and_deviation(data):
        # 确保'close'列是数值类型
        data['close'] = pd.to_numeric(data['close'], errors='coerce')

        # 处理缺失值
        data = data.dropna(subset=['close'])

        # 计算5日和10日移动平均线
        data['MA5'] = data['close'].rolling(window=5).mean()
        data['MA10'] = data['close'].rolling(window=10).mean()

        # 计算偏离度
        data['偏离度5'] = (data['close'] - data['MA5']) / data['MA5'] * 100
        data['偏离度10'] = (data['close'] - data['MA10']) / data['MA10'] * 100

        return data

    def get_data(self):
        # 获取连板和涨停的数据
        self.lianban_data = adata.stock.market.get_market_concept_ths(index_code=self.lianban_code, k_type=1)
        self.zhangting_data = adata.stock.market.get_market_concept_ths(index_code=self.zhangting_code, k_type=1)

        # 计算移动平均线和偏离度
        self.lianban_data = self.calculate_ma_and_deviation(self.lianban_data)
        self.zhangting_data = self.calculate_ma_and_deviation(self.zhangting_data)

    def apply_strategy(self):
        # 计算昨日是否符合条件
        self.lianban_data['昨日10日线条件'] = (
            (self.lianban_data['close'].shift(1) < self.lianban_data['MA10'].shift(1)) &
            (self.lianban_data['close'] >= self.lianban_data['MA10'])
        )
        self.zhangting_data['昨日5日线条件'] = (
            (self.zhangting_data['close'].shift(1) < self.zhangting_data['MA5'].shift(1)) &
            (self.zhangting_data['close'] >= self.zhangting_data['MA5'])
        )

        # 策略 1: 积极操作信号
        self.lianban_data['积极信号'] = self.lianban_data['昨日10日线条件']
        self.zhangting_data['积极信号'] = self.zhangting_data['昨日5日线条件']

        # 策略 2: 回落信号
        self.lianban_data['回落信号'] = self.lianban_data['偏离度5'] > 9
        self.zhangting_data['回落信号'] = self.zhangting_data['偏离度5']>6

        # 结合两个板块的信号
        self.积极操作信号 = self.lianban_data['积极信号'] & self.zhangting_data['积极信号']
        self.回落操作信号 = self.lianban_data['回落信号'] & self.zhangting_data['回落信号']
        # 将布尔值转换为字符串
        self.积极操作信号 = self.积极操作信号.map({True: '干', False: '忍住'})
        self.回落操作信号 = self.回落操作信号.map({True: '跑', False: '稳住'})
    def print_results(self):
        # 将积极操作信号和回落操作信号作为新列加入到lianban_data中
        self.lianban_data['积极操作信号'] = self.积极操作信号
        self.lianban_data['回落操作信号'] = self.回落操作信号

        # 打印含有这些新信息的lianban_data的最后30行
        df=self.lianban_data[['trade_date','open','close','low','high','volume','change_pct','MA5','MA10','偏离度5','积极操作信号', '回落操作信号']].tail(60)
        # 将 DataFrame 转换为 ECharts 需要的格式
        k_data = [df['trade_date'].tolist(),df['open'].tolist(),
                df['close'].tolist(),df['low'].tolist(),df['high'].tolist(),df['volume'].tolist(),
                   df['change_pct'].tolist(), df['MA5'].tolist(), df['MA10'].tolist(), df['偏离度5'].tolist(),
                  df['积极操作信号'].tolist(),df['回落操作信号'].tolist()]

        return k_data


# 使用类

