import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from datetime import datetime, timedelta
import chinese_calendar as calendar
# 当前日期
def future_days():
    # 当前日期
    current_date = datetime.now()
    # 未来日期列表
    future_dates = []
    # 当我们找到足够的工作日时停止循环
    while len(future_dates) < 6:
        # 如果不是周末并且不是节假日，则将其添加到未来日期列表
        if not calendar.is_holiday(current_date) and not calendar.is_in_lieu(current_date):
            future_dates.append(current_date)
        # 无论如何，我们都将日期增加一天
        current_date += timedelta(days=1)

    # 格式化日期
    future_dates_formatted = [date.strftime('%Y-%m-%d') for date in future_dates]
    return future_dates_formatted[1:]


#股价数据集和
class StockPriceDataset(Dataset):
    def __init__(self, stock_prices, input_seq_len, output_seq_len):
        self.stock_prices = stock_prices
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len

    def __len__(self):
        return len(self.stock_prices) - self.input_seq_len - self.output_seq_len + 1

    def __getitem__(self, idx):
        src = self.stock_prices[idx:idx+self.input_seq_len]
        tgt = self.stock_prices[idx+self.input_seq_len:idx+self.input_seq_len+self.output_seq_len]
        return torch.tensor(src, dtype=torch.float32).unsqueeze(-1), torch.tensor(tgt, dtype=torch.float32).unsqueeze(-1)


class StockPriceTransformer(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(StockPriceTransformer, self).__init__()
        self.input_linear = nn.Linear(1, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_layers, dropout=dropout)
        self.output_linear = nn.Linear(d_model, 1)

    def forward(self, src, tgt):
        src = self.input_linear(src)
        tgt = self.input_linear(tgt)
        output = self.transformer(src, tgt)
        output = self.output_linear(output)
        return output
#batch_size 的提高可以提高效率 但是占内存  16 32 64
def train_stock_price_model(stock_prices, input_seq_len=10, output_seq_len=5, epochs=30, lr=0.001, batch_size=64):
    # 创建数据集实例
    dataset = StockPriceDataset(stock_prices, input_seq_len, output_seq_len)

    # 创建 DataLoader 实例
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    num_days = len(stock_prices)
    num_samples = num_days - input_seq_len - output_seq_len + 1

    # 数据预处理
    src_data = torch.tensor([stock_prices[i:i+input_seq_len] for i in range(num_samples)]).unsqueeze(-1).float()
    tgt_data = torch.tensor([stock_prices[i+input_seq_len:i+input_seq_len+output_seq_len] for i in range(num_samples)]).unsqueeze(-1).float()

    model = StockPriceTransformer()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for src_batch, tgt_batch in dataloader:
            src_batch = src_batch.transpose(0, 1)
            tgt_batch = tgt_batch.transpose(0, 1)

            optimizer.zero_grad()
            output = model(src_batch, tgt_batch[:-1])
            loss = criterion(output, tgt_batch[1:])
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    return model

def predict_future_stock_prices(stock_prices, trained_model, input_seq_len=10, num_days_to_predict=5):
    # 预测未来股票价格
    src = torch.tensor(stock_prices[-input_seq_len:]).unsqueeze(-1).unsqueeze(1).float()
    tgt = torch.zeros(num_days_to_predict, 1, 1)

    with torch.no_grad():
        for i in range(num_days_to_predict):
            prediction = trained_model(src, tgt[:i+1])
            tgt[i] = prediction[-1]

    output = tgt.squeeze().tolist()
    output=[round(i,2) for i in output]

    return output


# a=future_days()
# print(a)


# 输入股票价格数据
# num_days = 200
# stock_prices = [20.53, 20.4, 20.56, 20.12, 19.8, 19.96, 19.8, 20.93, 20.6, 22.66, 22.89, 22.27, 22.61, 22.78, 22.77, 22.57, 22.14, 22.01, 22.13, 21.87, 21.56, 20.94, 21.1, 19.79, 19.56, 20.07, 20.22, 20.61, 20.4, 20.03]

# # 训练模型
# trained_model = train_stock_price_model(stock_prices)
#
# # 预测未来 5 天的股票价格
# future_predictions = predict_future_stock_prices(stock_prices, trained_model, input_seq_len=10, num_days_to_predict=5)
# print(f"Next 5 days of stock prices:", future_predictions)