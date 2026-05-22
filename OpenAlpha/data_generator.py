import numpy as np
import pandas as pd
import os

np.random.seed(42)

# 配置参数
n_stocks = 500  # CSI 500 成分股数量
n_dates = 500   # 交易日数量（约2年数据）
start_date = pd.Timestamp('2020-01-01')
dates = pd.bdate_range(start=start_date, periods=n_dates)
datestr = dates.strftime('%Y%m%d').astype(str)

# 生成股票代码列表
stock_list = np.array([f'{600000 + i:06d}.SH' if i % 2 == 0 else f'{1 + i:06d}.SZ'
                       for i in range(n_stocks)])

# 生成价格数据（使用几何布朗运动模拟）
def generate_price_series(n_stocks, n_dates, mu=0.0002, sigma=0.02, init_price=10.0):
    """生成模拟股价序列"""
    dt = 1.0
    returns = np.random.normal(mu * dt, sigma * np.sqrt(dt), size=(n_stocks, n_dates))
    log_prices = np.cumsum(returns, axis=1)
    prices = init_price * np.exp(log_prices)
    return prices

# 生成基础收盘价
close = generate_price_series(n_stocks, n_dates)

# 从收盘价派生其他价格
open_price = close * (1 + np.random.normal(0, 0.005, size=(n_stocks, n_dates)))
high = np.maximum(open_price, close) * (1 + np.abs(np.random.normal(0, 0.01, size=(n_stocks, n_dates))))
low = np.minimum(open_price, close) * (1 - np.abs(np.random.normal(0, 0.01, size=(n_stocks, n_dates))))

# 确保价格逻辑
high = np.maximum(high, np.maximum(open_price, close))
low = np.minimum(low, np.minimum(open_price, close))

# 生成成交量和成交额
volume = np.random.lognormal(15, 0.5, size=(n_stocks, n_dates))
amount = volume * close * np.random.uniform(0.9, 1.1, size=(n_stocks, n_dates))

# 生成 CSI 500 指数数据（市值加权的平均）
csi_500_weight = np.random.uniform(0.001, 0.01, size=(n_stocks, n_dates))
csi_500_weight = csi_500_weight / csi_500_weight.sum(axis=0, keepdims=True)

csi_500_close = (close * csi_500_weight).sum(axis=0, keepdims=True).repeat(n_stocks, axis=0)
csi_500_open = (open_price * csi_500_weight).sum(axis=0, keepdims=True).repeat(n_stocks, axis=0)
csi_500_high = (high * csi_500_weight).sum(axis=0, keepdims=True).repeat(n_stocks, axis=0)
csi_500_low = (low * csi_500_weight).sum(axis=0, keepdims=True).repeat(n_stocks, axis=0)
csi_500_volume = volume.sum(axis=0, keepdims=True).repeat(n_stocks, axis=0)
csi_500_amount = amount.sum(axis=0, keepdims=True).repeat(n_stocks, axis=0)

# 创建数据目录
data_dir = './data/20251231'
os.makedirs(data_dir, exist_ok=True)

# 保存所有数据为 parquet 文件
fields = {
    'close': close,
    'open': open_price,
    'high': high,
    'low': low,
    'volume': volume,
    'amount': amount,
    'csi_500_close': csi_500_close,
    'csi_500_open': csi_500_open,
    'csi_500_high': csi_500_high,
    'csi_500_low': csi_500_low,
    'csi_500_volume': csi_500_volume,
    'csi_500_amount': csi_500_amount,
    'csi_500_weight': csi_500_weight,
}

for name, data in fields.items():
    df = pd.DataFrame(data, index=stock_list, columns=datestr)
    df.to_parquet(f'{data_dir}/{name}.parquet')
    print(f"Saved {name}: shape {data.shape}")

# 保存日期和股票列表
pd.DataFrame(datestr).to_parquet(f'{data_dir}/datestr.parquet')
pd.DataFrame(stock_list).to_parquet(f'{data_dir}/stock_list.parquet')

print(f"\n数据生成完成！保存至 {data_dir}")
print(f"股票数量: {n_stocks}, 交易日数量: {n_dates}")
print(f"日期范围: {datestr[0]} ~ {datestr[-1]}")
