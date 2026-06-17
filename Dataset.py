import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class HouseTensorDataset(Dataset):
    def __init__(self, features, labels=None):
        self.features = torch.tensor(features, dtype=torch.float32)
        if labels is not None:
            self.labels = torch.tensor(labels, dtype=torch.float32).reshape(-1, 1)
        else:
            self.labels = None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx]

class KaggleHouse:
    def __init__(self, batch_size, train_df=None, val_df=None):
        self.batch_size = batch_size
        
        # K-Fold分割によって内部から呼び出された場合
        if train_df is not None:
            self.train = train_df.copy()
            self.val = val_df.copy() if val_df is not None else None
        else:
            # 初回起動時はローカルにある本物のCSVを直接読み込む
            try:
                self.raw_train = pd.read_csv('train.csv')
                self.raw_val = pd.read_csv('test.csv')  # Kaggleの本番テストデータ
            except FileNotFoundError:
                raise FileNotFoundError("エラー: 'train.csv' または 'test.csv' が見つかりません。スクリプトと同じフォルダに配置してください。")
            self.train = None
            self.val = None

    def preprocess(self):
        if self.train is not None:
            return 
            
        label = 'SalePrice'
        #Id列と正解ラベルを切り離して、訓練とテストを結合
        features = pd.concat((
            self.raw_train.drop(columns=['Id', label]),
            self.raw_val.drop(columns=['Id'])
        ))
        
        features['TotalSF'] = features['1stFlrSF'] + features['2ndFlrSF'] + features['TotalBsmtSF']
        features['HouseAge'] = features['YrSold'] - features['YearBuilt']
        # 2. 数値データの標準化
        numeric_features = features.dtypes[features.dtypes != 'object'].index
        features[numeric_features] = features[numeric_features].apply(
            lambda x: (x - x.mean()) / (x.std() if x.std() > 0 else 1)
        )
        
        # 3. 欠損値（NaN）を0で埋める
        features[numeric_features] = features[numeric_features].fillna(0)
        features = pd.get_dummies(features, dummy_na=True, dtype=float)
        
        # 5. 前処理が終わったデータを訓練データと本番テストデータに再切り分け
        n_train = self.raw_train.shape[0]
        self.train = features.iloc[:n_train, :].copy()
        self.train[label] = self.raw_train[label].values # 訓練データにのみ価格を戻す
        self.val = features.iloc[n_train:, :].copy()     # 本番テストデータ（価格列なし）

    def get_dataloader(self, train):
        label = 'SalePrice'
        
        if train:
            # 訓練時、またはK-foldの検証時（SalePriceが存在する）
            data = self.train
            X = data.drop(columns=[label]).values
            y = data[label].values
            
            dataset = HouseTensorDataset(X, y)
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        else:
            # Kaggle本番予測用（SalePriceが存在しない）
            data = self.val
            X = data.values
            
            dataset = HouseTensorDataset(X, None)
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)