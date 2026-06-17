import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from Dataset import KaggleHouse
from MLP import MLP

def k_fold_data(data, k):
    rets = []
    fold_size = data.train.shape[0] // k
    for j in range(k):
        val_idx = list(range(j * fold_size, (j + 1) * fold_size))
        train_df = data.train.drop(index=val_idx).reset_index(drop=True)
        val_df = data.train.iloc[val_idx].reset_index(drop=True)
        rets.append(KaggleHouse(data.batch_size, train_df=train_df, val_df=val_df))
    return rets

# スコア（Log-RMSE）を計算する関数
def compute_rmse(model, features, log_labels):
    model.eval()
    with torch.no_grad():
        preds = model(features)
        mse = nn.MSELoss()(preds, log_labels)
        return torch.sqrt(mse).item()
    
def train_fold(model, train_loader, val_features, val_labels_log, epochs, lr, weight_decay, fold_num):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            # ターゲットの対数化
            y_batch_log = torch.log1p(y_batch)
            
            output = model(X_batch)
            loss = criterion(output, y_batch_log)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            
        # 💡 20エポックごとに現在の学習状況を画面に強制出力（フリーズ対策）
        if (epoch + 1) % 20 == 0 or (epoch + 1) == epochs:
            avg_loss = epoch_loss / batch_count
            print(f"  [Fold {fold_num}] Epoch {epoch+1:3d}/{epochs} | Train MSE Loss: {avg_loss:.4f}", flush=True)
            
    val_score = compute_rmse(model, val_features, val_labels_log)
    return val_score

def main():
    # --- ハイパーパラメータの設定 ---
    k = 5                # 5-Fold 
    batch_size = 64
    epochs = 200     # 2層MLPをじっくり学習させる
    lr = 0.01          # 学習率
    weight_decay = 0.01  # L2正則化（過学習の抑制）
    
    # MLPのノード数とドロップアウト率
    num_hiddens_1 = 256
    num_hiddens_2 = 128
    dropout_1 = 0.1
    dropout_2 = 0.2

    data_module = KaggleHouse(batch_size=batch_size)
    data_module.preprocess()
    
    folds = k_fold_data(data_module, k=k)
    val_scores = []
    trained_models = []
    
    print(f"=== {k}-Fold 検証開始===", flush=True)
    for i, data_fold in enumerate(folds):
        train_loader = data_fold.get_dataloader(train=True)
        val_label_all = data_fold.val['SalePrice'].values
        val_labels_log = torch.tensor(np.log1p(val_label_all), dtype=torch.float32).reshape(-1, 1)
        val_features = torch.tensor(data_fold.val.drop(columns=['SalePrice']).values, dtype=torch.float32)
        model = MLP(num_hiddens_1, num_hiddens_2, dropout_1, dropout_2)
        model.eval()
        model(torch.zeros(1, val_features.shape[1])) 
        model.train()
        score = train_fold(model, train_loader, val_features, val_labels_log, epochs, lr, weight_decay, i + 1)
        print(f"Fold {i+1} | Validation Log-RMSE: {score:.4f}\n", flush=True)
        
        val_scores.append(score)
        trained_models.append(model)
        
    print(f"平均 Validation Log-RMSE = {sum(val_scores)/len(val_scores):.4f}", flush=True)

    # 【工程⑤】本番テストデータの予測（アンサンブル）
    print("\nすべての学習完了", flush=True)
    all_test_features = torch.tensor(data_module.val.values, dtype=torch.float32)
    test_preds_list = []
    
    for model in trained_models:
        model.eval()
        with torch.no_grad():
            preds_log = model(all_test_features).numpy().flatten()
            test_preds_list.append(preds_log)
    final_preds_log = np.mean(test_preds_list, axis=0)
    final_preds = np.expm1(final_preds_log)

    #submission.csvの作成
    submission = pd.DataFrame({
        'Id': data_module.raw_val['Id'], # テストデータのId
        'SalePrice': final_preds
    })
    
    submission.to_csv('submission.csv', index=False)
    print("提出ファイル作成完了")

if __name__ == "__main__":
    main()