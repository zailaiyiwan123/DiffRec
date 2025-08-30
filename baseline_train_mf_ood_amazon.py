import random
import time
import json
import gzip
import pickle
import os

import numpy as np
import omegaconf
import pandas as pd
import torch
import torch.nn as nn
import torch.optim
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from torch.utils.data.dataloader import DataLoader

# SwanLab for experiment tracking
try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    print("SwanLab not installed. Install with: pip install swanlab")
    SWANLAB_AVAILABLE = False

# from minigpt4.models.rec_model import MatrixFactorization, BinMF

# 直接定义矩阵分解模型，避免导入问题
class MatrixFactorization(nn.Module):
    """简化的矩阵分解模型"""
    def __init__(self, config):
        super(MatrixFactorization, self).__init__()
        self.user_num = config.user_num
        self.item_num = config.item_num
        self.embedding_size = config.embedding_size
        
        # 用户和物品嵌入
        self.user_embedding = nn.Embedding(self.user_num, self.embedding_size)
        self.item_embedding = nn.Embedding(self.item_num, self.embedding_size)
        
        # 偏置项
        self.user_bias = nn.Embedding(self.user_num, 1)
        self.item_bias = nn.Embedding(self.item_num, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # 初始化
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        nn.init.normal_(self.user_bias.weight, std=0.1)
        nn.init.normal_(self.item_bias.weight, std=0.1)
    
    def forward(self, user_ids, item_ids):
        # 获取嵌入
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # 获取偏置
        user_bias = self.user_bias(user_ids).squeeze()
        item_bias = self.item_bias(item_ids).squeeze()
        
        # 计算预测评分
        prediction = torch.sum(user_emb * item_emb, dim=1) + user_bias + item_bias + self.global_bias
        
        # 使用tanh将输出限制在0-5范围内，避免sigmoid的梯度消失问题
        # tanh输出范围(-1,1)，映射到(0,5)：(tanh(x) + 1) * 2.5
        prediction = (torch.tanh(prediction) + 1) * 2.5
        
        return prediction


# os.environ['CUDA_VISIBLE_DEVICES']='7'

def load_and_encode_all_data(data_dir):
    """
    加载所有数据集（数据已经包含uid, iid, rating字段）
    """
    print("加载所有数据集...")
    
    # 加载所有数据集
    train_path = os.path.join(data_dir, "train_ood2.pkl")
    valid_path = os.path.join(data_dir, "valid_ood2.pkl")
    test_path = os.path.join(data_dir, "test_ood2.pkl")
    
    datasets = {}
    max_user_id = 0
    max_item_id = 0
    
    for name, path in [("train", train_path), ("valid", valid_path), ("test", test_path)]:
        print(f"Loading {name} data from {path}...")
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, pd.DataFrame):
            print(f"Loaded DataFrame {name}, shape: {data.shape}")
            print(f"DataFrame columns: {list(data.columns)}")
            
            # 检查是否包含必要的列
            if 'uid' in data.columns and 'iid' in data.columns and 'rating' in data.columns:
                # 转换为numpy数组格式 [uid, iid, rating]
                numpy_data = data[['uid', 'iid', 'rating']].values
                datasets[name] = numpy_data
                
                # 记录最大的用户ID和物品ID
                max_user_id = max(max_user_id, data['uid'].max())
                max_item_id = max(max_item_id, data['iid'].max())
                
                print(f"{name} 数据集形状: {numpy_data.shape}")
                print(f"{name} 用户ID范围: {data['uid'].min()}-{data['uid'].max()}")
                print(f"{name} 物品ID范围: {data['iid'].min()}-{data['iid'].max()}")
                print(f"{name} 评分范围: {data['rating'].min()}-{data['rating'].max()}")
            else:
                raise ValueError(f"数据格式不正确：{name} 缺少uid, iid, rating字段")
        else:
            raise ValueError(f"期望pandas DataFrame，但得到 {type(data)}")
    
    user_num = max_user_id + 1
    item_num = max_item_id + 1
    
    print(f"总用户数: {user_num}")
    print(f"总物品数: {item_num}")
    
    return datasets, user_num, item_num


def load_pkl_data(file_path):
    """
    加载pkl格式的数据（保持向后兼容）
    """
    print(f"Loading data from {file_path}...")
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    # 检查数据类型
    if isinstance(data, pd.DataFrame):
        print(f"警告：使用单独的数据加载可能导致编码不一致，建议使用load_and_encode_all_data")
        print(f"Loaded DataFrame from {file_path}, shape: {data.shape}")
        print(f"DataFrame columns: {list(data.columns)}")
        
        # 检查是否包含必要的列（uid, iid, rating）
        if 'uid' in data.columns and 'iid' in data.columns and 'rating' in data.columns:
            print("数据包含uid, iid, rating字段")
            
            print(f"用户ID范围: {data['uid'].min()}-{data['uid'].max()}")
            print(f"物品ID范围: {data['iid'].min()}-{data['iid'].max()}")
            print(f"评分范围: {data['rating'].min()}-{data['rating'].max()}")
            
            # 转换为numpy数组格式 [uid, iid, rating]
            numpy_data = data[['uid', 'iid', 'rating']].values
            print(f"转换后的numpy数组形状: {numpy_data.shape}")
            
            return numpy_data
        else:
            print(f"数据列不匹配，期望包含uid, iid, rating，实际包含: {list(data.columns)}")
            raise ValueError("数据格式不正确")
    else:
        # 如果是numpy数组格式，保持原有逻辑
        print(f"Loaded numpy array from {file_path}, shape: {data.shape}")
        print(f"Data columns: uid, iid, rating")
        print(f"Data range - uid: {data[:, 0].min()}-{data[:, 0].max()}, "
              f"iid: {data[:, 1].min()}-{data[:, 1].max()}, "
              f"rating: {data[:, 2].min()}-{data[:, 2].max()}")
        
        return data

def calculate_rmse_mae(user, predict, rating):
    """计算RMSE和MAE指标"""
    if not isinstance(predict, np.ndarray):
        predict = np.array(predict)
    if not isinstance(rating, np.ndarray):
        rating = np.array(rating)
    predict = predict.squeeze()
    rating = rating.squeeze()

    start_time = time.time()
    u, inverse, counts = np.unique(user, return_inverse=True, return_counts=True)
    index = np.argsort(inverse)
    candidates_dict = {}
    k = 0
    total_num = 0
    only_one_interaction = 0
    computed_u = []
    
    for u_i in u:
        start_id, end_id = total_num, total_num + counts[k]
        u_i_counts = counts[k]
        index_ui = index[start_id:end_id]
        if u_i_counts == 1:
            only_one_interaction += 1
            total_num += counts[k]
            k += 1
            continue
        candidates_dict[u_i] = [predict[index_ui], rating[index_ui]]
        total_num += counts[k]
        k += 1
    
    print("only one interaction users:", only_one_interaction)
    user_rmse = []
    user_mae = []
    
    for ui, pre_and_true in candidates_dict.items():
        pre_i, rating_i = pre_and_true
        ui_rmse = np.sqrt(mean_squared_error(rating_i, pre_i))
        ui_mae = mean_absolute_error(rating_i, pre_i)
        user_rmse.append(ui_rmse)
        user_mae.append(ui_mae)
        computed_u.append(ui)
    
    user_rmse = np.array(user_rmse)
    user_mae = np.array(user_mae)
    print("computed user:", user_rmse.shape[0])
    avg_rmse = user_rmse.mean()
    avg_mae = user_mae.mean()
    print("User-wise RMSE:", avg_rmse, "User-wise MAE:", avg_mae, "Cost:", time.time() - start_time)
    return avg_rmse, avg_mae, computed_u, user_rmse, user_mae


class early_stoper(object):
    def __init__(self, ref_metric='valid_rmse', increase=False, patience=20) -> None:
        self.ref_metric = ref_metric
        self.best_metric = None
        self.increase = increase  # RMSE越小越好，所以设为False
        self.reach_count = 0
        self.patience = patience

    def _registry(self, metrics):
        self.best_metric = metrics

    def update(self, metrics):
        if self.best_metric is None:
            self._registry(metrics)
            return True
        else:
            if self.increase and metrics[self.ref_metric] > self.best_metric[self.ref_metric]:
                self.best_metric = metrics
                self.reach_count = 0
                return True
            elif not self.increase and metrics[self.ref_metric] < self.best_metric[self.ref_metric]:
                self.best_metric = metrics
                self.reach_count = 0
                return True
            else:
                self.reach_count += 1
                return False

    def is_stop(self):
        if self.reach_count >= self.patience:
            return True
        else:
            return False


# set random seed
def run_a_trail(train_config, log_file=None, save_mode=False, save_file=None, need_train=True, warm_or_cold=None):
    seed = 2023
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Initialize SwanLab
    if SWANLAB_AVAILABLE:
        # Create experiment name
        exp_name = f"Amazon_MF_lr{train_config['lr']}_wd{train_config['wd']}_emb{train_config['embedding_size']}"
        swanlab.init(
            project="Amazon-Matrix-Factorization",
            experiment_name=exp_name,
            config=train_config,
            description=f"Matrix Factorization on Amazon dataset with {train_config['embedding_size']}D embeddings"
        )

    # load dataset from pkl files
    data_dir = "/root/autodl-tmp/dataset/amazon/"
    
    # 使用统一的数据加载和编码方法
    try:
        datasets, user_num, item_num = load_and_encode_all_data(data_dir)
        train_data = datasets['train']
        valid_data = datasets['valid']
        test_data = datasets['test']
    except Exception as e:
        print(f"使用新方法加载数据失败: {e}")
        print("回退到原有方法...")
        # 回退到原有的加载方法
        train_data = load_pkl_data(data_dir + "train_ood2.pkl")
        valid_data = load_pkl_data(data_dir + "valid_ood2.pkl")
        test_data = load_pkl_data(data_dir + "test_ood2.pkl")
        
        user_num = max(train_data[:, 0].max(), valid_data[:, 0].max(), test_data[:, 0].max()) + 1
        item_num = max(train_data[:, 1].max(), valid_data[:, 1].max(), test_data[:, 1].max()) + 1

    print("train data:", train_data.shape, "valid:", valid_data.shape, "test:", test_data.shape)
    print("user nums:", user_num, "item nums:", item_num)

    # 注意：当前数据格式不包含not_cold字段，如果需要冷启动测试，需要额外处理
    if warm_or_cold is not None:
        print(f"Warning: warm_or_cold filtering not implemented for current data format")
        print(f"Using full test set instead")

    mf_config = {
        "user_num": int(user_num),
        "item_num": int(item_num),
        "embedding_size": int(train_config['embedding_size'])
    }
    mf_config = omegaconf.OmegaConf.create(mf_config)

    train_data_loader = DataLoader(train_data, batch_size=train_config['batch_size'], shuffle=True)
    valid_data_loader = DataLoader(valid_data, batch_size=train_config['batch_size'], shuffle=False)
    test_data_loader = DataLoader(test_data, batch_size=train_config['batch_size'], shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MatrixFactorization(mf_config).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=train_config['lr'], weight_decay=train_config['wd'])
    early_stop = early_stoper(ref_metric='valid_rmse', increase=False, patience=train_config['patience'])
    
    # 使用MSE损失函数进行回归
    criterion = nn.MSELoss()

    if not need_train:
        model.load_state_dict(torch.load(save_file))
        model.eval()
        pre = []
        rating = []
        users = []
        for batch_id, batch_data in enumerate(valid_data_loader):
            batch_data = batch_data.to(device)
            ui_matching = model(batch_data[:, 0].long(), batch_data[:, 1].long())
            users.extend(batch_data[:, 0].cpu().numpy())
            pre.extend(ui_matching.detach().cpu().numpy())
            rating.extend(batch_data[:, -1].cpu().numpy())
        
        valid_rmse = np.sqrt(mean_squared_error(rating, pre))
        valid_mae = mean_absolute_error(rating, pre)
        valid_urmse, valid_umae, _, _, _ = calculate_rmse_mae(users, pre, rating)

        pre = []
        rating = []
        users = []
        for batch_id, batch_data in enumerate(test_data_loader):
            batch_data = batch_data.to(device)
            ui_matching = model(batch_data[:, 0].long(), batch_data[:, 1].long())
            pre.extend(ui_matching.detach().cpu().numpy())
            rating.extend(batch_data[:, -1].cpu().numpy())
            users.extend(batch_data[:, 0].cpu().numpy())
        
        test_rmse = np.sqrt(mean_squared_error(rating, pre))
        test_mae = mean_absolute_error(rating, pre)
        test_urmse, test_umae, _, _, _ = calculate_rmse_mae(users, pre, rating)

        print("valid_rmse:{}, valid_mae:{}, valid_urmse:{}, valid_umae:{}, test_rmse:{}, test_mae:{}, test_urmse:{}, test_umae:{}".format(
            valid_rmse, valid_mae, valid_urmse, valid_umae, test_rmse, test_mae, test_urmse, test_umae))
        return

    for epoch in range(train_config['epoch']):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for bacth_id, batch_data in enumerate(train_data_loader):
            batch_data = batch_data.to(device)
            ui_matching = model(batch_data[:, 0].long(), batch_data[:, 1].long())
            loss = criterion(ui_matching, batch_data[:, -1].float())
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            epoch_loss += loss.item()
            num_batches += 1

        # Calculate average training loss
        avg_train_loss = epoch_loss / num_batches
        
        # Log training loss every epoch
        if SWANLAB_AVAILABLE:
            swanlab.log({"train_loss": avg_train_loss, "epoch": epoch})

        if epoch % train_config['eval_epoch'] == 0:
            model.eval()
            pre = []
            rating = []
            users = []
            for batch_id, batch_data in enumerate(valid_data_loader):
                batch_data = batch_data.to(device)
                ui_matching = model(batch_data[:, 0].long(), batch_data[:, 1].long())
                users.extend(batch_data[:, 0].cpu().numpy())
                pre.extend(ui_matching.detach().cpu().numpy())
                rating.extend(batch_data[:, -1].cpu().numpy())
            
            valid_rmse = np.sqrt(mean_squared_error(rating, pre))
            valid_mae = mean_absolute_error(rating, pre)
            valid_urmse, valid_umae, _, _, _ = calculate_rmse_mae(users, pre, rating)

            pre = []
            rating = []
            users = []
            for batch_id, batch_data in enumerate(test_data_loader):
                batch_data = batch_data.to(device)
                ui_matching = model(batch_data[:, 0].long(), batch_data[:, 1].long())
                users.extend(batch_data[:, 0].cpu().numpy())
                pre.extend(ui_matching.detach().cpu().numpy())
                rating.extend(batch_data[:, -1].cpu().numpy())
            
            test_rmse = np.sqrt(mean_squared_error(rating, pre))
            test_mae = mean_absolute_error(rating, pre)
            test_urmse, test_umae, _, _, _ = calculate_rmse_mae(users, pre, rating)

            # Log validation and test metrics
            if SWANLAB_AVAILABLE:
                swanlab.log({
                    "valid_rmse": valid_rmse,
                    "valid_mae": valid_mae,
                    "valid_urmse": valid_urmse,
                    "valid_umae": valid_umae,
                    "test_rmse": test_rmse,
                    "test_mae": test_mae,
                    "test_urmse": test_urmse,
                    "test_umae": test_umae,
                    "early_stop_count": early_stop.reach_count,
                    "epoch": epoch
                })

            updated = early_stop.update(
                {'valid_rmse': valid_rmse, 'valid_mae': valid_mae, 'valid_urmse': valid_urmse, 'valid_umae': valid_umae,
                 'test_rmse': test_rmse, 'test_mae': test_mae, 'test_urmse': test_urmse, 'test_umae': test_umae,
                 'epoch': epoch})
            if updated and save_mode:
                # 保存当前最佳模型
                try:
                    # 确保保存目录存在
                    os.makedirs(os.path.dirname(save_file), exist_ok=True)
                    
                    # 只保存主要的模型文件，避免复杂的文件名
                    torch.save(model.state_dict(), save_file)
                    print(f"New best model saved at epoch {epoch} with valid_rmse: {valid_rmse:.4f}")
                    print(f"Model saved to: {save_file}")
                    
                except Exception as save_error:
                    print(f"Warning: Failed to save model: {save_error}")
                    print("继续训练...")

            print("epoch:{}, valid_rmse:{:.4f}, test_rmse:{:.4f}, early_count:{}".format(epoch, valid_rmse, test_rmse,
                                                                               early_stop.reach_count))
            if early_stop.is_stop():
                print("early stop is reached....!")
                break
            if epoch > 500 and early_stop.best_metric[early_stop.ref_metric] > 2.0:
                print("training reaches to 500 epoch but the valid_rmse is still greater than 2.0")
                break
    print("train_config:", train_config, "\nbest result:", early_stop.best_metric)
    
    # 显示最终的最佳模型信息
    if save_mode and early_stop.best_metric is not None:
        best_epoch = early_stop.best_metric.get('epoch', 0)
        best_rmse = early_stop.best_metric.get('valid_rmse', 0)
        
        if best_epoch > 0:
            print(f"\n=== 最佳模型信息 ===")
            print(f"最佳模型在 epoch {best_epoch} 获得")
            print(f"最佳验证集 RMSE: {best_rmse:.4f}")
            print(f"模型已保存到: {save_file}")
            print(f"==================\n")
        else:
            print("警告：没有找到最佳模型，可能训练没有收敛")
    
    if log_file is not None:
        print("train_config:", train_config, "best result:", early_stop.best_metric, file=log_file)
        log_file.flush()
    
    # Log final best results
    if SWANLAB_AVAILABLE and early_stop.best_metric is not None:
        swanlab.log({
            "final_best_valid_rmse": early_stop.best_metric.get('valid_rmse', 0),
            "final_best_valid_mae": early_stop.best_metric.get('valid_mae', 0),
            "final_best_test_rmse": early_stop.best_metric.get('test_rmse', 0),
            "final_best_test_mae": early_stop.best_metric.get('test_mae', 0),
            "final_best_epoch": early_stop.best_metric.get('epoch', 0)
        })
        swanlab.finish()


# if __name__=='__main__':
#     # lr_ = [1e-1,1e-2,1e-3]
#     lr_=[1e-4]
#     dw_ = [1e-2,1e-3,1e-4,1e-5,1e-6,1e-7]
#     # embedding_size_ = [32, 64, 128, 156, 512]
#     embedding_size_ = [64,128,256]
#     try:
#         f = open("0923amazon-book-new-ood-v2-mf_search_lr"+str(lr_[0])+".log",'rw+')
#     except:
#         f = open("0923amazon-book-new-ood-v2-mf_search_lr"+str(lr_[0])+".log",'w+')
#     for lr in lr_:
#         for wd in dw_:
#             for embedding_size in embedding_size_:
#                 train_config={
#                     'lr': lr,
#                     'wd': wd,
#                     'embedding_size': embedding_size,
#                     "epoch": 5000,
#                     "eval_epoch":1,
#                     "patience":100,
#                     "batch_size": 2048*5
#                 }
#                 print(train_config)
#                 run_a_trail(train_config=train_config, log_file=f, save_mode=False)
#     f.close()


# {'lr': 0.001, 'wd': 0.0001, 'embedding_size': 256, 'epoch': 5000, 'eval_epoch': 1, 'patience': 100, 'batch_size': 2048},
#  {'valid_auc': 0.6760080227104877, 'valid_uauc': 0.6191863368703151, 'test_auc': 0.6482002627476354, 'test_uauc': 0.636100123360848, 'epoch': 465}
# save version....
# if __name__=='__main__':
#     # lr_ = [1e-1,1e-2,1e-3]
#     lr_=[1e-3] #1e-2
#     dw_ = [1e-6]
#     # embedding_size_ = [32, 64, 128, 156, 512]
#     embedding_size_ = [256]
#     save_path = "/data/zyang/LLM/PretrainedModels/mf/"
#     # save_path = "/home/sist/zyang/LLM/PretrainedModels/mf/"
#     # try:
#     #     f = open("rec_mf_search_lr"+str(lr_[0])+".log",'rw+')
#     # except:
#     #     f = open("rec_mf_search_lr"+str(lr_[0])+".log",'w+')
#     f=None
#     for lr in lr_:
#         for wd in dw_:
#             for embedding_size in embedding_size_:
#                 train_config={
#                     'lr': lr,
#                     'wd': wd,
#                     'embedding_size': embedding_size,
#                     "epoch": 5000,
#                     "eval_epoch":1,
#                     "patience":100,
#                     "batch_size":2048*5
#                 }
#                 print(train_config)
#                 save_path += "0923_book_oodv2_best_model_d" + str(embedding_size)+ 'lr-'+ str(lr) + "wd"+str(wd) + ".pth"
#                 print("save path: ", save_path)
#                 run_a_trail(train_config=train_config, log_file=f, save_mode=True,save_file=save_path)
#     f.close()


#### /data/zyang/LLM/PretrainedModels/mf/best_model_d128.pth
# with prtrain version:
if __name__ == '__main__':
    # 简化参数搜索，减少组合数量
    lr_ = [1e-3]  # 只用一个学习率
    dw_ = [1e-4]  # 只用一个权重衰减
    embedding_size_ = [64,128]  # 只用一个嵌入维度
    
    # 创建保存目录
    import os
    save_dir = "/root/autodl-tmp/pretrained/mf"
    os.makedirs(save_dir, exist_ok=True)
    print(f"创建目录: {save_dir}")

    f = None
    for lr in lr_:
        for wd in dw_:
            for embedding_size in embedding_size_:
                # 在 run_a_trail 函数调用之前
                train_config = {
                    'lr': lr,
                    'wd': wd,
                    'embedding_size': embedding_size,
                    "epoch": 50,   # 减少最大epoch数  
                    "eval_epoch": 1,
                    "patience": 5,  # 减少早停耐心值
                    "batch_size": 4096  # 增加batch size加速训练
                }
                print(f"\n=== 开始训练配置 ===")
                print(train_config)
                
                # 构建模型和日志文件的保存路径
                model_save_path = "/root/autodl-tmp/pretrained/mf/amazon_lr{0}_wd{1}_emb{2}.pth".format(lr, wd, embedding_size)
                log_file_path = "/root/autodl-tmp/pretrained/mf/amazon_lr{0}_wd{1}_emb{2}.log".format(lr, wd, embedding_size)
                
                # 确保目录存在
                os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
                
                print(f"模型将保存到: {model_save_path}")
                print(f"日志将保存到: {log_file_path}")
                print(f"==================\n")
                
                f = open(log_file_path, 'w+')

                run_a_trail(train_config=train_config, log_file=f, save_mode=True, save_file=model_save_path,
                            need_train=True, warm_or_cold=None)
                f.close()
    print("所有训练完成！")
