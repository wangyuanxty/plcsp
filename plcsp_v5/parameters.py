import pickle
import csv
import math
import psutil
import os
import json
import torch
import argparse
import copy
import random
import numpy as np
from torch.utils.data import DataLoader
import sys

from trainer.train_device import device


run_configs_pwd = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_run.json")  # [local fix]
# run_configs_pwd = "E:\PY-code\MO-FJSP-DRL\config.json"  # 这个是本地文件，上述是本地文件的在服务器的投影，内容是一样的！
if os.path.exists(run_configs_pwd):
    # Load config and init objects
    with open(run_configs_pwd, 'r') as load_f:
        run_dict = json.load(load_f)  # json方式来加载，形成conf的字典！
        

test_configs_pwd = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tester", "config_test.json")  # [local fix]
# test_configs_pwd = "E:\PY-code\MO-FJSP-DRL\config.json"  # 这个是本地文件，上述是本地文件的在服务器的投影，内容是一样的！
if os.path.exists(test_configs_pwd):
    # Load config and init objects
    with open(test_configs_pwd, 'r') as load_f1:
        test_dict = json.load(load_f1)  # json方式来加载，形成conf的字典！


# parse_dict 函数使用了 json.loads(s) 来将一个 JSON 格式的字符串 s 解析为一个 Python 字典。如果传递给 parse_dict 的字符串不符合 JSON 格式，将会引发 json.JSONDecodeError 异常。
def parse_dict(s):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        raise argparse.ArgumentTypeError("Invalid dictionary format")

parser = argparse.ArgumentParser()  # 创建一个ArgumentParser对象，这个对象将用于解析命令行参数




# ！  TODO：记得同步修改device  = D:/research/DeepReinforcementLearningScheduling/plcsp_v5/algorithm/ppo_algorithm.py
parser.add_argument('--device', type=str, default=device, help='training device type') # !BUG'cuda'是否不指定gpu？跨GPU? 原先"cuda:0"

parser.add_argument('--n_job', type=int, default=6, help='instance')
parser.add_argument('--n_machine', type=int, default=6, help='instance')
parser.add_argument('--n_edge', type=int, default=2, help='instance')

parser.add_argument('--weight_mk', type=float, default=0.4, help='instance')
parser.add_argument('--weight_ec', type=float, default=0.4, help='instance')
parser.add_argument('--weight_tt', type=float, default=0.2, help='instance')

parser.add_argument('--train_seed', type=int, default=0, help='instance')
parser.add_argument('--seed', type=int, default=-1, help='[v5] global RNG seed; -1 = off')
parser.add_argument('--eval_seed', type=int, default=1, help='instance') # 是否在更新ppo的时候采用学习率递减（线性递减）
parser.add_argument('--test_seed', type=int, default=3, help='instance')

# [[6,6,2],[10,6,2],[20,6,3],[10,10,2],[15,10,2],[20,10,5]],0=传入[6，6，2] + 对应的epi的数值
parser.add_argument('--mappo_scene', type=parse_dict, default=test_dict["mappo_id"]["Exist_jme"][0], help='hyper_paras') # iotj模型场景
parser.add_argument('--mappo_id', type=parse_dict, default=test_dict["mappo_id"]["Exist_epi"][0], help='hyper_paras') # iotj模型id
parser.add_argument('--esa_scene', type=parse_dict, default=test_dict["esa_id"]["Exist_jme"][0], help='hyper_paras') # eswa模型场景
parser.add_argument('--esa_id', type=parse_dict, default=test_dict["esa_id"]["Exist_epi"][0], help='hyper_paras') # eswa模型id


parser.add_argument('--mask_value', type=float, default=1, help='hyper_paras')  # job的候选节点的mask的赋值
parser.add_argument('--m_scaling', type=int, default=1, help='hyper_paras') # m选择中的reward放缩比例
parser.add_argument('--reward_scaling', type=parse_dict, default=run_dict["reward_scaling"], help='hyper_paras') # DG图中direction选择中的reward放缩比例

"""=========================================以上是test会用到的超参数========================================================="""

parser.add_argument('--env_batch', type=int, default=16, help='hyper_paras') # 按照2的次幂进行选择：太大没意义，反正都是要动态递减的
# parser.add_argument('--episode_num', type=int, default=4000, help='hyper_paras')# 训练次数
parser.add_argument('--resample_freq', type=int, default=5, help='hyper_paras')# 更换instance的频率，按照episode的个数进行选择
parser.add_argument('--buffer_size', type=int, default=5, help='hyper_paras') # 经验池replayBuffer中需要缓存多少个episode的数据
parser.add_argument('--K_epochs', type=int, default=5, help='hyper_paras') # ppo训练的epoch次数，all数据训练完一次叫做一个epoch: 10
parser.add_argument('--use_grad_clip', type=int, default=True, help='hyper_paras') # 是否在update更新网络的时候使用梯度裁剪
parser.add_argument('--CLIP_GRAD', type=float, default=0.5, help='hyper_paras') # 梯度裁剪，防止梯度爆炸
parser.add_argument('--eval_freq', type=int, default=10, help='hyper_paras') # 循环多少episode之后用validate进行检测
parser.add_argument('--eval_sample', type=int, default=100, help='hyper_paras') # 验证时：我用多少个随机的ability instance来进行验证(验证的样本数量)
parser.add_argument('--eval_data_type', type=str, default="random", help='train_paras') # 定义eval的时候的数据类型：same_data同数据 + same_samples同样本的后20% + random完全随机新dataset
parser.add_argument('--random_weight_type', type=str, default="01", help='hyper_paras') # 使用的随机权重的类型，01：就是[0,1)之间的随机数，0.1：就是一位小数的小值；eval：就是指定的权重


parser.add_argument('--LR', type=float, default=0.001, help='hyper_paras')  # learning rate 
parser.add_argument('--lr_eps', type=float, default=1e-5, help='train_paras') # # 修改adam优化器默认的eps，提高训练性能 BUG  写成了int TODO 注意：这里用的default，如果不是arg命令行修改，还是默认值，只有修改的时候才会报错，所以运行没问题，我也检查过参数的，细节问题
parser.add_argument('--use_lr_decay', type=int, default=False, help='hyper_paras') # 是否在更新ppo的时候采用学习率递减（线性递减）TODO 250509-发现不衰减更稳定和优
parser.add_argument('--decay_step_size', type=int, default=20, help='train_paras') 
parser.add_argument('--decay_ratio', type=float, default=0.96, help='train_paras')    # s BUG 写成了int
parser.add_argument('--GAMMA', type=float, default=0.99, help='hyper_paras') # reward discount 
# 常用0.95-0.99。GAE方法，将偏差控制在一定的范围（超参数可调节）01之间如果您希望优势函数更关注短期奖励，可以选择较小的 lambda 值；如果您更关注长期奖励，可以选择较大的 lambda 值。
parser.add_argument('--LAMDA', type=float, default=0.98, help='hyper_paras')
# 偏向于出现0001，那就调小！！！！  平衡熵正则化[0.01， 0.05， 0.1] 有的设置0.001
parser.add_argument('--ENTROPY_BETA', type=float, default=0.01, help='hyper_paras')  # 策略熵参数
parser.add_argument('--epsilon', type=float, default=0.2, help='hyper_paras') # 重要性采样的裁剪 BUG 250422-我擦，这个epsilon压根没用，int=0
parser.add_argument('--fig_scaling', type=int, default=100, help='hyper_paras') # 画图显示的时候放缩比例


parser.add_argument('--use_orthogonal', type=int, default=False, help='hyper_paras')  # 是否使用正交初始化
parser.add_argument('--neighbor_pooling_type', type=str, default="average", help='gcn_paras') # 定义pooling或aggregate的方式（暂时用average，比max什么的好）
parser.add_argument('--encoder_type', type=str, default='e2e', choices=['e2e', 'settrans', 'hybrid', 'hybrid_full', 'job_st'], help='[v5] e2e=GIN+GAT, settrans=ST+ST, hybrid=GIN job + ST machine, hybrid_full=also ST machine critic, job_st=ST job + original GAT machine')
parser.add_argument('--algo', type=str, default='ppo', choices=['ppo', 'grpo'], help='[v5] policy optimization: ppo (critics) or grpo (critic-free group-relative)')
parser.add_argument('--st_graph_bias', type=int, default=0, help='[v5] inject ST attention bias from the task graph')
parser.add_argument('--st_token_context', type=int, default=1, help='[v5] add edge-consequence features to ST task tokens')
parser.add_argument('--st_full_context', type=int, default=0, help='[v5] add full job/machine/edge context features to ST')
parser.add_argument('--st_task_features', type=int, default=1, help='[v5] add the 13 task-context features')
parser.add_argument('--st_edge_features', type=int, default=0, help='[v5] add the 4 edge-split features')
parser.add_argument('--machine_id_emb', type=int, default=0, help='[v5] add a learned machine-ID embedding to ST machine tokens')
parser.add_argument('--machine_no_attn', type=int, default=0, help='[v5] remove cross-machine attention; use per-machine MLP + mean')
parser.add_argument('--machine_no_sab', type=int, default=0, help='[v5] keep machine input projection + PMA, remove only the SAB layers')
parser.add_argument('--machine_logit_scale', type=float, default=1.0, help='[v5] scale the ST machine policy logits (GAT machine uses 10)')
parser.add_argument('--machine_bn', type=int, default=0, help='[v5] apply BatchNorm to ST machine node embeddings (GAT machine uses BN)')
parser.add_argument('--machine_intra_attn', type=int, default=1, help='[v5] ST machine: separate m1/m2/m3 projections + intra-machine attention before the inter-machine SAB (0 = joint-projection ablation)')
parser.add_argument('--run_tag', type=str, default='', help='[v5] optional output subdirectory tag')
parser.add_argument('--use_agv', type=int, default=0, help='[v5] Stage-0 AGV transport resource (0=off)')
parser.add_argument('--agv_rule', type=int, default=0, help='[v5] use earliest-free AGV rule instead of the AGV actor')
parser.add_argument('--agv_top_rule', type=int, default=0, help='[v5] use the top-level AGV rule head for dispatch (RS arms)')
parser.add_argument('--agv_dst_actor', type=int, default=1, help='[v5] let the AGV actor choose the destination-workshop AGV too (cross-workshop legs); 0 = destination stays earliest-free')
parser.add_argument('--machine_failure', type=int, default=0, help='[v5] enable machine failures (repair delays)')
parser.add_argument('--machine_mtbf', type=float, default=100.0, help='[v5] machine mean time between failures')
parser.add_argument('--machine_mttr', type=float, default=10.0, help='[v5] machine mean repair time')
parser.add_argument('--agv_failure', type=int, default=0, help='[v5] enable AGV failures (repair delays)')
parser.add_argument('--agv_mtbf', type=float, default=100.0, help='[v5] AGV mean time between failures')
parser.add_argument('--agv_mttr', type=float, default=10.0, help='[v5] AGV mean repair time')
parser.add_argument('--failure_seed', type=int, default=0, help='[v5] failure-event RNG seed')
parser.add_argument('--buffer_capacity', type=int, default=0, help='[v5] finite machine buffer capacity (0=off)')
parser.add_argument('--n_agv', type=int, default=6, help='[v5] total number of AGVs (split by workshop edge)')
parser.add_argument('--cross_delay', type=float, default=5.0, help='[v5] fixed cross-workshop delay; consumes no AGV')
parser.add_argument('--cross_leg_ratio', type=float, default=0.5, help='[v5] source share of remaining cross-workshop travel time')
parser.add_argument('--hier', type=int, default=0, help='[v5] top-level rule-pair PPO (0=off)')
parser.add_argument('--hier_interval', type=int, default=8, help='[v5] low-level steps per top-level rule')
parser.add_argument('--hier_alpha', type=float, default=0.5, help='[v5] rule logit-bias scale')
parser.add_argument('--hier_lr', type=float, default=1e-3, help='[v5] top-level PPO learning rate')
parser.add_argument('--hier_epochs', type=int, default=4, help='[v5] top-level PPO epochs per episode')
parser.add_argument('--hier_entropy', type=float, default=0.01, help='[v5] top-level entropy bonus')
parser.add_argument('--top_credit', type=str, default='team',
                    choices=['team', 'local', 'coma', 'hybrid'],
                    help='[v5] top-level credit assignment: shared team reward, per-head local rewards, COMA counterfactual advantage, or hybrid (COMA over per-head local returns)')
parser.add_argument('--pref_embed', type=int, default=0, help='[v5] replace raw lambda features with a learned preference embedding')
parser.add_argument('--pref_dim', type=int, default=16, help='[v5] preference embedding dimension')
parser.add_argument('--gcn_layer', type=int, default=3, help='gcn_paras') # 定义gcn的总层数，包括输入层
parser.add_argument('--mlp_fea_extract_layer', type=int, default=3, help='gcn_paras') # 定义特征提取要用的mlp的层数
parser.add_argument('--gcn_input_dim', type=int, default=12, help='gcn_paras') # 定义图卷积的最初输入的维度，即特征向量元素的个数
parser.add_argument('--gcn_hidden_dim', type=int, default=128, help='gcn_paras') # 定义gcn的隐藏层的维度
parser.add_argument('--learn_eps', type=int, default=False, help='gcn_paras') # 定义是否采用gin的学习参数e。（暂时就是简单对的图卷积）
parser.add_argument('--mlp_actor_layer', type=int, default=3, help='gcn_paras') # 定义对encoder之后的节点嵌入进行MLP的层数（输出logit，作为score），包括job和machine的mlp都用这个参数
parser.add_argument('--machine_hidden_dim', type=int, default=128, help='gcn_paras') # 定义选machine的FCL和MLP的隐藏层大小
parser.add_argument('--mlp_critic_layer', type=int, default=3, help='gcn_paras') # 定义critic的mlp的总层数（包括输入层）
parser.add_argument('--critic_input_dim', type=int, default=128, help='gcn_paras') # 定义critic的mlp的输入参数的维度
parser.add_argument('--critic_hidden_dim', type=int, default=128, help='gcn_paras') # 定义critic的mlp的隐藏层的大小

parser.add_argument('--log_to_wandb', '-w', type=bool, default=True)  # 是否使用wandb   
    
parser.add_argument('--load_pth', type=parse_dict, default=run_dict["load_pth"], help='paload_pth_parametersth') # 定义加载的模型的jme和episode
parser.add_argument('--use_load_model', type=int, default=False, help='paload_pth_parametersth') # 定义是否加载模型
parser.add_argument('--resume_ckpt', type=str, default='', help='[v5] full checkpoint .pt to resume training from')
parser.add_argument('--load_model_type', type=int, default=0, help='paload_pth_parametersth') # 0=best/top1,1=final,2=error选episode,3=top2/3
parser.add_argument('--trained_model_pth', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trained_model') + os.sep, help='paload_pth_parametersth')

"""计算episode的总次数，保证生成的instances数量都能跑完"""
epi_num = 12800 / parser.get_default('env_batch') * parser.get_default('resample_freq')    # 获取 --xxx 参数的默认值
parser.add_argument('--episode_num', type=int, default=int(epi_num), help='hyper_paras')# 训练次数

args = parser.parse_args()   # 解析命令行参数，并将结果存储在args变量中。