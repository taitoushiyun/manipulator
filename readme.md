mani_0   ppo_single   'distance_threshold': 0.02,  without max episode steps done  
mani_2   ppo_single   'distance_threshold': 0.02,  without max episode steps done  
mani_1   ppo_single   sparse goal 'distance_threshold': 0.02,  without max episode steps done  
mani_3   ppo_single  distance_threshold': 0.01,    log_std  nn.parameter  with max episode steps done  
mani_4   ppo_single  distance_threshold': 0.02,    log_std  nn.parameter   without max episode steps done  
mani_5   ppo_single  distance_threshold:  0.01,    log_std  nn.parameter   with max  espisode steps done  

mani_6   ppo_single  distance_threshold:  0.01    repair mu bug    with max episode steps done  
mani_7   ppo_single  distance_threshold:  0.02    repair  mu  bug      with max episode steps done 0, 20, 0, 20, 0, -10, 0, -15, 0, 20  
mani_16   ppo_single  distance_threshold:  0.02    repair  mu  bug    wiht max episode steps done 0, 20, 0, 20, 0, -10, 0, -15, 0, 20  
mani_8  ppo_single  distance_threshold:  0.02    repair  mu  bug    wiht max episode steps done goal  0, 20, 0, 15, 0, 20, 0, 20, 0, 20  
mani_9  ppo_single   distance_threshold: 0.02  goal 0, 20, 0, 10, 0, 20  
mani_10  ppo_single   distance_threshold: 0.01  goal 0, 20, 0, 10, 0, 20  
mani_11  ppo_single   distance_threshold: 0.02  goal 0, 20, 0, 10  
mani_12  ppo_single   distance_threshold: 0.01  goal 0, 20, 0, 10  
mani_13  ppo_single   distance_threshold: 0.02 0, 20, 0, 20, 0, -10, 0, -15, 0, 20, 0, 20, 0, 20, 0, -10, 0, -15, 0, 20  
mani_14  ppo_single   distance_threshold: 0.02 0, 20, 0, 20, 0, -10, 0, -15, 0, 20, 0, 20, 0, 20, 0, -10, 0, -15, 0, 20  
mani_15  ppo_single   distance_threshold: 0.02 0, 20, 0, 20, 0, -10, 0, -15, 0, 20, 0, 20, 0, 20, 0, -10, 0, -15, 0, 20  
mani_17 ppo_single  distance_threshold: 0.02   goal  0, -50, 0, -50, 0, -50, 0, 0, -20, -10   10000 episodes  5平面自由度  
mani_18 ppo_single reward 势函数 goal 0, 20, 0, 20, 0, -10, 0, -15, 0, 20  
mani_19  ppo_single  reward -d 势函数   goal 0, 20, 0, 15, 0, 20, 0, 20, 0, 20  
mani_20  ppo_single  reward -d 势函数  0, 20, 0, 20, 0, -10, 0, -15, 0, 20  gamma 0.99  
mani_21  ppo_single  reward -d 势函数  0, 20, 0, 20, 0, -10, 0, -15, 0, 20  gamma  0.8  
mani_22  ppo_single  reward -d 势函数  0, 20, 0, 15, 0, 20, 0, 20, 0, 20  gamma  0.8  

mani_23 ppo_single reward -d 势函数  easy  gamma 0.8  
mani_24 ppo_single reward -d 势函数  hard  gamma 0.8  
mani_25 ppo_single reward -d 势函数  super hard  gamma 0.8  (bug)  

mani_26 ppo_single reward -d 势函数  super hard  gamma 0.8  修正goal 标错bug  
mani_27 ppo_single reward -d 势函数  super hard  gamma 0.95  修正goal 标错bug  
mani_28 ppo_single reward -d 势函数  super hard  gamma 0.6  修正goal 标错bug  

mani_29 ppo_single reward -d 非势函数  super hard  gamma 0.9  
mani_31 ppo_single reward -d 非势函数  super hard  gamma 0.9  
mani_30 ppo_single reward -d 非势函数  super hard  gamma 0.6  
mani_32 ppo_single reward -d 势函数     super hard  gamma 0.99   with out max steps done true  


-------------------------------------------------remote-----------------------------------------------------------  
td3_2   lunarlander  td3_agent  
td3_3  lunarlander   td3_run    
td3_4   lunarlander   td3_ours  
td3_5  lunarlander   td3_ours  repair sigma bug when excute action  
td3_6  lunarlander   td3_ours  np.random.randint and random.random  

td3_8  easy goal  
td3_9   super hard goal gamma 0.99  
td3_10  super hard goal on ubuntu gamma 0.99  
td3_11  gamma  0.6  hard goal           distance threshold 0.05  
td3_12  gamma  0.6  super hard goal     distance threshold 0.05  
td3_13  gamma 0.9 or 0.6（not clear）  super hard goal  distance threshold 0.02  
td3_14  gamma 0.9  super hard goal distance threshold 0.02  非势函数  
td3_15  gamma 0.9  super hard goal distance threshold 0.02  势函数  

td3_17  multi goal gamma 0.9 非势函数  
td3_18  multi goal gamma 0.9 势函数  
td3_19  multi goal 正态train 均匀eval  gamma 0.99  

----------------------------td3_cc-------------------------------------  
td3_20  multi goal cc_model gamma 0.9  
td3_21 multi goal cc_model eval 确定性策略  
td3_22 multi goal cc_model eval 随机性策略  
td3_23 multi goal cc_model 去掉了getpingtime 其他和td3_20一样  
td3_24 multi goal td3_20的retry实验  

-----------------------------td3_pyrep-----------------------------  
td3_25 multi goal not cc_model train   pyrep version first try  
td3_27 multi goal not cc_model train   pyrep 兼容cc版本first try  
td3_28 multi goal not cc_model train pyrep 兼容cc版本 pr.stop+pr.start  
td3_29 eval of td3_28  
td3_30 td3_28 retry 10000 episodes  dense potential gamma=0.9 not cc_model plane model
修改reset方式  
td3_31 multi goal not cc_model reward dense distance gamma=0.9  
td3_32 multi goal not cc_model reward dense distance gamma=0.99  
td3_33 super hard goal not cc_model reward dense distance gamma=0.9 action_noise_drop_rate 1000  
td3_34 super hard goal not cc_model reward dense distance gamma=0.9 action_noise_drop_rate 500  
td3_35 multi goal gamma 0.9 action_noise_drop_rate 2000  reward dense potential not cc model 3D model 12 joints  
td3_36 hard goal gamma=0.9 action_noise_drop_rate 500 reward dense potential not cc model plane model  
td3_37 hard goal gamma=0.9 action_noise_drop_rate 500 reward dense distance not cc model plane model   
td3_38 hard goal gamma=0.9 action_noise_drop_rate 500 reward dense distance not cc model 3D model  
td3_39 hard goal gamma=0.9 action_noise-drop_rate 500 reward dense potential not cc model 3D model  
td3_40 hard goal gamma=0.99 action_noise_drop_rate 500 reward dense potential not cc model 3D model  
td3_41 hard goal gamma=0.6 action_noise_drop_rate 500 reward dense potential not cc model 3D model  
td3_42 hard goal gamma=0.6 action_noise_drop_rate 50 reward dense0 distance not cc model 3D model  
td3_43 super hard goal gamma=0.6 noise_decay_period 500 reward dense potential not cc model 3D model  
td3_44 random goal gamma=0.6 noise_decay_period 1000 reward dense potential not cc model 3D model  
td3_46 random goal gamma=0.6 noise_decay_period 4000 reward dense potential not cc model 3D model  
td3_45 hard goal gamma=0.9 noise_decay_period 1000 reward sparse not cc model 3D model  
td3_47 hard goal gamma=0.6 action_noise_drop_rate 500 reward dense potential not cc model 3D model
actor [64, 64, 32, 32],  ciritc [64, 64]  
td3_48 hard goal gamma=0.6 action_noise_drop_rate 500 reward dense potential not cc model 3D model
actor [100, 100],  ciritc [64, 64]  
td3_49 hard goal gamma=0.6 action_noise_drop_rate 500 reward dense potential not cc model 3D model
actor [100, 100],  ciritc [32, 32]  
td3_50 hard goal gamma=0.6 action_noise_drop_rate 500 reward dense potential not cc model 3D model
actor [64, 64],  ciritc [64, 64]  
td3_51 hard goal gamma=0.6 action_noise_drop_rate 500 reward dense distance not cc model 3D model
actor [64, 64],  ciritc [64, 64]  
td3_52 hard goal gamma=0.6 action_noise_drop_rate 500 reward dense distance not cc model 3D model
actor [100, 100],  ciritc [64, 64] done change  
td3_53 td3_54 hard goal gamma=0.99 action_noise_drop_rate 500 reward dense distance not cc model 3D model
actor [100, 100],  ciritc [64, 64] done change  
td3_55 hard goal gamma=0.6 action_noise_drop_rate 500 reward dense distance not cc model 3D model
actor [100, 100],  critic [64, 64]  
td3_56 hard goal gamma=0.6 action_noise_drop_rate 500 reward dense distance not cc model 3D model 
actor [100, 100],  critic [64, 64] reward +10  
td3_57 hard goal gamma=0.6 action_noise_drop_rate 500 reward dense distance not cc model 3D model 
actor [100, 100],  critic [64, 64] reward +100  
td3_58 hard goal gamma=0.6 action_noise_drop_rate 500 reward dense distance not cc model 3D model 
actor [200, 200],  critic [64, 64]  
td3_59 hard goal gamma=0.99 action_noise_drop_rate 500 reward dense distance not cc model 3D model 
actor [100, 100],  critic [64, 64]  reward +100  
td3_60 hard goal gamma=0.6 action_noise_drop_rate 500 reward dense potential not cc model 3D model 
actor [200, 200],  critic [64, 64]  
td3_61 super hard goal gamma=0.6 action_noise_drop_rate 500 reward dense potential not cc model 3D model 
actor [128, 128],  critic [64, 64]  
td3_62 random goal gamma=0.6 action_noise_drop_rate 500 reward dense potential not cc model 3D model 
actor [128, 128],  critic [64, 64]  
td3_63 super hard goal gamma=0.6 action_noise_drop_rate 500 reward dense potential not cc model 3D model 
actor [128, 128],  critic [64, 64] 模型最大关节角度50度  
td3_64 hard goal gamma=0.6 action_noise_drop_rate 500 reward dense potential not cc model 3D model 
actor [128, 128],  critic [64, 64] 状态近似归一化  base_pos bug 全为0  
td3_65 super hard goal gamma=0.6 action_noise_drop_rate 500 reward dense potential not cc model 3D model 
actor [128, 128],  critic [64, 64] 状态归一化  失败  
td3_66 super hard goal gamma=0.6 action_noise_drop_rate 500 reward dense potential not cc model 3D model 
actor [128, 128],  critic [64, 64] 状态近似归一化  
td3_67 super hard goal gamma=0.6 action_noise_drop_rate 500 reward dense potential not cc model 3D model 
actor [128, 128],  critic [64, 64] 状态归一化，包括末端位置 和速度
td3_68 super hard goal gamma=0.6 action_noise_drop_rate 500 reward dense potential not cc model 3D model 
actor [128, 128],  critic [64, 64] 关节角度角速度归一化  
td3_69 random goal gamma=0.6 action_noise_drop_rate 1000 reward dense potential not cc model 3D model 
actor [128, 128],  critic [64, 64] 状态归一化，包括末端位置 和速度  100%成功  

td3_70 random goal gamma=0.6 action_noise_drop_rate 1000 reward dense potential cc model 3D model 
actor [128, 128],  critic [64, 64] 状态归一化，包括末端位置 和速度  80%成功  
td3_71 super hard goal gamma=0.6 action_noise_drop_rate 1000 reward dense potential cc model plane model  



 


