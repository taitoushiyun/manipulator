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
mani_32 ppo_single reward -d 势函数    super hard  gamma 0.99   without max steps done true  

----------------------------------ppo pyrep --------------------------------------------  
mani_33 reward dense potential hard goal gamma 0.95 plane model  
mani_34 reward dense distance hard goal gamma 0.95 plane model  
mani_35 reward dense distance hard goal gamma 0.95 plane model run whole episode  
mani_36 reward dense distance hard goal gamma 0.95 3D model  
mani_37 reward dense distance hard goal gamma 0.6 3D model 


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
td3_28 multi goal not cc_model train   pyrep 兼容cc版本 pr.stop+pr.start  
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
actor [128, 128],  critic [64, 64] 状态归一化  失败 (done没有处理好)
td3_66 super hard goal gamma=0.6 action_noise_drop_rate 500 reward dense potential not cc model 3D model 
actor [128, 128],  critic [64, 64] 状态近似归一化,关节角度角速度rad表示 
td3_67 super hard goal gamma=0.6 action_noise_drop_rate 500 reward dense potential not cc model 3D model 
actor [128, 128],  critic [64, 64] 状态归一化，包括末端位置 和速度
td3_68 super hard goal gamma=0.6 action_noise_drop_rate 500 reward dense potential not cc model 3D model 
actor [128, 128],  critic [64, 64] 关节角度角速度归一化  
td3_69 random goal gamma=0.6 action_noise_drop_rate 1000 reward dense potential not cc model 3D model 
actor [128, 128],  critic [64, 64] 状态归一化，包括末端位置 和速度  100%成功  

td3_70 random goal gamma=0.6 action_noise_drop_rate 1000 reward dense potential cc model 3D model 
actor [128, 128],  critic [64, 64] 状态归一化，包括末端位置 和速度  80%成功  （带reset bug）
td3_71 super hard goal 'super hard': [0, -40, 0, -40, 0, -40, 0, 35, 0, 35, 0, 35]
gamma=0.6 action_noise_drop_rate 1000 reward dense potential cc model plane model 错误的goal  
td3_72 easy goal gamma 0.6 noise_drop_rate 500 reward dense potential cc model plane model  
td3_73 hard goal gamma 0.6 noise_drop_rate 500 reward dense potential cc model plane model  
td3_75 easy goal gamma 0.6 noise_drop_rate 500 reward dense potential cc model not plane model  
td3_76 hard goal gamma 0.6 noise_drop_rate 500 reward dense potential cc model not plane model  
td3_72_rt td3_73_rt td3_75 td3_76 retry (修复cc model 的 reset bug)  
td3_74_rt super hard goal简单版本 gamma 0.6 noise_drop_rate 500 reward  dense potential cc model plane model  
td3_74 super hard goal 困难版本 gamma 0.6 noise_drop_rate 500 reward dense  potential cc model plane model  
td3_77 super hard goal 困难版本 gamma 0.9 noise_drop_rate 500 reward dense  potential cc model plane model  
td3_78 super hard goal 困难版本 gamma 0.6 noise_drop_rate 500 reward dense  potential cc model plane model  不做collision检测  
td3_79 super hard goal 困难版本 gamma 0.99 noise_drop_rate 500 reward dense  potential cc model plane model  
td3_80 super hard goal 困难版本 gamma 0.6 noise_drop_rate 500 reward dense  potential cc model plane model  collision reward -0.1  


平面内gamma太小会导致得到的策略非累积折扣奖励最大，gamma大的时候学习效果好  
空间内gamma太大会导致学习缓慢且收敛不稳定，gamma小的时候反而学习效果要更好  

td3_81 hard goal gamma 0.99 noise 500 reward dense potential cc model plane model  collision reward -0.1 
td3_81_rt hard goal gamma 0.99 dense potential cc_model plane_model  
td3_82 super hard goal gamma 0.99 noise 500 dense potential cc modle 3D model  collision reward -0.1 
td3_82_rt super hard goal gamma 0.99 dense potential cc_model 3D model
td3_83 super hard goal gamma 0.6  noise 500 dense potential cc model 3D model collision reward -0.1 
td3_83_rt super hard goal gamma 0.6 dense potential cc_model 3D model

td3_84 hard goal num_joints 24 gamma 0.6 dense potential not cc model 3D model  
td3_85 hard goal num_joints 24 gamma 0.99 dense potential not cc model 3D model  
td3_86 hard goal num_joints 24 gamma 0.5 dense potential not cc model 3D model  

td3_90 hard goal num_joints 12 gamma 0.6 dense potential not cc model 3D model 不提前done 现象螺旋丸  
td3_91 hard goal num_joints 12 gamma 0.99 dense potential not cc model 3D model 不提前done  
td3_92 hard goal num_joints 12 gamma 0.6 dense potential not cc model 3D model 不提前done distance threshold 0.05
td3_93 hard goal num_joints 6 gamma 0.6 dense potential not cc model 3D model 不提前done 

---------------------------------------td3 mujoco env-------------------------------------  
td3_100 random  goal num_joints 12  gamma 0.6 dense potential 3D model max_episode_steps 50  
td3_101 random  goal num_joints 12  gamma 0.6 dense potential 3D model max_episode_steps 100  
td3_102 td3_100 eval  
td3_103 td3_101 eval  
td3_104 random goal num_joints 12  gamma 0.6 dense potential 3D model max_episode_steps 100 eval按照her来  
td3_105 random goal num_joints 12 gamma 0.6 dense potential 3D model max_episode_steps 100 distance-threshold 0.01 episodes 20000
td3_106 random goal num_joints 12 gamma 0.6 dense potential 3D model max_episode_steps 100 distance-threshold 0.005 episodes 20000
td3_107 random goal num_joints 24 gamma 0.6 dense potential 3D model max_episode_steps 100 distance-threshold 0.02 episode 20000  
td3_108 random goal num_joints 24 gamma 0.6 dense potential 3D model max_episode_steps 100 distance-threshold 0.01 episode 20000  

~~td3_111 hard goal num_joints 24 gamma 0.6 dense potential 3D model max_episode_steps 100 distance threshold 0.02 episode 2000~~  
td3_117 hard goal num_joints 24 gamma 0.6 dense potential 3D model max_episode_steps 100 distance threshold 0.02 episode 4000   
~~td3_112 hard goal num_joints 24 gamma 0.6 dense mix 3D model max_episode_steps 100 distance threshold 0.02 episode 2000~~  
td3_118 hard goal num_joints 24 gamma 0.6 dense mix 3D model max_episode_steps 100 distance threshold 0.02 episode 4000  
~~td3_113 hard goal num_joints 24 gamma 0.6 dense 2x 3D model max_episode_steps 100 distance threshold 0.02 episode 2000~~  
td3_116 hard goal num_joints 24 gamma 0.6 dense 2x 3D model max_episode_steps 100 distance threshold 0.02 episode 4000
td3_114 hard goal num_joints 24 gamma 0.6 dense 4x 3D model max_episode_steps 100 distance threshold 0.02 episode 2000
td3_115 hard goal num_joints 24 gamma 0.6 dense distance 3D model max_episode_steps 100 distance threshold 0.02 episode 2000  
<font color=#FF0000> **td3_111与td3_115对比表明势函数比-d好** </font>  
td3_117,td3_118,td3_115表明R>R+r>r,其中R表示势函数  
td3_117,td3_116,td3_114表明高次势函数效果差，可以补充低次势函数实验  


td3_119 hard goal num_joints 24 gamma 0.6 dense potential 3D model add_ta add_peb  验证time aware和peb,没有归一化  
td3_120 hard goal num_joints 24 gamma 0.6 dense potential 3D model 存在种子没对齐的bug
td3_121 hard goal num_joints 24 gamma 0.6 dense potential 3D model 种子与td3_111对齐，加入简单归一化，功能与td3_84对齐，效果和td3_111差不多  
td3_130 修改了一些不重要地方产生了新的随机序列  对齐 td3_121  
td3_122 hard goal num_joints 24 gamma 0.6 dense potential 3D model add_ta 
td3_123 hard goal num_joints 24 gamma 0.6 dense potential 3D model add_peb  
td3_124 hard goal num_joints 24 gamma 0.6 dense potential 3D model add_ta add_peb  
td3_121,td3_122,td3_123,td3_124消融实验，验证time-awareness和PEB效果,如果效果不明显可以考虑gamma=0.99的情况效果怎么样   

td3_126 hard goal num_joints 24 gamma 0.99 dense potential 3D model add_ta add_peb  
td3_126 和 td3_124对照实验证明空间0.99gamma不可取  
td3_125 eval td3_122/1990.pth on 1000 epsilon decay rate  1 -> 0.05
td3_127 eval td3_122/1830.pth on 1000 epsilon decay rate  1 -> 0.05  
td3_128 eval td3_122/1830.pth on 1000 epsilon decay rate  1 -> 0  
td3_129 eval td3_122/1990.pth on 1000 epsilon decay rate  1 -> 0  

td3_131  td3_130加入ASF

-------------------------------------her -------------------------------------------------  
her_3 her on pyrep env num_joints 12  
her_4 her on pyrep env num_joints 24  
her_5 her on mujoco env  num_joints 12  
her_6 her on mujoco env  num_joints 24  
her_7 her on mujoco env num_joints 12 hard goal   
her_8 her on mujoco env num_joints 12 block0 goal  
her_9 her on mujoco env num_joints 12 block0 goal block env  
her_10 her on mujococ env num_joints 12 block1 goal  
her_11 her on mujoco env num_joints 12 block1 goal block env  
her_12 her on mujoco env num_joints 12 block2 goal block env  
her_13 her on mujoco env num_joints 12 block2 goal  
her_14 her on mujoco env num_joints 24 block3 goal
her_15 her on mujoco env num_joints 24 block3 goal block env
her_16 her on mujoco env num_joints 24 block0 goal block env plane model
her_17 her on mujoco env num_joints 24 block0 goal block env plane model with heatmap
her_21 her on mujoco env num_joints 24 random goal distance-threshold 0.02 epoch 500
her_22 her on mujoco env num_joints 24 random goal distance-threshold 0.015 epoch 1000
her_20 her on mujoco env num_joints 24 random goal distance-threshold 0.01 epoch 1000  
her_19 her on mujoco env num_joints 24 random goal distance-threshold 0.005 epoch 1000 
her_23 her on mujoco env num_joints 24 random goal distance-threshold 0.02 epoch 500 均匀 sample method   
her_24 her on mujoco env num_joints 24 random goal distance-threshold 0.02 epoch 500 尖峰 sample method  
her_25 her_6 9800 eval
her_26 her on mujoco env num_joints 24 random goal distance-threshold 0.02 epoch 500 DenseNet简化版本  seed 1  
her_27 her on mujoco env num_joints 24 random goal distance-threshold 0.02 epoch 500 DenseNet seed 1  
her_28 her on mujoco env num_joints 24 random goal distance-threshold 0.015 epoch 1000 DenseNet seed  1  
her_29 her on mujoco env num_joints 24 random goal distance-threshold 0.01 epoch 1000 DenseNet seed  1  
 
her_30 her on mujoco env num_joints 24 random goal distance-threshold 0.02 epoch 1000 DenseNet  random-initial-state  seed  1  
her_31 her on mujoco env num_joints 12 random goal distance-threshold 0.02 epoch 500 DenseNet 添加末端距离矢量特征  跑错了实验  seed  1     
her_32 her on mujoco env num_joints 24 random goal distance-threshold 0.02 epoch 500 DenseNet 添加末端距离矢量特征  seed  1   

her_33 her on mujoco env num_joints 4 random goal distance-threshold 0.02 epoch 500 DenseNet plane model   
her_34 her on mujoco env num_joints 4 random goal distance-threshold 0.02 epoch 500 DenseNet plane model random initial action_l2 0  
her_35 her on mujoco env num_joints 4 random goal distance-threshold 0.02 epoch 500 DenseNet plane model random initial action_l2 1  
her_36 her on mujoco env num_joints 4 random goal distance-threshold 0.02 epoch 500 DenseNet plane model random initial action_l2 0.1  
her_37 her on mujoco env num_joints 6 random goal distance-threshold 0.02 epoch 500 DenseNet plane model   
her_38 her on mujoco env num_joints 6 random goal distance-threshold 0.02 epoch 500 DenseNet plane model random initial  
her_39 her on mujoco env num_joints 8 random goal distance-threshold 0.02 epoch 500 DenseNet plane model   
her_40 her on mujoco env num_joints 8 random goal distance-threshold 0.02 epoch 500 DenseNet plane model random initial  
her_41 her on mujoco env num_joints 10 random goal distance-threshold 0.02 epoch 500 DenseNet plane model   
her_42 her on mujoco env num_joints 10 random goal distance-threshold 0.02 epoch 500 DenseNet plane model random initial  
her_43 her on mujoco env num_joints 12 random goal distance-threshold 0.02 epoch 500 DenseNet plane model  
her_44 her on mujoco env num_joints 12 random goal distance-threshold 0.02 epoch 500 DenseNet plane model random initial  
her_45 her on mujoco env num_joints 16 random goal distance-threshold 0.02 epoch 500 DenseNet plane model  
her_46 her on mujoco env num_joints 16 random goal distance-threshold 0.02 epoch 500 DenseNet plane model random initial  
her_47 her on mujoco env num_joints 24 random goal distance-threshold 0.02 epoch 500 DenseNet plane model  
her_48 her on mujoco env num_joints 24 random goal distance-threshold 0.02 epoch 500 DenseNet plane model random initial  
her_49 her on mujoco env num_joints 18 random goal distance-threshold 0.02 epoch 500 DenseNet plane model random initial  
her_50 her on mujoco env num_joints 20 random goal distance-threshold 0.02 epoch 500 DenseNet plane model random initial  
her_51 her on mujoco env num_joints 20 random goal distance-threshold 0.02 epoch 500 DenseNet plane model random initial reset period 10  
her_52 her on mujoco env num_joints 20 random goal distance-threshold 0.02 epoch 500 DenseNet plane model random initial reset period 10 ASF+Densenet  
her_53 her on mujoco env num_joints 20 random goal distance-threshold 0.02 epoch 500 DenseNet plane model random initial reset period 10 ASF+Densenet 两层attention  
her_54 joints 20 random goal dt 0.02 denseASF hidden [16] random init_10 plane  
her_55 joints 20 random goal dt 0.02 denseASF hidden [1] random init_10 plane   
her_56 joints 20 random goal dt 0.02 denseASF hidden [1] random init_10 plane  修改denseASF结构  

her_57 joints 20 random goal dt .02 denseASF hidden [16] random init_10 plane  critic2-ratio 0.01  
her_58 joints 20 random goal dt .02 denseASF hidden [16] random init_10 plane  critic2-ratio 0.1  
her_59 joints 20 random goal dt .02 denseASF hidden [16] random init_10 plane  critic2-ratio 0.5  
her_60 joints 20 random goal dt .02 denseASF hidden [16] random init_10 plane  critic2-ratio 1  
her_61 joints 20 random goal dt .02 denseASF hidden [16] random init_10 plane  critic2-ratio 10  
her_62 joints 4 random goal dt .02 denseASF hidden [16] random init_10 3D  critic2-ratio 0  
her_63 joints 6 random goal dt .02 denseASF hidden [16] random init_10 3D  critic2-ratio 0  
her_64 joints 8 random goal dt .02 denseASF hidden [16] random init_10 3D  critic2-ratio 0  
her_65 joints 10 random goal dt .02 denseASF hidden [16] random init_10 3D  critic2-ratio 0  
her_66 joints 12 random goal dt .02 denseASF hidden [16] random init_10 3D  critic2-ratio 0  
her_67 joints 14 random goal dt .02 denseASF hidden [16] random init_10 3D  critic2-ratio 0  
her_68 joints 16 random goal dt .02 denseASF hidden [16] random init_10 3D  critic2-ratio 0  
her_69 joints 18 random goal dt .02 denseASF hidden [16] random init_10 3D  critic2-ratio 0  
her_70 joints 20 random goal dt .02 denseASF hidden [16] random init_10 3D  critic2-ratio 0  

her_71 joints 10 random goal dt .02 denseASF hidden [16] random init_10 3D  critic2-ratio 0 add_dtt distance to target  
her_72 joints 8 random goal dt .02 denseASF hidden [16] random init_10 3D  critic2-ratio 0 add_dtt max_joint_speed 20  
her_73 joints 8 random goal dt .02 denseASF hidden [16] random init_10 3D  critic2-ratio 0 add_dtt 1500 episode sample cnt decay
her_74 retry her_66
her_75 joints 12 random goal dt .02 denseASF hidden [16] random init_1-->10(30) 3D add_dtt 
her_76 joints 24 random goal dt .02 denseASF hidden [16] action_l2 1 3D add_dtt
 


