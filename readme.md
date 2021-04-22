mani_0   ppo_single   'distance_threshold': 0.02,  without max episode steps done  
mani_2   ppo_single   'distance_threshold': 0.02,  without max episode steps done  
mani_1   ppo_single   sparse goal 'distance_threshold': 0.02,  without max episode steps done  
mani_3   ppo_single  distance_threshold': 0.01,    log_std  nn.parameter  with max episode steps done  
mani_4   ppo_single  distance_threshold': 0.02,    log_std  nn.parameter   without max episode steps done  
mani_5   ppo_single  distance_threshold:  0.01,    log_std  nn.parameter   with max  espisode steps done  

# 平面2自由度  
mani_11  ppo_single   distance_threshold: 0.02  goal 0, 20, 0, 10  
mani_12  ppo_single   distance_threshold: 0.01  goal 0, 20, 0, 10  
# 平面3自由度  
mani_9   ppo_single   distance_threshold: 0.02  goal 0, 20, 0, 10, 0, 20  
mani_10  ppo_single   distance_threshold: 0.01  goal 0, 20, 0, 10, 0, 20  
# 平面10自由度  
mani_13  ppo_single  distance_threshold: 0.02 goal 0, 20, 0, 20, 0, -10, 0, -15, 0, 20, 0, 20, 0, 20, 0, -10, 0, -15, 0, 20  
mani_14  ppo_single  distance_threshold: 0.02 goal 0, 20, 0, 20, 0, -10, 0, -15, 0, 20, 0, 20, 0, 20, 0, -10, 0, -15, 0, 20  
mani_15  ppo_single  distance_threshold: 0.02 goal 0, 20, 0, 20, 0, -10, 0, -15, 0, 20, 0, 20, 0, 20, 0, -10, 0, -15, 0, 20  
# 5自由度  
~~mani_7   ppo_single  distance_threshold:  0.02  easy goal 0, 20, 0, 20, 0, -10, 0, -15, 0, 20~~   
# easy goal
mani_16  ppo_single  distance_threshold:  0.02  easy goal 0, 20, 0, 20, 0, -10, 0, -15, 0, 20   
mani_6   ppo_single  distance_threshold:  0.01  easy goal 0, 20, 0, 20, 0, -10, 0, -15, 0, 20  
~~mani_18  ppo_single  distance_threshold:  0.02  easy goal 0, 20, 0, 20, 0, -10, 0, -15, 0, 20  reward 势函数  gamma 0.99~~ [reward零点没矫正]  
~~mani_20  ppo_single  distance_threshold:  0.02  easy goal 0, 20, 0, 20, 0, -10, 0, -15, 0, 20  reward 势函数  gamma 0.99~~ [矫正失败]  
~~mani_21  ppo_single  distance_threshold:  0.02  easy goal 0, 20, 0, 20, 0, -10, 0, -15, 0, 20  reward 势函数  gamma  0.8~~ [矫正失败]    

# hard goal
mani_8   ppo_single  distance_threshold:  0.02  hard goal 0, 20, 0, 15, 0, 20, 0, 20, 0, 20  gamma 0.99  
mani_19  ppo_single  distance_threshold:  0.02  hard goal 0, 20, 0, 15, 0, 20, 0, 20, 0, 20  gamma 0.99  reward 势函数   
mani_22  ppo_single  distance_threshold:  0.02  hard goal 0, 20, 0, 15, 0, 20, 0, 20, 0, 20  gamma  0.8  reward 势函数    
~~mani_17  ppo_single  distance_threshold:  0.02  super hard goal 0, -50, 0, -50, 0, -50, 0, 0, -20, -10   10000 episodes~~ [目标错误]   
~~mani_23 ppo_single reward -d 势函数  easy  gamma 0.8~~  [矫正失败]  
~~mani_24 ppo_single reward -d 势函数  hard  gamma 0.8~~  [矫正失败]  
~~mani_25 ppo_single reward -d 势函数  super hard  gamma 0.8  (bug)~~   [矫正失败]  
~~mani_27 ppo_single reward 势函数  super hard  gamma 0.95~~  [retry]  [对照结果反的]  
~~mani_26 ppo_single reward 势函数  super hard  gamma 0.8~~   [retry]  [对照结果反的]  
~~mani_28 ppo_single reward 势函数  super hard  gamma 0.6~~   [retry]  [对照结果反的]  
# 5自由度平面 不同 gamma 不同 reward类型
mani_32 ppo_single reward 势函数  super hard  gamma 0.99   without max steps done true  

mani_66 ppo dense potential super hard goal gamma 0.95 
mani_67 ppo dense potential super hard goal gamma 0.8 
mani_68 ppo dense potential super hard goal gamma 0.6 
# 有点看出gamma越小越好，不明显  上下对应归一下的作用明显
mani_69 ppo dense potential super hard goal gamma 0.95  归一化  
mani_70 ppo dense potential super hard goal gamma 0.8   归一化  
mani_71 ppo dense potential super hard goal gamma 0.6   归一化  

~~mani_29 ppo_single reward 非势函数  super hard  gamma 0.9~~  [retry]
~~mani_31 ppo_single reward 非势函数  super hard  gamma 0.9~~  
mani_63 ppo dense distance super hard goal gamma 0.95  
mani_64 ppo dense distance super hard goal gamma 0.8   
mani_65 ppo dense distance super hard goal gamma 0.6   

mani_72 ppo dense distance super hard goal gamma 0.95  归一化  
mani_73 ppo dense distance super hard goal gamma 0.8   归一化  
mani_74 ppo dense distance super hard goal gamma 0.6   归一化  
# 有点看出gamma越小越好，不明显 
~~mani_30 ppo_single reward 非势函数  super hard  gamma 0.6~~  [retry]

# 20自由度平面 验证归一化
mani_75 ppo 20 joints plane dense distance super hard goal gamma 0.95  归一化
mani_76 ppo 20 joints plane dense distance super hard goal gamma 0.95 

# mani63, mani72, mani75, mani76验证归一化的影响，出图

----------------------------------ppo pyrep --------------------------------------------  
# 6节臂
# 6节臂gamma确定下reward 类型和在不同空间下的影响
mani_33 reward dense potential hard goal gamma 0.95 plane model  [平面环境下ppo势函数和普通reward差别不大]
mani_34 reward dense distance  hard  goal gamma 0.95 plane model  
~~mani_35 reward dense distance hard goal gamma 0.95 plane model run whole episode~~ [跑全局没什么意义]   
  
mani_36 reward dense potential hard goal gamma 0.95 3D  [这里证明在3D环境下ppo势函数明显比普通要好]  
mani_59 reward dense distance  hard goal gamma 0.95 3D  

~~mani_37 reward dense distance hard goal gamma 0.6 3D model~~  

# 对比6节臂和12节臂 在不同gamma，不同reward type 不同空间类型的效果，出图
# 6节臂
mani_40 reward dense potential gamma 0.99 plane   [对比表明平面内gamma影响不大， 平面成功率好，reward尖峰噪声多了点]  
main_41 reward dense potential gamma 0.9 plane 
mani_42 reward dense potential gamma 0.8 plane
mani_43 reward dense potential gamma 0.6 plane  

mani_53 reward dense potential gamma 0.99 3D hard max-episode-steps 20  [对比表明空间内gamma越小越好， 空间成功率图不好，reward图好]  
mani_54 reward dense potential gamma 0.9 3D hard max-episode-steps 20 
mani_55 reward dense potential gamma 0.8 3D hard max-episode-steps 20 
mani_56 reward dense potential gamma 0.6 3D hard max-episode-steps 20 

# 新补充实验
# 6节臂
mani_89 reward dense distance gamma 0.95 hard plane
mani_90 reward dense distance gamma 0.8  hard plane 
mani_91 reward dense distance gamma 0.6  hard plane
mani_92 reward dense distance gamma 0.95 hard 3D
mani_93 reward dense distance gamma 0.8  hard 3D
mani_94 reward dense distance gamma 0.6  hard 3D

mani_95 reward dense potential gamma 0.95 hard plane
mani_96 reward dense potential gamma 0.8  hard plane 
mani_97 reward dense potential gamma 0.6  hard plane
mani_98 reward dense potential gamma 0.95 hard 3D
mani_99 reward dense potential gamma 0.8  hard 3D
mani_100 reward dense potential gamma 0.6  hard 3D


~~mani_57 reward dense potential gamma 0.0 3D hard max-episode-steps 20~~ 
# 12节臂
mani_77 plane reward dense distance hard goal gamma 0.95  [对比表明平面内gamma也是越小越好,reward和rate表现一致]
mani_78 plane reward dense distance hard goal gamma 0.8  
mani_79 plane reward dense distance hard goal gamma 0.6  
mani_80 3D reward dense distance hard goal gamma 0.95   [没学起来，但gamma小一点好一些]
mani_81 3D reward dense distance hard goal gamma 0.8  
mani_82 3D reward dense distance hard goal gamma 0.6  

mani_83 plane reward dense potential hard goal gamma 0.95 [rate gamma越小越好]
mani_84 plane reward dense potential hard goal gamma 0.8  
mani_85 plane reward dense potential hard goal gamma 0.6  
mani_86 3D reward dense potential hard goal gamma 0.95   
mani_87 3D reward dense potential hard goal gamma 0.8  
mani_88 3D reward dense potential hard goal gamma 0.6  

gamma_sapce_6.png 6 节 dense potential  
gamma_sapce_12_1.png 12 节 dense distance 
gamma_sapce_12_2.png 12 节 dense potential

# 对比goal参数
~~mani_44 reward dense potential gamma 0.8  3D easy~~   [对比中噪声已经很大了]  
~~mani_45 reward dense potential gamma 0.8  3D hard~~
~~mani_46 reward dense potential gamma 0.8  3D super hard~~
# 对比max episode steps参数
~~mani_47 reward dense potential gamma 0.8 3D hard max-episode-steps 20~~  [都很差了]
~~mani_48 reward dense potential gamma 0.8 3D hard max-episode-steps 30~~ 
~~mani_49 reward dense potential gamma 0.8 3D hard max-episode-steps 50~~ 
# 对比batch size 参数
~~mani_50 reward dense potential gamma 0.8 3D hard batch_size 16~~ [越大越差]
~~mani_51 reward dense potential gamma 0.8 3D hard batch_size 64~~
~~mani_52 reward dense potential gamma 0.8 3D hard batch_size 128~~


mani_58 reward dense potential gamma 0.8 3D random goal max-episode-steps 30  [1500 episodes 还没达到25%]


-------------------------------------------------remote-----------------------------------------------------------  
td3_2   lunarlander  td3_agent  
td3_3  lunarlander   td3_run    
td3_4   lunarlander   td3_ours  
td3_5  lunarlander   td3_ours  repair sigma bug when excute action  
td3_6  lunarlander   td3_ours  np.random.randint and random.random  

~~td3_8   easy goal~~   
~~td3_9   super hard goal gamma 0.99~~  
~~td3_10  super hard goal on ubuntu gamma 0.99~~  
~~td3_11  gamma  0.6  hard goal    distance threshold 0.05~~  
~~td3_12  gamma  0.6  super hard goal     distance threshold 0.05~~    
~~td3_13  gamma 0.9 or 0.6（not clear）  super hard goal  distance threshold 0.02~~  
~~td3_18 joint 10 multi goal dense potential gamma 0.9~~   

----------------------------td3_cc-------------------------------------  
~~td3_20  multi goal cc_model gamma 0.9~~  
~~td3_21 multi goal cc_model eval 确定性策略~~  
~~td3_22 multi goal cc_model eval 随机性策略~~  
~~td3_23 multi goal cc_model 去掉了getpingtime 其他和td3_20一样~~  
~~td3_24 multi goal td3_20的retry实验~~  

-----------------------------td3_pyrep-----------------------------  
~~td3_25 multi goal not cc_model train   pyrep version first try~~  
~~td3_27 multi goal not cc_model train   pyrep 兼容cc版本first try~~  [初始化存在漂移]
~~td3_28 multi goal not cc_model train   pyrep 兼容cc版本 pr.stop+pr.start~~  
~~td3_29 eval of td3_28~~    

~~修改reset方式~~  

~~td3_33 super hard goal dense distance gamma=0.9 action_noise_drop_rate 1000~~  

~~td3_44 random goal gamma=0.6 noise_decay_period 1000 reward dense potential 3D model~~  
 
~~td3_47 hard goal gamma=0.6 action_noise_drop_rate 500 reward dense potential not cc model 3D model
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
actor [128, 128],  critic [64, 64] 关节角度角速度归一化~~  

~~td3_70 random goal gamma=0.6 action_noise_drop_rate 1000 reward dense potential cc model 3D model 
actor [128, 128],  critic [64, 64] 状态归一化，包括末端位置 和速度  80%成功  （带reset bug） 
td3_71 super hard goal 'super hard': 0, -40, 0, -40, 0, -40, 0, 35, 0, 35, 0, 35 
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
td3_80 super hard goal 困难版本 gamma 0.6 noise_drop_rate 500 reward dense  potential cc model plane model  collision reward -0.1~~  
~~平面内gamma太小会导致得到的策略非累积折扣奖励最大，gamma大的时候学习效果好  
空间内gamma太大会导致学习缓慢且收敛不稳定，gamma小的时候反而学习效果要更好~~  
~~td3_81 hard goal gamma 0.99 noise 500 reward dense potential cc model plane model  collision reward -0.1 
td3_81_rt hard goal gamma 0.99 dense potential cc_model plane_model  
td3_82 super hard goal gamma 0.99 noise 500 dense potential cc modle 3D model  collision reward -0.1 
td3_82_rt super hard goal gamma 0.99 dense potential cc_model 3D model
td3_83 super hard goal gamma 0.6  noise 500 dense potential cc model 3D model collision reward -0.1 
td3_83_rt super hard goal gamma 0.6 dense potential cc_model 3D model~~  

~~td3_90 hard goal num_joints 12 gamma 0.6 dense potential not cc model 3D model  不提前done 现象螺旋丸  
td3_91 hard goal num_joints 12 gamma 0.99 dense potential not cc model 3D model 不提前done  
td3_92 hard goal num_joints 12 gamma 0.6 dense potential not cc model 3D model  不提前done distance threshold 0.05
td3_93 hard goal num_joints 6 gamma 0.6 dense potential not cc model 3D model 不提前done~~   

td3_45 hard goal gamma=0.9 noise_decay_period 1000 reward sparse 3D model 

~~td3_14 super hard goal gamma 0.9 dense distance~~  
td3_34 super hard goal gamma=0.9 dense distance  
td3_15 super hard goal gamma 0.9 dense potential  
~~td3_43 super hard goal gamma=0.6 dense potential 3D model~~  

td3_36 hard goal gamma=0.9 dense potential plane
td3_37 hard goal gamma=0.9 dense distance plane

td3_38 hard goal gamma=0.9  dense distance 3D model  
td3_42 hard goal gamma=0.6  dense distance 3D model 
 
td3_40 hard goal gamma=0.99 dense potential 3D model  [这里明显gamma越小越好]
td3_39 hard goal gamma=0.9  dense potential 3D model  
td3_41 hard goal gamma=0.6  dense potential 3D model  

# 平面内可以泛化，空间内泛化不了，除非归一化
~~td3_17 random goal dense distance gamma=0.9   plane~~  
td3_31 random goal dense distance gamma=0.9   plane
~~td3_32 random goal dense distance gamma=0.99  plane~~
td3_30 random goal dense potential gamma=0.9  plane 

~~td3_35 random goal gamma 0.9 dense potential 3D~~
td3_46 random goal gamma=0.6 noise_decay_period 4000 dense potential 3D model 
td3_69 random goal gamma=0.6 noise_decay_period 1000 reward dense potential not cc model 3D model actor [128, 128],  critic [64, 64] 状态归一化，包括末端位置 和速度  100%成功 
# 差一个归一化之后gamma 0.9泛化的实验

td3_19 joitn 10 multi goal 正态train 均匀eval  gamma 0.99  

td3_85 hard goal num_joints 24 gamma 0.99 dense potential not cc model 3D model  
td3_84 hard goal num_joints 24 gamma 0.6  dense potential not cc model 3D model   
td3_86 hard goal num_joints 24 gamma 0.5  dense potential not cc model 3D model  

---------------------------------------td3 mujoco env-------------------------------------  
~~td3_100 random  goal num_joints 12  gamma 0.6 dense potential 3D model max_episode_steps 50~~  

~~td3_102 td3_100 eval~~  
~~td3_103 td3_101 eval~~  
~~td3_104 random goal num_joints 12 gamma 0.6 dense potential 3D model max_episode_steps 100 eval按照her来~~  
#----------------------------测试最大精度,6节臂最大可以到0.5cm精度------------------------------------------------------------------------------
#这部分没有用归一化
td3_101 random goal num_joints 12 gamma 0.6 dense potential 3D model max_episode_steps 100    
td3_105 random goal num_joints 12 gamma 0.6 dense potential 3D model max_episode_steps 100 distance-threshold 0.01 episodes 20000
td3_106 random goal num_joints 12 gamma 0.6 dense potential 3D model max_episode_steps 100 distance-threshold 0.005 episodes 20000
td3_107 random goal num_joints 24 gamma 0.6 dense potential 3D model max_episode_steps 100 distance-threshold 0.02 episode 20000  
~~td3_108 random goal num_joints 24 gamma 0.6 dense potential 3D model max_episode_steps 100 distance-threshold 0.01 episode 20000~~ [数据太短]  
#-----------------------------------------------------------------------------------------------------------------------------------------  


#----------------------------验证不同reward的效果-------------------------------------------------------------------------------------------
#以下是不带归一化
td3_115 hard goal num_joints 24 gamma 0.6 dense distance 3D model max_episode_steps 100 distance threshold 0.02 episode 2000  
td3_117 hard goal num_joints 24 gamma 0.6 dense potential 3D model max_episode_steps 100 distance threshold 0.02 episode 4000   
td3_118 hard goal num_joints 24 gamma 0.6 dense mix 3D model max_episode_steps 100 distance threshold 0.02 episode 4000  
td3_116 hard goal num_joints 24 gamma 0.6 dense 2x 3D model max_episode_steps 100 distance threshold 0.02 episode 4000
td3_114 hard goal num_joints 24 gamma 0.6 dense 4x 3D model max_episode_steps 100 distance threshold 0.02 episode 2000
<font color=#FF0000> **td3_111与td3_115对比表明势函数比-d好** </font>  
td3_117,td3_118,td3_115表明R>R+r>r,其中R表示势函数  
td3_117,td3_116,td3_114表明高次势函数效果差，可以补充低次势函数实验  
#以下是在归一化条件和PEB条件下新补充的实验
td3_165 hard goal num_joints 24 gamma 0.6 add peb 3D model dense distance 
td3_167 hard goal num_joints 24 gamma 0.6 add peb 3D model dense potential 
td3_168 hard goal num_joints 24 gamma 0.6 add peb 3D model dense mix 
td3_169 hard goal num_joints 24 gamma 0.6 add peb 3D model dense 2x 
td3_170 hard goal num_joints 24 gamma 0.6 add peb 3D model dense 4x 
#这里实验表明mix效果会更好一点
#----------------------------------------------------------------------------------------------------------------------------------------


#-----------------------------下验证peb和ta作用--------------------------------------------------------------------------------------------
td3_119 hard goal num_joints 24 gamma 0.6 dense potential 3D model add_ta add_peb  验证time aware和peb,没有归一化  
td3_120 hard goal num_joints 24 gamma 0.6 dense potential 3D model 存在种子没对齐的bug
td3_121 hard goal num_joints 24 gamma 0.6 dense potential 3D model 种子与td3_111对齐，加入简单归一化，功能与td3_84对齐，效果和td3_111差不多  
td3_130 修改了一些不重要地方产生了新的随机序列  对齐 td3_121  
td3_122 hard goal num_joints 24 gamma 0.6 dense potential 3D model add_ta 
td3_123 hard goal num_joints 24 gamma 0.6 dense potential 3D model add_peb  
td3_124 hard goal num_joints 24 gamma 0.6 dense potential 3D model add_ta add_peb  
td3_121,td3_122,td3_123,td3_124消融实验，验证time-awareness和PEB效果,如果效果不明显可以考虑gamma=0.99的情况效果怎么样   
td3_126 hard goal num_joints 24 gamma 0.99 dense potential 3D model add_ta add_peb  
#td3_126 和 td3_124对照实验证明空间0.99gamma不可取  
#---------------------------------------------------------------------------------------------------------------------------------------

~~td3_125 eval td3_122/1990.pth on 1000 epsilon decay rate  1 -> 0.05
td3_127 eval td3_122/1830.pth on 1000 epsilon decay rate  1 -> 0.05  
td3_128 eval td3_122/1830.pth on 1000 epsilon decay rate  1 -> 0  
td3_129 eval td3_122/1990.pth on 1000 epsilon decay rate  1 -> 0~~  
~~td3_131  td3_130加入ASF~~

#--------------------------------------测试action-q的作用---------------------------------------------------------------------------------
td3_165 hard goal num_joints 24 gamma 0.6 add peb 3D model dense distance action_q 0
td3_164 hard goal num_joints 24 gamma 0.6 add peb 3D model dense distance action_q 0.1
td3_171 hard goal num_joints 24 gamma 0.6 add peb 3D model dense distance action_q 0.5
td3_172 hard goal num_joints 24 gamma 0.6 add peb 3D model dense distance action_q 1
td3_173 hard goal num_joints 24 gamma 0.6 add peb 3D model dense distance action_q 10
#---------------------------------------------------------------------------------------------------------------------------------------


#---------------------------------------测试gamma space type的影响-----------------------------------------------------------------
td3_140 hard goal num_joints 12 gamma 0.95 add peb plane dense distance 
td3_141 hard goal num_joints 12 gamma 0.8  add peb plane dense distance
td3_142 hard goal num_joints 12 gamma 0.6  add peb plane dense distance
td3_143 hard goal num_joints 12 gamma 0.95 add peb 3D dense distance 
td3_144 hard goal num_joints 12 gamma 0.8  add peb 3D dense distance
td3_145 hard goal num_joints 12 gamma 0.6  add peb 3D dense distance

td3_146 hard goal num_joints 12 gamma 0.95 add peb plane dense potential 
td3_147 hard goal num_joints 12 gamma 0.8  add peb plane dense potential
td3_148 hard goal num_joints 12 gamma 0.6  add peb plane dense potential
td3_149 hard goal num_joints 12 gamma 0.95 add peb 3D dense potential 
td3_150 hard goal num_joints 12 gamma 0.8  add peb 3D dense potential
td3_151 hard goal num_joints 12 gamma 0.6  add peb 3D dense potential

td3_152 hard goal num_joints 24 gamma 0.95 add peb plane dense distance 
td3_153 hard goal num_joints 24 gamma 0.8  add peb plane dense distance
td3_154 hard goal num_joints 24 gamma 0.6  add peb plane dense distance
td3_155 hard goal num_joints 24 gamma 0.95 add peb 3D dense distance 
td3_156 hard goal num_joints 24 gamma 0.8  add peb 3D dense distance
td3_157 hard goal num_joints 24 gamma 0.6  add peb 3D dense distance

td3_158 hard goal num_joints 24 gamma 0.95 add peb plane dense potential 
td3_159 hard goal num_joints 24 gamma 0.8  add peb plane dense potential
td3_160 hard goal num_joints 24 gamma 0.6  add peb plane dense potential
td3_161 hard goal num_joints 24 gamma 0.95 add peb 3D dense potential 
td3_162 hard goal num_joints 24 gamma 0.8  add peb 3D dense potential
td3_163 hard goal num_joints 24 gamma 0.6  add peb 3D dense potential

td3_185 hard goal num_joints 24 gamma 0.0  add peb 3D dense distance
td3_186 hard goal num_joints 24 gamma 0.0  add peb plane dense distance
#--------------------------------------------------------------------------------------------------------------------

td3_187 one step max_angle_vel 50
td3_188 one step max_angle_vel 40
td3_189 one step max_angle_vel 30
td3_190 one step max_angle_vel 20
td3_191 one step max_angle_vel 10
td3_192 one step max_angle_vel 60
td3_193 one step max_angle_vel 70
~~td3_194 one step max_angle_vel 70~~
td3_195 one step max_angle_vel 70  reward * 10
#----------------------------------------------------------------------------------------------------------------------


td3_200 random goal joints 24 dense distance  3D add peb 归一化 gamma=0.6
td3_201 random goal joints 24 dense potential 3D add peb 归一化 special goal random eval goal gamma=0.6
td3_202 random goal joints 24 dense potential 3D add peb 归一化 special1 goal special1 eval goal gamma=0.6
td3_203 random goal joints 24 dense potential 3D add peb 归一化 special1 goal special1 eval goal gamma=0.0

-------------------------------------her -------------------------------------------------  
her_3 her on pyrep env num_joints 12  
her_4 her on pyrep env num_joints 24  
her_5 her on mujoco env  num_joints 12  
her_31 her on mujoco env num_joints 12 random goal distance-threshold 0.02 epoch 500 DenseNet add_dtt  跑错了实验  seed  1 
her_6 her on mujoco env  num_joints 24  
her_7 her on mujoco env num_joints 12 hard goal   
#------------------------------------------------------------测试block------------------------------------------------------------
her_8 her on mujoco env num_joints 12 block0 goal  
block_100 her on mujoco env num_joints 12 block0 goal  noise eval  
her_9 her on mujoco env num_joints 12 block0 goal block env  

her_10 her on mujococ env num_joints 12 block1 goal  
block_101 her on mujoco env num_joints 12 block1 goal  noise eval  
her_11 her on mujoco env num_joints 12 block1 goal block env  

her_13 her on mujoco env num_joints 12 block2 goal 
block_102 her on mujoco env num_joints 12 block2 goal  noise eval
block_103 her on mujoco env num_joints 12 block2 goal  noise eval not add-dtt
her_12 her on mujoco env num_joints 12 block2 goal block env  

her_14 her on mujoco env num_joints 24 block3 goal
her_15 her on mujoco env num_joints 24 block3 goal block env
her_16 her on mujoco env num_joints 24 block0 goal block env plane model
her_17 her on mujoco env num_joints 24 block0 goal block env plane model with heatmap

her_86 joints 24 3D block4 goal dt .02 dense action_l2 1 block4_env
her_87 joints 24 3D block4 goal dt .02 dense action_l2 1 
her_115 joints 24 3D dt .02 dense block4 goal  带loss显示

#---------------------------------------------------测试不同网络下不同精度的效果--------------------------------------------------------------
her_21 her on mujoco env num_joints 24 random goal distance-threshold 0.02 epoch 500
her_22 her on mujoco env num_joints 24 random goal distance-threshold 0.015 epoch 1000
her_20 her on mujoco env num_joints 24 random goal distance-threshold 0.01 epoch 1000  
her_19 her on mujoco env num_joints 24 random goal distance-threshold 0.005 epoch 1000  

her_110 her on mujoco env num_joints 24 random goal distance-threshold 0.02  epoch 1000 DenseNet seed  1 add_dtt  td3
her_32  her on mujoco env num_joints 24 random goal distance-threshold 0.02  epoch  500 DenseNet seed  1 add_dtt  
her_116 her on mujoco env num_joints 24 random goal distance-threshold 0.02  epoch  500 DenseNet seed  1 add_dtt  带loss显示，种子有变
her_27 her on mujoco env num_joints 24 random goal distance-threshold 0.02  epoch  500 DenseNet seed  1  
her_28 her on mujoco env num_joints 24 random goal distance-threshold 0.015 epoch 1000 DenseNet seed  1  
her_29 her on mujoco env num_joints 24 random goal distance-threshold 0.01  epoch 1000 DenseNet seed  1  

her_26 her on mujoco env num_joints 24 random goal distance-threshold 0.02 epoch  500  DenseNetSimple seed 1  
her_90 her on mujoco env num_joints 24 random goal distance-threshold 0.015 epoch 1000 DenseNetSimple seed 1 添加末端距离矢量特征
her_93 her on mujoco env num_joints 24 random goal distance-threshold 0.01  epoch 1000 DenseNetSimple seed 1 添加末端距离矢量特征
#----------------------------------------------------------------------------------------------------------------------------------

her_23 her on mujoco env num_joints 24 random goal distance-threshold 0.02 epoch 500 均匀 sample method   
her_24 her on mujoco env num_joints 24 random goal distance-threshold 0.02 epoch 500 尖峰 sample method  
her_91 her on mujoco env num_joints 24 random goal distance-threshold 0.02 epoch 500 尖峰学，均匀评估 sample method  
her_25 her_6 9800 eval
#------------------------------------------------------------------------------------------------------------------------------

#---------------------------------------------------测试 random reset-----------------------------------------------------------
her_33 her on mujoco env num_joints 4 random goal distance-threshold 0.02 epoch 500 DenseNet plane model   
her_35 her on mujoco env num_joints 4 random goal distance-threshold 0.02 epoch 500 DenseNet plane model random initial action_l2 1  
her_36 her on mujoco env num_joints 4 random goal distance-threshold 0.02 epoch 500 DenseNet plane model random initial action_l2 0.1  
her_34 her on mujoco env num_joints 4 random goal distance-threshold 0.02 epoch 500 DenseNet plane model random initial action_l2 0  
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
~~her_72 joints 8 random goal dt .02 denseASF hidden [16] random init_10 3D  critic2-ratio 0 add_dtt max_joint_speed 20~~  
~~her_73 joints 8 random goal dt .02 denseASF hidden [16] random init_10 3D  critic2-ratio 0 add_dtt 1500 episode sample cnt decay~~
~~her_74 retry her_66~~

~~her_76 joints 24 3D random goal dt .02 denseASF hidden [16] action_l2 1   add_dtt~~  
~~her_77 joints 24 3D random goal dt .02 denseASF hidden [16] action_l2 0.1 add_dtt~~  
 
her_30 joints 12 3D random goal dt .02 dense action_l2 1 random-initial-state 永不复位 
her_75 joints 12 3D random goal dt .02 denseASF hidden [16] random init_1-->10(30)  add_dtt  

her_98 joints 24 3D random goal dt .02 dense action_l2 1 random-initial-state-10 fixed reset
her_78 joints 24 3D random goal dt .02 dense action_l2 1 random_initial(100--30-->10)  种子和her_32一致
her_79 joints 24 3D random goal dt .02 dense action_l2 1 random_initial(50--30-->10)  
her_81 joints 24 3D random goal dt .02 dense action_l2 1 random_initial(20--30-->10)  
her_80 joints 24 3D random goal dt .02 dense action_l2 1 random_initial(0--30-->10)  999.pth二校门 
her_118 joints 24 3D random goal dt .02 dense action_l2 1 (random_initial0--30-->10)  the same as her_80
her_100 joints 24 3D random goal dt .02 mlp action_l2 1 random_initial(0--30-->10) 

her_82 joints 24 3D random goal dt .02 dense action_l2 0.1 random_initial(50--30-->10)  
her_84 joints 24 3D random goal dt .02 dense action_l2 0.1 random_initial(50--30-->10)  her_82数据丢了重跑了一遍  
her_83 joints 24 3D random goal dt .02 dense action_l2 1 random_initial(20--30-->10)  double_q critic_ratio 0.1  
her_85 joints 24 3D random goal dt .02 dense action_l2 1 random_initial(50--30-->10)  double_q critic_ratio 1



her_94 joints 24 3D special1 goal eval special1 goal 
her_95 joints 24 3D random   goal eval special1 goal
her_96 joints 24 3D random   goal eval special1 goal
her_97 joints 24 3D random   goal eval special1 goal


her_101 jonins 24 3D dt .02 dense random goal action_l2 0.1
her_103 jonins 24 3D dt .02 dense random goal action_l2 5
her_102 jonins 24 3D dt .02 dense random goal action_l2 10

#这里q-action是反的
~~her_104 jonins 24 3D dt .02 dense random goal q-action 0.1~~ 二范数形式  
~~her_105 jonins 24 3D dt .02 dense random goal q-action 5~~  二范数形式  
~~her_106 jonins 24 3D dt .02 dense random goal q-action 10~~  二范数形式  
#q-action正的
her_107 jonins 24 3D dt .02 dense random goal q-action 0.1 二范数形式   负面影响大 
her_108 jonins 24 3D dt .02 dense random goal q-action 1   二范数形式

her_111 joints 24 3D dt .02 dense random goal add-dtt q-action 0.1 残差形式  负面影响小 阶次不确定
her-117 joints 24 3D dt .02 dense random goal add-dtt q-action 10  残差形式  负面影响大，完全学不出 四阶
her-119 joints 24 3D dt .02 dense random goal random_initial(0--30-->10)  add-dtt q-action  1e-4 一言难尽  四阶
her-120 joints 24 3D dt .02 dense random goal random_initial(0--30-->10)  goal special1 eval goal special1 一言难尽
her-121 joints 24 3D dt .02 dense random goal goal special1 eval goal special1 
her-122 joints 24 3D dt .02 dense random goal goal special1 eval goal random


------------------------------------------------------------------------------------------------------------------- 

block0_1_env_6 一块平面长板
block0_2_env_6 两块间隔0.1长板
block0_3_env_6 两块间隔0.2长板
block0_4_env_6 两块间隔0.15长板
block0_5_env_6 两块间隔0.05长板


~~block_0  dt 0.03 joint_goal block_env block5
block_1  dt 0.03 joint_goal 
block_2  dt 0.02 goal       block5
block_3  dt 0.1  joint_goal 
block_4  dt 0.2  joint_goal~~
block_5  dt 0.02  goal     sparse reward  td3  
block_6  dt 0.02 block5 goal block-env 3D
block_7  dt 0.02 block3 goal block-env plane
block_8  dt 0.02 block3 goal non-block-env plane
block_9  dt 0.02 block3 goal non-block-env plane actor MLP 
block_10 dt 0.02 block0 goal block-env block0_1_env_6    
block_11 dt 0.02 block0_1 goal block-env block0_1_env_6 
block_12 dt 0.02 block0_2 goal block-env block0_1_env_6
block_13 dt 0.02 block0_2 goal block-env block0_2_env_6
block_14 dt 0.02 block0_2 goal block-env block0_3_env_6   0.75~0.9
block_15 dt 0.02 block0_2 goal block-env block0_3_env_6   0.7~0.9
# HER_RND
block_16 dt 0.02 block0_2 goal block-env block0_1_env_6 add-dtt 带normalization action_l2 0.1 
~~block_17 dt 0.02 block0_2 goal block-env block0_1_env_6 add-dtt action_l2 0~~ 
block_18 dt 0.02 block0_2 goal block_env block0_1_env_6 add-dtt action_l2 0 
block_19 dt 0.02 block0_2 goal block_env block0_1_env_6 add-dtt action_l2 0 pop_art

# 无效乌鸡哥
~~block_20 dt 0.02 block0_2 goal block_env block0_1_env_6 add-dtt action_l2 0 pop_art beta 3e-4
block_21 dt 0.02 block0_2 goal block_env block0_1_env_6 add-dtt action_l2 0 pop_art beta 3e-3
block_22 dt 0.02 block0_2 goal block_env block0_1_env_6 add-dtt action_l2 0 pop_art beta 3e-2
block_23 dt 0.02 block0_2 goal block_env block0_1_env_6 add-dtt action_l2 0 pop_art beta 0.1
block_24 dt 0.02 block0_2 goal block_env block0_1_env_6 add-dtt action_l2 0 pop_art beta 3e-5
block_25 dt 0.02 block0_2 goal block_env block0_1_env_6 add-dtt action_l2 0 pop_art beta 3e-6~~

~~block_27 dt 0.02 block0_2 goal block-env block0_1_env_6 action_l2 0   same as block_17 block_18~~
block_29 dt 0.02 block0_2 goal block-env block0_1_env_6 add-dtt action_l2 0   same as block_17 block_18
block_28 dt 0.02 block0_2 goal block-env block0_1_env_6 action_l2 0 pop_art beta 3e-4 

block_30 dt 0.02 block0_2 goal block-env block0_1_env_6 action_l2 0 beta 1e-4 
~~block_31 dt 0.02 block0_2 goal block-env block0_1_env_6 action_l2 0 art beta 1e-4~~  
block_32 dt 0.02 block0_2 goal block-env block0_1_env_6 action_l2 0 art beta 1e-4  target含义有改变，为未归一化时的target

#有效乌鸡哥
block_34 dt 0.02 block0_2 goal block-env block0_1_env_6 action_l2 0 art beta 1e-5 min_step 800000
block_35 dt 0.02 block0_2 goal block-env block0_1_env_6 action_l2 0 art beta 1e-4 min_step 800000
block_36 dt 0.02 block0_2 goal block-env block0_1_env_6 action_l2 0 art beta 1e-5 min_step 100000


block_37 dt 0.02 block0_2 goal block-env block0_1_env_6 action_l2 0 art beta 1e-4 min_step 100000
block_38 dt 0.02 block0_2 goal block-env block0_1_env_6 action_l2 0.1 art beta 1e-4 min_step 100000
block_39 dt 0.02 block0_2 goal block-env block0_1_env_6 action_l2 0.01 art beta 1e-4 min_step 100000
block_40 dt 0.02 block0_2 goal block-env block0_1_env_6 action_l2 1 art beta 1e-4 min_step 100000

block_41 dt 0.02 block0_2 goal block-env block0_3_env_6 action_l2 0.1 art beta 1e-4 min_step 100000
block_42 dt 0.02 block0_2 goal block-env block0_4_env_6 action_l2 0.1 art beta 1e-4 min_step 100000
#正式加入curiosity
block_48 dt 0.02 block3 goal block-env block3_env_6 reward_weight 0.8 explore_weight 0.2 eval使用不同的策略
#两快相隔0.15平面板
block_43 dt 0.02 block0_2 goal block-env block0_4_env_6 reward_weight 1.0 explore_weight 0.0 eval使用train时的相同策略
block_44 dt 0.02 block0_2 goal block-env block0_4_env_6 reward_weight 1.0 explore_weight 0.0 eval使用不同的策略
block_45 dt 0.02 block0_2 goal block-env block0_4_env_6 reward_weight 0.8 explore_weight 0.2 eval使用不同的策略

#两快相隔0.1的平面板
~~block_46 dt 0.02 block0_2 goal block-env block0_2_env_6 reward_weight 0.8 explore_weight 0.2 eval使用不同的策略~~
block_47 dt 0.02 block0_2 goal block-env block0_2_env_6 reward_weight 0.8 explore_weight 0.2 eval使用不同的策略 same as block_46 46中获取的是train的策略
block_49 dt 0.02 block0_2 goal block-env block0_2_env_6 reward_weight 1 explore_weight 0 eval使用不同的策略

#两块平面板相距0.05米
block_52 dt 0.02 block0_5 goal block-env block0_5_env_6 dense critic  fail
block_51 dt 0.02 block0_5 goal block-env block0_5_env_6 reward_weight 1.0 explore_weight 0.0 eval使用不同的策略 不稳定popart fail
block_50 dt 0.02 block0_5 goal block-env block0_5_env_6 reward_weight 0.8 explore_weight 0.2 eval使用不同的策略 不稳定popart success
~~block_53 dt 0.02 block0_5 goal block-env block0_5_env_6 reward_weight 0.8 explore_weight 0.2 eval使用不同的策略 不稳定popart success~~
~~block_57 block0_5 goal block-env block0_5_env_6 reward_weight 1 explore_weight 0 stable 0.05 min_step 100000~~ 
block_58 block0_5 goal block-env block0_5_env_6 reward_weight 1 explore_weight 0 stable 0.005 min_step 100000 
block_56 dt 0.02 block0_5 goal block-env block0_5_env_6 reward_weight 0.8 explore_weight 0.2 eval使用不同的策略 pop-art 直接进入pop-art reward pop art 发散
block_55 dt 0.02 block0_5 goal block-env block0_5_env_6 reward_weight 0.8 explore_weight 0.2 eval使用不同的策略 art fail
~~block_54 dt 0.02 block0_5 goal block-env block0_5_env_6 reward_weight 0.5 explore_weight 0.5 eval使用不同的策略~~  

block_59 block0_5 goal block-env block0_5_env_6 reward_weight 0.8 explore_weight 0.2 forward dynamic stable success
block_65 block0_5 goal block-env block0_5_env_6 reward_weight 0.8 explore_weight 0.2 RND 稳定版本pop-art

#终极两块平面板
block_61 block0_5 goal block-env block0_6_env_6 reward_weight 0.8 explore_weight 0.2 forward dynamic 向左，同样是很慢
block_63 block0_5 goal block-env block0_6_env_6 reward_weight 0.5 explore_weight 0.5 forward dynamic 和block_64差不多，向右但是速度很慢
block_64 block0_5 goal block-env block0_6_env_6 reward_weight 0.5 explore_weight 0.5 RND  没学到， 有学到的迹象，向右转而不是向左转， 但是跑得很慢导致50步没出结果
block_66 block0_5 goal block-env block0_6_env_6 reward_weight 0.2 explore_weight 0.8 RND  没学到

#平面钻孔
block_60 block3   goal block-env block3_env_12 reward_weight 0.8 explore_weight 0.2 stable 0.005 min_step 100000 plane
block_62 block3   goal block-env block3_env_12 reward_weight 0.5 explore_weight 0.5 forward dynamic plane

block_69 reward 0.5 explore 0.5              fail 
block_73 reward 0.8 explore 0.2              success
block_70 mlp-256 RND reward 0.8 explore 0.2  success
block_72 mlp-10  RND reward 0.8 explore 0.2  fail

# predict dynamic compare
block_73 forward dynamic reward 1.0 explore 0.0  block0_5_env_6  fail
block_74 forward dynamic reward 0.8 explore 0.2  block0_5_env_6  success 
block_75 forward dynamic reward 0.5 explore 0.5  block0_5_env_6  fail q有上升迹象

block_76 RND MLP         reward 1.0 explore 0.0  block0_5_env_6  fail
block_77 RND MLP         reward 0.8 explore 0.2  block0_5_env_6  suceess 
block_78 RND MLP         reward 0.5 explore 0.5  block0_5_env_6  fail

block_79 RND densenet    reward 1.0 explore 0.0  block0_5_env_6
block_80 RND densenet    reward 0.8 explore 0.2  block0_5_env_6
block_81 RND densenet    reward 0.5 explore 0.5  block0_5_env_6

#测试action-l2的影响
block_82 block2 mani_env_6 action_l2 0.1 pop-art
~~block_85 block2 mani_env_6 action_l2 0.1 pop-art~~
block_86 block2 mani_env_6 action_l2 0.5 pop-art
block_87 block2 mani_env_6 action_l2 1.0 pop-art

block_83 block2 mani_env_6 action_l2 0.1  不要popart
blcok_84 blcok2 mani_env_6 action_l2 1    不要popart

#---------------------测试rms形式explore--------------------------------------------
block_88 block2 mani_env_6 action_l2 1 rms  reward 0.9 explore 0.1 

block_84 block2 mani_env_6 action_l2 1 rms  reward 1.0 explore 0 
block_90 block2 mani_env_6 action_l2 1 rms  reward 1.0 explore 0.01 
block_91 block2 mani_env_6 action_l2 1 rms  reward 1.0 explore 0.1 
block_92 block2 mani_env_6 action_l2 1 rms  reward 1.0 explore 1 
block_93 block2 mani_env_6 action_l2 1 rms  reward 1.0 explore 10 
block_94 block2 mani_env_6 action_l2 1 rms  reward 1.0 explore 100 
block_95 block2 mani_env_6 action_l2 1 rms  reward 1.0 explore 1000 
# 在无障碍环境中表现可以，有障碍时不行
block_99 block0_5 block0_5_env_6 action_l2 1 rms  reward 1.0 explore 0.1
block_96 block0_5 block0_5_env_6 action_l2 1 rms  reward 1.0 explore 1
block_97 block0_5 block0_5_env_6 action_l2 1 rms  reward 1.0 explore 10 
block_98 block0_5 block0_5_env_6 action_l2 1 rms  reward 1.0 explore 100 