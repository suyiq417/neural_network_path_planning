from typing import List, Tuple, Dict, Optional, Set # 用于类型注解，增强代码可读性和健壮性
import numpy as np  # 用于数值计算，特别是数组操作
import pandas as pd # 用于数据处理，尤其是在从xlsx文件加载数据时（虽然在此代码段中未直接使用，但EvacuationEntity.from_xlsx可能使用）
from evacuation_entity import EvacuationEntity, EntityType # 导入自定义的实体类和实体类型枚举
from path_generation import PathGenerator # 确保可以导入
from model import BPNeuralNetworkModel # 导入自定义的BP神经网络模型类
import random # 用于生成随机数，例如选择紧急区域
import copy # 用于深拷贝字典等
import os # 导入os模块用于路径处理
import multiprocessing # 导入多进程库，后续进行并行自学习
import matplotlib.pyplot as plt # 导入 matplotlib
import visualization as viz # 导入可视化模块
import torch # 导入 PyTorch，不使用tensflow，因为cuda版本限制
import tempfile # 导入 tempfile

BETA = 0.8 # 公式 (18) 中的拥堵减少系数
MIN_VOLUME = 2.0 # 为面积为0的房间或非房间实体设置最小体积，防止除零
FLOOR_HEIGHT = 3.5 # 定义楼层高度常量 (与 evacuation_entity.py 中默认值一致，3.5m)

def get_entity_volume(entity: EvacuationEntity) -> float:
    """
    获取实体的容量/体积。
    对于 Room 类型，使用其面积（最小为 MIN_VOLUME）。
    对于其他类型，使用默认的 MIN_VOLUME。
    :param entity: 实体对象
    :return: 实体的体积（大于0）
    """
    if entity.entity_type == EntityType.Room:
        # 如果是房间，使用其面积，但确保不小于 MIN_VOLUME
        return max(entity.area, MIN_VOLUME)
    else:
        # 对于非房间类型（门、走廊、楼梯、出口），使用默认最小体积
        return MIN_VOLUME

# 定义路径训练类
class PathTraining:
    # 初始化方法
    def __init__(self, training_data: List[Tuple[List[EvacuationEntity], List[EvacuationEntity], float]],
                 all_entities: List[EvacuationEntity],
                 hidden_dim: int = 128, # 神经网络隐藏层维度，默认为128
                 learning_rate: float = 0.001): # 移除了 weight_decay 参数
        """
        初始化路径训练器。
        :param training_data: 训练数据列表，每个元素是一个元组 (路径, 紧急实体列表, 路径得分)
        :param all_entities: 所有实体对象的列表
        :param hidden_dim: 神经网络隐藏层的大小
        :param learning_rate: 神经网络的学习率
        """
        self.training_data = training_data # 存储训练数据
        self.all_entities = all_entities # 存储所有实体
        self.N = len(all_entities) # 获取实体总数
        # 创建一个从实体ID到其在 all_entities 列表中索引的映射字典
        self.entity_id_to_index: Dict[int, int] = {entity.id: i for i, entity in enumerate(all_entities)}

        # 定义神经网络的输入维度和输出维度
        # 输入维度 = 当前位置(N) +紧急位置(N) + 偏置(1) = 2*N + 1
        input_dim = 2 * self.N + 1
        # 输出维度 = 每个可能的目标位置的得分(N)
        output_dim = self.N

        # 实例化BP神经网络模型 
        self.model = BPNeuralNetworkModel(input_dim=input_dim,
                                          hidden_dim=hidden_dim,
                                          output_dim=output_dim,
                                          learning_rate=learning_rate)

    # 准备训练数据的方法（得到所有训练集路径的每一步的输入和输出向量构建）
    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        将原始训练数据转换为神经网络可以使用的输入和目标 NumPy 数组。
        :return: 一个包含输入数组和目标数组的元组 (inputs, targets)
        """
        inputs = [] # 初始化输入数据列表
        targets = [] # 初始化目标数据列表

        # 遍历每条训练路径及其相关信息
        for path, emergency_entities, score in self.training_data:
            # 获取该路径对应的紧急实体ID集合
            emergency_ids = {entity.id for entity in emergency_entities}

            # 遍历路径中的每一步（除了最后一步）
            for i in range(len(path) - 1):
                current_entity = path[i] # 当前实体
                next_entity = path[i + 1] # 下一步的实体

                # 创建输入向量，初始化为全零
                # 维度：N (当前位置) + N (紧急位置) + 1 (偏置)
                input_vector = np.zeros(2 * self.N + 1)

                # 设置当前位置的输入特征
                if current_entity.id in self.entity_id_to_index:
                    current_index = self.entity_id_to_index[current_entity.id]
                    input_vector[current_index] = 1 # 将当前实体对应的索引位置设为1
                else:
                    # 如果当前实体ID不在映射中，打印警告并跳过此步骤
                    print(f"警告：当前实体 ID {current_entity.id} 不在索引映射中。跳过此步骤。")
                    continue

                # 设置紧急位置的输入特征
                for emergency_id in emergency_ids:
                    if emergency_id in self.entity_id_to_index:
                        emergency_index = self.entity_id_to_index[emergency_id]
                        # 将紧急实体对应的索引位置（在输入向量的后半部分）设为1
                        input_vector[self.N + emergency_index] = 1

                # 设置偏置项（通常设为1）
                input_vector[2 * self.N] = 1

                # 创建目标向量，初始化为全零
                # 维度：N (每个可能的目标位置的得分)
                target_vector = np.zeros(self.N, dtype=float)
                # 设置目标向量中，实际下一步实体对应的位置的值为该路径的得分
                if next_entity.id in self.entity_id_to_index:
                    next_index = self.entity_id_to_index[next_entity.id]
                    target_vector[next_index] = score # 将下一步实体对应的索引位置设为路径得分
                else:
                    # 如果下一步实体ID不在映射中，打印警告并跳过此步骤
                    print(f"警告：下一个实体 ID {next_entity.id} 不在索引映射中。跳过此步骤。")
                    continue

                # 只有当下一个实体也在索引中时，才添加输入和目标向量
                if next_entity.id in self.entity_id_to_index:
                    inputs.append(input_vector)
                    targets.append(target_vector)

        # 如果没有生成任何有效的输入/输出对，打印警告并返回空数组
        if not inputs:
             print("警告：没有生成有效的训练输入/输出对。请检查训练数据和实体映射。")
             return np.array([]), np.array([])

        # 将列表转换为 NumPy 数组并返回
        return np.array(inputs), np.array(targets)

    # 训练模型的方法
    def train(self, epochs: int = 100, batch_size: int = 32, validation_split: float = 0.2):
        """
        准备数据并调用 PyTorch 模型的 train_model 方法。
        :param epochs: 训练的轮数
        :param batch_size: 每批训练的数据量大小
        :param validation_split: 验证集划分比例
        :return: 包含训练历史的字典 (如 {'loss': [...], 'val_loss': [...]}) 或 None
        """
        # 准备训练数据
        inputs, targets = self.prepare_training_data()
        # 检查是否有有效的训练数据
        if inputs.size == 0 or targets.size == 0:
            print("无法进行训练，因为没有准备好有效的训练数据。")
            return None # 返回 None 表示训练未进行
        # 打印准备好的数据的形状信息
        print(f"准备训练数据：输入形状 {inputs.shape}, 目标形状 {targets.shape}")

        # 调用 PyTorch 模型的训练方法
        # 这里调用的是 model 实例的 train_model 方法
        history = self.model.train_model(inputs, targets, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

        # 直接返回 PyTorch train_model 返回的 history 字典
        return history

# 使用训练好的模型生成路径的函数
def generate_path_with_model(model: BPNeuralNetworkModel,
                             start_entity: EvacuationEntity, # 起始实体
                             emergency_entities: List[EvacuationEntity], # 紧急实体列表
                             all_entities: List[EvacuationEntity], # 所有实体列表
                             entity_id_to_index: Dict[int, int], # 实体ID到索引的映射
                             index_to_entity_id: Dict[int, int], # 索引到实体ID的映射
                             max_length: int = 100) -> Optional[List[EvacuationEntity]]: # 最大路径长度限制
    """
    使用训练好的神经网络模型生成一条从起点到出口的疏散路径。
    :param model: 训练好的BP神经网络模型
    :param start_entity: 路径的起始实体
    :param emergency_entities: 当前场景下的紧急实体列表
    :param all_entities: 所有实体对象的列表
    :param entity_id_to_index: 实体ID到其在 all_entities 列表中索引的映射
    :param index_to_entity_id: 索引到实体ID的映射
    :param max_length: 生成路径的最大长度限制
    :return: 生成的路径（实体列表），如果无法生成或未找到出口则返回 None
    """
    path = [start_entity] # 初始化路径，包含起始实体
    visited_entity_ids = {start_entity.id} # 记录已访问的实体ID，防止循环
    current_entity = start_entity # 当前所在实体
    N = len(all_entities) # 实体总数
    emergency_ids = {entity.id for entity in emergency_entities} # 紧急实体ID集合
    # 创建一个从实体ID到实体对象的映射，方便快速查找
    all_entities_map = {entity.id: entity for entity in all_entities}

    # 循环生成路径，直到到达出口或达到最大长度
    while current_entity.entity_type != EntityType.Exit and len(path) < max_length:
        # 创建输入向量
        input_vector = np.zeros(2 * N + 1)
        # 检查当前实体ID是否存在于映射中
        if current_entity.id not in entity_id_to_index:
            print(f"警告：当前实体 ID {current_entity.id} 不在索引映射中，无法生成路径。")
            return None # 无法继续生成，返回None
        current_index = entity_id_to_index[current_entity.id]
        input_vector[current_index] = 1 # 设置当前位置

        # 设置紧急位置
        for emergency_id in emergency_ids:
            if emergency_id in entity_id_to_index:
                emergency_index = entity_id_to_index[emergency_id]
                input_vector[N + emergency_index] = 1
        input_vector[2 * N] = 1 # 设置偏置项

        # 使用模型预测下一步的得分
        predicted_scores = model.predict(np.array([input_vector]))[0] # 获取预测结果 (N维向量)

        next_entity = None # 初始化下一步实体
        best_score = -np.inf # 初始化最高得分

        # 只考虑相邻且未访问的实体
        if hasattr(current_entity, 'connected_entity_ids') and current_entity.connected_entity_ids:
            candidate_neighbor_ids = current_entity.connected_entity_ids
        else:
            # 如果没有连接信息，或者连接列表为空，则无法继续
            print(f"警告：实体 {current_entity.id} 没有连接信息或没有邻居，无法确定下一步。")
            return None # 或者可以尝试其他策略，但通常意味着路径中断

        # 遍历物理上相邻的实体ID
        for neighbor_id in candidate_neighbor_ids:
            # 检查邻居是否未被访问过
            if neighbor_id not in visited_entity_ids:
                # 检查邻居ID是否存在于索引映射中 (理论上应该存在)
                if neighbor_id in entity_id_to_index:
                    neighbor_index = entity_id_to_index[neighbor_id]
                    score = predicted_scores[neighbor_index] # 获取该邻居的模型预测得分

                    # 如果得分更高，更新最佳选择
                    if score > best_score:
                        # 确保邻居ID确实存在于 all_entities_map 中
                        if neighbor_id in all_entities_map:
                            best_score = score
                            next_entity = all_entities_map[neighbor_id]

        if next_entity is None:
            # 如果所有相邻实体都已被访问或得分极低，则无法继续
            return None # 无法继续，返回None

        path.append(next_entity)
        visited_entity_ids.add(next_entity.id) # 标记为已访问
        current_entity = next_entity # 更新当前实体

    # 循环结束后，检查是否成功到达出口
    if current_entity.entity_type == EntityType.Exit:
        return path # 成功到达出口，返回完整路径
    else:
        return None # 未到达出口，返回None

# 用于动态规划的路径生成函数
def generate_path_with_dynamic_indices(model: BPNeuralNetworkModel,
                                       start_entity: EvacuationEntity,
                                       emergency_indices: Dict[int, float], # 使用紧急指数字典
                                       all_entities: List[EvacuationEntity],
                                       entity_id_to_index: Dict[int, int],
                                       index_to_entity_id: Dict[int, int],
                                       all_entities_map: Dict[int, EvacuationEntity], # 传入预先计算好的map
                                       max_length: int = 100) -> Optional[List[EvacuationEntity]]:
    """
    (此函数用于动态规划过程，使用连续的紧急指数)
    使用训练好的模型和当前的动态紧急指数生成一条路径。
    :param model: 训练好的BP神经网络模型
    :param start_entity: 路径的起始实体
    :param emergency_indices: 包含实体ID到其当前紧急指数（连续值）的字典
    :param all_entities: 所有实体对象的列表
    :param entity_id_to_index: 实体ID到其在 all_entities 列表中索引的映射
    :param index_to_entity_id: 索引到实体ID的映射
    :param all_entities_map: 实体ID到实体对象的映射
    :param max_length: 生成路径的最大长度限制
    :return: 生成的路径（实体列表），如果无法生成或未找到出口则返回 None
    """
    path = [start_entity]
    visited_entity_ids = {start_entity.id}
    current_entity = start_entity
    N = len(all_entities)

    while current_entity.entity_type != EntityType.Exit and len(path) < max_length:
        input_vector = np.zeros(2 * N + 1)
        if current_entity.id not in entity_id_to_index:
            return None
        current_index = entity_id_to_index[current_entity.id]
        input_vector[current_index] = 1

        # 设置紧急位置 (使用连续的紧急指数)
        for entity_id, emer_index_value in emergency_indices.items():
            if entity_id in entity_id_to_index:
                emergency_vector_index = entity_id_to_index[entity_id]
                input_vector[N + emergency_vector_index] = min(emer_index_value, 1.0) # 简单地将指数限制在[0, 1]

        input_vector[2 * N] = 1 # 偏置项

        predicted_scores = model.predict(np.array([input_vector]))[0]

        next_entity = None
        best_score = -np.inf

        # 只考虑相邻且未访问的实体
        if hasattr(current_entity, 'connected_entity_ids') and current_entity.connected_entity_ids:
            candidate_neighbor_ids = current_entity.connected_entity_ids
        else:
            return None # 没有邻居无法继续

        for neighbor_id in candidate_neighbor_ids:
            if neighbor_id not in visited_entity_ids:
                if neighbor_id in entity_id_to_index:
                    neighbor_index = entity_id_to_index[neighbor_id]
                    score = predicted_scores[neighbor_index]
                    if score > best_score:
                        if neighbor_id in all_entities_map:
                            best_score = score
                            next_entity = all_entities_map[neighbor_id]

        if next_entity is None:
            return None

        path.append(next_entity)
        visited_entity_ids.add(next_entity.id)
        current_entity = next_entity

    if current_entity.entity_type == EntityType.Exit:
        return path
    else:
        return None

# 动态多路径规划函数 
def plan_dynamic_paths(model: BPNeuralNetworkModel,
                       start_entities: List[EvacuationEntity], # 多个起点
                       initial_emergency_entities: List[EvacuationEntity], # 初始紧急区域
                       all_entities: List[EvacuationEntity],
                       entity_id_to_index: Dict[int, int],
                       index_to_entity_id: Dict[int, int],
                       max_iterations: int = 10, # 最大迭代次数
                       occupants_per_start: int = 1) -> Tuple[Dict[int, Optional[List[EvacuationEntity]]], Dict[int, Optional[List[EvacuationEntity]]]]:
    """
    执行动态多路径规划迭代过程。
    :param model: 训练好的模型
    :param start_entities: 疏散人员的起始实体列表
    :param initial_emergency_entities: 初始的紧急实体列表 (如火源点)
    :param all_entities: 所有实体列表
    :param entity_id_to_index: ID到索引映射
    :param index_to_entity_id: 索引到ID映射
    :param max_iterations: 防止无限循环的最大迭代次数
    :param occupants_per_start: 假设每个起点出发的人数 (用于计算拥堵)
    :return: 一个元组，包含两个字典：(初始路径结果, 最终路径结果)。每个字典键是起始实体的ID，值是规划出的路径（实体列表）或None
    """
    print(f"\n--- 开始动态多路径规划 (起点数: {len(start_entities)}, 最大迭代: {max_iterations}) ---")
    N = len(all_entities)
    all_entities_map = {entity.id: entity for entity in all_entities}

    # 初始化紧急指数 (字典：entity_id -> index_value)
    # 初始紧急区域指数设为 1.0，其他为 0.0
    current_emergency_indices: Dict[int, float] = {
        entity.id: 1.0 for entity in initial_emergency_entities if entity.id in entity_id_to_index
    }
    # 确保所有实体都有一个初始指数（即使是0）
    for entity_id in entity_id_to_index:
        if entity_id not in current_emergency_indices:
            current_emergency_indices[entity_id] = 0.0


    previous_paths: Dict[int, List[int]] = {} # 存储上一轮各起点的路径 (仅ID列表)
    final_paths: Dict[int, Optional[List[EvacuationEntity]]] = {start.id: None for start in start_entities} # 存储最终路径
    initial_paths_first_iter: Dict[int, Optional[List[EvacuationEntity]]] = {} # 新增：存储第一次迭代的结果

    for iteration in range(max_iterations):
        print(f"  动态规划迭代 {iteration + 1}/{max_iterations}...")
        current_paths_this_iter: Dict[int, Optional[List[EvacuationEntity]]] = {} # 存储本轮生成的路径
        next_emergency_indices = copy.deepcopy(current_emergency_indices) # 基于当前指数开始计算下一轮

        # 步骤 1: 独立路径规划 (使用当前的 emergency_indices)
        print(f"    步骤 1: 为 {len(start_entities)} 个起点生成路径...")
        paths_generated_this_iter: List[Tuple[int, List[EvacuationEntity]]] = [] # 存储成功生成的路径 (start_id, path)
        for start_entity in start_entities:
            path = generate_path_with_dynamic_indices(model,
                                                      start_entity,
                                                      current_emergency_indices, # 使用当前迭代的指数
                                                      all_entities,
                                                      entity_id_to_index,
                                                      index_to_entity_id,
                                                      all_entities_map,
                                                      max_length=150) # 可调最大长度
            current_paths_this_iter[start_entity.id] = path
            if path:
                paths_generated_this_iter.append((start_entity.id, path))

        # 新增：保存第一次迭代的结果作为初始路径
        if iteration == 0:
            initial_paths_first_iter = copy.deepcopy(current_paths_this_iter) # 保存第一次迭代的路径

        # 步骤 2: 更新紧急条件 (基于本轮生成的路径)
        print(f"    步骤 2: 根据 {len(paths_generated_this_iter)} 条生成路径更新紧急指数...")
        # 先重置由路径引起的拥堵指数增加量（保留初始紧急区域的基础指数）
        temp_emergency_indices: Dict[int, float] = {
             entity.id: 1.0 for entity in initial_emergency_entities if entity.id in entity_id_to_index
        }
        # 确保所有实体都有一个初始指数（即使是0）
        for entity_id in entity_id_to_index:
            if entity_id not in temp_emergency_indices:
                temp_emergency_indices[entity_id] = 0.0

        # 根据公式 (18) 累加拥堵指数
        for start_id, path in paths_generated_this_iter:
            N_path = occupants_per_start # 每个路径的使用人数
            for entity in path:
                if entity.id in all_entities_map: # 确保实体有效
                    volume = get_entity_volume(entity)
                    if volume > 0: # 避免除以零
                        update = BETA * N_path / volume
                        # 累加到临时指数上 (如果实体已存在则累加，否则新增)
                        current_val = temp_emergency_indices.get(entity.id, 0.0)
                        temp_emergency_indices[entity.id] = min(current_val + update, 1.0) # 简单上限为1

        # 更新下一轮要使用的紧急指数
        next_emergency_indices = temp_emergency_indices

        # 步骤 3: 检查收敛性
        print(f"    步骤 3: 检查路径是否稳定...")
        current_paths_ids: Dict[int, List[int]] = {}
        for start_id, path in current_paths_this_iter.items():
            current_paths_ids[start_id] = [e.id for e in path] if path else []

        if current_paths_ids == previous_paths:
            print(f"  路径在迭代 {iteration + 1} 稳定，动态规划结束。")
            final_paths = current_paths_this_iter # 使用本轮生成的路径作为最终结果
            break # 收敛，退出循环
        else:
            previous_paths = current_paths_ids # 更新上一轮路径记录
            current_emergency_indices = next_emergency_indices # 更新紧急指数以进行下一轮迭代
            if iteration == max_iterations - 1:
                print(f"  达到最大迭代次数 {max_iterations}，规划结束。")
                final_paths = current_paths_this_iter # 使用最后一轮的结果

    return initial_paths_first_iter, final_paths

# 用于并行生成的辅助函数 
def generate_and_score_path_scenario(args: Tuple) -> Optional[Tuple[List[EvacuationEntity], List[EvacuationEntity], float]]:
    """
    为一个随机选择的场景生成、评分并返回路径。设计为可被 multiprocessing 使用。
    :param args: 包含所需参数的元组 (all_entities, model_path, entity_id_to_index, index_to_entity_id, score_threshold, max_path_length) # 修改了第二个参数
    :return: 如果生成了高分路径，则返回 (路径, 紧急实体列表, 分数)，否则返回 None。
    """
    # 加载模型
    all_entities, model_path, entity_id_to_index, index_to_entity_id, score_threshold, max_path_length = args
    try:
        # 在子进程中加载模型
        model = BPNeuralNetworkModel.load_model(model_path)
    except Exception as e:
        print(f"  [子进程 {os.getpid()}] 加载模型 {model_path} 时出错: {e}")
        return None

    # 随机选择一个起始区域
    selected_starts = EvacuationEntity.select_starting_areas(all_entities, num_areas=1)
    if not selected_starts:
        return None
    start_entity = selected_starts[0]

    # 随机选择一些紧急区域
    # 先筛选出除起始点外的所有实体
    potential_emergency_candidates = [e for e in all_entities if e.id != start_entity.id]
    # 从筛选后的候选中选择房间作为紧急区域
    room_candidates = [e for e in potential_emergency_candidates if e.entity_type == EntityType.Room]
    if not room_candidates: # 防止没有可选房间
        return None
    max_possible_emergencies = len(room_candidates)
    num_emergency = random.randint(3, min(6, max_possible_emergencies))
    # 从房间候选中选择紧急区域
    emergency_entities = EvacuationEntity.select_emergency_areas(room_candidates, num_areas=num_emergency)
    if not emergency_entities:
         return None

    # 使用模型生成路径 (现在使用在子进程中加载的 model)
    generated_path = generate_path_with_model(model,
                                              start_entity,
                                              emergency_entities,
                                              all_entities, # 仍然使用 all_entities 进行路径生成
                                              entity_id_to_index,
                                              index_to_entity_id,
                                              max_length=max_path_length)

    # 如果成功生成了路径并评分
    if generated_path:
        score = EvacuationEntity.score_path(generated_path, emergency_entities, all_entities)
        if score >= score_threshold:
            return (generated_path, emergency_entities, score)
    return None

# 主程序入口
if __name__ == "__main__":
    # 检查 PyTorch GPU 可用性
    print("\n--- 检查 PyTorch GPU 可用性 ---")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"检测到 PyTorch CUDA 设备: {torch.cuda.get_device_name(0)}")
        print("PyTorch 将尝试使用 GPU。")
    else:
        device = torch.device("cpu")
        print("未检测到 PyTorch CUDA 设备。PyTorch 将使用 CPU。")

    try:
        # 路径设置和实体加载
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        data_dir = os.path.join(project_root, 'data') # 定义数据目录
        os.makedirs(data_dir, exist_ok=True) # 确保数据目录存在
        data_file_path = os.path.join(data_dir, 'building_layout.xlsx')
        model_dir = os.path.join(project_root, 'models')
        os.makedirs(model_dir, exist_ok=True)
        model_save_path = os.path.join(model_dir, 'final_trained_model.pth') # 修改模型保存路径
        initial_data_save_path = os.path.join(data_dir, 'initial_training_data.csv') # 初始训练数据保存路径

        print(f"尝试从以下路径加载实体数据: {data_file_path}")
        all_entities = EvacuationEntity.from_xlsx(data_file_path)
        print(f"成功加载 {len(all_entities)} 个实体。")
        if not all_entities:
            print("错误：加载的实体列表为空。")
            exit()
        entity_id_to_index: Dict[int, int] = {entity.id: i for i, entity in enumerate(all_entities)}
        index_to_entity_id: Dict[int, int] = {i: entity.id for i, entity in enumerate(all_entities)}
        N_entities = len(all_entities)
    except FileNotFoundError:
        print(f"错误：找不到实体数据文件。请确保 '{data_file_path}' 存在。")
        exit()
    except Exception as e:
        print(f"加载实体数据时出错: {e}")
        exit()

    # --- 1. 生成初始训练数据 ---
    print("\n--- 步骤 1: 生成初始训练数据 ---")
    initial_training_data = []
    generator = None # 初始化 generator
    try:
        generator = PathGenerator(all_entities)
        print("正在生成初始训练路径...")
        initial_training_data = generator.prepare_training_data(num_paths_to_attempt=6000)
    except ImportError:
        print("错误：无法导入 PathGenerator。请确保 path_generation.py 在 src 目录中且无误。")
        exit()
    except Exception as e:
        print(f"使用 PathGenerator 生成初始数据时出错: {e}")
        exit()

    if initial_training_data:
        print(f"成功生成 {len(initial_training_data)} 条初始训练路径。")
        if generator:
            try:
                generator.save_training_data(initial_training_data, initial_data_save_path)
            except Exception as e:
                print(f"保存初始训练数据时出错: {e}")
        else:
            print("警告：PathGenerator 未成功初始化，无法保存初始训练数据。")
    else:
        print("错误：未能生成足够的初始训练数据。请检查 PathGenerator 或增加尝试次数。")
        exit()

    # --- 配置训练参数 ---
    num_iterations = 5
    num_paths_per_iteration = 1000
    score_threshold = 0.6
    hidden_dim_config = 700
    learning_rate_config = 0.0005
    epochs_initial_training = 100 # 初始训练的 epoch 数量
    epochs_per_iteration = 50     # 每次迭代训练的 epoch 数量
    batch_size_config = 32
    max_attempts_multiplier = 10

    # --- 2. 初始模型训练 ---
    print("\n--- 步骤 2: 初始模型训练 ---")
    current_training_data = initial_training_data
    current_model = None
    last_loss = None

    if current_training_data:
        trainer = PathTraining(training_data=current_training_data,
                               all_entities=all_entities,
                               hidden_dim=hidden_dim_config,
                               learning_rate=learning_rate_config) 
        print(f"开始初始训练 (Epochs: {epochs_initial_training}, Batch Size: {batch_size_config})...")
        # 调用 train 方法，它会调用 model.train_model
        history = trainer.train(epochs=epochs_initial_training, batch_size=batch_size_config)
        current_model = trainer.model

        if history and 'loss' in history and history['loss']:
            last_loss = history['loss'][-1]
            print(f"初始训练完成 {epochs_initial_training} 个周期。最终训练损失: {last_loss:.6f}")
            if 'val_loss' in history and history['val_loss']:
                last_val_loss = history['val_loss'][-1]
                print(f"最终验证损失: {last_val_loss:.6f}")

    if not current_model:
        print("错误：初始模型训练失败。")
        exit()

    # --- 3. 自学习迭代过程 ---
    print("\n--- 步骤 3: 开始自学习迭代 ---")
    for iteration in range(num_iterations):
        print(f"\n--- 开始迭代 {iteration + 1}/{num_iterations} ---")

        # 3a. 使用当前模型生成新的路径数据 (并行化)
        print(f"  阶段 A: 使用当前模型并行生成新路径...")
        newly_generated_paths_data = []
        generated_count = 0
        total_attempts_limit = num_paths_per_iteration * max_attempts_multiplier
        attempted_scenarios = 0

        # 创建一个临时文件来保存模型状态
        temp_model_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pth", dir=model_dir)
        temp_model_path = temp_model_file.name
        temp_model_file.close() # 关闭文件句柄，以便 save_model 可以写入
        try:
            current_model.save_model(temp_model_path)
            print(f"  临时模型已保存到: {temp_model_path}")

            # 将模型路径而不是模型对象传递给子进程
            generation_args = (all_entities, temp_model_path, entity_id_to_index, index_to_entity_id, score_threshold, 150)
        except Exception as e:
            print(f"  保存临时模型时出错: {e}")
            try: # 尝试清理临时文件
                os.remove(temp_model_path)
            except OSError:
                pass
            continue # 跳到下一次迭代

        try:
            num_worker_processes = max(1, os.cpu_count() // 2)
            with multiprocessing.Pool(processes=num_worker_processes) as pool:
                print(f"  使用 {pool._processes} 个进程并行生成路径...")
                # 使用 imap_unordered 以便更快地处理结果
                results_iterator = pool.imap_unordered(generate_and_score_path_scenario, [generation_args] * total_attempts_limit)
                for result in results_iterator:
                    attempted_scenarios += 1
                    if result:
                        newly_generated_paths_data.append(result)
                        generated_count += 1
                        if generated_count % 20 == 0:
                             print(f"    已生成 {generated_count}/{num_paths_per_iteration} 条高分路径...")
                        if generated_count >= num_paths_per_iteration:
                            print(f"    已达到目标 {num_paths_per_iteration} 条高分路径，停止生成。")
                            pool.terminate() # 提前终止其他进程
                            break # 退出循环
                    # 检查是否已达到尝试上限，即使未达到目标数量
                    if attempted_scenarios >= total_attempts_limit and generated_count < num_paths_per_iteration:
                        print(f"    已达到尝试上限 {total_attempts_limit} 次，但只生成了 {generated_count} 条高分路径。")
                        pool.terminate() # 提前终止其他进程
                        break # 退出循环

            print(f"  并行生成结束。总尝试次数: {attempted_scenarios}")
        except Exception as e:
            print(f"  并行生成路径时出错: {e}")
        finally:
            try:
                os.remove(temp_model_path)
                print(f"  临时模型文件已删除: {temp_model_path}")
            except OSError as e:
                print(f"  删除临时模型文件时出错: {e}")

        print(f"  本轮成功生成并筛选出 {len(newly_generated_paths_data)} 条高分路径 (阈值 > {score_threshold})。")

        # 3b. 将新生成的高分路径数据添加到训练数据集中
        if newly_generated_paths_data:
            current_training_data.extend(newly_generated_paths_data)
            print(f"  训练数据集大小更新为: {len(current_training_data)} 条路径。")
        else:
            print("  警告：本轮未能生成新的高分路径，训练数据未增加。")

        # 3c. 使用更新后的数据重新训练模型
        print(f"  阶段 B: 使用 {len(current_training_data)} 条路径数据重新训练模型...")
        if not current_training_data:
            print("  错误：训练数据为空，无法继续迭代。")
            break

        current_lr = learning_rate_config
        trainer = PathTraining(training_data=current_training_data,
                               all_entities=all_entities,
                               hidden_dim=hidden_dim_config,
                               learning_rate=current_lr) # 不再传递 weight_decay

        if current_model:
             current_weights_state_dict = current_model.get_weights()
             if current_weights_state_dict:
                 trainer.model.set_weights(current_weights_state_dict)
                 print("  已加载上一轮模型权重以继续训练。")
             else:
                 print("  警告：未能获取上一轮模型权重，将从头开始训练本轮。")
        else:
             print("  警告：无法加载上一轮模型权重（模型不存在），将从头开始训练。")

        print(f"  开始重新训练 (Epochs: {epochs_per_iteration}, Batch Size: {batch_size_config}, LR: {current_lr})...")
        # 调用 train 方法进行再训练
        history = trainer.train(epochs=epochs_per_iteration, batch_size=batch_size_config)
        current_model = trainer.model

        if history and 'loss' in history and history['loss']:
            last_loss = history['loss'][-1]
            print(f"  迭代 {iteration + 1} 训练完成 {epochs_per_iteration} 个周期。最终训练损失: {last_loss:.6f}")
            if 'val_loss' in history and history['val_loss']:
                last_val_loss = history['val_loss'][-1]
                print(f"  最终验证损失: {last_val_loss:.6f}")

    # --- 4. 最终模型评估与保存 ---
    print("\n--- 步骤 4: 自学习迭代完成与最终评估 ---")
    if current_model:
        print("最终模型训练完成。")
        if isinstance(last_loss, float):
             print(f"最后一次训练的最终损失: {last_loss:.6f}")
        else:
             print("未能记录最后一次训练的损失值。")

        try:
            current_model.save_model(model_save_path)
            print(f"最终模型状态字典已保存到 {model_save_path}")
        except Exception as e:
            print(f"保存最终模型时出错: {e}")

    else:
        print("训练过程未能生成有效模型，无法进行评估或保存。")