# 导入所需的库
from typing import List, Tuple
import pandas as pd
import numpy as np
import random # 新增：导入 random 模块
# 从 evacuation_entity 模块导入 EvacuationEntity 类
from evacuation_entity import EvacuationEntity, EntityType # 确保 EntityType 也被导入

# 定义路径生成器类
class PathGenerator:
    # 初始化方法，接收一个 EvacuationEntity 列表作为参数
    def __init__(self, entities: List[EvacuationEntity]):
        # 存储实体列表
        self.entities = entities
        # 存储实体 ID 到实体的映射，方便查找
        self.entity_map = {entity.id: entity for entity in entities}

    # 生成路径的方法
    # start_entity: 起始实体
    # max_length: 路径最大长度，默认为 100
    # nearest_exit_p: 选择最近出口的概率，默认为 0.5
    # 返回值：一个包含路径、对应分数和生成该路径时使用的紧急区域列表的元组列表
    def generate_paths(self, start_entity: EvacuationEntity, max_length: int = 100, nearest_exit_p: float = 0.5) -> List[Tuple[List[EvacuationEntity], List[EvacuationEntity], float]]:
        # 初始化存储路径、分数和紧急区域的列表
        paths_with_scores_and_emergencies = []
        # 选择紧急区域，这里示例选择 5 个
        # 注意：每次调用 generate_paths 都会重新选择紧急区域
        emergency_entities = self.select_emergency_areas(num_areas=5)

        # 调用 EvacuationEntity 的静态方法生成一条路径
        # 需要传入紧急区域列表给 generate_path
        path = EvacuationEntity.generate_path(start_entity, self.entities, emergency_entities, max_length, nearest_exit_p)

        # 检查路径是否成功生成
        if path:
            # 调用 EvacuationEntity 的静态方法为生成的路径打分
            # 需要传入紧急区域列表给 score_path
            score = EvacuationEntity.score_path(path, emergency_entities, self.entities)

            # 如果路径有效且分数大于 0.30，则将其添加到列表中
            if score > 0.3:
                paths_with_scores_and_emergencies.append((path, emergency_entities, score))

        # 返回包含路径、分数和紧急区域的列表
        return paths_with_scores_and_emergencies

    # 选择紧急区域的方法
    # num_areas: 需要选择的紧急区域数量，默认为 5
    # 返回值：一个包含紧急区域实体的列表
    def select_emergency_areas(self, num_areas: int = 5) -> List[EvacuationEntity]:
        # 调用 EvacuationEntity 的静态方法来选择紧急区域
        return EvacuationEntity.select_emergency_areas(self.entities, num_areas)

    # 准备训练数据的方法
    # num_paths_to_attempt: 尝试生成路径的次数，默认为 1000
    # 返回值：一个包含路径、对应分数和紧急区域列表的元组列表，作为训练数据
    # 注意：最终得到的训练数据量可能小于 num_paths_to_attempt，因为并非每次尝试都能生成有效路径
    def prepare_training_data(self, num_paths_to_attempt: int = 1000, max_emergency_areas: int = 6) -> List[Tuple[List[EvacuationEntity], List[EvacuationEntity], float]]:
        """
        准备训练数据，每次尝试随机选择紧急区域的数量和位置。

        Args:
            num_paths_to_attempt: 尝试生成有效路径的目标数量。
            max_emergency_areas: 每次尝试时随机选择的最大紧急区域数量。

        Returns:
            List[Tuple[List[EvacuationEntity], List[EvacuationEntity], float]]: 训练数据列表。
        """
        # 初始化训练数据列表
        training_data = []
        # 循环尝试生成指定数量的路径
        generated_count = 0
        attempts = 0
        max_attempts = num_paths_to_attempt * 10 # 增加尝试次数上限，防止死循环

        # 计算可作为紧急区域的最大实体数（例如，所有房间）
        potential_emergency_candidates = [e for e in self.entities if e.entity_type == EntityType.Room]
        max_possible_emergencies = len(potential_emergency_candidates)

        if max_possible_emergencies == 0:
            print("警告：数据中没有房间类型的实体，无法选择紧急区域。")
            return [] # 如果没有房间，无法生成带紧急区域的训练数据

        # 确定实际的最大紧急区域数，不超过总房间数和设定的上限
        actual_max_emergency = min(max_emergency_areas, max_possible_emergencies)
        if actual_max_emergency < 1:
             print("警告：可选择的最大紧急区域数小于1。")
             actual_max_emergency = 1 # 至少尝试选择1个

        while generated_count < num_paths_to_attempt and attempts < max_attempts:
            attempts += 1

            # 1. 随机确定本次迭代的紧急区域数量 (至少为1)
            num_emergency = random.randint(3, actual_max_emergency)

            # 2. 选择本次迭代的紧急区域
            # select_emergency_areas 内部会处理数量超过可选实体的情况
            current_emergency_entities = self.select_emergency_areas(num_areas=num_emergency)
            if not current_emergency_entities:
                 # 如果 select_emergency_areas 返回空（理论上在有房间的情况下不应发生），跳过
                 print(f"警告: 在第 {attempts} 次尝试中未能选择紧急区域。")
                 continue

            # 3. 使用 EvacuationEntity 的方法选择一个非紧急的起始区域
            # 传入当前选择的紧急区域列表
            selected_starts = EvacuationEntity.select_starting_areas(
                self.entities,
                num_areas=1,
                emergency_entities=current_emergency_entities # 传递紧急区域
            )

            # 如果没有可选的起始区域（例如所有室内区域都是紧急区域），则跳过本次迭代
            if not selected_starts:
                # print(f"警告: 在第 {attempts} 次尝试中未能找到非紧急的起始区域。") # 减少打印频率
                continue

            start_entity = selected_starts[0] # 获取选中的起始实体

            # 4. 为选定的起始实体生成路径、分数和紧急区域信息
            path = EvacuationEntity.generate_path(
                start_entity,
                self.entities,
                current_emergency_entities, # 使用本次迭代选定的紧急区域
                max_length=150,
                nearest_exit_p=0.5
            )

            if path:
                score = EvacuationEntity.score_path(
                    path,
                    current_emergency_entities, # 使用相同的紧急区域进行评分
                    self.entities
                )
                # 保持分数阈值筛选
                if score > 0.3:
                    training_data.append((path, current_emergency_entities, score))
                    generated_count += 1 # 增加成功生成的计数
            # --- 修改方案结束 ---

        if attempts == max_attempts and generated_count < num_paths_to_attempt:
             print(f"达到最大尝试次数 {max_attempts}，生成了 {len(training_data)} 条有效路径。")
        else:
             print(f"尝试 {attempts} 次生成路径，目标生成 {num_paths_to_attempt} 条，实际成功生成 {len(training_data)} 条有效路径用于训练。")
        # 返回准备好的训练数据
        return training_data

    # 保存训练数据的方法
    # training_data: 包含路径、分数和紧急区域列表的训练数据列表
    # file_path: 保存数据的文件路径
    def save_training_data(self, training_data: List[Tuple[List[EvacuationEntity], List[EvacuationEntity], float]], file_path: str):
        # 初始化用于保存的数据列表
        data_to_save = []
        # 遍历训练数据中的每条路径、分数和紧急区域列表
        for path, emergency_entities , score in training_data :
            # 提取路径中每个实体的 ID
            path_ids = [entity.id for entity in path]
            # 提取生成该路径时使用的紧急区域实体的 ID
            emergency_ids = [entity.id for entity in emergency_entities]
            # 将路径 ID 列表、分数和紧急区域 ID 列表作为一个元组添加到待保存列表中
            data_to_save.append((path_ids, emergency_ids, score))

        # 使用 pandas 创建 DataFrame
        # 添加了 'emergency_ids' 列
        df = pd.DataFrame(data_to_save, columns=['path', 'emergency_ids', 'score'])
        # 将 DataFrame 保存为 CSV 文件，不包含索引
        df.to_csv(file_path, index=False)
        print(f"训练数据已保存到 {file_path}")
