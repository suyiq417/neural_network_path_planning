from enum import Enum
from typing import List, Tuple, Optional, Dict, Any
import pandas as pd
import numpy as np
import time

class EntityType(Enum):
    """
    枚举类型，表示建筑物中的实体类型
    Door: 房间门
    Exit: 建筑物出口
    Corridor: 楼道
    Stair: 楼梯口
    Room: 室内
    """
    Door = 1  # 房间门 
    Exit = 2  # 建筑物出口
    Corridor = 3  # 楼道
    Stair = 4  # 楼梯口
    Room = 5  # 室内

class EvacuationEntity:
    """
    表示建筑物中的实体，如房间、楼梯口、出口等
    """
    # 初始化实体
    def __init__(
        self,
        id: int,  # 识别号
        name: str,  # 实体名称
        area: float,  # 面积
        center: Tuple[float, float],  # 实体中心坐标
        entity_type: EntityType,  # 实体类型
        connected_entity_ids: List[int] = None,  # 连接的实体ID列表
        current_occupancy: int = 0,  # 当前实体内人数
        max_capacity: int = 0,  # 实体最大容纳人数
        safety_level: int = 0,  # 实体安全等级
        floor: int = 0  # 楼层
    ):
        self.id = id
        self.name = name
        self.area = area
        self.center = center
        self.entity_type = entity_type
        self.connected_entity_ids = connected_entity_ids or []
        self.current_occupancy = current_occupancy
        self.max_capacity = max_capacity
        self.safety_level = safety_level
        self.floor = floor
        # 额外的属性，在子类中可能会用到
        self.additional_properties = {}
    
    # 从Excel文件中读取数据构建疏散实体列表
    @classmethod
    def from_xlsx(cls, file_path: str) -> List['EvacuationEntity']:
        """
        从Excel文件中读取数据构建疏散实体列表
        
        Args:
            file_path: Excel文件路径
        
        Returns:
            List[EvacuationEntity]: 疏散实体列表
        """
        # 读取Excel文件
        df = pd.read_excel(file_path)
        
        entities = []
        for index, row in df.iterrows():
            # 根据pointtype确定实体类型
            point_type = str(row.get('pointtype', '')).lower() if not pd.isna(row.get('pointtype', '')) else ''
            
            if '房间门' in point_type:
                entity_type = EntityType.Door
            elif '建筑物出口' in point_type:
                entity_type = EntityType.Exit
            elif '楼道' in point_type:
                entity_type = EntityType.Corridor
            elif '楼梯口' in point_type:
                entity_type = EntityType.Stair
            elif '室内' in point_type:
                entity_type = EntityType.Room
            else:
                # 默认为室内
                entity_type = EntityType.Room
            
            # 处理ID
            entity_id = index if pd.isna(row.get('id')) else int(row.get('id'))
            
            # 处理面积
            area = 0.0 if pd.isna(row.get('area')) else float(row.get('area'))
            
            # 处理坐标
            x_coord = 0.0 if pd.isna(row.get('x坐标')) else float(row.get('x坐标'))
            y_coord = 0.0 if pd.isna(row.get('y坐标')) else float(row.get('y坐标'))
            
            # 处理楼层
            floor = 0 if pd.isna(row.get('lc')) else int(row.get('lc'))

            # 处理连接的实体ID
            connected_ids = []
            if 'glgx' in row and not pd.isna(row['glgx']):
                glgx_str = str(row['glgx'])
                for id_str in glgx_str.split(','):
                    id_str = id_str.strip()
                    if id_str and id_str.isdigit():
                        connected_ids.append(int(id_str))
            
            # 计算最大容量
            max_capacity = int(area * 4)  # 假设每1平方米可容纳4人
            
            # 创建实体
            entity = cls(
                id=entity_id,
                name=f"{entity_type.name}_{entity_id}",
                area=area,
                center=(x_coord, y_coord),
                entity_type=entity_type,
                connected_entity_ids=connected_ids,
                max_capacity=max_capacity,
                floor=floor
            )
            
            # 存储额外属性
            for key, value in row.items():
                if key not in ['id', 'area', 'x坐标', 'y坐标', 'glgx', 'pointtype', 'lc', 'length'] and not pd.isna(value):
                    entity.additional_properties[key] = value
            
            entities.append(entity)
        
        return entities

    # 计算两个实体之间的欧几里得距离
    @staticmethod
    def calculate_distance(entity1: 'EvacuationEntity', entity2: 'EvacuationEntity', floor_height: float = 3.5) -> float:
        """
        计算两个实体中心点之间的欧几里得距离 (考虑楼层)
        
        Args:
            entity1: 第一个实体
            entity2: 第二个实体
            floor_height: 每层楼的高度 (米)
            
        Returns:
            float: 两个实体间的距离
        """
        x1, y1 = entity1.center
        x2, y2 = entity2.center
        floor_diff = abs(entity1.floor - entity2.floor)
        vertical_distance = floor_diff * floor_height
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + vertical_distance ** 2)

    # 计算两个实体之间的曼哈顿距离
    # 曼哈顿距离 = |x2 - x1| + |y2 - y1| + |floor_diff| * floor_height
    @staticmethod
    def calculate_manhattan_distance(entity1: 'EvacuationEntity', entity2: 'EvacuationEntity', floor_height: float = 3.5) -> float:
        """
        计算两个实体中心点之间的曼哈顿距离 (考虑楼层)
        
        Args:
            entity1: 第一个实体
            entity2: 第二个实体
            floor_height: 每层楼的高度 (米)
        Returns:
            float: 两个实体间的曼哈顿距离
        """
        x1, y1 = entity1.center
        x2, y2 = entity2.center
        floor_diff = abs(entity1.floor - entity2.floor)
        vertical_distance = floor_diff * floor_height
        return abs(x2 - x1) + abs(y2 - y1) + vertical_distance

    # 起始区域选择
    @staticmethod
    def select_starting_areas(entities: List['EvacuationEntity'], 
                              num_areas: int = 3, 
                              emergency_entities: List['EvacuationEntity'] = None) -> List['EvacuationEntity']:
        """
        根据人口容量（面积占比）在非紧急的室内实体中选择起始区域
        
        Args:
            entities: 所有实体的列表
            num_areas: 要选择的起始区域数量
            emergency_entities: 紧急区域实体列表，这些区域将被排除
            
        Returns:
            List[EvacuationEntity]: 选择的起始区域列表
        """
        # 使用基于时间的随机种子
        EvacuationEntity.set_random_seed(None)
            
        if emergency_entities is None:
            emergency_entities = []
        emergency_ids = {e.id for e in emergency_entities}
            
        # 过滤出所有非紧急的室内实体
        potential_start_entities = [
            entity for entity in entities 
            if entity.entity_type == EntityType.Room and entity.id not in emergency_ids
        ]
        
        if not potential_start_entities:
            # 如果没有非紧急的室内区域可选，可以考虑返回空列表或选择其他类型的区域
            # 这里返回空列表
            return []
        
        if len(potential_start_entities) <= num_areas:
            return potential_start_entities
        
        # 计算总面积 (仅限非紧急室内区域)
        total_area = sum(entity.area for entity in potential_start_entities)
        
        # 如果总面积为0，则平均分配概率
        if total_area == 0:
            probabilities = [1.0 / len(potential_start_entities)] * len(potential_start_entities)
        else:
            # 根据面积占比计算每个实体的选择概率
            probabilities = [entity.area / total_area for entity in potential_start_entities]
        
        # 根据概率选择起始区域（不放回抽样）
        selected_indices = np.random.choice(
            range(len(potential_start_entities)), 
            size=min(num_areas, len(potential_start_entities)), 
            replace=False, 
            p=probabilities
        )
        
        return [potential_start_entities[i] for i in selected_indices]

    # 紧急区域选择
    @staticmethod
    def select_emergency_areas(entities: List['EvacuationEntity'], num_areas: int = 2) -> List['EvacuationEntity']:
        """
        在室内实体中随机选择几个作为紧急区域
        
        Args:
            entities: 所有实体的列表
            num_areas: 要选择的紧急区域数量
            
        Returns:
            List[EvacuationEntity]: 选择的紧急区域列表
        """
        # 使用基于时间的随机种子
        EvacuationEntity.set_random_seed(None)
            
        # 过滤出所有室内实体
        room_entities = [entity for entity in entities if entity.entity_type == EntityType.Room]
        
        if not room_entities:
            return []
        
        if len(room_entities) <= num_areas:
            return room_entities
        
        # 随机选择紧急区域
        selected_indices = np.random.choice(len(room_entities), size=num_areas, replace=False)
        return [room_entities[i] for i in selected_indices]
    
    # 选择最近的出口
    @staticmethod
    def select_nearest_exit(current_entity: 'EvacuationEntity', entities: List['EvacuationEntity'], 
                          num_exits: int = 1, p: float = 0.5) -> List['EvacuationEntity']:
        """
        为当前实体选择最近的出口，基于公式：
        W(entity) = {
            p + (1-p)/n, if entity is the nearest one
            (1-p)/n,     Others
        }
        其中0 < p < 1，n是出口的总数量
        
        Args:
            current_entity: 当前实体
            entities: 所有实体的列表
            num_exits: 要选择的出口数量
            p: 最近出口的概率偏好参数(0<p<1)，越大表示越倾向于选择最近出口
            
        Returns:
            List[EvacuationEntity]: 选择的出口列表
        """
        # 使用基于时间的随机种子
        EvacuationEntity.set_random_seed(None)
            
        # 过滤出所有出口实体
        exit_entities = [entity for entity in entities if entity.entity_type == EntityType.Exit]
        
        if not exit_entities:
            return []
        
        if len(exit_entities) <= num_exits:
            return exit_entities
        
        # 计算当前实体到每个出口的距离
        distances = [EvacuationEntity.calculate_distance(current_entity, exit_entity) for exit_entity in exit_entities]
        
        # 找出最近的出口的索引
        nearest_idx = np.argmin(distances)
        n = len(exit_entities)
        
        # 根据公式计算权重
        weights = [(1-p)/n] * n
        weights[nearest_idx] = p + (1-p)/n
        
        # 根据权重选择出口
        selected_indices = np.random.choice(
            range(len(exit_entities)), 
            size=num_exits, 
            replace=False, 
            p=weights
        )
        
        return [exit_entities[i] for i in selected_indices]
    
    # 设置随机数生成器的种子
    @staticmethod
    def set_random_seed(seed: int = None):
        """
        设置随机数生成器的种子，如果不提供种子则使用当前时间
        
        Args:
            seed: 随机种子，如果为None则使用当前时间戳
        """
        if seed is None:
            # 使用当前时间戳作为随机种子
            seed = int(time.time() * 1000) % 10000000
        np.random.seed(seed)

    # 剪枝死胡同规则下的候选实体筛选
    @staticmethod
    def prune_dead_ends(current_entity: 'EvacuationEntity',
                         connected_entities: List['EvacuationEntity'],
                         all_entities: List['EvacuationEntity'],
                         visited_entities: List['EvacuationEntity'],
                         target_entity: Optional['EvacuationEntity'] = None) -> List['EvacuationEntity']:
        """
        排除已访问的实体并剪枝可能导致死胡同的候选实体，但保留目标实体（如出口）
        
        Args:
            current_entity: 当前实体
            connected_entities: 当前实体连接的所有实体列表
            all_entities: 所有实体的列表
            visited_entities: 已访问的实体列表
            target_entity: 目标实体，确保不会被剪枝掉
            
        Returns:
            List[EvacuationEntity]: 剪枝后的候选实体列表
        """
        # 首先排除已访问的实体
        candidate_entities = [entity for entity in connected_entities if entity not in visited_entities]
        
        if not candidate_entities:
            return []
            
        pruned_candidates = [] # 剪枝后的候选实体列表
        uncertain_entities = [] # 不确定的实体列表
        
        # 第一步：根据度数进行初步筛选
        for entity in candidate_entities:
            # 如果是目标出口或其他出口类型，直接添加到候选列表，不进行剪枝
            if entity.entity_type == EntityType.Exit:
                pruned_candidates.append(entity)
                continue
                
            # 计算度数 - 连接实体的数量（不包括已访问的和当前实体）
            connected_entities = [e for e in all_entities if e.id in entity.connected_entity_ids 
                                 and e not in visited_entities and e != current_entity]
            degree = len(connected_entities)
            
            if degree == 0:
                # 该实体没有其他出路，是死胡同，不添加
                continue
            elif degree == 1:
                # 该实体只有一个出路，需要进一步检查
                uncertain_entities.append((entity, connected_entities))
            else:
                # 度数大于1，至少有两条出路，直接添加到候选列表
                pruned_candidates.append(entity)
        
        # 第二步：处理不确定的实体，检查它们是否通向死胡同
        for entity, connected_entities in uncertain_entities:
            next_entity = connected_entities[0]  # 只有一个连接的实体
            
            # 如果下一个实体是出口，当前实体可以添加
            if next_entity.entity_type == EntityType.Exit:
                pruned_candidates.append(entity)
                continue
                
            # 检查下一层实体的度数
            next_connected_entities = [e for e in all_entities if e.id in next_entity.connected_entity_ids 
                                     and e not in visited_entities and e != entity]
            next_degree = len(next_connected_entities)
            
            if next_degree > 0:
                # 下一层实体至少有一条出路，可以添加
                pruned_candidates.append(entity)
            # 否则是死胡同，不添加
        
        return pruned_candidates

    # 选择下一个实体
    @staticmethod
    def select_next_entity(current_entity: 'EvacuationEntity', 
                          all_entities: List['EvacuationEntity'], 
                          target_entity: Optional['EvacuationEntity'] = None,
                          visited_entities: List['EvacuationEntity'] = None,
                          emergency_entities: List['EvacuationEntity'] = None, # 添加紧急实体列表参数
                          stair_weight: float = 30,
                          distance_weight: float = 60.0,
                          base_weight: float = 1.0,
                          safety_penalty: float = -40.0) -> Optional['EvacuationEntity']:
        """
        从当前实体选择下一个要访问的实体 (考虑紧急实体惩罚)
        
        Args:
            current_entity: 当前实体
            all_entities: 所有实体的列表
            target_entity: 目标实体（用于计算方向向量）
            visited_entities: 已访问的实体列表
            emergency_entities: 紧急区域实体列表
            stair_weight: 楼梯权重系数
            distance_weight: 距离权重系数
            base_weight: 基础权重系数
            safety_penalty: 紧急实体的惩罚值
            
        Returns:
            Optional[EvacuationEntity]: 选择的下一个实体，如果没有可选实体则返回None
        """
        # 使用基于时间的随机种子
        EvacuationEntity.set_random_seed(None)
            
        if visited_entities is None:
            visited_entities = []
        if emergency_entities is None:
            emergency_entities = [] # 初始化为空列表以防万一
            
        # 创建紧急实体ID集合以便快速查找
        emergency_ids = {e.id for e in emergency_entities}
            
        # 获取当前实体连接的所有实体ID
        connected_ids = current_entity.connected_entity_ids
        if not connected_ids:
            return None
            
        # 找出所有连接的实体
        connected_entities = [entity for entity in all_entities if entity.id in connected_ids]
        
        # 直接应用剪枝策略，同时处理已访问实体排除和死胡同剪枝
        candidate_entities = EvacuationEntity.prune_dead_ends(
            current_entity,
            connected_entities,
            all_entities,
            visited_entities,
            target_entity
        )
        
        if not candidate_entities:
            return None
            
        # 只有一个候选实体，直接返回
        if len(candidate_entities) == 1:
            return candidate_entities[0]
        
        # 计算从当前实体到目标实体的方向向量
        target_vector = None
        if target_entity:
            current_x, current_y = current_entity.center
            target_x, target_y = target_entity.center
            target_vector = (target_x - current_x, target_y - current_y)
            # 标准化向量
            target_vector_length = np.sqrt(target_vector[0]**2 + target_vector[1]**2)
            if target_vector_length > 0:
                target_vector = (target_vector[0]/target_vector_length, target_vector[1]/target_vector_length)
            
        # 获取目标楼层（如果目标实体存在）
        target_floor = target_entity.floor if target_entity else None
        
        # 判断当前是否已经在目标楼层
        same_floor_as_target = (target_floor is not None and current_entity.floor == target_floor)
        
        # 计算每个候选实体的权重
        weights = []
        for entity in candidate_entities:
            # 初始化基础权重和楼梯权重
            base_component = base_weight
            stair_component = 0
            safety_component = 0.0 # 初始化安全分量
            
            # 处理楼梯特殊情况
            if entity.entity_type == EntityType.Stair:
                if same_floor_as_target:
                    base_component = 0.1  # 已在目标楼层时，楼梯的基础权重很低
                    stair_component = 0.1  # 已在目标楼层时，楼梯的楼梯权重很低
                elif target_floor is not None:
                    current_to_target = target_floor - current_entity.floor
                    entity_to_current = entity.floor - current_entity.floor
                    
                    if (current_to_target > 0 and entity_to_current > 0) or (current_to_target < 0 and entity_to_current < 0):
                        base_component = 100.0  # 正确方向楼梯的基础权重
                        stair_component = 100.0  # 正确方向楼梯的楼梯权重
                    else:
                        base_component = 0.1
                        stair_component = 0.1
                else:
                    stair_component = stair_weight
            
            # 计算距离权重 C_i^d (使用余弦夹角)
            distance_component = 0
            if target_entity and target_vector:
                current_x, current_y = current_entity.center
                candidate_x, candidate_y = entity.center
                candidate_vector = (candidate_x - current_x, candidate_y - current_y)
                candidate_vector_length = np.sqrt(candidate_vector[0]**2 + candidate_vector[1]**2)
                if candidate_vector_length > 0:
                    candidate_vector = (candidate_vector[0]/candidate_vector_length, candidate_vector[1]/candidate_vector_length)
                    cos_angle = target_vector[0]*candidate_vector[0] + target_vector[1]*candidate_vector[1]
                    distance_component = distance_weight * cos_angle
            
            # 计算安全权重 C_i^l
            if entity.id in emergency_ids:
                safety_component = safety_penalty # 应用惩罚值

            # 综合权重 W(can_i) = C_i^0 + C_i^s + C_i^d + C_i^l
            total_weight = base_component + stair_component + distance_component + safety_component
            weights.append(max(0.001, total_weight))  # 确保权重为正
            
        # 归一化权重
        total_weight = sum(weights)
        if total_weight == 0:
            probabilities = [1.0 / len(weights)] * len(weights)
        else:
            probabilities = [w / total_weight for w in weights]
            
        # 根据权重进行概率抽样
        selected_index = np.random.choice(len(candidate_entities), p=probabilities)
        return candidate_entities[selected_index]

    # 生成疏散路径
    @staticmethod
    def generate_path(start_entity: 'EvacuationEntity',
                      all_entities: List['EvacuationEntity'],
                      emergency_entities: List['EvacuationEntity'], # 添加紧急实体列表参数
                      max_length: int = 100,
                      nearest_exit_p: float = 0.5) -> Optional[List['EvacuationEntity']]:
        """
        生成从起点出发的疏散路径，每一步动态选择最近的出口作为引导目标。
        如果路径达到最大长度仍未找到出口，则认为路径生成失败。

        Args:
            start_entity: 起始实体
            all_entities: 所有实体的列表
            emergency_entities: 紧急区域实体列表
            max_length: 路径最大长度，防止无限循环
            nearest_exit_p: select_nearest_exit 的 p 参数

        Returns:
            Optional[List[EvacuationEntity]]: 生成的有效路径列表（到达出口），
                                             如果达到最大长度仍未找到出口或无法继续前进，则返回 None。
        """
        path = [start_entity]
        visited_entities = [start_entity]  # 使用列表记录访问顺序
        current_entity = start_entity

        while current_entity.entity_type != EntityType.Exit and len(path) < max_length:
            # 1. 为当前实体选择一个目标出口 (动态)
            target_exits = EvacuationEntity.select_nearest_exit(
                current_entity,
                all_entities,
                num_exits=1,
                p=nearest_exit_p
            )

            current_target_exit = target_exits[0]

            # 2. 使用动态选择的出口作为目标，选择下一个实体 (传入 emergency_entities)
            next_entity = EvacuationEntity.select_next_entity(
                current_entity,
                all_entities,
                current_target_exit,
                visited_entities,
                emergency_entities=emergency_entities # 传递紧急实体列表
            )

            if next_entity is None:
                return None # 无法找到下一个有效节点，路径失败

            path.append(next_entity)
            visited_entities.append(next_entity)
            current_entity = next_entity

        # 循环结束后检查是否到达出口
        if current_entity.entity_type != EntityType.Exit:
             return None # 未能在最大长度内到达出口，路径失败

        # 成功到达出口
        return path

    # 评估路径得分
    @staticmethod
    def score_path(path: List['EvacuationEntity'],
                   emergency_entities: List['EvacuationEntity'],
                   all_entities: List['EvacuationEntity'],
                   mu1: float = 0.32,
                   mu2: float = 0.004,
                   floor_height: float = 3.5) -> float: # 添加 floor_height 参数
        """
        根据给定的规则评估路径得分
        S = min{1.00，μ1 * (长度得分 + 紧急得分) + μ2 * 加分数和}

        Args:
            path: 要评估的路径 (实体列表)
            emergency_entities: 紧急区域实体列表
            all_entities: 所有实体的列表
            mu1: 长度和紧急得分的权重系数
            mu2: 加分分数的权重系数
            floor_height: 层高，用于距离计算

        Returns:
            float: 路径的最终得分
        """
        if not path or len(path) < 2:
            return 0.0  # 无效路径或只有一个节点的路径得分为0

        start_entity = path[0]
        end_entity = path[-1]

        # 1. 计算路径长度 (欧几里得距离之和，考虑楼层)
        path_length = 0.0
        for i in range(len(path) - 1):
            # 使用更新后的距离计算函数
            path_length += EvacuationEntity.calculate_distance(path[i], path[i+1], floor_height)

        if path_length == 0:
            # 如果起点和终点是同一个实体且路径只有一个节点，长度为0
            # 或者路径只包含重叠的实体，长度也可能为0
             if len(path) == 1 and start_entity == end_entity:
                 return 0.0 # 单节点路径得分为0
             else:
                 # 路径多于一个节点但长度为0，可能表示数据问题或所有点重合
                 # 给予一个极小的长度避免除零，或者返回0分
                 path_length = 1e-6 # 设定一个极小值

        # 2. 计算起点到终点的曼哈顿距离 (考虑楼层)
        # 使用更新后的距离计算函数
        manhattan_dist = EvacuationEntity.calculate_manhattan_distance(start_entity, end_entity, floor_height)

        # 3. 计算长度得分 = 曼哈顿距离 / 路径长度
        length_score = abs(manhattan_dist / path_length)

        # 4. 计算紧急得分
        emergency_score = 1.00
        emergency_ids = {e.id for e in emergency_entities}
        all_entities_map = {e.id: e for e in all_entities}
        path_ids = {e.id for e in path}
        processed_neighbors = set()

        for entity in path:
            if entity.id in emergency_ids:
                emergency_score -= 0.3

            for neighbor_id in entity.connected_entity_ids:
                neighbor_pair = tuple(sorted((entity.id, neighbor_id)))

                if neighbor_id in all_entities_map and \
                   neighbor_id in emergency_ids and \
                   neighbor_id not in path_ids and \
                   neighbor_pair not in processed_neighbors:
                    emergency_score -= 0.1
                    processed_neighbors.add(neighbor_pair)

        # 5. 计算加分分数
        num_emergency_in_path = sum(1 for entity in path if entity.id in emergency_ids)
        bonus_score = len(path) + num_emergency_in_path

        # 6. 计算最终得分 (根据新规则)
        score_term1 = mu1 * (length_score + emergency_score)
        score_term2 = mu2 * bonus_score

        final_score = min(1.00, score_term1 + score_term2)
        # 考虑是否需要将最终得分限制在非负范围内
        # final_score = max(0.0, final_score)
        return final_score
