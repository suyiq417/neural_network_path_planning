from typing import List, Tuple, Dict, Optional, Set
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from evacuation_entity import EvacuationEntity, EntityType
from model import BPNeuralNetworkModel
from training import generate_path_with_model, plan_dynamic_paths, get_entity_volume, BETA
import visualization as viz
import networkx as nx
from collections import Counter, defaultdict
from scipy.interpolate import griddata

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 常量定义
FLOOR_HEIGHT = 3.5
MIN_VOLUME = 2.0

def validate_path_topology(path: List[EvacuationEntity]) -> bool:
    """
    检查路径中的每一步是否对应于有效的实体连接。
    """
    if not path or len(path) < 2:
        return True
    for i in range(len(path) - 1):
        current_entity = path[i]
        next_entity = path[i+1]
        if not hasattr(current_entity, 'connected_entity_ids'):
            return False
        if next_entity.id not in current_entity.connected_entity_ids:
            return False
    return True

def calculate_shortest_paths(all_entities: List[EvacuationEntity]) -> Dict[int, Dict[int, List[int]]]:
    """
    使用Dijkstra算法计算所有实体之间的最短路径
    
    返回: 字典，{起点ID: {终点ID: [路径节点ID列表]}}
    """
    # 构建图
    G = nx.Graph()
    
    # 添加节点
    for entity in all_entities:
        G.add_node(entity.id)
    
    # 添加边
    for entity in all_entities:
        if hasattr(entity, 'connected_entity_ids'):
            for connected_id in entity.connected_entity_ids:
                # 每条边权重为1（简化模型）
                G.add_edge(entity.id, connected_id, weight=1)
    
    # 计算所有节点对之间的最短路径
    shortest_paths = {}
    
    # 找出所有出口
    exit_ids = [e.id for e in all_entities if e.entity_type == EntityType.Exit]
    
    # 计算每个节点到所有出口的最短路径
    for entity in all_entities:
        shortest_paths[entity.id] = {}
        for exit_id in exit_ids:
            if nx.has_path(G, entity.id, exit_id):
                try:
                    path = nx.shortest_path(G, source=entity.id, target=exit_id, weight='weight')
                    shortest_paths[entity.id][exit_id] = path
                except nx.NetworkXNoPath:
                    continue
    
    return shortest_paths

def get_shortest_to_any_exit(shortest_paths: Dict[int, Dict[int, List[int]]], entity_id: int) -> Optional[List[int]]:
    """获取从指定实体到任意出口的最短路径"""
    if entity_id not in shortest_paths:
        return None
    
    paths_to_exits = shortest_paths[entity_id]
    if not paths_to_exits:
        return None
    
    # 返回最短的路径
    return min(paths_to_exits.values(), key=len)

def calculate_entity_congestion(path_dict: Dict[int, List[EvacuationEntity]], 
                               all_entities: List[EvacuationEntity], 
                               occupants_per_path: int = 1) -> Dict[int, float]:
    """
    计算根据当前路径规划导致的各实体拥堵指数
    
    参数:
        path_dict: 路径字典，key是起点ID，value是路径实体列表
        all_entities: 所有实体列表
        occupants_per_path: 每条路径的人数设置
    
    返回:
        字典，key是实体ID，value是拥堵指数 
    """
    congestion_indices = defaultdict(float)
    entity_visit_count = Counter()
    
    # 计算每个实体被访问的次数
    for _, path in path_dict.items():
        if not path:
            continue
        for entity in path:
            entity_visit_count[entity.id] += occupants_per_path
    
    # 转换为拥堵指数
    for entity_id, count in entity_visit_count.items():
        # 找出对应的实体
        entity = next((e for e in all_entities if e.id == entity_id), None)
        if entity:
            volume = get_entity_volume(entity)
            if volume > 0:
                congestion_indices[entity_id] = min(BETA * count / volume, 1.0)
    
    return congestion_indices

def evaluate_path_quality(path: List[EvacuationEntity], 
                          emergency_entities: List[EvacuationEntity],
                          shortest_path_ids: Optional[List[int]] = None) -> Dict:
    """
    评估单条路径的质量
    
    参数:
        path: 路径实体列表
        emergency_entities: 紧急实体列表
        shortest_path_ids: 理论最短路径ID列表（可选）
    
    返回:
        包含各评估指标的字典
    """
    if not path:
        return {
            'valid': False,
            'length': 0,
            'emergency_distance': 0,
            'shortest_path_ratio': None,
            'unnecessary_detour': 0
        }
    
    # 路径有效性检查
    is_valid = validate_path_topology(path)
    
    # 路径长度
    path_length = len(path)
    
    # 计算与紧急区域的距离总和
    emergency_distance = 0
    emergency_ids = {e.id for e in emergency_entities}
    
    for entity in path:
        # 如果实体本身是紧急区域，增加风险值
        if entity.id in emergency_ids:
            emergency_distance -= 10  # 穿过紧急区域有显著负面评分
        else:
            # 检查当前节点与任何紧急区域之间的连接
            connected_to_emergency = False
            if hasattr(entity, 'connected_entity_ids'):
                for connected_id in entity.connected_entity_ids:
                    if connected_id in emergency_ids:
                        connected_to_emergency = True
                        emergency_distance -= 1  # 靠近紧急区域有轻微负面评分
            
            if not connected_to_emergency:
                emergency_distance += 1  # 远离紧急区域有轻微正面评分
    
    # 与最短路径对比
    shortest_path_ratio = None
    unnecessary_detour = 0
    
    if shortest_path_ids:
        shortest_length = len(shortest_path_ids)
        if shortest_length > 0:
            shortest_path_ratio = path_length / shortest_length
            
            # 计算不必要的绕路
            path_ids = [e.id for e in path]
            extra_nodes = set(path_ids) - set(shortest_path_ids)
            unnecessary_detour = len(extra_nodes)
    
    return {
        'valid': is_valid,
        'length': path_length,
        'emergency_distance': emergency_distance,
        'shortest_path_ratio': shortest_path_ratio,
        'unnecessary_detour': unnecessary_detour
    }

def evaluate_dynamic_planning_performance(initial_paths: Dict[int, List[EvacuationEntity]],
                                         final_paths: Dict[int, List[EvacuationEntity]],
                                         shortest_paths: Dict[int, Dict[int, List[int]]],
                                         all_entities: List[EvacuationEntity],
                                         emergency_entities: List[EvacuationEntity],
                                         occupants_per_start: int = 1) -> Dict:
    """
    评估动态规划性能
    """
    all_entities_map = {entity.id: entity for entity in all_entities}
    
    # 计算总体拓扑有效性
    initial_valid_count = 0
    final_valid_count = 0
    
    # 初始路径评估结果
    initial_path_evaluations = {}
    for start_id, path in initial_paths.items():
        if path:
            # 获取从start_id到最近出口的理论最短路径
            shortest_path_ids = None
            if start_id in shortest_paths:
                shortest_path_ids = get_shortest_to_any_exit(shortest_paths, start_id)
            
            # 评估路径质量
            eval_result = evaluate_path_quality(path, emergency_entities, shortest_path_ids)
            initial_path_evaluations[start_id] = eval_result
            
            if eval_result['valid']:
                initial_valid_count += 1
    
    # 最终路径评估结果
    final_path_evaluations = {}
    for start_id, path in final_paths.items():
        if path:
            # 获取从start_id到最近出口的理论最短路径
            shortest_path_ids = None
            if start_id in shortest_paths:
                shortest_path_ids = get_shortest_to_any_exit(shortest_paths, start_id)
                
            # 评估路径质量
            eval_result = evaluate_path_quality(path, emergency_entities, shortest_path_ids)
            final_path_evaluations[start_id] = eval_result
            
            if eval_result['valid']:
                final_valid_count += 1
    
    # 计算拥堵分析 - 修改这两行
    initial_congestion = calculate_entity_congestion(initial_paths, all_entities, occupants_per_start)
    final_congestion = calculate_entity_congestion(final_paths, all_entities, occupants_per_start)
    
    # 计算拥堵改善率
    congestion_improvement = {}
    all_entities_with_congestion = set(initial_congestion.keys()) | set(final_congestion.keys())
    
    for entity_id in all_entities_with_congestion:
        initial_value = initial_congestion.get(entity_id, 0.0)
        final_value = final_congestion.get(entity_id, 0.0)
        if initial_value > 0:
            improvement = (initial_value - final_value) / initial_value * 100  # 百分比改善
        else:
            improvement = 0 if final_value == 0 else -100  # 如果初始为0，最终不为0，则为-100%恶化
        congestion_improvement[entity_id] = improvement
    
    # 计算平均路径长度
    initial_avg_length = sum(eval_result['length'] for eval_result in initial_path_evaluations.values() if eval_result['length'] > 0) / max(len([e for e in initial_path_evaluations.values() if e['length'] > 0]), 1)
    final_avg_length = sum(eval_result['length'] for eval_result in final_path_evaluations.values() if eval_result['length'] > 0) / max(len([e for e in final_path_evaluations.values() if e['length'] > 0]), 1)
    
    # 计算路径长度变化率
    length_change_percent = (final_avg_length - initial_avg_length) / initial_avg_length * 100 if initial_avg_length > 0 else 0
    
    # 计算避开紧急区域程度
    initial_avg_emergency_dist = sum(eval_result['emergency_distance'] for eval_result in initial_path_evaluations.values()) / max(len(initial_path_evaluations), 1)
    final_avg_emergency_dist = sum(eval_result['emergency_distance'] for eval_result in final_path_evaluations.values()) / max(len(final_path_evaluations), 1)
    
    # 计算成功率
    initial_success_rate = sum(1 for path in initial_paths.values() if path) / len(initial_paths) * 100 if initial_paths else 0
    final_success_rate = sum(1 for path in final_paths.values() if path) / len(final_paths) * 100 if final_paths else 0
    
    # 计算平均绕路程度
    initial_avg_detour = sum(eval_result.get('unnecessary_detour', 0) for eval_result in initial_path_evaluations.values()) / max(len(initial_path_evaluations), 1)
    final_avg_detour = sum(eval_result.get('unnecessary_detour', 0) for eval_result in final_path_evaluations.values()) / max(len(final_path_evaluations), 1)
    
    # 汇总结果
    return {
        'initial_valid_ratio': initial_valid_count / max(len(initial_paths), 1) * 100,
        'final_valid_ratio': final_valid_count / max(len(final_paths), 1) * 100,
        'initial_avg_length': initial_avg_length,
        'final_avg_length': final_avg_length,
        'length_change_percent': length_change_percent,
        'initial_success_rate': initial_success_rate,
        'final_success_rate': final_success_rate,
        'initial_avg_emergency_dist': initial_avg_emergency_dist,
        'final_avg_emergency_dist': final_avg_emergency_dist,
        'emergency_avoidance_improvement': final_avg_emergency_dist - initial_avg_emergency_dist,
        'initial_avg_detour': initial_avg_detour,
        'final_avg_detour': final_avg_detour,
        'detour_change': final_avg_detour - initial_avg_detour,
        'initial_congestion': initial_congestion,
        'final_congestion': final_congestion,
        'congestion_improvement': congestion_improvement,
        'initial_path_evaluations': initial_path_evaluations,
        'final_path_evaluations': final_path_evaluations
    }

def visualize_evaluation_results(evaluation_results: Dict, all_entities: List[EvacuationEntity], model_dir: str):
    """
    可视化评估结果
    
    参数:
        evaluation_results: 评估指标字典
        all_entities: 所有实体列表
        model_dir: 保存可视化结果的目录
    """
    # 1. 创建路径有效性和成功率的柱状图
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = ['拓扑有效率 (%)', '路径成功率 (%)']
    initial_values = [evaluation_results['initial_valid_ratio'], evaluation_results['initial_success_rate']]
    final_values = [evaluation_results['final_valid_ratio'], evaluation_results['final_success_rate']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax.bar(x - width/2, initial_values, width, label='初始路径')
    ax.bar(x + width/2, final_values, width, label='最终动态路径')
    
    ax.set_title('路径有效性和成功率')
    ax.set_ylabel('百分比 (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    # 添加数值标签
    for i, v in enumerate(initial_values):
        ax.text(i - width/2, v + 1, f'{v:.1f}%', ha='center')
    for i, v in enumerate(final_values):
        ax.text(i + width/2, v + 1, f'{v:.1f}%', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'path_validity_success_rate.png'), dpi=300)
    plt.close(fig)
    
    # 2. 创建路径长度和绕路程度的柱状图
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = ['平均路径长度', '平均绕路程度']
    initial_values = [evaluation_results['initial_avg_length'], evaluation_results['initial_avg_detour']]
    final_values = [evaluation_results['final_avg_length'], evaluation_results['final_avg_detour']]
    
    x = np.arange(len(metrics))
    
    ax.bar(x - width/2, initial_values, width, label='初始路径')
    ax.bar(x + width/2, final_values, width, label='最终动态路径')
    
    ax.set_title('路径长度和绕路分析')
    ax.set_ylabel('节点数量')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    # 添加百分比变化标签
    ax.text(0, max(initial_values[0], final_values[0]) + 0.5, 
            f'变化: {evaluation_results["length_change_percent"]:.1f}%', ha='center')
    ax.text(1, max(initial_values[1], final_values[1]) + 0.5, 
            f'变化: {evaluation_results["detour_change"]:.1f}', ha='center')
    
    # 添加数值标签
    for i, v in enumerate(initial_values):
        ax.text(i - width/2, v + 0.2, f'{v:.1f}', ha='center')
    for i, v in enumerate(final_values):
        ax.text(i + width/2, v + 0.2, f'{v:.1f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'path_length_detour_analysis.png'), dpi=300)
    plt.close(fig)
    
    # 3. 创建拥堵改善热力图
    # 首先确保有一个实体ID到其在平面上坐标的映射
    entity_positions = {}
    entity_floors = {}
    for entity in all_entities:
        entity_positions[entity.id] = entity.center
        entity_floors[entity.id] = entity.floor
    
    # 为每个楼层创建一个热力图
    floors = sorted(set(entity_floors.values()))
    
    for floor in floors:
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 筛选当前楼层的实体
        floor_entities = [entity for entity in all_entities if entity.floor == floor]
        
        # 获取该楼层内最大和最小的xy值以设置合适的图表范围
        x_vals = [entity.center[0] for entity in floor_entities]
        y_vals = [entity.center[1] for entity in floor_entities]
        
        if not x_vals or not y_vals:
            continue  # 跳过没有实体的楼层
            
        x_min, x_max = min(x_vals), max(x_vals)
        y_min, y_max = min(y_vals), max(y_vals)
        
        # 添加余量
        margin = max((x_max - x_min) * 0.1, (y_max - y_min) * 0.1, 1)
        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)
        
        # 绘制楼层布局基础图
        for entity in floor_entities:
            style = viz.ENTITY_STYLE.get(entity.entity_type, viz.DEFAULT_STYLE)
            ax.scatter(entity.center[0], entity.center[1], 
                      c=style['color'], marker=style['marker'], s=style['size'])
            
            # 绘制连接线
            if hasattr(entity, 'connected_entity_ids'):
                for connected_id in entity.connected_entity_ids:
                    if connected_id in entity_positions and entity_floors.get(connected_id) == floor:
                        connected_pos = entity_positions[connected_id]
                        ax.plot([entity.center[0], connected_pos[0]], 
                                [entity.center[1], connected_pos[1]], 
                                color='gray', linewidth=0.5, alpha=0.3)
        
        # 创建拥堵改善热力图
        congestion_improvement = evaluation_results['congestion_improvement']
        
        # 筛选当前楼层的拥堵改善数据
        floor_congestion_data = []
        for entity_id, improvement in congestion_improvement.items():
            if entity_id in entity_floors and entity_floors[entity_id] == floor:
                pos = entity_positions.get(entity_id)
                if pos:
                    floor_congestion_data.append((pos[0], pos[1], improvement))
        
        if floor_congestion_data:
            x_cong = [item[0] for item in floor_congestion_data]
            y_cong = [item[1] for item in floor_congestion_data]
            z_cong = [item[2] for item in floor_congestion_data]
            
            # 创建自定义颜色映射：负值(恶化)为红色，正值(改善)为绿色，中间值为黄色
            colors = [(0.8, 0, 0), (1, 1, 0), (0, 0.8, 0)]  # 红-黄-绿
            cmap_name = 'congestion_improvement'
            cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)
            
            # 计算插值网格
            margin = max((x_max - x_min) * 0.05, (y_max - y_min) * 0.05, 0.5)
            xi = np.linspace(x_min - margin, x_max + margin, 100)
            yi = np.linspace(y_min - margin, y_max + margin, 100)
            xi, yi = np.meshgrid(xi, yi)
            
            # 排除z_cong中的NaN或inf值
            valid_indices = [i for i, z in enumerate(z_cong) if np.isfinite(z)]
            if valid_indices:
                x_valid = [x_cong[i] for i in valid_indices]
                y_valid = [y_cong[i] for i in valid_indices]
                z_valid = [z_cong[i] for i in valid_indices]
                
                try:
                    # 使用线性插值方法
                    zi = griddata((x_valid, y_valid), z_valid, (xi, yi), method='linear', fill_value=0)
                    
                    # 绘制热力图
                    im = ax.imshow(zi, origin='lower', extent=(x_min-margin, x_max+margin, y_min-margin, y_max+margin), 
                                  cmap=cm, alpha=0.6, vmin=-100, vmax=100)
                    
                    # 添加颜色条
                    cbar = plt.colorbar(im, ax=ax)
                    cbar.set_label('拥堵改善率 (%)')
                except Exception as e:
                    print(f"插值或热力图生成错误: {e}")
                    # 如果插值失败，使用散点图显示
                    scatter = ax.scatter(x_valid, y_valid, c=z_valid, cmap=cm, 
                                       vmin=-100, vmax=100, s=100, alpha=0.7)
                    cbar = plt.colorbar(scatter, ax=ax)
                    cbar.set_label('拥堵改善率 (%)')
        
        # 添加图例
        legend_elements = []
        entity_types_added = set()
        for entity_type in EntityType:
            if entity_type not in entity_types_added:
                style = viz.ENTITY_STYLE.get(entity_type, viz.DEFAULT_STYLE)
                legend_elements.append(Patch(facecolor=style['color'], 
                                           label=entity_type.name))
                entity_types_added.add(entity_type)
        
        ax.legend(handles=legend_elements, title="实体类型", 
                 loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
        
        ax.set_title(f'第 {floor} 层拥堵改善分析')
        ax.set_xlabel('X坐标')
        ax.set_ylabel('Y坐标')
        ax.grid(True, linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, f'congestion_improvement_floor_{floor}.png'), dpi=300)
        plt.close(fig)
    
    # 4. 创建路径长度分布柱状图
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 收集路径长度数据
    initial_lengths = [eval_result['length'] for eval_result in evaluation_results['initial_path_evaluations'].values() if eval_result['length'] > 0]
    final_lengths = [eval_result['length'] for eval_result in evaluation_results['final_path_evaluations'].values() if eval_result['length'] > 0]
    
    # 设置直方图参数
    bins = range(0, max(max(initial_lengths, default=10), max(final_lengths, default=10)) + 5, 2)
    
    # 绘制路径长度分布直方图
    ax.hist(initial_lengths, bins=bins, alpha=0.5, label='初始路径', color='blue')
    ax.hist(final_lengths, bins=bins, alpha=0.5, label='最终动态路径', color='orange')
    
    ax.set_title('路径长度分布对比')
    ax.set_xlabel('路径长度 (节点数)')
    ax.set_ylabel('路径数量')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'path_length_distribution.png'), dpi=300)
    plt.close(fig)
    
    # 5. 创建综合评估雷达图
    categories = ['拓扑有效率',
                 '路径成功率',
                 '平均路径长度指标', 
                 '紧急区域避开程度', 
                 '平均绕路优化', 
                 '拥堵控制']
    
    # 规范化数据以适应雷达图 (所有指标转为0-100之间)
    # 注意这里用了一些指标转换以统一所有指标为"越高越好"
    
    # 先计算长度索引 - 越短越好，但要反转为指标
    max_path_length = max(evaluation_results['initial_avg_length'], evaluation_results['final_avg_length'])
    initial_length_index = (max_path_length - evaluation_results['initial_avg_length']) / max_path_length * 100
    final_length_index = (max_path_length - evaluation_results['final_avg_length']) / max_path_length * 100
    
    # 拥堵控制指标 - 从拥堵改善率计算
    avg_initial_congestion = sum(evaluation_results['initial_congestion'].values()) / max(len(evaluation_results['initial_congestion']), 1)
    avg_final_congestion = sum(evaluation_results['final_congestion'].values()) / max(len(evaluation_results['final_congestion']), 1)
    
    # 拥堵控制指标 - 值越低越好，换算为0-100的指标
    initial_congestion_index = (1 - avg_initial_congestion) * 100
    final_congestion_index = (1 - avg_final_congestion) * 100
    
    # 紧急区域避开程度 - 值越高越好
    emergency_avoid_max = max(evaluation_results['initial_avg_emergency_dist'], evaluation_results['final_avg_emergency_dist'], 1)
    initial_emergency_index = (evaluation_results['initial_avg_emergency_dist'] / emergency_avoid_max) * 100
    final_emergency_index = (evaluation_results['final_avg_emergency_dist'] / emergency_avoid_max) * 100
    # 对负值进行处理
    if emergency_avoid_max < 0:
        initial_emergency_index = abs(evaluation_results['initial_avg_emergency_dist'] / emergency_avoid_max) * 100
        final_emergency_index = abs(evaluation_results['final_avg_emergency_dist'] / emergency_avoid_max) * 100
    
    # 绕路优化指标 - 绕路越少越好
    detour_max = max(evaluation_results['initial_avg_detour'], evaluation_results['final_avg_detour'], 1)
    initial_detour_index = (1 - evaluation_results['initial_avg_detour'] / detour_max) * 100
    final_detour_index = (1 - evaluation_results['final_avg_detour'] / detour_max) * 100
    
    # 创建雷达图数据
    initial_data = [
        evaluation_results['initial_valid_ratio'],  # 拓扑有效率
        evaluation_results['initial_success_rate'],  # 路径成功率
        initial_length_index,                        # 路径长度指标
        initial_emergency_index,                     # 紧急区域避开程度
        initial_detour_index,                        # 平均绕路优化
        initial_congestion_index                     # 拥堵控制
    ]
    
    final_data = [
        evaluation_results['final_valid_ratio'],     # 拓扑有效率
        evaluation_results['final_success_rate'],    # 路径成功率
        final_length_index,                         # 路径长度指标
        final_emergency_index,                      # 紧急区域避开程度
        final_detour_index,                         # 平均绕路优化
        final_congestion_index                      # 拥堵控制
    ]
    
    # 创建雷达图
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    
    # 调整角度从顶部开始
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # 闭合多边形
    
    # 添加数据
    initial_data += initial_data[:1]  # 闭合初始数据
    final_data += final_data[:1]      # 闭合最终数据
    
    # 绘制雷达图
    ax.plot(angles, initial_data, 'b-', linewidth=1.5, label='初始路径')
    ax.fill(angles, initial_data, 'b', alpha=0.1)
    ax.plot(angles, final_data, 'r-', linewidth=1.5, label='最终动态路径')
    ax.fill(angles, final_data, 'r', alpha=0.1)
    
    # 添加类别标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    
    # 设置雷达图外观
    ax.set_ylim(0, 100)
    ax.set_title('路径规划综合评估指标雷达图', va='bottom')
    ax.grid(True)
    
    # 添加图例
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'comprehensive_radar_chart.png'), dpi=300)
    plt.close(fig)
    
    # 6. 创建评估报告摘要表格
    summary_data = {
        '评估指标': ['拓扑有效率 (%)', '路径成功率 (%)', '平均路径长度', '紧急区域避开指数', 
                   '平均绕路程度', '平均拥堵指数'],
        '初始路径': [
            f"{evaluation_results['initial_valid_ratio']:.1f}%",
            f"{evaluation_results['initial_success_rate']:.1f}%",
            f"{evaluation_results['initial_avg_length']:.2f}",
            f"{evaluation_results['initial_avg_emergency_dist']:.2f}",
            f"{evaluation_results['initial_avg_detour']:.2f}",
            f"{avg_initial_congestion:.3f}"
        ],
        '最终动态路径': [
            f"{evaluation_results['final_valid_ratio']:.1f}%",
            f"{evaluation_results['final_success_rate']:.1f}%",
            f"{evaluation_results['final_avg_length']:.2f}",
            f"{evaluation_results['final_avg_emergency_dist']:.2f}",
            f"{evaluation_results['final_avg_detour']:.2f}",
            f"{avg_final_congestion:.3f}"
        ],
        '变化': [
            f"{evaluation_results['final_valid_ratio'] - evaluation_results['initial_valid_ratio']:.1f}%",
            f"{evaluation_results['final_success_rate'] - evaluation_results['initial_success_rate']:.1f}%",
            f"{evaluation_results['length_change_percent']:.1f}%",
            f"{evaluation_results['emergency_avoidance_improvement']:.2f}",
            f"{evaluation_results['detour_change']:.2f}",
            f"{avg_final_congestion - avg_initial_congestion:.3f}"
        ]
    }
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # 创建颜色映射，根据"变化"值设置单元格颜色
    cell_colors = []
    for i in range(len(summary_data['评估指标'])):
        row_colors = ['w', 'w', 'w', 'w']  # 默认白色
        
        # 获取变化值并解析
        change_text = summary_data['变化'][i]
        try:
            if '%' in change_text:  # 百分比类型的变化
                change_value = float(change_text.strip('%'))
            else:  # 普通数值类型的变化
                change_value = float(change_text)
            
            # 根据变化方向设置颜色
            # 前两行(拓扑有效率, 路径成功率)和第四行(紧急区域避开)为值越高越好
            if i in [0, 1, 3]:
                if change_value > 0:
                    row_colors[3] = '#d5f5d5'  # 浅绿色表示改善
                elif change_value < 0:
                    row_colors[3] = '#f5d5d5'  # 浅红色表示恶化
            # 其余行(路径长度、绕路程度、拥堵指数)为值越低越好
            else:
                if change_value < 0:
                    row_colors[3] = '#d5f5d5'  # 浅绿色表示改善
                elif change_value > 0:
                    row_colors[3] = '#f5d5d5'  # 浅红色表示恶化
        except ValueError:
            pass  # 如果解析失败，保持默认白色
        
        cell_colors.append(row_colors)
    
    table = ax.table(cellText=[
                         summary_data['评估指标'], 
                         summary_data['初始路径'],
                         summary_data['最终动态路径'],
                         summary_data['变化']
                     ],
                     rowLabels=['评估指标', '初始路径', '最终动态路径', '变化'],
                     cellColours=list(map(list, zip(*cell_colors))),  # 转置颜色列表以匹配表格
                     loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    plt.title('路径规划评估摘要', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'evaluation_summary_table.png'), dpi=300)
    plt.close(fig)

def evaluate_model_performance(model, 
                              start_entities,
                              initial_emergency_entities,
                              all_entities,
                              entity_id_to_index,
                              index_to_entity_id,
                              model_dir,
                              max_iterations=50,
                              occupants_setting=1):
    """
    全面评估模型性能并生成评估报告
    """
    print("\n--- 开始全面模型评估 ---")
    start_time = time.time()
    
    # 1. 预计算最短路径
    print("计算理论最短路径...")
    shortest_paths = calculate_shortest_paths(all_entities)
    
    # 2. 执行动态规划
    print(f"执行动态多路径规划 (最大迭代: {max_iterations})...")
    initial_paths, final_paths, *_ = plan_dynamic_paths(
        model=model,
        start_entities=start_entities,
        initial_emergency_entities=initial_emergency_entities,
        all_entities=all_entities,
        entity_id_to_index=entity_id_to_index,
        index_to_entity_id=index_to_entity_id,
        max_iterations=max_iterations,
        occupants_per_start=occupants_setting
    )
    
    # 3. 评估动态规划性能
    print("评估动态规划性能...")
    evaluation_results = evaluate_dynamic_planning_performance(
        initial_paths=initial_paths,
        final_paths=final_paths,
        shortest_paths=shortest_paths,
        all_entities=all_entities,
        emergency_entities=initial_emergency_entities,
        occupants_per_start=occupants_setting
    )
    
    # 4. 可视化评估结果
    print("生成评估可视化...")
    visualize_evaluation_results(evaluation_results, all_entities, model_dir)
    
    # 5. 保存评估结果摘要
    print("保存评估摘要...")
    summary = {
        'initial_valid_ratio': evaluation_results['initial_valid_ratio'],
        'final_valid_ratio': evaluation_results['final_valid_ratio'],
        'initial_success_rate': evaluation_results['initial_success_rate'],
        'final_success_rate': evaluation_results['final_success_rate'],
        'initial_avg_length': evaluation_results['initial_avg_length'],
        'final_avg_length': evaluation_results['final_avg_length'],
        'length_change_percent': evaluation_results['length_change_percent'],
        'emergency_avoidance_improvement': evaluation_results['emergency_avoidance_improvement'],
        'avg_congestion_reduction': sum(evaluation_results['congestion_improvement'].values()) / max(len(evaluation_results['congestion_improvement']), 1)
    }
    
    # 保存评估摘要到CSV
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(model_dir, 'evaluation_summary.csv'), index=False)
    
    end_time = time.time()
    print(f"--- 模型评估完成，用时 {end_time - start_time:.2f} 秒 ---")
    
    return evaluation_results, summary

# 运行测试示例
if __name__ == "__main__":
    print("此模块提供了模型评估功能。请在test_dynamic_planning.py中导入并使用。")