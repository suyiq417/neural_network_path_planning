from typing import List, Tuple, Dict, Optional, Set
import numpy as np
from evacuation_entity import EvacuationEntity, EntityType
from model import BPNeuralNetworkModel
import os
import matplotlib.pyplot as plt
import visualization as viz
from training import generate_path_with_dynamic_indices, plan_dynamic_paths
import model_evaluation

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

BETA = 0.8 # 公式 (18) 中的减少系数
MIN_VOLUME = 2.0 # 为面积为0的房间或非房间实体设置最小体积，防止除零
FLOOR_HEIGHT = 3.5 # 定义楼层高度常量

def get_entity_volume(entity: EvacuationEntity) -> float:
    """
    获取实体的容量/体积。
    """
    if entity.entity_type == EntityType.Room:
        return max(entity.area, MIN_VOLUME)
    else:
        return MIN_VOLUME

def validate_path_topology(path: List[EvacuationEntity]) -> bool:
    """
    检查路径中的每一步是否对应于有效的实体连接。
    每个 EvacuationEntity 实例都有一个 'connected_entity_ids' 属性，包含连接的实体ID列表/集合。
    """
    if not path or len(path) < 2:
        return True # 空路径或单点路径视为拓扑有效
    for i in range(len(path) - 1):
        current_entity = path[i]
        next_entity = path[i+1]
        # 检查 next_entity.id 是否在 current_entity 的连接列表connected_entity_ids中
        if not hasattr(current_entity, 'connected_entity_ids'):
            # 添加调试信息
            print(f"  [调试] 拓扑错误: 实体 {current_entity.id} 缺少 'connected_entity_ids' 属性。")
            return False
        if next_entity.id not in current_entity.connected_entity_ids:
            # 添加调试信息
            print(f"  [调试] 拓扑错误: 实体 {next_entity.id} 不在实体 {current_entity.id} 的连接列表 {current_entity.connected_entity_ids} 中。")
            return False
    return True

# 可视化辅助函数 ---
def plot_paths_3d(paths_dict: Dict[int, Optional[List[EvacuationEntity]]],
                  all_entities: List[EvacuationEntity],
                  emergency_entities: List[EvacuationEntity],
                  title: str,
                  filename: str,
                  model_dir: str):
    """生成并保存给定路径的 3D 可视化图。"""
    print(f"\n--- 可视化: {title} ---")
    if not paths_dict or not all_entities:
        print("  没有路径结果或实体数据可供可视化。")
        return

    valid_paths_count = sum(1 for path in paths_dict.values() if path)
    if valid_paths_count == 0:
        print("  没有有效的路径可供绘制。")
        return

    print(f"  正在为 {valid_paths_count} 条有效路径生成 3D 可视化...")
    fig = plt.figure(figsize=(18, 15))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    # 设置一个默认视角
    ax.view_init(elev=30., azim=-60)
    
    # 绘制布局
    viz.plot_layout_3d(all_entities, floor_height=FLOOR_HEIGHT, ax=ax, show_connections=True, show_labels=False)

    # 高亮紧急点
    if emergency_entities:
        viz.highlight_entities_3d(emergency_entities, ax, floor_height=FLOOR_HEIGHT, 
                                 color='red', marker='s', size=200, label='Initial Emergency')

    # 绘制路径
    color_idx = 0
    plotted_starts = set()
    valid_path_count = 0
    invalid_path_count = 0
    valid_path_line = None
    invalid_path_line = None

    for start_id, path in paths_dict.items():
        if path:
            is_valid_topology = validate_path_topology(path)
            
            # 根据拓扑有效性设置颜色
            if is_valid_topology:
                path_color = 'green'  # 有效路径使用绿色
                line_width = 2.5
                valid_path_count += 1
                path_label = '_nolegend_'
                # 绘制路径
                line = viz.plot_path_on_layout_3d(path, ax, floor_height=FLOOR_HEIGHT, 
                                                 color=path_color, linewidth=line_width, label=path_label)
                valid_path_line = line
            else:
                path_color = 'red'  # 无效路径使用红色
                line_width = 1.5
                invalid_path_count += 1
                path_label = '_nolegend_'
                # 绘制路径
                line = viz.plot_path_on_layout_3d(path, ax, floor_height=FLOOR_HEIGHT, 
                                                 color=path_color, linewidth=line_width, label=path_label)
                invalid_path_line = line

            # 绘制起点，确保只绘制一次图例标签
            if start_id not in plotted_starts:
                start_label = 'Start Point(s)' if not plotted_starts else "_nolegend_"
                viz.highlight_entities_3d([path[0]], ax, floor_height=FLOOR_HEIGHT, 
                                         color='lime', marker='s', size=100, label=start_label)
                plotted_starts.add(start_id)

    # 优化图例处理
    handles, labels = ax.get_legend_handles_labels()
    by_label = {}
    
    # 手动创建图例项
    from matplotlib.lines import Line2D
    
    # 添加有效和无效路径的图例
    if valid_path_count > 0:
        valid_legend = Line2D([0], [0], color='green', lw=2.5, label=f'Valid Path ({valid_path_count})')
        handles.append(valid_legend)
        labels.append(valid_legend.get_label())
    
    if invalid_path_count > 0:
        invalid_legend = Line2D([0], [0], color='red', lw=1.5, label=f'Invalid Path ({invalid_path_count})')
        handles.append(invalid_legend)
        labels.append(invalid_legend.get_label())
    
    # 过滤和去重图例项
    legend_labels_to_keep = {'Start Point(s)', 'Initial Emergency', 
                            f'Valid Path ({valid_path_count})', f'Invalid Path ({invalid_path_count})'}
    
    for h, l in zip(handles, labels):
        if l in legend_labels_to_keep and l not in by_label:
            by_label[l] = h

    if by_label:
        ax.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1.02, 1), 
                 borderaxespad=0., title="Legend")

    # 调整坐标轴和网格
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    full_filename = os.path.join(model_dir, filename)
    try:
        plt.savefig(full_filename, dpi=300)
        print(f"  3D 可视化已保存到: {full_filename}")
    except Exception as e:
        print(f"  保存 3D 可视化时出错: {e}")
    plt.close(fig)

# 主程序入口
if __name__ == "__main__":
    try:
        # --- 路径设置 ---
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        data_dir = os.path.join(project_root, 'data')
        model_dir = os.path.join(project_root, 'models')
        data_file_path = os.path.join(data_dir, 'building_layout.xlsx')
        model_load_path = os.path.join(model_dir, 'final_trained_model.pth')

        # --- 加载实体数据 ---
        print(f"尝试从以下路径加载实体数据: {data_file_path}")
        if not os.path.exists(data_file_path):
             print(f"错误：找不到实体数据文件 '{data_file_path}'。")
             exit()
        all_entities = EvacuationEntity.from_xlsx(data_file_path)
        print(f"成功加载 {len(all_entities)} 个实体。")
        if not all_entities:
            print("错误：加载的实体列表为空。")
            exit()

        entity_id_to_index: Dict[int, int] = {entity.id: i for i, entity in enumerate(all_entities)}
        index_to_entity_id: Dict[int, int] = {i: entity.id for i, entity in enumerate(all_entities)}
        N_entities = len(all_entities)

        # --- 加载预训练模型 ---
        print(f"\n--- 加载预训练模型 ---")
        if not os.path.exists(model_load_path):
            print(f"错误：找不到预训练模型文件 '{model_load_path}'。请先运行 training.py 完成训练。")
            exit()
        try:
            print(f"正在从 {model_load_path} 加载模型...")
            loaded_model = BPNeuralNetworkModel.load_model(model_load_path)
            print("模型加载成功。")
            print(f"  模型结构: Input={loaded_model.input_dim}, Hidden={loaded_model.hidden_dim}, Output={loaded_model.output_dim}")
            print(f"  模型设备: {loaded_model.device}")
        except Exception as e:
            print(f"加载模型时出错: {e}")
            exit()

        # --- 动态多路径规划示例 ---
        print("\n--- 开始动态多路径规划示例 ---")
        num_starts = 300
        num_initial_emergencies = 2
        max_dynamic_iterations = 50
        occupants_setting = 1

        example_emergency_entities = EvacuationEntity.select_emergency_areas(all_entities, num_areas=num_initial_emergencies)
        example_start_entities = EvacuationEntity.select_starting_areas(all_entities,  num_areas=num_starts, emergency_entities=example_emergency_entities)
        initial_paths_result = {}
        final_dynamic_paths = {}

        if example_start_entities and example_emergency_entities:
            print(f"示例场景：{len(example_start_entities)} 个起点, {len(example_emergency_entities)} 个初始紧急点。")
            print(f"  起点IDs: {[e.id for e in example_start_entities]}")
            print(f"  紧急点IDs: {[e.id for e in example_emergency_entities]}")

            try:
                initial_paths_result, final_dynamic_paths = plan_dynamic_paths(
                    model=loaded_model,
                    start_entities=example_start_entities,
                    initial_emergency_entities=example_emergency_entities,
                    all_entities=all_entities,
                    entity_id_to_index=entity_id_to_index,
                    index_to_entity_id=index_to_entity_id,
                    max_iterations=max_dynamic_iterations,
                    occupants_per_start=occupants_setting
                )

                # --- 验证和打印初始路径结果 ---
                print("\n--- 初始路径规划结果 (动态调整前) ---")
                initial_successful_paths = 0 # 成功路径计数
                initial_valid_topology_paths = 0 # 有效拓扑路径计数
                if initial_paths_result:
                    total_initial_paths = len(initial_paths_result) # 计算总路径数
                    # 遍历每个起点的路径
                    for start_id, path in initial_paths_result.items():
                        if path:
                            initial_successful_paths += 1 # 成功路径计数
                            is_valid = validate_path_topology(path) # 验证路径拓扑
                            # 如果路径有效，增加有效拓扑路径计数
                            if is_valid:
                                initial_valid_topology_paths += 1
                            valid_str = "有效拓扑" if is_valid else "无效拓扑"
                            print(f"  起点 {start_id}: 路径 (长度 {len(path)}, {valid_str}) -> {[e.id for e in path]}")
                        else:
                            print(f"  起点 {start_id}: 未能规划出路径。")

                    print(f"\n成功为 {initial_successful_paths}/{total_initial_paths} 个起点规划了初始路径。")
                    if initial_successful_paths > 0:
                        initial_accuracy = (initial_valid_topology_paths / initial_successful_paths) * 100
                        print(f"初始路径拓扑准确率: {initial_valid_topology_paths}/{initial_successful_paths} = {initial_accuracy:.2f}%")
                    else:
                        print("没有成功生成的初始路径可供评估准确率。")
                else:
                    print("  未能获取初始路径结果。")

                # --- 验证和打印最终动态规划结果 ---
                print("\n--- 最终动态规划结果 (调整后) ---")
                final_successful_paths = 0
                final_valid_topology_paths = 0
                if final_dynamic_paths:
                    total_final_paths = len(final_dynamic_paths)
                    for start_id, path in final_dynamic_paths.items():
                        if path:
                            final_successful_paths += 1
                            is_valid = validate_path_topology(path)
                            if is_valid:
                                final_valid_topology_paths += 1
                            valid_str = "有效拓扑" if is_valid else "无效拓扑"
                            print(f"  起点 {start_id}: 路径 (长度 {len(path)}, {valid_str}) -> {[e.id for e in path]}")
                        else:
                            print(f"  起点 {start_id}: 未能规划出路径。")

                    print(f"\n成功为 {final_successful_paths}/{total_final_paths} 个起点规划了最终动态路径。")
                    if final_successful_paths > 0:
                        final_accuracy = (final_valid_topology_paths / final_successful_paths) * 100
                        print(f"最终路径拓扑准确率: {final_valid_topology_paths}/{final_successful_paths} = {final_accuracy:.2f}%")
                    else:
                        print("没有成功生成的最终路径可供评估准确率。")
                else:
                     print("  未能获取最终动态路径结果。")

            except Exception as e:
                 print(f"执行动态规划时出错: {e}")
                 import traceback
                 traceback.print_exc()

        #     # --- 全面模型评估 ---
        #     print("\n--- 开始全面模型评估 ---")
        #     try:
        #         # 使用新的评估模块进行全面评估
        #         evaluation_results, evaluation_summary = model_evaluation.evaluate_model_performance(
        #             model=loaded_model,
        #             start_entities=example_start_entities,
        #             initial_emergency_entities=example_emergency_entities,
        #             all_entities=all_entities,
        #             entity_id_to_index=entity_id_to_index,
        #             index_to_entity_id=index_to_entity_id,
        #             model_dir=model_dir,
        #             max_iterations=max_dynamic_iterations,
        #             occupants_setting=occupants_setting
        #         )
                
        #         # 打印评估摘要
        #         print("\n--- 模型评估摘要 ---")
        #         print(f"初始路径拓扑有效率: {evaluation_summary['initial_valid_ratio']:.2f}%")
        #         print(f"最终路径拓扑有效率: {evaluation_summary['final_valid_ratio']:.2f}%")
        #         print(f"初始路径成功率: {evaluation_summary['initial_success_rate']:.2f}%")
        #         print(f"最终路径成功率: {evaluation_summary['final_success_rate']:.2f}%")
        #         print(f"平均路径长度变化: {evaluation_summary['length_change_percent']:.2f}%")
        #         print(f"紧急区域避开改善程度: {evaluation_summary['emergency_avoidance_improvement']:.2f}")
        #         print(f"平均拥堵减少率: {evaluation_summary['avg_congestion_reduction']:.2f}%")
                
        #     except Exception as e:
        #         print(f"进行模型评估时出错: {e}")
        #         import traceback
        #         traceback.print_exc()

        # else:
        #     print("未能为动态规划示例选择足够的起点或初始紧急点。")

        # # --- 可视化初始和最终结果 ---
        # plot_paths_3d(initial_paths_result,
        #               all_entities,
        #               example_emergency_entities,
        #               "Initial Path Planning Results (Before Dynamic Adjustment)",
        #               "initial_dynamic_paths_test_3d.png",
        #               model_dir)

        # plot_paths_3d(final_dynamic_paths,
        #               all_entities,
        #               example_emergency_entities,
        #               "Final Dynamic Path Planning Results (After Adjustment)",
        #               "final_dynamic_paths_test_3d.png",
        #               model_dir)

    except FileNotFoundError as e:
         print(f"文件未找到错误: {e}")
    except ImportError as e:
         print(f"导入错误: {e}。请确保所有依赖项已安装且在 Python 路径中。")
    except AttributeError as e:
         print(f"属性错误: {e}。请检查 EvacuationEntity 是否包含 'connections' 属性以及其初始化是否正确。")
    except Exception as e:
         print(f"执行测试时发生未知错误: {e}")
         import traceback
         traceback.print_exc()
