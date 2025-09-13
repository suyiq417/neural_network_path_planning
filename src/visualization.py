import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # 新增：导入 3D 绘图工具
from matplotlib.collections import LineCollection
from typing import List, Dict, Optional
from evacuation_entity import EvacuationEntity, EntityType
import numpy as np

# 定义不同实体类型的颜色和标记
ENTITY_STYLE = {
    EntityType.Room: {'color': 'lightblue', 'marker': 's', 'size': 50},
    EntityType.Corridor: {'color': 'lightgrey', 'marker': 's', 'size': 30},
    EntityType.Door: {'color': 'brown', 'marker': 'o', 'size': 20},
    EntityType.Stair: {'color': 'purple', 'marker': '^', 'size': 40},
    EntityType.Exit: {'color': 'red', 'marker': '*', 'size': 100},
}
DEFAULT_STYLE = {'color': 'gray', 'marker': '.', 'size': 10}

def plot_layout_3d(entities: List[EvacuationEntity], floor_height: float = 3.5, ax: Optional[plt.Axes] = None, show_connections: bool = True, show_labels: bool = False):
    """
    绘制建筑布局（3D）。

    Args:
        entities: 所有实体的列表。
        floor_height: 每层楼的高度。
        ax: 用于绘图的 matplotlib 3D Axes 对象。如果为 None，则创建新的 Figure 和 Axes。
        show_connections: 是否绘制实体间的连接线。
        show_labels: 是否显示实体 ID 标签。
    """
    if ax is None:
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("Building Layout (3D)")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_zlabel("Floor Level (Height)")

    if not entities:
        print("警告: 没有实体用于绘制。")
        return ax

    entity_map = {e.id: e for e in entities}

    # 设置方形标记作为默认标记
    default_marker = 's'
    
    # 更新实体样式使用方形标记
    updated_entity_style = {
        EntityType.Room: {'color': 'lightblue', 'marker': default_marker, 'size': 80},
        EntityType.Corridor: {'color': 'lightgrey', 'marker': default_marker, 'size': 60},
        EntityType.Door: {'color': 'brown', 'marker': default_marker, 'size': 40},
        EntityType.Stair: {'color': 'purple', 'marker': default_marker, 'size': 70},
        EntityType.Exit: {'color': 'red', 'marker': default_marker, 'size': 100},
    }

    # 绘制实体
    plotted_labels = set()
    for entity in entities:
        style = updated_entity_style.get(entity.entity_type, {'color': 'gray', 'marker': default_marker, 'size': 40})
        x, y = entity.center
        z = entity.floor * floor_height
        label = f"{entity.entity_type.name}"
        if label not in plotted_labels:
            ax.scatter(x, y, z,
                       c=style['color'], marker=style['marker'], s=style['size'],
                       label=label)
            plotted_labels.add(label)
        else:
             ax.scatter(x, y, z,
                       c=style['color'], marker=style['marker'], s=style['size'])

        if show_labels:
            ax.text(x, y, z + 0.1, str(entity.id), fontsize=8)

    # 绘制连接线
    if show_connections:
        lines_3d = []
        for entity in entities:
            for connected_id in entity.connected_entity_ids:
                connected_entity = entity_map.get(connected_id)
                if connected_entity:
                    p1 = (*entity.center, entity.floor * floor_height)
                    p2 = (*connected_entity.center, connected_entity.floor * floor_height)
                    lines_3d.append([p1, p2])

        # 在 3D 中绘制线段
        for line in lines_3d:
            xs = [line[0][0], line[1][0]]
            ys = [line[0][1], line[1][1]]
            zs = [line[0][2], line[1][2]]
            ax.plot(xs, ys, zs, color='gray', linewidth=0.5, alpha=0.7)

    # 添加图例
    ax.legend(title="Entity Types")
    ax.grid(True, linestyle='--', alpha=0.5)

    # 调整视角
    ax.view_init(elev=20., azim=-65)

    return ax

def plot_path_on_layout_3d(path: List[EvacuationEntity], ax: plt.Axes, floor_height: float = 3.5, color: str = 'green', linewidth: float = 2.0, label: Optional[str] = None, marker: str = '>', markersize: int = 6):
    """
    在给定的 3D Axes 上绘制一条路径。

    Args:
        path: 实体列表表示的路径。
        ax: 用于绘图的 matplotlib 3D Axes 对象。
        floor_height: 每层楼的高度。
        color: 路径颜色。
        linewidth: 路径线宽。
        label: 路径标签（用于图例）。
        marker: 路径方向标记。
        markersize: 标记大小。
    """
    if not path or len(path) < 2:
        return

    x_coords = [e.center[0] for e in path]
    y_coords = [e.center[1] for e in path]
    z_coords = [e.floor * floor_height for e in path] # 计算 Z 坐标

    # 绘制路径线段
    ax.plot(x_coords, y_coords, z_coords, color=color, linewidth=linewidth, label=label, marker=marker, markersize=markersize, linestyle='-')

    # 标记起点和终点 (使用 3D 高亮函数)
    highlight_entities_3d([path[0]], ax, floor_height, color='lime', marker='o', size=100, label='Start')
    highlight_entities_3d([path[-1]], ax, floor_height, color='magenta', marker='X', size=150, label='End (Actual)')

def highlight_entities_3d(entities_to_highlight: List[EvacuationEntity], ax: plt.Axes, floor_height: float = 3.5, color: str = 'yellow', marker: str = 's', size: int = 100, label: Optional[str] = None, alpha: float = 0.8):
    """
    在给定的 3D Axes 上高亮显示指定的实体。
    使用方形标记来表示实体。

    Args:
        entities_to_highlight: 要高亮的实体列表。
        ax: 用于绘图的 matplotlib 3D Axes 对象。
        floor_height: 每层楼的高度。
        color: 高亮颜色。
        marker: 高亮标记 ('s' 代表正方形)。
        size: 高亮标记大小。
        label: 标签（用于图例）。
        alpha: 透明度。
    """
    if not entities_to_highlight:
        return

    x_coords = [e.center[0] for e in entities_to_highlight]
    y_coords = [e.center[1] for e in entities_to_highlight]
    z_coords = [e.floor * floor_height for e in entities_to_highlight]

    # 使用 scatter 绘制高亮标记
    # 检查标签是否已存在，避免重复图例项
    current_labels = [l.get_label() for l in ax.get_legend_handles_labels()[0]] if hasattr(ax, 'get_legend_handles_labels') else []
    if label and label in current_labels:
        label = None
    
    ax.scatter(x_coords, y_coords, z_coords, c=color, marker=marker, s=size, 
              label=label, alpha=alpha, edgecolors='black', depthshade=True, zorder=5)

