from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np


if "MPLBACKEND" not in os.environ:  # 判断是否满足当前处理的条件
    matplotlib.use("TkAgg")  # 启用支持鼠标拖拽的Tk图形后端

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from OCC.Core.Geom import Geom_BSplineCurve, Geom_BSplineSurface
from OCC.Core.gp import gp_Pnt
from OCC.Core.TColgp import TColgp_Array1OfPnt, TColgp_Array2OfPnt
from OCC.Core.TColStd import (
    TColStd_Array1OfInteger,
    TColStd_Array1OfReal,
    TColStd_Array2OfReal,
)
from occwl.compound import Compound

from new_g_extractor import NewGraphExtractor


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]  # 指定中文标题优先使用微软雅黑或黑体
plt.rcParams["axes.unicode_minus"] = False  # 让坐标轴负号使用正常字符显示


# SCRIPT_DIR = Path(__file__).resolve().parent  # 取得当前脚本目录作为默认文件位置
# DEFAULT_STEP = SCRIPT_DIR / "_model1.stp"  # 指定脚本目录中的默认STEP模型


def _real_array(values: Any) -> TColStd_Array1OfReal:
    values = np.asarray(values, dtype=float).ravel()  # 把输入展平为OpenCascade需要的一维浮点序列
    result = TColStd_Array1OfReal(1, len(values))  # 按OpenCascade的一基索引分配实数数组
    for index, value in enumerate(values, start=1):
        result.SetValue(index, float(value))  # 把当前数值写入OpenCascade数组
    return result  # 返回OpenCascade一维实数数组


def _integer_array(values: Any) -> TColStd_Array1OfInteger:
    values = np.asarray(values, dtype=int).ravel()  # 把输入展平为OpenCascade需要的一维整数序列
    result = TColStd_Array1OfInteger(1, len(values))  # 按OpenCascade的一基索引分配整数数组
    for index, value in enumerate(values, start=1):
        result.SetValue(index, int(value))  # 把当前数值写入OpenCascade数组
    return result  # 返回OpenCascade一维整数数组


def evaluate_surface(feature: dict[str, Any], sample_count: int = 22) -> np.ndarray:

    poles = np.asarray(feature["poles"], float)  # 分配或保存NURBS控制点坐标
    weights = np.asarray(feature["weights"], float)  # 分配或保存每个控制点的有理权重
    nu, nv = poles.shape[:2]  # 完成由NURBS参数重建并采样周期或非周期曲面中的当前计算
    occ_poles = TColgp_Array2OfPnt(1, nu, 1, nv)  # 创建OpenCascade格式的控制点数组
    occ_weights = TColStd_Array2OfReal(1, nu, 1, nv)  # 创建OpenCascade格式的权重数组
    for u in range(nu):
        for v in range(nv):
            occ_poles.SetValue(u + 1, v + 1, gp_Pnt(*map(float, poles[u, v])))  # 写入一个OpenCascade格式控制点
            occ_weights.SetValue(u + 1, v + 1, float(weights[u, v]))  # 写入对应控制点的NURBS权重

    surface = Geom_BSplineSurface(  # 用控制点、权重、节点、次数和周期性重建曲面
        occ_poles,  # 补充由NURBS参数重建并采样周期或非周期曲面所需参数
        occ_weights,  # 补充由NURBS参数重建并采样周期或非周期曲面所需参数
        _real_array(feature["u_knots"]),  # 传入曲面U方向节点序列
        _real_array(feature["v_knots"]),  # 传入曲面V方向节点序列
        _integer_array(feature["u_multiplicities"]),  # 传入曲面U方向节点重数
        _integer_array(feature["v_multiplicities"]),  # 传入曲面V方向节点重数
        int(feature["u_degree"]),  # 传入曲面U方向次数
        int(feature["v_degree"]),  # 传入曲面V方向次数
        bool(feature["u_periodic"]),  # 传入曲面U方向周期标志
        bool(feature["v_periodic"]),  # 传入曲面V方向周期标志
    )  # 完成由NURBS参数重建并采样周期或非周期曲面的数据构造
    u_first, u_last, v_first, v_last = surface.Bounds()  # 读取曲面U、V两个方向的有效参数范围
    u_values = np.linspace(u_first, u_last, sample_count)  # 在曲面U参数范围内均匀取值
    v_values = np.linspace(v_first, v_last, sample_count)  # 在曲面V参数范围内均匀取值
    result = np.empty((sample_count, sample_count, 3), dtype=float)  # 保存当前函数最终输出
    for u_index, u_value in enumerate(u_values):
        for v_index, v_value in enumerate(v_values):
            point = surface.Value(float(u_value), float(v_value))  # 计算当前UV参数对应的曲面空间点
            result[u_index, v_index] = point.X(), point.Y(), point.Z()  # 把曲面采样点的XYZ坐标写入点阵
    return result  # 返回规则采样的三维NURBS曲面点阵


def evaluate_curve(feature: dict[str, Any], sample_count: int = 64) -> np.ndarray:

    poles = np.asarray(feature["poles"], float)  # 分配或保存NURBS控制点坐标
    weights = np.asarray(feature["weights"], float)  # 分配或保存每个控制点的有理权重
    occ_poles = TColgp_Array1OfPnt(1, len(poles))  # 创建OpenCascade格式的控制点数组
    for index, pole in enumerate(poles, start=1):
        occ_poles.SetValue(index, gp_Pnt(*map(float, pole)))  # 写入一个OpenCascade格式控制点

    curve = Geom_BSplineCurve(  # 用控制点、权重、节点、次数和周期性重建曲线
        occ_poles,  # 补充由NURBS参数重建并采样周期或非周期曲线所需参数
        _real_array(weights),  # 补充由NURBS参数重建并采样周期或非周期曲线所需参数
        _real_array(feature["knots"]),  # 传入曲线唯一节点序列
        _integer_array(feature["multiplicities"]),  # 传入曲线各节点重数
        int(feature["degree"]),  # 传入曲线次数
        bool(feature["periodic"]),  # 传入曲线周期闭合标志
    )  # 完成由NURBS参数重建并采样周期或非周期曲线的数据构造
    parameters = np.linspace(curve.FirstParameter(), curve.LastParameter(), sample_count)  # 在曲线首尾参数之间生成均匀采样值
    result = np.empty((sample_count, 3), dtype=float)  # 保存当前函数最终输出
    for index, parameter in enumerate(parameters):
        point = curve.Value(float(parameter))  # 计算当前参数对应的曲线空间点
        result[index] = point.X(), point.Y(), point.Z()  # 把曲线采样点的XYZ坐标写入结果数组
    return result  # 返回沿参数域采样的三维NURBS曲线点


def _edge_signature(feature: dict[str, Any]) -> tuple[float, ...]:

    poles = np.round(np.asarray(feature["poles"], float), 6)  # 分配或保存NURBS控制点坐标
    weights = np.round(np.asarray(feature["weights"], float), 6)  # 分配或保存每个控制点的有理权重
    forward = tuple(np.concatenate((poles.ravel(), weights)).tolist())  # 按原控制点方向构造边签名
    reverse = tuple(np.concatenate((poles[::-1].ravel(), weights[::-1])).tolist())  # 按反向控制点顺序构造边签名
    return min(forward, reverse)  # 返回正反方向一致的边几何签名


def _unique_edges(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[tuple[float, ...], dict[str, Any]] = {}  # 保存每种实体边几何的唯一副本
    for feature in features:
        unique.setdefault(_edge_signature(feature), feature)  # 用方向无关签名识别重复的有向边
    return list(unique.values())  # 返回去除有向重复后的实体边列表


def load_backbone(step_file: Path, feature_file: Path | None) -> dict[str, Any]:

    if feature_file is None:  # 未提供缓存特征时直接从STEP提取
        data = NewGraphExtractor(step_file).process()  # 保存读取或提取的主干特征字典
    else:  # 处理前述条件之外的数据
        with feature_file.open("r", encoding="utf-8") as stream:  # 以UTF-8文本方式读取特征JSON
            data = json.load(stream)  # 解析JSON中的NURBS和拓扑特征

    required = ("face_nurbs", "edge_nurbs", "edge_face_incidence", "normalization")  # 限定主干可视化需要读取的四类特征
    body = Compound.load_from_step(str(step_file))  # 读取STEP并得到BRep复合体
    vertices, triangles = body.get_triangles()  # 三角化原始BRep并保留面的裁剪边界
    center = np.asarray(data["normalization"]["center"], dtype=float)  # 计算或保存模型包围盒中心
    scale = float(data["normalization"]["scale"])  # 计算或保存统一坐标缩放尺度

    result = {key: data[key] for key in required}  # 保存当前函数最终输出
    result["model_vertices"] = (np.asarray(vertices, float) - center) / scale  # 转换为固定类型的NumPy数组
    result["model_triangles"] = np.asarray(triangles, dtype=int)  # 转换为固定类型的NumPy数组
    return result  # 返回NURBS主干特征和原模型显示网格


def _weight_norm(data: dict[str, Any]) -> Normalize:
    arrays = [np.asarray(item["weights"]).ravel() for item in data["face_nurbs"]]  # 收集面和边的全部控制点权重
    arrays += [np.asarray(item["weights"]).ravel() for item in data["edge_nurbs"]]  # 转换为固定类型的NumPy数组
    values = np.concatenate(arrays) if arrays else np.array([1.0])  # 组合当前实体需要输出的数值通道
    low, high = float(values.min()), float(values.max())  # 完成统计控制点权重并建立统一颜色范围中的当前计算
    if np.isclose(low, high):  # 权重完全相同时扩展色域避免除零
        low, high = low - 0.5, high + 0.5  # 完成统计控制点权重并建立统一颜色范围中的当前计算
    return Normalize(low, high)  # 返回控制点权重的颜色归一化器


def _equal_axes(ax: Any, point_sets: list[np.ndarray]) -> None:

    xyz = np.concatenate([points.reshape(-1, 3) for points in point_sets])  # 合并所有点以计算统一显示范围
    low, high = xyz.min(axis=0), xyz.max(axis=0)  # 完成统一三维坐标轴尺度以防模型显示变形中的当前计算
    center = (low + high) / 2.0  # 计算或保存模型包围盒中心
    radius = max(float((high - low).max()) / 2.0, 1e-6)  # 取最大轴跨度的一半作为显示半径
    ax.set_xlim(center[0] - radius, center[0] + radius)  # 按统一中心和半径限定X轴范围
    ax.set_ylim(center[1] - radius, center[1] + radius)  # 按统一中心和半径限定Y轴范围
    ax.set_zlim(center[2] - radius, center[2] + radius)  # 按统一中心和半径限定Z轴范围
    ax.set_box_aspect((1, 1, 1))  # 令XYZ三轴等比例以避免模型变形


def _draw_parameter_arrow(
    ax: Any, origin: np.ndarray, direction: np.ndarray,
    label: str, color: str, ratio: float,
) -> None:
    length = float(np.linalg.norm(direction))  # 计算相邻控制点间距以识别退化方向
    if length <= 1e-10:  # 重合控制点不能确定清晰的参数正方向
        return  # 不为退化控制段绘制错误箭头
    arrow = direction * ratio  # 让箭头沿控制段前进且不遮住下一个控制点
    ax.quiver(*origin, *arrow, color=color, linewidth=2.3,  # 以当前控制点作为箭头起点
              arrow_length_ratio=0.18)  # 保持箭头头部在缩放后仍清晰可见
    label_position = origin + arrow * 1.12  # 将参数名称放在箭头尖端外侧
    ax.text(*label_position, label, color=color, fontsize=10,  # 标出当前控制段对应的参数正方向
            fontweight="bold", ha="center", va="center")  # 加粗并居中显示参数名称


def _draw_uv_directions(ax: Any, poles: np.ndarray) -> None:
    if poles.shape[0] < 2 or poles.shape[1] < 2:  # U、V方向都至少需要两个控制点
        return  # 控制网格退化时不绘制双参数方向
    origin = poles[0, 0]  # 选择P00控制点作为U、V箭头的共同起点
    u_direction = poles[1, 0] - origin  # P00指向P10表示U索引增大的方向
    v_direction = poles[0, 1] - origin  # P00指向P01表示V索引增大的方向
    _draw_parameter_arrow(ax, origin, u_direction, "U", "#7c3aed", 0.42)  # 在U控制线上绘制紫色正向箭头
    _draw_parameter_arrow(ax, origin, v_direction, "V", "#16a34a", 0.42)  # 在V控制线上绘制绿色正向箭头


def _draw_edge_u_direction(ax: Any, poles: np.ndarray) -> None:
    if len(poles) < 2:  # 曲线参数方向至少需要两个控制点确定
        return  # 单控制点退化边不绘制参数箭头
    origin = poles[0]  # 选择曲线第一个控制点作为箭头起点
    direction = poles[1] - origin  # 第一个控制点指向第二个控制点表示曲线U正方向
    _draw_parameter_arrow(ax, origin, direction, "U", "#7c3aed", 0.42)  # 在边控制多边形上绘制U方向


def draw_geometry(
    ax: Any, data: dict[str, Any], norm: Normalize, mode: str
) -> None:

    weight_cmap = plt.get_cmap("viridis")  # 选择控制点权重的颜色映射
    model_vertices = np.asarray(data["model_vertices"], dtype=float)  # 读取归一化后的原模型网格顶点
    model_triangles = np.asarray(data["model_triangles"], dtype=int)  # 读取原模型的三角形连接索引
    point_sets: list[np.ndarray] = [model_vertices]  # 收集模型和控制点用于统一坐标范围


    ax.plot_trisurf(  # 绘制保留裁剪边界的原始BRep三角网格
        model_vertices[:, 0], model_vertices[:, 1], model_vertices[:, 2],  # 向三角网格绘图函数传入XYZ顶点坐标
        triangles=model_triangles, color="#94a3b8", alpha=0.30,  # 按原模型三角形索引连接网格顶点
        linewidth=0, edgecolor="none", shade=True,  # 隐藏三角片边线并启用表面明暗
    )  # 完成把NURBS控制结构叠加到原始裁剪模型的数据构造


    face_heights = [  # 计算每个面控制网格中心的全局Z高度
        np.asarray(feature["poles"], float)[..., 2].mean()  # 用全部面控制点的平均Z代表该面高度
        for feature in data["face_nurbs"]
    ]  # 得到与面编号顺序一致的高度列表
    top_face_index = int(np.argmax(face_heights))  # 选择平均Z最大的面作为最高面

    if mode == "face_control":  # 第一张子图只绘制最高面的控制结构
        feature = data["face_nurbs"][top_face_index]  # 读取最高面对应的NURBS参数
        poles = np.asarray(feature["poles"], float)  # 读取最高面的二维控制点网格
        weights = np.asarray(feature["weights"], float).ravel()  # 读取与控制点逐一对应的权重
        point_sets.append(poles)  # 将最高面控制点加入等比例坐标范围计算
        _draw_uv_directions(ax, poles)  # 在P00控制点上标出最高面的U、V正方向
        for row in poles:
            ax.plot(*row.T, color="#2563eb", linewidth=1.5)  # 连接同一U索引下的V方向控制点
        for column in poles.transpose(1, 0, 2):
            ax.plot(*column.T, color="#2563eb", linewidth=1.5)  # 连接同一V索引下的U方向控制点
        flat = poles.reshape(-1, 3)  # 展平最高面的控制网格用于一次性绘制控制点
        ax.scatter(  # 绘制最高面的全部NURBS控制点
            *flat.T, c=weights, cmap=weight_cmap, norm=norm,  # 用权重颜色保持控制点物理含义可见
            s=42 + 22 * norm(weights), edgecolors="#1e3a8a",  # 用权重调节点大小并添加蓝色边框
            linewidths=0.6, depthshade=False,  # 固定边框宽度并关闭深度引起的颜色变化
        )  # 完成最高面控制点绘制

    if mode == "edge_control":  # 第二张子图只绘制最高面边界上的边控制结构
        incidence = np.asarray(data["edge_face_incidence"], dtype=int)  # 读取每条有向边及其两侧面编号
        top_edge_ids = [  # 收集与最高面相邻的有向边记录编号
            int(edge_id)  # 使用关联表第一列索引对应的edge_nurbs记录
            for edge_id, source, target in incidence
            if source == top_face_index or target == top_face_index  # 保留任一侧属于最高面的边
        ]  # 同一物理边可能因正反面对偶方向出现两次
        top_edges = _unique_edges([data["edge_nurbs"][edge_id] for edge_id in top_edge_ids])  # 按控制点几何去除正反方向重复边
        for feature in top_edges:
            curve = evaluate_curve(feature)  # 从NURBS参数重建最高面的一条边界曲线
            poles = np.asarray(feature["poles"], float)  # 读取该边界曲线的控制点序列
            weights = np.asarray(feature["weights"], float)  # 读取各边控制点的有理权重
            point_sets.extend((curve, poles))  # 将边曲线和控制点加入等比例坐标范围计算
            _draw_edge_u_direction(ax, poles)  # 在第一个边控制点上标出曲线U正方向
            ax.plot(*curve.T, color="#dc2626", linewidth=2.0)  # 绘制最高面的真实NURBS边界曲线
            ax.plot(*poles.T, color="#f59e0b", linestyle="--", linewidth=1.6)  # 绘制该边的控制多边形
            ax.scatter(  # 绘制最高面边界上的曲线控制点
                *poles.T, c=weights, cmap=weight_cmap, norm=norm, marker="s",  # 用方形点区别于面控制点
                s=40 + 20 * norm(weights), edgecolors="#92400e",  # 用权重调节点大小并添加棕色边框
                linewidths=0.6, depthshade=False,  # 固定边框宽度并保持权重颜色不受深度影响
            )  # 完成最高面边控制点绘制

    _equal_axes(ax, point_sets)  # 完成把NURBS控制结构叠加到原始裁剪模型中的当前计算
    ax.view_init(elev=24, azim=-54)  # 完成把NURBS控制结构叠加到原始裁剪模型中的当前计算
    ax.set_xlabel("X")  # 标明归一化模型的X坐标轴
    ax.set_ylabel("Y")  # 标明归一化模型的Y坐标轴
    ax.set_zlabel("Z")  # 标明归一化模型的Z坐标轴
    ax.grid(True, alpha=0.20)  # 完成把NURBS控制结构叠加到原始裁剪模型中的当前计算


def create_figure(data: dict[str, Any], output: Path, dpi: int, show: bool) -> None:

    norm = _weight_norm(data)  # 建立面和边控制点共享的权重色域
    fig = plt.figure(figsize=(16, 7.5))  # 创建Matplotlib整张画布
    panels = (  # 定义两个子图的模式和中文标题
        ("face_control", "最高面的控制网格、控制点与U/V方向"),  # 补充创建两个可交互子图并配置图片保存所需参数
        ("edge_control", "最高面边界的控制点与U方向"),  # 补充创建两个可交互子图并配置图片保存所需参数
    )  # 完成创建两个可交互子图并配置图片保存的数据构造
    axes = []  # 收集两个三维坐标轴供公共色条使用
    for position, (mode, title) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 2, position, projection="3d")  # 在一行两列布局中创建当前三维子图
        draw_geometry(ax, data, norm, mode)  # 完成创建两个可交互子图并配置图片保存中的当前计算
        ax.set_title(title, fontsize=11)  # 为当前控制结构子图显示中文标题
        axes.append(ax)  # 把当前计算结果加入输出序列


    fig.subplots_adjust(left=0.03, right=0.88, bottom=0.06, top=0.92,  # 为两个子图和右侧色条预留画布空间
                        wspace=0.08)  # 控制左右两个子图之间的水平间距
    colorbar_ax = fig.add_axes((0.92, 0.22, 0.015, 0.56))  # 在画布右侧创建独立色条区域
    colorbar = fig.colorbar(  # 创建解释NURBS权重颜色的公共色条
        ScalarMappable(norm=norm, cmap="viridis"), cax=colorbar_ax,  # 补充创建两个可交互子图并配置图片保存所需参数
    )  # 完成创建两个可交互子图并配置图片保存的数据构造
    colorbar.set_label("NURBS 权重")  # 标明颜色代表NURBS控制点权重
    fig.canvas.manager.set_window_title("BRep JEPA 主干特征")  # 为交互窗口显示主干特征标题

    output.parent.mkdir(parents=True, exist_ok=True)  # 确保图片或特征输出目录已经存在

    def save_current_view(_event: Any = None) -> None:
        fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")  # 按当前相机视角和指定DPI保存整张图


    def on_key(event: Any) -> None:
        if event.key and event.key.lower() == "s":  # 检测用户是否按下保存快捷键S
            save_current_view(event)  # 完成响应键盘按键并触发当前视角保存中的当前计算

    fig.canvas.mpl_connect("key_press_event", on_key)  # 绑定键盘事件以支持按S保存视角
    save_current_view()  # 完成创建两个可交互子图并配置图片保存中的当前计算

    if show:  # 判断是否满足创建两个可交互子图并配置图片保存的条件
        plt.show()  # 进入Matplotlib鼠标交互窗口
    else:  # 处理前述条件之外的数据
        plt.close(fig)  # 完成创建两个可交互子图并配置图片保存中的当前计算


def main() -> None:
    parser = argparse.ArgumentParser(description="交互显示并保存 JEPA 主干 NURBS 特征")  # 创建命令行参数解析器
    parser.add_argument("step_file", nargs="?", type=Path, default='C:\\Users\\shinny\\Desktop\\labeled\\data\\_model1.stp')  # 把命令行字符串自动转换为文件路径
    parser.add_argument("--features", type=Path, help="可选：特征提取器生成的 JSON")  # 允许直接读取已生成的图特征
    parser.add_argument("--output", type=Path, help="图片路径，默认保存在 STEP 文件旁")  # 把命令行字符串自动转换为文件路径
    parser.add_argument("--dpi", type=int, default=220)  # 定义命令行参数"--dpi"
    parser.add_argument("--no-show", action="store_true", help="只保存，不打开交互窗口")  # 说明该命令行参数的用途和默认规则
    args = parser.parse_args()  # 读取用户提供的命令行参数

    step_file = args.step_file.resolve()  # 保存STEP文件的绝对路径
    output = (args.output or step_file.with_name(  # 确定最终图片或特征文件的保存路径
        f"{step_file.stem}_backbone_features.png")).resolve()  # 完成解析命令行参数并运行当前程序中的当前计算
    features = args.features.resolve() if args.features else None  # 确定可选的预提取特征文件路径
    data = load_backbone(step_file, features)  # 保存读取或提取的主干特征字典
    create_figure(data, output, args.dpi, show=not args.no_show)  # 完成解析命令行参数并运行当前程序中的当前计算


if __name__ == "__main__":  # 判断是否满足当前处理的条件
    main()  # 完成backbone_features处理流程中的当前计算
