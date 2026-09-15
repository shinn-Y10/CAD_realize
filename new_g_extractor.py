from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_NurbsConvert
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.GeomConvert import geomconvert
from OCC.Core.TopoDS import topods
from occwl.compound import Compound
from occwl.edge import Edge
from occwl.edge_data_extractor import EdgeDataExtractor
from occwl.graph import face_adjacency
from occwl.uvgrid import uvgrid


SCRIPT_DIR = Path(__file__).resolve().parent 
DEFAULT_STEP = SCRIPT_DIR / "_model1.stp" 
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results" / "graphs" 


def _xyz(point: Any) -> list[float]:

    return [float(point.X()), float(point.Y()), float(point.Z())]  # 返回可直接存入NumPy数组的XYZ坐标


def _normalize(points: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:

    return (points - center) / scale  # 返回中心位于原点且统一缩放后的坐标


class NewGraphExtractor:


    def __init__(self, step_file: str | Path, face_samples: int = 10,
                 edge_samples: int = 10):
        self.step_file = Path(step_file).resolve()  # 保存STEP文件的绝对路径
        self.face_samples = face_samples  # 保存面在每个参数方向的采样数
        self.edge_samples = edge_samples  # 保存每条边的采样数

    def process(self) -> dict[str, Any]:

        body = Compound.load_from_step(str(self.step_file))  # 读取STEP并得到BRep复合体
        shape = body.topods_shape()  # 取得OpenCascade底层拓扑形状
        center, scale = self._normalization(shape)  # 计算模型统一归一化所需的中心和尺度
        graph = face_adjacency(body)  # 根据共享边建立面之间的邻接图

        face_nurbs = []  # 收集每个面的NURBS参数
        face_targets = []  # 收集每个面的规则采样监督目标
        node_to_face = {}  # 建立图节点编号到输出面编号的映射


        for node_id in sorted(graph.nodes):
            face = graph.nodes[node_id]["face"]  # 取得当前图节点对应的BRep面
            node_to_face[node_id] = len(face_nurbs)  # 记录当前图节点在面特征列表中的位置
            face_nurbs.append(  # 加入当前面的NURBS编码器输入
                self._face_nurbs(face.topods_shape(), center, scale)  # 提取当前面的控制点、权重、节点和次数
            )  # 完成读取STEP并组织面、边、拓扑及几何特征的数据构造
            face_targets.append(self._face_target(face, center, scale))  # 生成当前面的坐标、法向和内部掩码监督

        edge_nurbs = []  # 收集每条边的NURBS参数
        edge_targets = []  # 收集每条边的规则采样监督目标
        edge_target_valid = []  # 记录边监督数据是否有效
        incidence = []  # 收集边与两侧面的拓扑连接关系


        for src, dst, edge_info in graph.edges(data=True):
            edge = edge_info["edge"]  # 取得当前邻接关系对应的BRep边
            if not edge.has_curve():  # 跳过没有有效三维曲线的退化边
                continue  # 忽略当前无效实体并处理下一个

            edge_id = len(edge_nurbs)  # 确定当前edge的稳定编号
            edge_nurbs.append(  # 加入当前边的NURBS编码器输入
                self._edge_nurbs(edge.topods_shape(), center, scale)  # 提取当前边的控制点、权重、节点和次数
            )  # 完成读取STEP并组织面、边、拓扑及几何特征的数据构造
            target, valid = self._edge_target(body, edge, center, scale)  # 生成当前边的十二通道监督及有效标记
            edge_targets.append(target)  # 加入当前边的十二通道监督目标
            edge_target_valid.append(valid)  # 保存当前边监督数据的有效标记
            incidence.append([edge_id, node_to_face[src], node_to_face[dst]])  # 记录当前边及其左右相邻面编号

        return {  # 返回包含几何、拓扑和监督目标的特征字典
            "uid": self.step_file.stem,  # 保存模型唯一名称
            "face_nurbs": face_nurbs,  # 保存全部面NURBS输入特征
            "edge_nurbs": edge_nurbs,  # 保存全部边NURBS输入特征
            "edge_face_incidence": np.asarray(incidence, dtype=np.int64).reshape(-1, 3),  # 保存边连接两侧面的拓扑关系
            "face_grid_target": self._stack(  # 保存七通道面采样监督目标
                face_targets, (7, self.face_samples, self.face_samples)  # 按坐标3、法向3和掩码1组织面监督形状
            ),  # 完成读取STEP并组织面、边、拓扑及几何特征的数据构造
            "edge_grid_target": self._stack(  # 保存十二通道边采样监督目标
                edge_targets, (12, self.edge_samples)  # 按坐标3、切向3和双侧法向6组织边监督形状
            ),  # 完成读取STEP并组织面、边、拓扑及几何特征的数据构造
            "edge_target_valid": np.asarray(edge_target_valid, dtype=np.bool_),  # 保存边监督有效性标记

            "normalization": {  # 保存恢复原始物理坐标所需参数
                "center": center.astype(np.float64),  # 保存归一化使用的模型中心
                "scale": float(scale),  # 保存归一化使用的统一尺度
                "unit": "STEP_source_unit",  # 说明坐标沿用STEP源文件单位
            },  # 完成读取STEP并组织面、边、拓扑及几何特征的数据构造
        }  # 完成读取STEP并组织面、边、拓扑及几何特征的数据构造

    @staticmethod  # 该方法不依赖对象内部状态
    def _normalization(shape: Any) -> tuple[np.ndarray, float]:

        box = Bnd_Box()  # 创建用于累积模型空间范围的包围盒
        brepbndlib.Add(shape, box)  # 把完整BRep形状加入包围盒计算
        xmin, ymin, zmin, xmax, ymax, zmax = box.Get()  # 读取包围盒的XYZ最小值和最大值
        lower = np.array([xmin, ymin, zmin], dtype=np.float64)  # 把包围盒下界转换为双精度向量
        upper = np.array([xmax, ymax, zmax], dtype=np.float64)  # 把包围盒上界转换为双精度向量
        center = (lower + upper) / 2.0  # 计算或保存模型包围盒中心
        scale = float(np.max(upper - lower) / 2.0)  # 计算或保存统一坐标缩放尺度
        return center, scale  # 返回模型中心和半最大边长尺度

    @staticmethod  # 该方法不依赖对象内部状态
    def _face_nurbs(face_occ: Any, center: np.ndarray,
                    scale: float) -> dict[str, Any]:

        converted = topods.Face(BRepBuilderAPI_NurbsConvert(face_occ).Shape())  # 把解析几何或样条几何统一转换为NURBS
        surface = geomconvert.SurfaceToBSplineSurface(BRep_Tool.Surface(converted))  # 取得统一的B-Spline曲面对象便于读取参数

        nu, nv = surface.NbUPoles(), surface.NbVPoles()  # 读取曲面U、V方向的控制点数量
        poles = np.empty((nu, nv, 3), dtype=np.float64)  # 分配或保存NURBS控制点坐标
        weights = np.empty((nu, nv), dtype=np.float64)  # 分配或保存每个控制点的有理权重
        for u in range(1, nu + 1):
            for v in range(1, nv + 1):
                poles[u - 1, v - 1] = _xyz(surface.Pole(u, v))  # 读取当前UV位置对应的曲面控制点
                weights[u - 1, v - 1] = surface.Weight(u, v)  # 读取当前曲面控制点的有理权重

        return {  # 返回当前面的NURBS参数字典
            "poles": _normalize(poles, center, scale).astype(np.float32),  # 保存NURBS控制点坐标
            "weights": weights.astype(np.float32),  # 保存NURBS有理控制点权重
            "u_knots": np.array(  # 保存曲面U方向唯一节点
                [surface.UKnot(i) for i in range(1, surface.NbUKnots() + 1)],  # 按顺序读取曲面U方向的唯一节点值
                dtype=np.float32,  # 指定紧凑且稳定的数组数据类型
            ),  # 完成把BRep面转换为NURBS并提取完整参数的数据构造
            "v_knots": np.array(  # 保存曲面V方向唯一节点
                [surface.VKnot(i) for i in range(1, surface.NbVKnots() + 1)],  # 按顺序读取曲面V方向的唯一节点值
                dtype=np.float32,  # 指定紧凑且稳定的数组数据类型
            ),  # 完成把BRep面转换为NURBS并提取完整参数的数据构造
            "u_multiplicities": np.array(  # 保存U方向各节点重数
                [surface.UMultiplicity(i) for i in range(1, surface.NbUKnots() + 1)],  # 按顺序读取曲面U方向的节点重数
                dtype=np.int16,  # 指定紧凑且稳定的数组数据类型
            ),  # 完成把BRep面转换为NURBS并提取完整参数的数据构造
            "v_multiplicities": np.array(  # 保存V方向各节点重数
                [surface.VMultiplicity(i) for i in range(1, surface.NbVKnots() + 1)],  # 按顺序读取曲面V方向的节点重数
                dtype=np.int16,  # 指定紧凑且稳定的数组数据类型
            ),  # 完成把BRep面转换为NURBS并提取完整参数的数据构造
            "u_degree": int(surface.UDegree()),  # 保存曲面U方向次数
            "v_degree": int(surface.VDegree()),  # 保存曲面V方向次数
            "u_periodic": bool(surface.IsUPeriodic()),  # 标记曲面U方向是否周期闭合
            "v_periodic": bool(surface.IsVPeriodic()),  # 标记曲面V方向是否周期闭合
        }  # 完成把BRep面转换为NURBS并提取完整参数的数据构造

    @staticmethod  # 该方法不依赖对象内部状态
    def _edge_nurbs(edge_occ: Any, center: np.ndarray,
                    scale: float) -> dict[str, Any]:

        converted = topods.Edge(BRepBuilderAPI_NurbsConvert(edge_occ).Shape())  # 把解析几何或样条几何统一转换为NURBS
        curve, first, last = BRep_Tool.Curve(converted)  # 完成把BRep边转换为NURBS并提取完整参数中的当前计算
        curve = geomconvert.CurveToBSplineCurve(curve)  # 取得统一的B-Spline曲线对象便于读取参数

        poles = np.array(  # 分配或保存NURBS控制点坐标
            [_xyz(curve.Pole(i)) for i in range(1, curve.NbPoles() + 1)],  # 读取当前曲线控制点的空间坐标
            dtype=np.float64,  # 指定紧凑且稳定的数组数据类型
        )  # 完成把BRep边转换为NURBS并提取完整参数的数据构造
        weights = np.array(  # 分配或保存每个控制点的有理权重
            [curve.Weight(i) for i in range(1, curve.NbPoles() + 1)],  # 读取当前曲线控制点的有理权重
            dtype=np.float32,  # 指定紧凑且稳定的数组数据类型
        )  # 完成把BRep边转换为NURBS并提取完整参数的数据构造
        return {  # 返回当前边的NURBS参数字典
            "poles": _normalize(poles, center, scale).astype(np.float32),  # 保存NURBS控制点坐标
            "weights": weights,  # 保存NURBS有理控制点权重
            "knots": np.array(  # 保存曲线唯一节点
                [curve.Knot(i) for i in range(1, curve.NbKnots() + 1)],  # 按顺序读取曲线的唯一节点值
                dtype=np.float32,  # 指定紧凑且稳定的数组数据类型
            ),  # 完成把BRep边转换为NURBS并提取完整参数的数据构造
            "multiplicities": np.array(  # 保存曲线各节点重数
                [curve.Multiplicity(i) for i in range(1, curve.NbKnots() + 1)],  # 按顺序读取曲线节点重数
                dtype=np.int16,  # 指定紧凑且稳定的数组数据类型
            ),  # 完成把BRep边转换为NURBS并提取完整参数的数据构造
            "degree": int(curve.Degree()),  # 保存曲线次数
            "periodic": bool(curve.IsPeriodic()),  # 标记曲线是否周期闭合
            "param_range": np.array([first, last], dtype=np.float32),  # 保存边曲线的有效参数范围
        }  # 完成把BRep边转换为NURBS并提取完整参数的数据构造

    def _face_target(self, face: Any, center: np.ndarray,
                     scale: float) -> np.ndarray:

        points = uvgrid(face, self.face_samples, self.face_samples, method="point")  # 在参数域中采样实际几何坐标
        normals = uvgrid(face, self.face_samples, self.face_samples, method="normal")  # 在参数域中采样单位法向量
        inside = uvgrid(face, self.face_samples, self.face_samples, method="inside")  # 采样点是否位于裁剪面内部
        values = np.concatenate([_normalize(points, center, scale), normals, inside], axis=2)  # 组合当前实体需要输出的数值通道
        return np.transpose(values, (2, 0, 1)).astype(np.float32)  # 返回通道顺序为坐标、法向、掩码的面数组

    def _edge_target(self, body: Any, edge: Edge, center: np.ndarray,
                     scale: float) -> tuple[np.ndarray, bool]:

        faces = list(body.faces_from_edge(edge))  # 按拓扑顺序收集模型中的全部面
        data = EdgeDataExtractor(  # 保存读取或提取的主干特征字典
            edge, faces, num_samples=self.edge_samples, use_arclength_params=True  # 完成采样边及相邻面方向信息作为监督目标中的当前计算
        )  # 完成采样边及相邻面方向信息作为监督目标的数据构造
        if not data.good:  # 判断是否满足采样边及相邻面方向信息作为监督目标的条件
            return np.zeros((12, self.edge_samples), dtype=np.float32), False  # 返回十二通道边数组及其有效标记

        values = np.concatenate(  # 组合当前实体需要输出的数值通道
            [  # 开始组织采样边及相邻面方向信息作为监督目标的数据
                _normalize(data.points, center, scale),  # 补充采样边及相邻面方向信息作为监督目标所需参数
                data.tangents,  # 补充采样边及相邻面方向信息作为监督目标所需参数
                data.left_normals,  # 补充采样边及相邻面方向信息作为监督目标所需参数
                data.right_normals,  # 补充采样边及相邻面方向信息作为监督目标所需参数
            ],  # 完成采样边及相邻面方向信息作为监督目标的数据构造
            axis=1,  # 沿通道方向拼接各类边几何特征
        )  # 完成采样边及相邻面方向信息作为监督目标的数据构造
        return values.T.astype(np.float32), True  # 返回十二通道边数组及其有效标记

    @staticmethod  # 该方法不依赖对象内部状态
    def _stack(items: list[np.ndarray], shape: tuple[int, ...]) -> np.ndarray:

        return np.stack(items) if items else np.empty((0, *shape), dtype=np.float32)  # 返回堆叠数组或指定形状的空数组


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):  # 数组不能直接交给标准JSON编码器
        return value.tolist()  # 把任意维数组转换为嵌套列表
    if isinstance(value, np.generic):  # NumPy标量同样不是Python原生标量
        return value.item()  # 转换为int、float或bool等原生类型
    if isinstance(value, dict):  # 字典内部可能继续包含NumPy对象
        return {key: _json_ready(item) for key, item in value.items()}  # 递归转换每个字段
    if isinstance(value, (list, tuple)):  # 列表和元组内部也可能包含数组
        return [_json_ready(item) for item in value]  # 递归转换每个序列元素
    return value  # 字符串和Python标量可直接写入JSON


def main():
    parser = argparse.ArgumentParser(description="提取 NURBS-BRep 自监督训练数据")
    parser.add_argument("step_file", nargs="?", type=Path, default='_model1.stp', help="输入 .step/.stp 文件")  # 省略路径时自动读取_model1.stp
    parser.add_argument("-o", "--output", type=Path, help="可选的JSON输出路径") 
    parser.add_argument("--face-samples", type=int, default=10)
    parser.add_argument("--edge-samples", type=int, default=10) 
    args = parser.parse_args() 

    data = NewGraphExtractor(  # 保存读取或提取的主干特征字典
        args.step_file, args.face_samples, args.edge_samples  # 完成解析命令行参数并运行当前程序中的当前计算
    ).process()  # 完成解析命令行参数并运行当前程序中的当前计算
    output = args.output or DEFAULT_OUTPUT_DIR / f"{args.step_file.stem}_graph.json"  # 按模型名称生成默认JSON文件名
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream: 
        json.dump(_json_ready(data), stream, ensure_ascii=False, indent=2) 
    print(f"特征已保存：{output}") 


if __name__ == "__main__":  
    main()
