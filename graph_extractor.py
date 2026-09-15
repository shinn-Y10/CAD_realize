"""把 STEP 三维模型转换成图结构和几何特征。

可以把一个 CAD 模型理解成“面和边组成的网络”：面是图中的节点，
相邻面之间的公共边是图中的连接。本文件负责读取模型、检查模型是否合法，
再提取每个面和每条边的类型、尺寸以及采样点，供后续神经网络使用。
"""

# NumPy 用于拼接、转置和创建几何采样数组。
import numpy as np
# 下面两个标准库在当前文件中暂未直接使用，保留它们是为了兼容原项目代码。
import json
from pathlib import Path

# occwl 对 Open Cascade 做了一层更易用的 Python 封装。
from occwl.compound import Compound
from occwl.solid import Solid
from occwl.graph import face_adjacency
from occwl.uvgrid import uvgrid, ugrid
from occwl.edge import Edge
from occwl.face import Face
from occwl.edge_data_extractor import EdgeDataExtractor, EdgeConvexity

# OCC 提供底层 B-Rep 几何、拓扑遍历和曲面/曲线类型判断能力。
from OCC.Core.BRep import BRep_Tool
from OCC.Extend import TopologyUtils
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCC.Core.GeomAbs import (GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
                              GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BezierSurface,
                              GeomAbs_BSplineSurface, GeomAbs_SurfaceOfRevolution,
                              GeomAbs_Line, GeomAbs_Circle, GeomAbs_Ellipse)
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop_LinearProperties, brepgprop_SurfaceProperties
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.BRepCheck import BRepCheck_Analyzer


class TopologyChecker:
    """检查 STEP 模型的拓扑是否适合继续提取图数据。"""

    # 这部分检查逻辑修改自 BRepNet：
    # https://github.com/AutodeskAILab/BRepNet/blob/master/pipeline/extract_brepnet_data_from_step.py
    def __init__(self):
        """创建一个不保存额外状态的拓扑检查器。"""

        # 当前检查器不需要保存额外状态，因此构造函数留空。
        pass

    def find_edges_from_wires(self, top_exp):
        """收集所有轮廓线（wire）中实际使用到的边。"""

        # 用集合去重，同一条边即使被遍历到多次也只保留一份。
        edge_set = set()
        for wire in top_exp.wires():
            # WireExplorer 会按照边在轮廓中的连接顺序进行遍历。
            wire_exp = TopologyUtils.WireExplorer(wire)
            for edge in wire_exp.ordered_edges():
                edge_set.add(edge)
        return edge_set

    def find_edges_from_top_exp(self, top_exp):
        """直接收集整个模型拓扑中登记的全部边。"""

        edge_set = set(top_exp.edges())
        return edge_set

    def check_closed(self, body):
        """检查模型是否闭合，即有没有没有被任何轮廓使用的开口边。"""

        # 若一条边出现在模型边列表里，却没有出现在任何 wire 中，
        # 通常说明它没有与周围面正确连接，模型可能存在开口。
        top_exp = TopologyUtils.TopologyExplorer(body, ignore_orientation=False)
        edges_from_wires = self.find_edges_from_wires(top_exp)
        edges_from_top_exp = self.find_edges_from_top_exp(top_exp)
        # “全部边 - 轮廓中的边”就是没有被轮廓使用的可疑边。
        missing_edges = edges_from_top_exp - edges_from_wires
        return len(missing_edges) == 0

    def check_manifold(self, top_exp):
        """检查模型是否为流形，避免同一个面被多个壳重复占用。"""

        faces = set()
        for shell in top_exp.shells():
            for face in top_exp._loop_topo(TopAbs_FACE, shell):
                # 同一个面第二次出现，说明模型的连接关系不符合这里的处理要求。
                if face in faces:
                    return False
                faces.add(face)
        return True

    def check_unique_coedges(self, top_exp):
        """检查“边 + 方向”的组合在各轮廓中是否唯一。"""

        # coedge 可理解为“带方向的边”；同一几何边正向和反向是两个不同组合。
        coedge_set = set()
        for loop in top_exp.wires():
            wire_exp = TopologyUtils.WireExplorer(loop)
            for coedge in wire_exp.ordered_edges():
                orientation = coedge.Orientation()
                tup = (coedge, orientation)
                # 如果完全相同的带方向边再次出现，说明多个轮廓重复使用了它。
                if tup in coedge_set:
                    return False
                coedge_set.add(tup)
        return True

    def __call__(self, body):
        """依次执行全部检查；全部通过时返回 True。"""

        # 这里忽略面的方向，只关心模型中有哪些拓扑对象。
        top_exp = TopologyUtils.TopologyExplorer(body, ignore_orientation=True)
        # 连一个面都没有的对象不能作为有效三维模型处理。
        if top_exp.number_of_faces() == 0:
            print('Empty shape')
            return False
        # 先使用 OCC 自带工具检查基本的拓扑和几何合法性。
        analyzer = BRepCheck_Analyzer(body)
        if not analyzer.IsValid(body):
            print('BRepCheck_Analyzer found defects')
            return False
        # 再执行本项目额外要求的三项检查：流形、闭合、coedge 唯一。
        if not self.check_manifold(top_exp):
            print("Non-manifold bodies are not supported")
            return False
        if not self.check_closed(body):
            print("Bodies which are not closed are not supported")
            return False
        if not self.check_unique_coedges(top_exp):
            print("Bodies where the same coedge is uses in multiple loops are not supported")
            return False
        return True


class GraphExtractor:
    """从一个 STEP 文件中提取面邻接图及面、边的数值特征。"""

    def __init__(self, step_file, attribute_config, scale_body=True):
        """保存输入路径和特征配置，并读取 UV 采样网格的大小。"""

        # STEP 文件是要处理的 CAD 模型；配置决定具体提取哪些特征。
        self.step_file = step_file
        self.attribute_config = attribute_config
        # 为 True 时，模型会按比例缩放进单位包围盒，便于不同大小模型统一训练。
        self.scale_body = scale_body

        # 配置中包含 “UV-grid” 时，除了普通属性，还会采样曲面和曲线上的点。
        self.use_uv = "UV-grid" in self.attribute_config.keys()
        self.checker = TopologyChecker()
        if self.use_uv:
            # 曲面使用 U、V 两个参数方向，因此需要两个采样数量。
            self.num_srf_u = self.attribute_config["UV-grid"]["num_srf_u"]
            self.num_srf_v = self.attribute_config["UV-grid"]["num_srf_v"]
            # 曲线只有一个参数方向，因此只需要一个采样数量。
            self.num_crv_u = self.attribute_config["UV-grid"]["num_crv_u"]
        # 模型会在 process() 中真正加载。
        self.body = None

    def process(self):
        """执行完整流程，返回可直接保存或送入图神经网络的数据字典。"""

        # 第一步：从 STEP 文件读取模型，并在提取特征前检查拓扑质量。
        self.body = Compound.load_from_step(self.step_file)
        assert self.body is not None, "the shape {} is non-manifold or open".format(self.step_file)
        assert self.checker(self.body.topods_shape()), "the shape {} has wrong topology".format(self.step_file)

        # 可选地将最长边缩放到统一范围，同时保持模型原来的长宽高比例。
        if self.scale_body:
            self.body = self.body.scale_to_unit_box(copy=True)
            # print(self.body.volume())

        # 第二步：以“面”为节点，以“两个面共享的边”为连接，建立邻接图。
        try:
            graph = face_adjacency(self.body)
        except Exception as e:
            # 建图失败通常意味着输入模型仍存在异常拓扑，打印底层错误便于排查。
            print(e)
            assert False, 'Wrong shape {} when create face adjacency'.format(self.step_file)

        # 第三步：分别保存每个面的普通属性和 UV 网格特征。
        graph_face_attr = []
        graph_face_grid = []
        # 通常一个配置项产生一个数值；面心会产生 x、y、z 三个数值，
        # 所以启用 FaceCentroidAttribute 后，实际长度要在配置项数量上再加 2。
        len_of_face_attr = len(self.attribute_config["face_attributes"]) + 2 if "FaceCentroidAttribute" in \
                                                                                self.attribute_config[
                                                                                    "face_attributes"] else 0
        for face_idx in graph.nodes:
            # 从图节点中取出 occwl.Face，再转为底层 OCC 面对象供属性函数使用。
            face = graph.nodes[face_idx]["face"]
            face_occ = face.topods_shape()

            # occwl 无法正常解析的曲面可能返回 float；这种异常面直接跳过。
            if type(face.surface()) is float:
                continue

            # 根据配置依次提取曲面类型、面积、面心等属性。
            face_attr = self.extract_attributes_from_face(face_occ)
            # 属性数量不一致往往代表配置名称写错或提取函数漏返回了数据。
            assert len_of_face_attr == len(face_attr)
            graph_face_attr.append(face_attr)

            # 若开启 UV 特征，则在曲面上按规则网格采样点、法向量和裁剪掩码。
            if self.use_uv and self.num_srf_u and self.num_srf_v:
                uv_grid = self.extract_face_point_grid(face)
                # 7 个通道分别是 xyz、法向量 ijk 和 inside mask。
                assert uv_grid.shape[0] == 7
                graph_face_grid.append(uv_grid.tolist())

        # 第四步：分别保存每条邻接边的普通属性和沿曲线采样的网格特征。
        graph_edge_attr = []
        graph_edge_grid = []
        for edge_idx in graph.edges:
            edge = graph.edges[edge_idx]["edge"]
            edge_occ = edge.topods_shape()
            # 退化边没有可采样的真实曲线，例如圆锥顶点处长度为零的边，应跳过。
            if not edge.has_curve():
                continue

            # 根据配置提取边的凹凸性、长度和曲线类型等属性。
            edge_attr = self.extract_attributes_from_edge(edge_occ)
            assert len(self.attribute_config["edge_attributes"]) == len(edge_attr)
            graph_edge_attr.append(edge_attr)
            # 若开启 UV 特征，则沿边采样点、切向量及左右相邻面的法向量。
            if self.use_uv and self.num_crv_u:
                u_grid = self.extract_edge_point_grid(edge)
                # 12 个通道 = 3 点坐标 + 3 切向量 + 3 左法向量 + 3 右法向量。
                assert u_grid.shape[0] == 12
                graph_edge_grid.append(u_grid.tolist())

        # 第五步：把 NetworkX 图改写成更容易序列化的“起点列表 + 终点列表”。
        edges = list(graph.edges)
        src = [e[0] for e in edges]
        dst = [e[1] for e in edges]
        graph = {
            'edges': (src, dst),
            'num_nodes': len(graph.nodes)
        }

        # 最终结果同时包含图连接关系、面特征和边特征。
        return {
            'graph': graph,
            'graph_face_attr': graph_face_attr,
            'graph_face_grid': graph_face_grid,
            'graph_edge_attr': graph_edge_attr,
            'graph_edge_grid': graph_edge_grid,
        }

    def extract_attributes_from_face(self, face_occ) -> list:
        """按配置提取一个面的类型、面积和中心点等特征。"""

        # 下列类型函数采用 one-hot 思路：属于该曲面类型返回 1.0，否则返回 0.0。
        def plane_attribute(face):
            """判断当前面是不是平面。"""
            surf_type = BRepAdaptor_Surface(face).GetType()
            if surf_type == GeomAbs_Plane:
                return 1.0
            return 0.0

        def cylinder_attribute(face):
            """判断当前面是不是圆柱面。"""
            surf_type = BRepAdaptor_Surface(face).GetType()
            if surf_type == GeomAbs_Cylinder:
                return 1.0
            return 0.0

        def cone_attribute(face):
            """判断当前面是不是圆锥面。"""
            surf_type = BRepAdaptor_Surface(face).GetType()
            if surf_type == GeomAbs_Cone:
                return 1.0
            return 0.0

        def sphere_attribute(face):
            """判断当前面是不是球面。"""
            surf_type = BRepAdaptor_Surface(face).GetType()
            if surf_type == GeomAbs_Sphere:
                return 1.0
            return 0.0

        def torus_attribute(face):
            """判断当前面是不是圆环面。"""
            surf_type = BRepAdaptor_Surface(face).GetType()
            if surf_type == GeomAbs_Torus:
                return 1.0
            return 0.0

        def revolution_attribute(face):
            """判断当前面是不是由一条曲线绕轴旋转生成的旋转面。"""
            surf_type = BRepAdaptor_Surface(face).GetType()
            if surf_type == GeomAbs_SurfaceOfRevolution:
                return 1.0
            return 0.0

        def extrusion_attribute(face):
            """判断当前面是不是由截面沿直线拉伸生成的拉伸面。"""
            if Face(face).surface_type() == "extrusion":
                return 1.0
            return 0.0

        def offset_attribute(face):
            """判断当前面是不是从另一个面等距偏移得到的偏置面。"""
            if Face(face).surface_type() == "offset":
                return 1.0
            return 0.0

        def other_attribute(face):
            """标记不属于上述已知类别的其他曲面。"""
            if Face(face).surface_type() == "other":
                return 1.0
            return 0.0

        def area_attribute(face):
            """计算面的实际面积。"""

            # OCC 把面积保存在通用质量属性对象的 Mass 字段中。
            geometry_properties = GProp_GProps()
            brepgprop_SurfaceProperties(face, geometry_properties)
            return geometry_properties.Mass()

        def rational_nurbs_attribute(face):
            """判断 Bezier/B-Spline 曲面是否使用了有理权重。"""

            # 只有 Bezier 和 B-Spline 曲面才需要继续检查控制点权重。
            surf = BRepAdaptor_Surface(face)
            if surf.GetType() == GeomAbs_BSplineSurface:
                bspline = surf.BSpline()
            elif surf.GetType() == GeomAbs_BezierSurface:
                bspline = surf.Bezier()
            else:
                bspline = None

            # U 或 V 任一方向为有理形式，就把该面标记为有理 NURBS 面。
            if bspline is not None:
                if bspline.IsURational() or bspline.IsVRational():
                    return 1.0
            return 0.0

        def centroid_attribute(face):
            """计算面的几何中心，返回 x、y、z 三个坐标。"""

            mass_props = GProp_GProps()
            brepgprop_SurfaceProperties(face, mass_props)
            gPt = mass_props.CentreOfMass()

            return gPt.Coord()

        # 严格按照配置文件中的顺序组装特征，保证模型输入列顺序固定。
        face_attributes = []
        for attribute in self.attribute_config["face_attributes"]:
            # 每个配置名都映射到上面一个具体的提取函数。
            if attribute == "Plane":
                face_attributes.append(plane_attribute(face_occ))
            elif attribute == "Cylinder":
                face_attributes.append(cylinder_attribute(face_occ))
            elif attribute == "Cone":
                face_attributes.append(cone_attribute(face_occ))
            elif attribute == "Sphere":
                face_attributes.append(sphere_attribute(face_occ))
            elif attribute == "Torus":
                face_attributes.append(torus_attribute(face_occ))
            elif attribute == "Revolution":
                face_attributes.append(revolution_attribute(face_occ))
            elif attribute == "Extrusion":
                face_attributes.append(extrusion_attribute(face_occ))
            elif attribute == "Offset":
                face_attributes.append(offset_attribute(face_occ))
            elif attribute == "Other":
                face_attributes.append(other_attribute(face_occ))
            elif attribute == "FaceAreaAttribute":
                face_attributes.append(area_attribute(face_occ))
            elif attribute == "RationalNurbsFaceAttribute":
                face_attributes.append(rational_nurbs_attribute(face_occ))
            elif attribute == "FaceCentroidAttribute":
                # 面心有三个坐标，因此要用 extend 展开，而不是作为一个元组 append。
                face_attributes.extend(centroid_attribute(face_occ))
            else:
                # 遇到未知名称立即停止，防止悄悄生成顺序错误的训练数据。
                assert False, "Unknown face attribute"
        return face_attributes

    def extract_face_point_grid(self, face) -> np.array:
        """在一个面上生成 UV-Net 使用的规则采样网格。

        返回形状为 ``[7, num_pts_u, num_pts_v]`` 的数组。7 个通道依次为：
        点坐标 x/y/z、法向量 i/j/k，以及该采样点是否在裁剪边界内的掩码。
        """
        # 使用完全相同的 U、V 位置，分别取得空间点、法向量和有效区域掩码。
        points = uvgrid(face, self.num_srf_u, self.num_srf_v, method="point")  # num_u * num_v * 3
        normals = uvgrid(face, self.num_srf_u, self.num_srf_v, method="normal")  # # num_u * num_v * 3
        mask = uvgrid(face, self.num_srf_u, self.num_srf_v, method="inside")  # # num_u * num_v * 1

        # 在最后一个维度拼接后，形状是 [num_u, num_v, 7]。
        single_grid = np.concatenate([points, normals, mask], axis=2)

        # 神经网络通常要求“通道在前”，所以转换为 [7, num_u, num_v]。
        return np.transpose(single_grid, (2, 0, 1))

    def extract_attributes_from_edge(self, edge_occ) -> list:
        """按配置提取一条边的凹凸性、长度和曲线类型等特征。"""

        def find_edge_convexity(edge, faces):
            """结合边两侧的面，判断这条边是凸边、凹边还是平滑连接。"""

            # EdgeDataExtractor 会在边上取样，并根据左右两侧面的法向量判断夹角。
            edge_data = EdgeDataExtractor(Edge(edge),
                                          faces, use_arclength_params=False)
            if not edge_data.good:
                # 球面极点等退化边无法获得可靠数据，这里用 0.0 作为兜底值。
                print("edge data not good")
                return 0.0
            # 小于 5 度的法向变化会被视为平滑连接；数值单位是弧度。
            angle_tol_rads = 0.0872664626
            convexity = edge_data.edge_convexity(angle_tol_rads)
            return convexity

        def convexity_attribute(convexity, attribute):
            """把枚举形式的凹凸性转换成配置需要的布尔特征。"""

            if attribute == "Convex edge":
                return convexity == EdgeConvexity.CONVEX
            if attribute == "Concave edge":
                return convexity == EdgeConvexity.CONCAVE
            if attribute == "Smooth":
                return convexity == EdgeConvexity.SMOOTH
            # 配置写入了未知凹凸类型时立即报错。
            assert False, "Unknown convexity"
            return 0.0

        def edge_length_attribute(edge):
            """计算边对应曲线的实际长度。"""

            # OCC 仍通过 Mass() 返回这里计算出的线性长度。
            geometry_properties = GProp_GProps()
            brepgprop_LinearProperties(edge, geometry_properties)
            return geometry_properties.Mass()

        def circular_edge_attribute(edge):
            """判断当前边是不是圆或圆弧。"""
            brep_adaptor_curve = BRepAdaptor_Curve(edge)
            curv_type = brep_adaptor_curve.GetType()
            if curv_type == GeomAbs_Circle:
                return 1.0
            return 0.0

        def closed_edge_attribute(edge):
            """判断当前边是否首尾闭合，例如完整的圆。"""
            if BRep_Tool().IsClosed(edge):
                return 1.0
            return 0.0

        def elliptical_edge_attribute(edge):
            """判断当前边是不是椭圆或椭圆弧。"""
            brep_adaptor_curve = BRepAdaptor_Curve(edge)
            curv_type = brep_adaptor_curve.GetType()
            if curv_type == GeomAbs_Ellipse:
                return 1.0
            return 0.0

        def straight_edge_attribute(edge):
            """判断当前边是不是直线段。"""
            brep_adaptor_curve = BRepAdaptor_Curve(edge)
            curv_type = brep_adaptor_curve.GetType()
            if curv_type == GeomAbs_Line:
                return 1.0
            return 0.0

        def hyperbolic_edge_attribute(edge):
            """判断当前边是不是双曲线。"""
            if Edge(edge).curve_type() == "hyperbola":
                return 1.0
            return 0.0

        def parabolic_edge_attribute(edge):
            """判断当前边是不是抛物线。"""
            if Edge(edge).curve_type() == "parabola":
                return 1.0
            return 0.0

        def bezier_edge_attribute(edge):
            """判断当前边是不是 Bezier 曲线。"""
            if Edge(edge).curve_type() == "bezier":
                return 1.0
            return 0.0

        def non_rational_bspline_edge_attribute(edge):
            """判断当前边是不是非有理 B-Spline 曲线。"""
            occwl_edge = Edge(edge)
            if occwl_edge.curve_type() == "bspline" and not occwl_edge.rational():
                return 1.0
            return 0.0

        def rational_bspline_edge_attribute(edge):
            """判断当前边是不是带权重的有理 B-Spline 曲线。"""
            occwl_edge = Edge(edge)
            if occwl_edge.curve_type() == "bspline" and occwl_edge.rational():
                return 1.0
            return 0.0

        def offset_edge_attribute(edge):
            """判断当前边是不是由另一条曲线偏移得到的曲线。"""
            if Edge(edge).curve_type() == "offset":
                return 1.0
            return 0.0

        def other_edge_attribute(edge):
            """标记不属于上述已知类别的其他曲线。"""
            if Edge(edge).curve_type() == "other":
                return 1.0
            return 0.0

        # 找出与当前边相邻的面；判断凹凸性和左右法向量都需要这些面。
        # top_exp = TopologyUtils.TopologyExplorer(self.body, ignore_orientation=True)
        # faces_of_edge = [Face(f) for f in top_exp.faces_from_edge(edge)]
        faces_of_edge = [f for f in self.body.faces_from_edge(Edge(edge_occ))]

        # 仅当配置确实需要凹凸性时才执行较耗时的几何计算。
        attribute_list = self.attribute_config["edge_attributes"]
        if "Concave edge" in attribute_list or \
                "Convex edge" in attribute_list or \
                "Smooth" in attribute_list:
            convexity = find_edge_convexity(edge_occ, faces_of_edge)

        # 按配置给出的固定顺序组装边特征。
        edge_attributes = []
        for attribute in attribute_list:
            # 每个配置名都映射到上面一个具体的提取函数。
            if attribute == "Concave edge":
                edge_attributes.append(convexity_attribute(convexity, attribute))
            elif attribute == "Convex edge":
                edge_attributes.append(convexity_attribute(convexity, attribute))
            elif attribute == "Smooth":
                edge_attributes.append(convexity_attribute(convexity, attribute))
            elif attribute == "EdgeLengthAttribute":
                edge_attributes.append(edge_length_attribute(edge_occ))
            elif attribute == "CircularEdgeAttribute":
                edge_attributes.append(circular_edge_attribute(edge_occ))
            elif attribute == "ClosedEdgeAttribute":
                edge_attributes.append(closed_edge_attribute(edge_occ))
            elif attribute == "EllipticalEdgeAttribute":
                edge_attributes.append(elliptical_edge_attribute(edge_occ))
            elif attribute == "StraightEdgeAttribute":
                edge_attributes.append(straight_edge_attribute(edge_occ))
            elif attribute == "HyperbolicEdgeAttribute":
                edge_attributes.append(hyperbolic_edge_attribute(edge_occ))
            elif attribute == "ParabolicEdgeAttribute":
                edge_attributes.append(parabolic_edge_attribute(edge_occ))
            elif attribute == "BezierEdgeAttribute":
                edge_attributes.append(bezier_edge_attribute(edge_occ))
            elif attribute == "NonRationalBSplineEdgeAttribute":
                edge_attributes.append(non_rational_bspline_edge_attribute(edge_occ))
            elif attribute == "RationalBSplineEdgeAttribute":
                edge_attributes.append(rational_bspline_edge_attribute(edge_occ))
            elif attribute == "OffsetEdgeAttribute":
                edge_attributes.append(offset_edge_attribute(edge_occ))
            elif attribute == "Other":
                edge_attributes.append(other_edge_attribute(edge_occ))
            else:
                # 未知名称说明配置与代码不匹配，继续运行会造成特征错位。
                assert False, "Unknown face attribute"
        return edge_attributes

    def extract_edge_point_grid(self, edge) -> np.array:
        """沿带方向的边生成形状为 ``[12, num_u]`` 的采样网格。

        12 个通道依次为点坐标、沿 coedge 方向的切向量、左侧面法向量和
        右侧面法向量，每一组都包含 x、y、z 三个分量。
        """

        # 先取得边相邻的面，后面需要用它们计算边左右两侧的法向量。
        # top_exp = TopologyUtils.TopologyExplorer(self.body, ignore_orientation=True)
        # faces_of_edge = [Face(f) for f in top_exp.faces_from_edge(edge)]
        faces_of_edge = [f for f in self.body.faces_from_edge(edge)]

        # 按弧长均匀取样，避免曲线参数不均匀导致采样点挤在某一段。
        edge_data = EdgeDataExtractor(edge, faces_of_edge, num_samples=self.num_crv_u, use_arclength_params=True)
        if not edge_data.good:
            # 球面极点等退化边没有可用几何数据，返回同形状的全零数组保持尺寸一致。
            return np.zeros((12, self.num_crv_u))

        # 每个采样点的四组三维向量在最后一个维度拼接，得到 [num_u, 12]。
        single_grid = np.concatenate(
            [
                edge_data.points,
                edge_data.tangents,
                edge_data.left_normals,
                edge_data.right_normals
            ],
            axis=1
        )

        # 转成神经网络常用的“通道在前”格式 [12, num_u]。
        return np.transpose(single_grid, (1, 0))
