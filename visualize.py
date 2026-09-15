
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Display.SimpleGui import init_display
from OCC.Extend.TopologyUtils import TopologyExplorer


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_STEP = SCRIPT_DIR / "_model1.stp"


def _same_face_index(face: Any, faces: list[Any]) -> int | None:
    """寻找传递出的 ADVANCED_FACE 在完整模型中的位置。"""
    for index, candidate in enumerate(faces):
        if face.IsSame(candidate):
            return index
    return None


def load_step_faces(step_path: Path) -> tuple[Any, list[Any], dict[int, str]]:
    """读取 STEP，并建立“OCC面位置 -> #ADVANCED_FACE实体号”映射。"""
    reader = STEPControl_Reader()
    if reader.ReadFile(str(step_path)) != IFSelect_RetDone:
        raise RuntimeError(f"STEP 文件读取失败：{step_path}")
    if reader.TransferRoots() == 0:
        raise RuntimeError(f"STEP 文件没有可传递的模型：{step_path}")

    # 先保存完整模型；后面会逐个传递 ADVANCED_FACE 以取得对应 TopoDS_Face。
    shape = reader.OneShape()
    faces = list(TopologyExplorer(shape, ignore_orientation=True).faces())
    model = reader.WS().Model()
    step_ids: dict[int, str] = {}

    for entity_index in range(1, model.NbEntities() + 1):
        entity = model.Value(entity_index)
        if "AdvancedFace" not in entity.DynamicType().Name():
            continue

        before = reader.NbShapes()
        if not reader.TransferEntity(entity) or reader.NbShapes() <= before:
            continue
        transferred_face = reader.Shape(reader.NbShapes())
        face_index = _same_face_index(transferred_face, faces)
        if face_index is not None:
            # StringLabel 返回 STEP 文件中的真实标签，例如 #104。
            step_ids[face_index] = model.StringLabel(entity).ToCString()

    return shape, faces, step_ids


def load_json_face_ids(json_path: Path | None) -> dict[int, int]:
    """读取“面遍历位置 -> JSON face_id”映射。"""
    if json_path is None or not json_path.is_file():
        return {}
    with json_path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)

    result: dict[int, int] = {}
    if isinstance(data, dict) and isinstance(data.get("faces"), list):
        for row in data["faces"]:

            face_index = int(row.get("graph_node_id", row.get("attribute_row", 0)))
            result[face_index] = int(row.get("face_id", face_index))
        return result

    # 同时兼容 MFTReNet 的 [uid, data] 图 JSON。
    if isinstance(data, list) and len(data) == 2:
        data = data[1]
    if isinstance(data, dict):
        ids = data.get("graph_face_ids", data.get("graph", {}).get("face_ids", []))
        result = {index: int(face_id) for index, face_id in enumerate(ids)}
    return result


def _face_center(face: Any) -> Any:
    """使用面积质心放置面标签。"""
    properties = GProp_GProps()
    brepgprop.SurfaceProperties(face, properties)
    return properties.CentreOfMass()


def visualize(step_path: Path, json_path: Path | None, image_path: Path) -> None:
    """显示模型、标注面编号，并把当前视图保存为图片。"""
    if not step_path.is_file():
        raise FileNotFoundError(f"找不到 STEP 文件：{step_path}")

    shape, faces, step_ids = load_step_faces(step_path)
    json_ids = load_json_face_ids(json_path)

    print(f"STEP：{step_path}")
    print(f"面数：{len(faces)}")
    print("面编号对应关系：")
    for face_index in range(len(faces)):
        step_id = step_ids.get(face_index, "?")
        json_id = json_ids.get(face_index, "-")
        print(f"  STEP:{step_id}  JSON:{json_id}")

    display, start_display, _, _ = init_display()
    display.DisplayShape(
        shape,
        color=Quantity_Color(0.78, 0.82, 0.88, Quantity_TOC_RGB),
        transparency=0.20,
        update=False,
    )

    # 只显示面标签，不显示任何边编号。
    for face_index, face in enumerate(faces):
        step_id = step_ids.get(face_index, "?")
        json_id = json_ids.get(face_index, "-")
        label = f"F STEP:{step_id} JSON:{json_id}"
        display.DisplayMessage(
            _face_center(face),
            label,
            height=17,
            message_color=(0.85, 0.05, 0.05),
            update=False,
        )

    display.FitAll()
    display.Repaint()

    # Dump 会保存当前 OCC 视窗，面标签也会一并写入图片。
    image_path.parent.mkdir(parents=True, exist_ok=True)
    if not display.View.Dump(str(image_path)):
        raise RuntimeError(f"图片保存失败：{image_path}")
    print(f"图片已保存：{image_path}")

    start_display()


def main() -> None:
    parser = argparse.ArgumentParser(description="显示 STEP 的真实面实体号与 JSON 编号")
    parser.add_argument("step_file", nargs="?", type=Path, default=DEFAULT_STEP)
    parser.add_argument(
        "--json",
        type=Path,
        help="面编号 JSON；默认自动查找 <STEP文件名>_faces.json",
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="输出图片路径；默认保存为 <STEP文件名>_faces.png",
    )
    args = parser.parse_args()

    step_path = args.step_file.resolve()
    json_path = (
        args.json or step_path.with_name(f"{step_path.stem}_faces.json")
    ).resolve()
    image_path = (
        args.image or step_path.with_name(f"{step_path.stem}_faces.png")
    ).resolve()
    visualize(step_path, json_path, image_path)


if __name__ == "__main__":
    main()
