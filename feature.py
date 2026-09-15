"""读取 STEP/STP 文件，并为每个面生成与 graph_extractor 一致的编号。"""

import argparse
import json
from pathlib import Path

from occwl.compound import Compound
from occwl.graph import face_adjacency


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_STEP = SCRIPT_DIR / "_model1.stp"


def load_and_number_faces(step_path, scale_body=True):
    step_path = Path(step_path).resolve()
    if not step_path.is_file():
        raise FileNotFoundError(f"找不到 STEP/STP 文件：{step_path}")
    if step_path.suffix.lower() not in {".step", ".stp"}:
        raise ValueError(f"文件扩展名不是 .step 或 .stp：{step_path.name}")

    # 将 STEP 文件中的 B-Rep 实体转换成 Open Cascade 的 Compound 对象。
    body = Compound.load_from_step(step_path)
    if body is None:
        raise RuntimeError(f"STEP 文件读取失败：{step_path}")

    # graph_extractor 默认也会执行相同的归一化，因此这里保持一致。
    if scale_body:
        body = body.scale_to_unit_box(copy=True)

    # face_adjacency 会遍历所有面，并通过 EntityMapper 从 0 开始编号。
    graph = face_adjacency(body)
    if graph is None:
        raise RuntimeError("无法建立面邻接图，STEP 可能不闭合或不是流形实体")

    faces = []
    for attribute_row, face_id in enumerate(sorted(graph.nodes)):
        face = graph.nodes[face_id]["face"]
        faces.append(
            {
                "face_id": int(face_id),
                "graph_node_id": int(face_id),
                "attribute_row": attribute_row,
                "surface_type": face.surface_type()
            }
        )

    return body, faces


def save_face_numbers(step_path, faces, output_path=None):
    """把面编号保存为 JSON，并返回输出文件路径。"""

    step_path = Path(step_path).resolve()
    if output_path is None:
        output_path = step_path.with_name(f"{step_path.stem}_faces.json")
    else:
        output_path = Path(output_path).resolve()

    data = {
        "step_file": step_path.name,
        "face_count": len(faces),
        "numbering_rule": "Open Cascade traversal order; face_id equals graph node id",
        "scaled_to_unit_box": True,
        "faces": faces
    }

    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False)

    return output_path


def number_step_file(step_path, output_path=None):
    """完成读取、面编号和 JSON 保存三个步骤。"""

    _, faces = load_and_number_faces(step_path, scale_body=True)
    saved_path = save_face_numbers(step_path, faces, output_path)
    return saved_path, faces


def main():
    parser = argparse.ArgumentParser(
        description="读取 STEP/STP，并为每个面生成与 graph_extractor 一致的编号"
    )
    parser.add_argument(
        "step_file",
        nargs="?",
        default=DEFAULT_STEP,
        help="要读取的 STEP/STP 文件；默认处理同目录的 _model1.stp"
    )
    parser.add_argument(
        "-o",
        "--output",
        help="面编号 JSON 的输出路径；默认输出为 <STEP文件名>_faces.json"
    )
    args = parser.parse_args()

    output_path, faces = number_step_file(args.step_file, args.output)

    print(f"读取完成：{Path(args.step_file).resolve()}")
    print(f"共识别出 {len(faces)} 个面")
    for face in faces:
        print(
            f"面 {face['face_id']}："
            f"图节点 {face['graph_node_id']}，"
            f"类型 {face['surface_type']}"
        )
    print(f"面编号已保存到：{output_path}")


if __name__ == "__main__":
    main()
