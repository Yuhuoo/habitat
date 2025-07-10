import os
import sys
import math
import argparse
import shortuuid
from typing import Sequence, Mapping

import numpy as np
import torch
import trimesh
from tqdm import tqdm

# -------------------------------------------------------------
# 🟢 project-specific paths — make sure these are reachable
# -------------------------------------------------------------
# sys.path.append("/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/xieyuan-24039/gls")

from segmentator import segment_point, compute_vn
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, tokenizer_special_token
from llava.pc_utils import referseg_transform_eval, Compose
from llava.conversation import conv_templates
from llava.utils import disable_torch_init
from pointgroup_ops import voxelization_idx

from collections.abc import Sequence, Mapping
from torch.utils.data.dataloader import default_collate

class ModelWrapper:
    def __init__(self, model_path, model_base, pointcloud_tower_name=None, conv_mode="llava_v1", temperature=0.2, top_p=None, num_beams=1):
        disable_torch_init()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conv_mode = conv_mode
        self.temperature = temperature
        self.top_p = top_p
        self.num_beams = num_beams

        model_name = get_model_name_from_path(model_path)
        tokenizer, model, _, _ = load_pretrained_model(model_path, model_base, model_name, pointcloud_tower_name=pointcloud_tower_name)
        self.model = model.eval().to(self.device)
        self.tokenizer = tokenizer

        self.templates = "<image>\n Please output the segmentation mask according to the following description. \n{description}"

        print("[✓] 模型加载完成")

    def ponder_collate_fn(self, batch, max_point=-1):
        if not isinstance(batch, Sequence):
            raise TypeError(f"{type(batch)} is not supported.")

        if max_point > 0:
            accum = 0
            kept = []
            for sample in batch:
                num = sample["coord"].shape[0]
                if accum + num > max_point:
                    continue
                accum += num
                kept.append(sample)
            return self.ponder_collate_fn(kept)

        if isinstance(batch[0], torch.Tensor):
            return torch.cat(list(batch))
        if isinstance(batch[0], np.ndarray):
            return torch.from_numpy(np.concatenate(batch, axis=0))
        if isinstance(batch[0], str):
            return list(batch)

        if isinstance(batch[0], Sequence):
            for b in batch:
                b.append(torch.tensor([b[0].shape[0]]))
            collated = [self.ponder_collate_fn(samples) for samples in zip(*batch)]
            collated[-1] = torch.cumsum(collated[-1], dim=0).int()
            return collated

        if isinstance(batch[0], Mapping):
            collated = {k: self.ponder_collate_fn([d[k] for d in batch]) for k in batch[0]}
            for k in collated:
                if "offset" in k:
                    collated[k] = torch.cumsum(collated[k], dim=0)
            return collated

        return default_collate(batch)

    def preprocess_pointcloud(self, ply_path):
        mesh = trimesh.load_mesh(ply_path)
        coords = mesh.vertices
        colors = mesh.visual.vertex_colors[:, :3]
        vertices = torch.from_numpy(coords.astype(np.float32))
        normals = torch.from_numpy(compute_vn(mesh).astype(np.float32))
        edges = torch.from_numpy(mesh.edges.astype(np.int64))
        superpoint_mask = segment_point(vertices, normals, edges).numpy()

        transform = Compose(referseg_transform_eval)
        pc_data_dict = dict(coord=coords, color=colors, superpoint_mask=superpoint_mask)
        pc_data_dict = transform(pc_data_dict)

        grid_coord = pc_data_dict['grid_coord']
        grid_coord = torch.cat([torch.LongTensor(grid_coord.shape[0], 1).fill_(0), grid_coord], 1)
        pc_data_dict['grid_coord'] = grid_coord

        spatial_shape = np.clip((grid_coord.max(0)[0][1:] + 1).numpy(), 128, None)
        voxel_coords, p2v_map, v2p_map = voxelization_idx(grid_coord, 1, 4)

        for key in ["coord", "grid_coord", "feat", "offset", "condition"]:
            if key in pc_data_dict:
                pc_data_dict[key] = self.ponder_collate_fn([pc_data_dict[key]])

        if "feat" in pc_data_dict and pc_data_dict["feat"].dim() == 3 and pc_data_dict["feat"].shape[0] == 1:
            pc_data_dict["feat"] = pc_data_dict["feat"].squeeze(0)

        return coords, colors, pc_data_dict, voxel_coords, p2v_map, v2p_map, spatial_shape, superpoint_mask

    def inference(self, pc_data_dict, voxel_coords, p2v_map, v2p_map, spatial_shape, query_text, superpoint_mask):
        conv = conv_templates[self.conv_mode].copy()
        query = self.templates.format(description=query_text)
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_special_token(prompt, self.tokenizer, return_tensors='pt').unsqueeze(0).to(self.device)
        coord = pc_data_dict["coord"].to(self.device, dtype=torch.bfloat16)
        offset = pc_data_dict["offset"].to(self.device)
        feat = pc_data_dict["feat"].to(self.device, dtype=torch.bfloat16)
        voxel_coords = voxel_coords.to(self.device)
        p2v_map = p2v_map.to(self.device)
        v2p_map = v2p_map.to(self.device)
        superpoint_tensor = [torch.tensor(superpoint_mask).to(self.device)]

        with torch.inference_mode():
            pred_mask = self.model.generate(
                input_ids,
                coord=coord,
                grid_coord=voxel_coords,
                offset=offset,
                feat=feat,
                p2v_map=p2v_map,
                v2p_map=v2p_map,
                spatial_shape=spatial_shape,
                superpoint_mask=superpoint_tensor,
                conditions=pc_data_dict["condition"],
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                top_p=self.top_p,
                num_beams=self.num_beams,
                max_new_tokens=64,
                tokenizer=self.tokenizer,
                click_mask=[[]],
                use_cache=True
            )
        return pred_mask.cpu().numpy().astype(bool)[0]

    def visualize_and_save(self, coords, colors, pred_mask, output_path):
        new_colors = colors.copy()
        highlight = np.array([255, 0, 0], dtype=np.uint8)
        new_colors[pred_mask] = highlight
        pc = trimesh.PointCloud(vertices=coords, colors=new_colors)
        pc.export(output_path)
        print(f"[✓] 可视化结果已保存到 {output_path}")

if __name__ == "__main__":
    model = ModelWrapper(
        model_path="/inspire/hdd/global_user/xieyuan-24039/sjm/3D-LLaVA/checkpoints/finetune-3d-llava-lora-0611-V1-bs-2-gpu8",
        model_base="/inspire/hdd/global_user/xieyuan-24039/sjm/LLaVA-3D-bf/llava-v1.5-7b",
        conv_mode="llava_v1"
    )

    ply_path = '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/xieyuan-24039/gls/segmentator/gls_add/pointcloud_mesh.ply'
    coords, colors, pc_data, voxel_coords, p2v_map, v2p_map, spatial_shape, sp_mask = model.preprocess_pointcloud(ply_path)

    while True:
        try:
            query = input("\n请输入目标描述（输入 exit 退出）：\n> ")
            if query.lower() == "exit":
                break
            pred = model.inference(pc_data, voxel_coords, p2v_map, v2p_map, spatial_shape, query, sp_mask)
            output_file ="/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/xieyuan-24039/gls/segmentator/gls_add/demo_results/{}.ply".format(query)
            model.visualize_and_save(coords, colors, pred, output_file)
        except:
            continue
