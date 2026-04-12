# Adapted from ARM
# Source: https://github.com/stepjam/ARM
# License: https://github.com/stepjam/ARM/LICENSE
from bift.helpers.preprocess_agent import PreprocessAgent

from bift.methods.bimanualshift.perceiver_lang_io import PerceiverVoxelLangEncoder
from bift.methods.bimanualshift.action_generator import ActionGenerator
from bift.methods.bimanualshift.qattention_peract_bc_agent import QAttentionPerActBCAgent
from bift.methods.bimanualshift.qattention_stack_agent import QAttentionStackAgent
from bift.methods.bimanualshift.visual_tracker import SkillManager, VisualTracker
from bift.methods.bimanualshift.skill_memory_bank import SkillMemoryBank
from omegaconf import DictConfig
import pickle
import torch
import os
from pathlib import Path
def create_agent(cfg: DictConfig):
    depth_0bounds = cfg.rlbench.scene_bounds
    cam_resolution = cfg.rlbench.camera_resolution

    num_rotation_classes = int(360.0 // cfg.method.rotation_resolution)
    qattention_agents = []

    repo_root = Path(__file__).resolve().parents[3]
    pkl_path = repo_root / "lang_token.pkl"
    with open(pkl_path, "rb") as f:
        embeddings_dict = pickle.load(f)
    flattened_embeddings = []
    for key in embeddings_dict.keys():
        embedding = torch.tensor(embeddings_dict[key]) 
        flattened_embedding = embedding.view(-1) 
        flattened_embeddings.append(flattened_embedding)
    embeddings_matrix = torch.stack(flattened_embeddings)  

    skill_manager = SkillManager(num_classes=18,embedding_matrix=embeddings_matrix)
    use_visual_tracker = bool(getattr(cfg.framework, "use_visual_tracker", False))
    visual_tracker = None
    if use_visual_tracker:
        visual_tracker = VisualTracker(
            backend=str(getattr(cfg.framework, "visual_tracker_backend", "grounded_sam")),
            enabled=True,
            repo_root=str(getattr(cfg.framework, "grounded_sam_repo", "")),
            device=str(getattr(cfg.framework, "visual_tracker_device", "cuda")),
            box_threshold=float(getattr(cfg.framework, "visual_tracker_box_threshold", 0.3)),
            text_threshold=float(getattr(cfg.framework, "visual_tracker_text_threshold", 0.25)),
            prompt=str(getattr(cfg.framework, "visual_tracker_prompt", "robotic arm.")),
            grounding_config=getattr(cfg.framework, "grounding_dino_config", None),
            grounding_checkpoint=getattr(cfg.framework, "grounding_dino_checkpoint", None),
            sam_checkpoint=getattr(cfg.framework, "sam_checkpoint", None),
            sam_version=str(getattr(cfg.framework, "sam_version", "vit_h")),
            bert_base_uncased_path=getattr(cfg.framework, "bert_base_uncased_path", None),
        )

    memory_bank = None
    if bool(getattr(cfg.framework, "use_memory_bank", False)):
        memory_bank = SkillMemoryBank(
            capacity=int(getattr(cfg.framework, "memory_bank_capacity", 256)),
            sim_threshold=float(getattr(cfg.framework, "memory_bank_sim_threshold", 0.65)),
            blend_alpha=float(getattr(cfg.framework, "memory_bank_blend_alpha", 0.8)),
        )
    action_generator = None
    if bool(getattr(cfg.framework, "use_action_generator", False)):
        action_generator = ActionGenerator(
            visual_dim=int(getattr(cfg.method, "final_dim", 64) * 2),
            lang_dim=int(getattr(cfg.method, "final_dim", 64) * 2),
            proprio_dim=int(getattr(cfg.method, "low_dim_size", 8)),
            hidden_dim=int(getattr(cfg.framework, "action_generator_hidden_dim", 256)),
        )
    for depth, vox_size in enumerate(cfg.method.voxel_sizes):
        last = depth == len(cfg.method.voxel_sizes) - 1
        if cfg.framework.use_skill:
            perceiver_encoder = PerceiverVoxelLangEncoder(
                depth=cfg.method.transformer_depth,
                iterations=cfg.method.transformer_iterations,
                voxel_size=vox_size,
                initial_dim=3 + 3 + 1 + 3,
                low_dim_size=cfg.method.low_dim_size,
                layer=depth,
                num_rotation_classes=num_rotation_classes if last else 0,
                num_grip_classes=2 if last else 0,
                num_collision_classes=2 if last else 0,
                input_axis=3,
                num_latents=cfg.method.num_latents,
                latent_dim=cfg.method.latent_dim,
                cross_heads=cfg.method.cross_heads,
                latent_heads=cfg.method.latent_heads,
                cross_dim_head=cfg.method.cross_dim_head,
                latent_dim_head=cfg.method.latent_dim_head,
                weight_tie_layers=False,
                activation=cfg.method.activation,
                pos_encoding_with_lang=cfg.method.pos_encoding_with_lang,
                input_dropout=cfg.method.input_dropout,
                attn_dropout=cfg.method.attn_dropout,
                decoder_dropout=cfg.method.decoder_dropout,
                lang_fusion_type=cfg.method.lang_fusion_type,
                voxel_patch_size=cfg.method.voxel_patch_size,
                voxel_patch_stride=cfg.method.voxel_patch_stride,
                no_skip_connection=cfg.method.no_skip_connection,
                no_perceiver=cfg.method.no_perceiver,
                no_language=cfg.method.no_language,
                final_dim=cfg.method.final_dim,
                skill_manager = skill_manager,
                visual_tracker=visual_tracker,
                action_generator=action_generator,
            )

        qattention_agent = QAttentionPerActBCAgent(
            layer=depth,
            coordinate_bounds=depth_0bounds,
            perceiver_encoder=perceiver_encoder,
            camera_names=cfg.rlbench.cameras,
            voxel_size=vox_size,
            bounds_offset=cfg.method.bounds_offset[depth - 1] if depth > 0 else None,
            image_crop_size=cfg.method.image_crop_size,
            lr=cfg.method.lr,
            training_iterations=cfg.framework.training_iterations,
            lr_scheduler=cfg.method.lr_scheduler,
            num_warmup_steps=cfg.method.num_warmup_steps,
            trans_loss_weight=cfg.method.trans_loss_weight,
            rot_loss_weight=cfg.method.rot_loss_weight,
            grip_loss_weight=cfg.method.grip_loss_weight,
            collision_loss_weight=cfg.method.collision_loss_weight,
            include_low_dim_state=True,
            image_resolution=cam_resolution,
            batch_size=cfg.replay.batch_size,
            voxel_feature_size=3,
            lambda_weight_l2=cfg.method.lambda_weight_l2,
            num_rotation_classes=num_rotation_classes,
            rotation_resolution=cfg.method.rotation_resolution,
            transform_augmentation=cfg.method.transform_augmentation.apply_se3,
            transform_augmentation_xyz=cfg.method.transform_augmentation.aug_xyz,
            transform_augmentation_rpy=cfg.method.transform_augmentation.aug_rpy,
            transform_augmentation_rot_resolution=cfg.method.transform_augmentation.aug_rot_resolution,
            optimizer_type=cfg.method.optimizer,
            num_devices=cfg.ddp.num_devices,
            load_exists_weights = cfg.framework.load_existing_weights,
            frozen = cfg.framework.frozen,
            cfg = cfg,
            aug_type=cfg.framework.augmentation_type,
            use_memory_bank=bool(getattr(cfg.framework, "use_memory_bank", False)),
            memory_bank=memory_bank,
        )
        qattention_agents.append(qattention_agent)

    rotation_agent = QAttentionStackAgent(
        qattention_agents=qattention_agents,
        rotation_resolution=cfg.method.rotation_resolution,
        camera_names=cfg.rlbench.cameras,
    )
    preprocess_agent = PreprocessAgent(pose_agent=rotation_agent)
    return preprocess_agent
