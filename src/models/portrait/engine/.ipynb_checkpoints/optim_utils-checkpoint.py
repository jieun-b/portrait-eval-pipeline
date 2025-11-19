import torch


# def build_optimizer(cfg, net, accelerator):
#     """
#     Custom optimizer builder:
#     - Rotation (oft_R): high lr, no weight decay
#     - MLP (lia.enc.fc): low lr, small weight decay
#     """
#     if cfg.solver.scale_lr:
#         base_lr = (
#             cfg.solver.learning_rate
#             * cfg.solver.gradient_accumulation_steps
#             * cfg.train_bs
#             * accelerator.num_processes
#         )
#     else:
#         base_lr = cfg.solver.learning_rate

#     # 8-bit AdamW 지원
#     if cfg.solver.use_8bit_adam:
#         try:
#             import bitsandbytes as bnb
#             optimizer_cls = bnb.optim.AdamW8bit
#         except ImportError:
#             raise ImportError(
#                 "Please install bitsandbytes to use 8-bit Adam: `pip install bitsandbytes`"
#             )
#     else:
#         optimizer_cls = torch.optim.AdamW

#     # -------------------------------
#     # Param grouping (lia 내부만)
#     # -------------------------------
#     rot_params = [
#         p for n, p in net.lia.named_parameters()
#         if "oft_R" in n and p.requires_grad
#     ]
#     mlp_params = [
#         p for n, p in net.lia.named_parameters()
#         if "enc.fc" in n and p.requires_grad
#     ]

#     # sanity check
#     print(f"[build_optimizer] rot_params: {len(rot_params)} tensors")
#     print(f"[build_optimizer] mlp_params: {len(mlp_params)} tensors")

#     # optimizer with param groups
#     # optimizer = optimizer_cls(
#     #     [
#     #         {"params": rot_params, "lr": 2e-4, "weight_decay": 0.0},   # rotation
#     #         {"params": mlp_params, "lr": base_lr, "weight_decay": cfg.solver.adam_weight_decay},  # MLP
#     #     ],
#     #     betas=(cfg.solver.adam_beta1, cfg.solver.adam_beta2),
#     #     eps=cfg.solver.adam_epsilon,
#     # )
#     optimizer = optimizer_cls(
#       [
#         {"params": rot_params, "lr": 2e-4, "weight_decay": 0.0},          # rotation
#         {"params": mlp_params, "lr": 5e-6, "weight_decay": 1e-3},         # MLP 매우 낮게
#       ],
#       betas=(cfg.solver.adam_beta1, cfg.solver.adam_beta2), eps=cfg.solver.adam_epsilon
#     )

    
#     return optimizer


def build_optimizer(cfg, trainable_params, accelerator):
    if cfg.solver.scale_lr:
        learning_rate = (
            cfg.solver.learning_rate
            * cfg.solver.gradient_accumulation_steps
            * cfg.train_bs
            * accelerator.num_processes
        )
    else:
        learning_rate = cfg.solver.learning_rate

    if cfg.solver.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam: `pip install bitsandbytes`"
            )
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        trainable_params,
        lr=learning_rate,
        betas=(cfg.solver.adam_beta1, cfg.solver.adam_beta2),
        weight_decay=cfg.solver.adam_weight_decay,
        eps=cfg.solver.adam_epsilon,
    )

    return optimizer
