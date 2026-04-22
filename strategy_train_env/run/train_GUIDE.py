import os
import sys
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import argparse
import csv
from datetime import datetime

# 添加项目根路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from bidding_train_env.common.utils import save_normalize_dict
from bidding_train_env.baseline.GUIDE.utils import EpisodeReplayBuffer
from bidding_train_env.baseline.GUIDE.dt_baselines import DecisionTransformer, Critic


def parse_args():
    parser = argparse.ArgumentParser(description="Train Decision Transformer with IDM and Critic")

    # === 数据相关 ===
    parser.add_argument('--data_path', type=str,
                        default="data/trajectory/trajectory_data.csv",
                        help="Path to trajectory CSV file")
    parser.add_argument('--sparse_data', action='store_true', default=True,
                        help="Whether to use sparse data mode (default: True)")
    parser.add_argument('--state_dim', type=int, default=16,
                        help="Dimension of state (default: 16)")
    parser.add_argument('--act_dim', type=int, default=1,
                        help="Dimension of action (default: 1)")

    # === 训练超参数 ===
    parser.add_argument('--step_num', type=int, default=16000,
                        help="Total training steps (default: 16000)")
    parser.add_argument('--batch_size', type=int, default=128,
                        help="Batch size (default: 128)")
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help="Learning rate for critic and actor (default: 3e-4)")
    parser.add_argument('--discount', type=float, default=0.99,
                        help="Discount factor (gamma) (default: 0.99)")
    parser.add_argument('--tau', type=float, default=0.005,
                        help="Target network update rate (EMA) (default: 0.005)")
    parser.add_argument('--alpha', type=float, default=1.0,
                        help="Weight for Q loss in actor objective (default: 1.0)")
    parser.add_argument('--detach_steps', type=int, default=500,
                        help="Steps before joint training (IDM detached before this) (default: 500)")

    # === 路径与保存 ===
    parser.add_argument('--replay_buffer_path', type=str, default="./replay_buffer.pkl",
                        help="Path to save/load replay buffer (default: saved_model/replay_buffer.pkl)")
    parser.add_argument('--results_dir_base', type=str, default="results",
                        help="Base directory for experiment results (default: results)")
    parser.add_argument('--model_save_dir', type=str, default="saved_model/GUIDE",
                        help="Directory to save final models and checkpoints")

    # === 设备 ===
    parser.add_argument('--device', type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Device to use (default: auto -> cuda if available else cpu)")

    return parser.parse_args()


def main():
    args = parse_args()

    # 自动设置设备
    if args.device == "auto":
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    # === 1. Replay Buffer ===
    os.makedirs(os.path.dirname(args.replay_buffer_path), exist_ok=True)
    if os.path.exists(args.replay_buffer_path):
        with open(args.replay_buffer_path, "rb") as f:
            replay_buffer = pickle.load(f)
    else:
        replay_buffer = EpisodeReplayBuffer(
            state_dim=args.state_dim,
            act_dim=args.act_dim,
            data_path=args.data_path,
            sparse_data=args.sparse_data
        )
        with open(args.replay_buffer_path, "wb") as f:
            pickle.dump(replay_buffer, f)

    save_normalize_dict(
        {"state_mean": replay_buffer.state_mean, "state_std": replay_buffer.state_std},
        os.path.dirname(args.replay_buffer_path)
    )
    replay_buffer.scale = 50

    # === 2. Model & Optimizer ===
    model = DecisionTransformer(
        state_dim=args.state_dim,
        act_dim=args.act_dim,
        state_mean=replay_buffer.state_mean,
        state_std=replay_buffer.state_std,
    ).to(device)

    critic = Critic(args.state_dim, args.act_dim, hidden_dim=256).to(device)
    critic_target = Critic(args.state_dim, args.act_dim, hidden_dim=256).to(device)
    critic_target.load_state_dict(critic.state_dict())

    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.learning_rate)
    actor_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=model.learning_rate,
        weight_decay=model.weight_decay
    )

    # === 3. DataLoader ===
    sampler = WeightedRandomSampler(
        weights=replay_buffer.p_sample,
        num_samples=args.step_num * args.batch_size,
        replacement=True
    )
    dataloader = DataLoader(replay_buffer, batch_size=args.batch_size, sampler=sampler)

    # === 4. Training Loop ===
    os.makedirs(args.model_save_dir, exist_ok=True)
    model.train()

    for i, batch in enumerate(dataloader):
        if i >= args.step_num:
            break

        states, actions, rewards, dones, rtg, timesteps, attention_mask, ctg, score_to_go, costs = batch
        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        dones = dones.to(device)
        rtg = rtg.to(device)
        timesteps = timesteps.to(device)
        attention_mask = attention_mask.to(device)
        ctg = ctg.to(device)
        score_to_go = score_to_go.to(device)
        costs = costs.to(device)

        _, action_preds, state_preds, _ = model.forward(
            states, actions, rewards,
            rtg[:, :-1], ctg[:, :-1], score_to_go[:, :-1],
            timesteps, attention_mask=attention_mask
        )

        # --- BC Loss & State Loss ---
        valid_mask = attention_mask.reshape(-1) > 0
        act_pred_vec = action_preds.reshape(-1, args.act_dim)[valid_mask]
        action_target_vec = actions.reshape(-1, args.act_dim)[valid_mask]
        bc_loss = F.mse_loss(act_pred_vec, action_target_vec)

        state_pred_vec = state_preds.reshape(-1, state_preds.shape[2])[valid_mask]
        state_target_vec = states.reshape(-1, states.shape[2])[valid_mask]
        state_loss = F.mse_loss(state_pred_vec, state_target_vec)

        # --- IDM Loss ---
        non_final_mask_idm = attention_mask[:, 1:].reshape(-1) > 0
        state_cur_idm = states[:, :-1, :].reshape(-1, args.state_dim)[non_final_mask_idm]
        state_pred_next_idm = state_preds[:, 1:, :].reshape(-1, args.state_dim)[non_final_mask_idm]
        action_target_idm = actions[:, 1:, :].reshape(-1, args.act_dim)[non_final_mask_idm]

        if i < args.detach_steps:
            with torch.no_grad():
                state_pred_next_detached = state_pred_next_idm.detach()
            action_pred_idm = model.id_model(state_cur_idm, state_pred_next_detached)
            idm_loss_predict = F.mse_loss(action_pred_idm, action_target_idm)

            model.id_optimizer.zero_grad()
            idm_loss_predict.backward()
            torch.nn.utils.clip_grad_norm_(model.id_model.parameters(), 1.0)
            model.id_optimizer.step()
        else:
            action_pred_idm = model.id_model(state_cur_idm, state_pred_next_idm)
            idm_loss_predict = F.mse_loss(action_pred_idm, action_target_idm)

        # --- Critic Update ---
        non_final_mask = attention_mask[:, :-1].reshape(-1) > 0
        states_q = states[:, :-1, :].reshape(-1, args.state_dim)[non_final_mask]
        actions_q = actions[:, :-1, :].reshape(-1, args.act_dim)[non_final_mask]
        current_q1, current_q2 = critic(states_q, actions_q)

        s_next = states[:, 1:, :].reshape(-1, args.state_dim)[non_final_mask]
        a_next = action_preds[:, 1:, :].reshape(-1, args.act_dim)[non_final_mask]
        with torch.no_grad():
            target_q1, target_q2 = critic_target(s_next, a_next)
            target_q = torch.min(target_q1, target_q2)

        reward_vec = rewards[:, :-1].reshape(-1, 1)[non_final_mask]
        done_vec = dones[:, :-1].reshape(-1, 1)[non_final_mask]
        td_target = reward_vec + args.discount * (1 - done_vec) * target_q
        critic_loss = F.mse_loss(current_q1, td_target) + F.mse_loss(current_q2, td_target)

        critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        critic_optimizer.step()

        # --- Actor Update ---
        states_all_valid = states.reshape(-1, args.state_dim)[valid_mask]
        act_pred_vec_full = action_preds.reshape(-1, args.act_dim)[valid_mask]
        q1_new, q2_new = critic(states_all_valid, act_pred_vec_full)
        actor_q_loss = -torch.mean(torch.min(q1_new, q2_new))

        if i < args.detach_steps:
            actor_loss = bc_loss + args.alpha * actor_q_loss + state_loss
        else:
            actor_loss = bc_loss + args.alpha * actor_q_loss + state_loss + idm_loss_predict

        actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        actor_optimizer.step()

        # --- Target Network EMA ---
        for param, target_param in zip(critic.parameters(), critic_target.parameters()):
            target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

        # --- Optional: Print progress (optional, not saved) ---
        if i % 1000 == 0:
            print(f"Step {i}/{args.step_num} | "
                  f"BC: {bc_loss.item():.4f}, "
                  f"Actor Q: {actor_q_loss.item():.4f}, "
                  f"Critic: {critic_loss.item():.4f}, "
                  f"IDM: {idm_loss_predict.item():.4f}")

        # --- Periodic Model Save (only models, no CSV) ---
        if i % 1000 == 0 or i == args.step_num - 1:
            ckpt_dir = os.path.join(args.model_save_dir, str(i))
            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_net(ckpt_dir)
            critic.save_net(ckpt_dir)
            model.save_idm(ckpt_dir)

        if hasattr(model, 'scheduler'):
            model.scheduler.step()

    # Final save
    model.save_net(args.model_save_dir)
    critic.save_net(args.model_save_dir)
    model.save_idm(args.model_save_dir)

    print(f"Training completed.")
    print(f"Models saved in: {args.model_save_dir}")


if __name__ == "__main__":
    main()
