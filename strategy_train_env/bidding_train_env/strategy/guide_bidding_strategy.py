import numpy as np
import os
from bidding_train_env.baseline.GUIDE.dt_baselines import DecisionTransformer, Critic
from bidding_train_env.strategy.base_bidding_strategy import BaseBiddingStrategy
import torch
import pickle

# Resolve paths relative to this file's location:
#   guide_bidding_strategy.py  →  strategy/  →  bidding_train_env/  →  strategy_train_env/
_STRATEGY_TRAIN_ENV_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class  GUIDEStrategy(BaseBiddingStrategy):
    def __init__(self, budget=100, name="GUIDE-Strategy", cpa=2, category=1,
                 load_dir=None, model_path=None, critic_path=None, idm_path=None):
        super().__init__(budget, name, cpa, category)

        if model_path is None:
            model_path = os.path.join(_STRATEGY_TRAIN_ENV_DIR, "saved_model", "GUIDE", "GUIDE.pt")
        if critic_path is None:
            critic_path = os.path.join(_STRATEGY_TRAIN_ENV_DIR, "saved_model", "GUIDE", "QGA_critic_inverse.pt")
        if idm_path is None:
            idm_path = os.path.join(_STRATEGY_TRAIN_ENV_DIR, "saved_model", "GUIDE", "GUIDE_idm.pt")
        picklePath = os.path.join(_STRATEGY_TRAIN_ENV_DIR, "saved_model", "GUIDE", "normalize_dict.pkl")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        import pickle
        with open(picklePath, 'rb') as f:
            normalize_dict = pickle.load(f)


        self.model = DecisionTransformer(
            state_dim=16, act_dim=1,
            state_mean=normalize_dict["state_mean"],
            state_std=normalize_dict["state_std"],
            target_return=16., target_ctg=16.,
        )
        print(f'Load DT from {model_path}, Critic from {critic_path}, IDM from {idm_path}')
        self.model.load_net(load_path=model_path,
                            critic_path=critic_path,
                            idm_path=idm_path)
        self.model.to(device)

        self.test_state_old = np.zeros(16)
        self.cpa = cpa
        self.budget = budget
        self.category = category
        self.remaining_budget_last = self.budget

    def reset(self):
        self.remaining_budget = self.budget

    def bidding(self, timeStepIndex, pValues, pValueSigmas, historyPValueInfo, historyBid,
                historyAuctionResult, historyImpressionResult, historyLeastWinningCost,
                actual_excuted_action=None):
        """
        Bids for all the opportunities in a delivery period (使用DT/IDM逆动力学二选一 with Critic).
        """

        self.cost_cur = self.remaining_budget_last - self.remaining_budget

        time_left = (48 - timeStepIndex) / 48
        budget_left = self.remaining_budget / self.budget if self.budget > 0 else 0
        history_xi = [result[:, 0] for result in historyAuctionResult]
        history_pValue = [result[:, 0] for result in historyPValueInfo]
        history_conversion = [result[:, 1] for result in historyImpressionResult]

        historical_xi_mean = np.mean([np.mean(xi) for xi in history_xi]) if history_xi else 0
        historical_conversion_mean = np.mean(
            [np.mean(reward) for reward in history_conversion]) if history_conversion else 0
        historical_LeastWinningCost_mean = np.mean(
            [np.mean(price) for price in historyLeastWinningCost]) if historyLeastWinningCost else 0
        historical_pValues_mean = np.mean([np.mean(value) for value in history_pValue]) if history_pValue else 0
        historical_bid_mean = np.mean([np.mean(bid) for bid in historyBid]) if historyBid else 0

        def mean_of_last_n_elements(history, n):
            l = len(history)
            last_n_data = history[max(0, l - n):l]
            if len(last_n_data) == 0:
                return 0
            else:
                return np.mean([np.mean(data) for data in last_n_data])

        last_three_xi_mean = mean_of_last_n_elements(history_xi, 3)
        last_three_conversion_mean = mean_of_last_n_elements(history_conversion, 3)
        last_three_LeastWinningCost_mean = mean_of_last_n_elements(historyLeastWinningCost, 3)
        last_three_pValues_mean = mean_of_last_n_elements(history_pValue, 3)
        last_three_bid_mean = mean_of_last_n_elements(historyBid, 3)

        current_pValues_mean = np.mean(pValues)
        current_pv_num = len(pValues)

        historical_pv_num_total = sum(len(bids) for bids in historyBid) if historyBid else 0
        last_three_pv_num_total = sum(
            [len(historyBid[i]) for i in range(max(0, timeStepIndex - 3), timeStepIndex)]) if historyBid else 0

        test_state = np.array([
            time_left, budget_left, historical_bid_mean, last_three_bid_mean,
            historical_LeastWinningCost_mean, historical_pValues_mean, historical_conversion_mean,
            historical_xi_mean, last_three_LeastWinningCost_mean, last_three_pValues_mean,
            last_three_conversion_mean, last_three_xi_mean,
            current_pValues_mean, current_pv_num, last_three_pv_num_total,
            historical_pv_num_total
        ])

        if timeStepIndex == 0:
            self.model.init_eval()
            self.test_state_old = np.zeros(16)

        self.test_state = test_state
        cost_constraint = self.cost_cur if len(history_conversion) != 0 else None

        # --------- 核心:DT+IDM二选一调用 ---------
        alpha = self.model.take_action_inverse(
            self.test_state, actual_excuted_action,
            pre_reward=sum(history_conversion[-1]) if len(history_conversion) != 0 else None,
            pre_cost=cost_constraint if len(history_conversion) != 0 else None,
            cpa_constrain=self.cpa
        )

        self.remaining_budget_last = self.remaining_budget
        bids = alpha * pValues
        return bids, alpha
