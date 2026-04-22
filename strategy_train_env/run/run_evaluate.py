import numpy as np
import math
import logging
import argparse
import os
import sys
import datetime
import random
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bidding_train_env.offline_eval.test_dataloader import TestDataLoader
from bidding_train_env.offline_eval.offline_env import OfflineEnv

from bidding_train_env.strategy.guide_bidding_strategy import GUIDEStrategy

def setup_logger():
    now = datetime.datetime.now()
    log_dir = os.path.join(".", "log", now.strftime("%Y%m%d"))
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log_{now.strftime('%H%M%S')}_QGA.log")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter("[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s")
    file_handler.setFormatter(file_format)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.info(f"Log file saved at: {log_file}")
    return logger


logger = None

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def getScore_neurips(reward, cpa, cpa_constraint):
    beta = 2
    penalty = 1
    if cpa > cpa_constraint:
        coef = cpa_constraint / (cpa + 1e-10)
        penalty = pow(coef, beta)
    return penalty * reward


def get_all_strategies():
    return {
        'GUIDE': GUIDEStrategy()
    }


def evaluate_strategy(agent, data_loader, env, keys, budget_ratio_f, test_dict):
    logger.info(f"Testing strategy: {agent.name}")

    total_reward = 0
    total_cost = 0
    total_score = 0

    selected_indices = [i for i in range(48)]
    advertiser_count = len(selected_indices)

    for idx in selected_indices:
        if idx >= len(keys):
            logger.warning(f"Index {idx} out of range, skipping")
            continue

        key = keys[idx]
        df = test_dict[key]

        budget = df["budget"].iloc[0] * budget_ratio_f
        cpa_constraint = df["CPAConstraint"].iloc[0]

        logger.info(f"Advertiser index {idx}: {key}, budget: {budget}, CPA constraint: {cpa_constraint}")

        num_timeStepIndex, pValues, pValueSigmas, leastWinningCosts = data_loader.mock_data(key)
        rewards = np.zeros(num_timeStepIndex)
        history = {
            'historyBids': [],
            'historyAuctionResult': [],
            'historyImpressionResult': [],
            'historyLeastWinningCost': [],
            'historyPValueInfo': []
        }

        agent.budget = budget
        agent.cpa = cpa_constraint
        agent.reset()

        for timeStep_index in range(num_timeStepIndex):
            pValue = pValues[timeStep_index]
            pValueSigma = pValueSigmas[timeStep_index]
            leastWinningCost = leastWinningCosts[timeStep_index]

            if agent.remaining_budget < env.min_remaining_budget:
                bid = np.zeros(pValue.shape[0])
            else:
                bid, _ = agent.bidding(timeStep_index, pValue, pValueSigma, history["historyPValueInfo"],
                                       history["historyBids"],
                                       history["historyAuctionResult"], history["historyImpressionResult"],
                                       history["historyLeastWinningCost"])

            tick_value, tick_cost, tick_status, tick_conversion = env.simulate_ad_bidding(pValue, pValueSigma, bid,
                                                                                          leastWinningCost)

            over_cost_ratio = max((np.sum(tick_cost) - agent.remaining_budget) / (np.sum(tick_cost) + 1e-4), 0)
            while over_cost_ratio > 0:
                pv_index = np.where(tick_status == 1)[0]
                dropped_pv_index = np.random.choice(pv_index, int(math.ceil(pv_index.shape[0] * over_cost_ratio)),
                                                    replace=False)
                bid[dropped_pv_index] = 0
                tick_value, tick_cost, tick_status, tick_conversion = env.simulate_ad_bidding(pValue, pValueSigma, bid,
                                                                                              leastWinningCost)
                over_cost_ratio = max((np.sum(tick_cost) - agent.remaining_budget) / (np.sum(tick_cost) + 1e-4), 0)

            agent.remaining_budget -= np.sum(tick_cost)
            rewards[timeStep_index] = np.sum(tick_conversion)
            temHistoryPValueInfo = [(pValue[i], pValueSigma[i]) for i in range(pValue.shape[0])]
            history["historyPValueInfo"].append(np.array(temHistoryPValueInfo))
            history["historyBids"].append(bid)
            history["historyLeastWinningCost"].append(leastWinningCost)
            temAuctionResult = np.array(
                [(tick_status[i], tick_status[i], tick_cost[i]) for i in range(tick_status.shape[0])])
            history["historyAuctionResult"].append(temAuctionResult)
            temImpressionResult = np.array([(tick_conversion[i], tick_conversion[i]) for i in range(pValue.shape[0])])
            history["historyImpressionResult"].append(temImpressionResult)

        advertiser_reward = np.sum(rewards)
        advertiser_cost = agent.budget - agent.remaining_budget
        advertiser_cpa = advertiser_cost / (advertiser_reward + 1e-10)
        advertiser_score = getScore_neurips(advertiser_reward, advertiser_cpa, cpa_constraint)

        logger.info(f'Strategy: {agent.name} - Advertiser index {idx}')
        logger.info(f'Reward: {advertiser_reward}')
        logger.info(f'Cost: {advertiser_cost}')
        logger.info(f'CPA: {advertiser_cpa}')
        logger.info(f'Score: {advertiser_score}')

        total_reward += advertiser_reward
        total_cost += advertiser_cost
        total_score += advertiser_score

    avg_reward = total_reward / advertiser_count
    avg_cost = total_cost / advertiser_count
    avg_cpa = avg_cost / (avg_reward + 1e-10)
    avg_score = total_score / advertiser_count

    strategy_result = {
        'name': agent.name,
        'reward': avg_reward,
        'cost': avg_cost,
        'cpa_real': avg_cpa,
        'cpa_constraint': sum([test_dict[key]["CPAConstraint"].iloc[0] for key in keys]) / advertiser_count,
        'score': avg_score,
        'total_reward': total_reward,
        'total_cost': total_cost,
        'total_score': total_score,
        'advertiser_count': advertiser_count
    }

    logger.info(f'Strategy: {agent.name} - Summary')
    logger.info(f'Avg Reward: {avg_reward}')
    logger.info(f'Avg Cost: {avg_cost}')
    logger.info(f'Avg CPA: {avg_cpa}')
    logger.info(f'Avg Score: {avg_score}')
    logger.info('-' * 50)

    return strategy_result


def run_test(args, period=None):
    if period is None:
        period = args.period
    file_path = os.path.join(args.data_dir, f"period-{period}.csv")

    print(file_path)
    data_loader = TestDataLoader(file_path=file_path)
    env = OfflineEnv()

    all_strategies = get_all_strategies()

    if args.flag == 1 and args.strategy:
        if args.strategy in all_strategies:
            strategies = [all_strategies[args.strategy]]
            logger.info(f"Running strategy: {args.strategy}")
        else:
            logger.error(f"Strategy not found: {args.strategy}")
            logger.info(f"Available strategies: {list(all_strategies.keys())}")
            return None
    else:
        strategies = list(all_strategies.values())
        logger.info("Running all strategies")

    keys, test_dict = data_loader.keys, data_loader.test_dict
    logger.info(f"Dataset contains {len(keys)} advertisers")

    all_results = {
        'reward': 0,
        'cost': 0,
        'cpa_real': 0,
        'total_score': 0,
        'strategies_results': [],
        'strategy_count': 0
    }

    for agent in strategies:
        for budget_ratio_f in [1.0]:
            strategy_result = evaluate_strategy(agent, data_loader, env, keys, budget_ratio_f, test_dict)
            all_results['strategies_results'].append(strategy_result)

            all_results['reward'] += strategy_result['reward']
            all_results['cost'] += strategy_result['cost']
            all_results['total_score'] += strategy_result['score']
            all_results['strategy_count'] += 1

        if all_results['strategy_count'] > 0:
            all_results['reward'] = all_results['reward'] / all_results['strategy_count']
            all_results['cost'] = all_results['cost'] / all_results['strategy_count']
            all_results['total_score'] = all_results['total_score'] / all_results['strategy_count']
            all_results['cpa_real'] = all_results['cost'] / (all_results['reward'] + 1e-10)

    return all_results


def run_all_period(args):
    results = []
    all_rewards = 0
    all_score = 0
    period_count = 0

    for period in range(7, 28):
        try:
            logger.info(f"Evaluating period-{period}")
            result = run_test(args, period)
            if result:
                results.append(result)
                all_rewards += result['reward']
                all_score += result['total_score']
                period_count += 1
            logger.info(f"Done period-{period}")
        except Exception as e:
            logger.error(f"Error in period-{period}: {str(e)}")

    logger.info("All evaluations done")

    if period_count > 0:
        avg_reward = all_rewards / period_count
        avg_score = all_score / period_count
        logger.info(f"Avg reward: {avg_reward}. Avg score: {avg_score}.")
        logger.info("=" * 50)
    else:
        logger.warning("No successful periods, cannot compute average")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bidding strategy evaluation')
    parser.add_argument('--flag', type=int, default=0, help='Set to 1 to run only the specified strategy')
    parser.add_argument('--strategy', type=str, help='Strategy name to run')
    parser.add_argument('--all_period', action='store_true', help='Run all periods (7-27)')
    parser.add_argument('--period', type=int, default=7, help='Period index to evaluate (default: 7)')
    parser.add_argument('--data_dir', type=str, default='data/trafficFinal',
                        help='Directory containing period-N.csv files (default: data/trafficFinal)')
    args = parser.parse_args()

    logger = setup_logger()
    set_seed(42)

    if args.all_period:
        run_all_period(args)
    else:
        run_test(args)
