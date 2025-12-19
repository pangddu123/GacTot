import numpy as np
import random
from typing import List, Set, Dict, Tuple
from dataclasses import dataclass

# 假设 logger 已经配置好，或者使用 print
import logging

logger = logging.getLogger("GAC_Evolution")


@dataclass
class ModelMetadata:
    model_id: str
    param_size: float  # 参数量 (Billion)
    domain_score: float  # 领域专精分数 (0-1)
    validation_accuracy: float  # 在验证集上的单体准确率
    error_set: Set[int]  # 验证集上错误样本的索引集合


class GACEvolutionaryOptimizer:
    def __init__(
            self,
            candidate_pool: List[ModelMetadata],
            subset_size_constraint: int,
            pop_size: int = 50,
            lambda_div: float = 0.5,
            gamma_gap: float = 0.2,
            mutation_rate: float = 0.1,
            crossover_rate: float = 0.8
    ):
        """
        初始化进化优化器
        :param candidate_pool: 所有候选模型的元数据列表
        :param subset_size_constraint: 协作子集的大小限制 (背包容量)
        :param lambda_div: 多样性权重的超参数
        :param gamma_gap: 性能差距惩罚的超参数
        """
        self.pool = candidate_pool
        self.n_models = len(candidate_pool)
        self.k = subset_size_constraint
        self.pop_size = pop_size
        self.lambda_div = lambda_div
        self.gamma_gap = gamma_gap
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def _stratified_initialization(self) -> List[List[int]]:
        """
        基于模型异构属性的分层初始化策略 (3.2节核心)
        从参数规模、领域专精两个维度对初始种群进行播种
        """
        population = []

        # 将模型按参数大小分层
        small_models = [i for i, m in enumerate(self.pool) if m.param_size < 7.0]
        large_models = [i for i, m in enumerate(self.pool) if m.param_size >= 7.0]

        # 将模型按领域分数分层 (假设 >0.8 为专家)
        experts = [i for i, m in enumerate(self.pool) if m.domain_score > 0.8]
        generalists = [i for i, m in enumerate(self.pool) if m.domain_score <= 0.8]

        for _ in range(self.pop_size):
            # 策略：混合不同层级的模型以保证多样性
            # 例如：确保至少包含 1 个大模型和 1 个专家模型，其余随机填充
            individual = set()

            if large_models:
                individual.add(random.choice(large_models))
            if experts:
                individual.add(random.choice(experts))

            # 填充剩余位置
            remaining_slots = self.k - len(individual)
            if remaining_slots > 0:
                available = list(set(range(self.n_models)) - individual)
                if len(available) >= remaining_slots:
                    individual.update(random.sample(available, remaining_slots))

            # 如果初始生成不满足 K，则补齐或截断
            individual = list(individual)
            if len(individual) > self.k:
                individual = individual[:self.k]

            population.append(individual)

        logger.info("Initialized population with stratified strategy.")
        return population

    def _calculate_diversity(self, subset_indices: List[int]) -> float:
        """
        计算 Div(S): 基于错误解耦的差异化项
        Div(S) = Average(1 - Jaccard(Ei, Ej))
        """
        if len(subset_indices) < 2:
            return 0.0

        total_div = 0.0
        count = 0

        for i in range(len(subset_indices)):
            for j in range(i + 1, len(subset_indices)):
                idx_i = subset_indices[i]
                idx_j = subset_indices[j]

                E_i = self.pool[idx_i].error_set
                E_j = self.pool[idx_j].error_set

                # 计算错误集的交集和并集
                intersection = len(E_i.intersection(E_j))
                union = len(E_i.union(E_j))

                if union == 0:
                    jaccard = 0.0  # 完全重叠或均无错误
                else:
                    jaccard = intersection / union

                # 互补性 = 1 - Jaccard (重叠越少，互补性越高)
                total_div += (1.0 - jaccard)
                count += 1

        if count == 0:
            return 0.0

        # 归一化：公式中分母是 |S|(|S|-1)，这里双重循环只跑了一半，所以除以 count 即可平均
        return total_div / count

    def _calculate_gap_penalty(self, subset_indices: List[int]) -> float:
        """
        计算 GapPenalty(S): 性能差距惩罚项 (方差或极差)
        这里使用准确率的方差
        """
        accuracies = [self.pool[i].validation_accuracy for i in subset_indices]
        if not accuracies:
            return 0.0
        return float(np.var(accuracies))

    def _estimate_ensemble_accuracy(self, subset_indices: List[int]) -> float:
        """
        估算子集集成后的基准预测准确率 Accuracy(S)
        简化模拟：假设多数投票 (Majority Voting)
        实际应用中应基于验证集真实 logits 计算
        """
        # 这里仅作逻辑演示，假设我们知道全集验证样本的总数
        # 实际代码需要传入验证集 ground truth
        total_samples = 1000  # 示例值

        # 模拟：如果一个样本不在多数模型的 error_set 中，则视为正确
        # 这里为了简化，直接使用平均单体准确率 + 协同增益的启发式模拟
        # 在真实场景中，这里必须是真实的集成评估函数
        avg_acc = np.mean([self.pool[i].validation_accuracy for i in subset_indices])
        return avg_acc  # 占位符

    def fitness_function(self, individual: List[int]) -> float:
        """
        F(S) = Accuracy(S) + lambda * Div(S) - gamma * GapPenalty(S)
        """
        acc = self._estimate_ensemble_accuracy(individual)
        div = self._calculate_diversity(individual)
        gap = self._calculate_gap_penalty(individual)

        score = acc + (self.lambda_div * div) - (self.gamma_gap * gap)
        return score

    def evolve(self, generations: int = 50) -> List[int]:
        """
        执行进化算法主循环
        """
        population = self._stratified_initialization()

        for gen in range(generations):
            # 评估适应度
            fitness_scores = [self.fitness_function(ind) for ind in population]

            # 精英保留
            sorted_indices = np.argsort(fitness_scores)[::-1]
            elite = population[sorted_indices[0]]

            logger.info(f"Generation {gen}: Best Fitness = {fitness_scores[sorted_indices[0]]:.4f}")

            new_population = [elite]

            # 选择与交叉
            while len(new_population) < self.pop_size:
                # 锦标赛选择
                parent1 = population[self._tournament_selection(fitness_scores)]
                parent2 = population[self._tournament_selection(fitness_scores)]

                # 交叉
                if random.random() < self.crossover_rate:
                    child = self._crossover(parent1, parent2)
                else:
                    child = parent1[:]

                # 变异
                if random.random() < self.mutation_rate:
                    child = self._mutate(child)

                new_population.append(child)

            population = new_population

        # 返回最优子集
        best_idx = np.argmax([self.fitness_function(ind) for ind in population])
        return population[best_idx]

    def _tournament_selection(self, scores: List[float], k: int = 3) -> int:
        indices = random.sample(range(len(scores)), k)
        best_i = indices[0]
        for i in indices[1:]:
            if scores[i] > scores[best_i]:
                best_i = i
        return best_i

    def _crossover(self, p1: List[int], p2: List[int]) -> List[int]:
        # 均匀交叉，并保持集合大小约束
        combined = list(set(p1) | set(p2))
        random.shuffle(combined)
        return combined[:self.k]

    def _mutate(self, individual: List[int]) -> List[int]:
        # 随机替换一个模型
        if len(individual) == 0: return individual
        idx_to_remove = random.choice(range(len(individual)))

        available = list(set(range(self.n_models)) - set(individual))
        if not available:
            return individual

        new_model = random.choice(available)
        individual[idx_to_remove] = new_model
        return individual