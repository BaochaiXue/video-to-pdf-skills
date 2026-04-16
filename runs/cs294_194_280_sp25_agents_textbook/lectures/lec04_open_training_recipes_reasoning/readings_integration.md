# Readings Integration

## Tulu 3

这篇 paper 对应 lecture 的主线。讲义中的 staged recipe、data curation、preference tuning 与 RLVR 都直接来自该论文及其 slides。它告诉我们 open post-training 不只是“公开 checkpoint”，而是公开一整条可被他人复做和质疑的工艺链。

教材化地说，Tulu 3 的真正价值有两层。第一层是把 SFT、preference tuning 和 reasoning-oriented RL 放回同一条 recipe 里，而不是把它们拆成互相竞争的口号。第二层是把 evaluator、data mixture 与 stage ordering 视为一等公民，这正是 lecture 里反复强调的系统工程视角。

## Unpacking DPO and PPO

lecture 对 DPO/PPO 的态度明显受这篇 paper 影响：不要把算法名词当成捷径，而要先搞清楚 preference data、reward model、prompt construction 和 evaluator 是否已经站得住。

这篇 reading 之所以重要，是因为它把“data 比 algorithm 更重要”从口头判断变成了系统消融结论。它也解释了 lecture 中那个看似保守但很工程化的结论：PPO 往往更强，但若团队算力和迭代速度有限，未必应该一开始就把主要精力花在 PPO 实现上。

## OpenScholar

OpenScholar 说明开放模型生态的价值会在下游 scientific agents 中体现出来。即便它不是这场 lecture 的中心算法，它仍然支撑了 lecture 的“开放基础设施可以加速研究与应用”这一论点。

它对本讲最重要的启发是：开放 recipe 的收益不会停在训练阶段，而会延伸到 retrieval、citation attribution 和 long-form grounded synthesis 这些下游 agent 能力上。换句话说，本讲讨论的 open recipe 是基础设施问题，而 OpenScholar 展示了这种基础设施如何转化为更可信的下游系统。
