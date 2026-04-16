# Notation Delta

- $R(	au)$: execution-based evaluation 给整条 trajectory 的成功信号。
- $p_	heta(t_{1:m}, a_{1:n}\mid x)$: CoTA-style multimodal action model 对 thought/action sequence 的联合建模。
- $a_t=(\mathrm{type}_t,x_t,y_t,\mathrm{arg}_t)$: GUI action 在统一视觉空间中的抽象。
- $z_{1:K}=f_	heta(v_{1:T})$: temporal encoder 对长视频做 token compression。
- $\hat{\mathcal{I}}=\{(i,r_i)\}$: GenS 输出的 frame relevance annotations。
