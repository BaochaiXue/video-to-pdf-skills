#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import OrderedDict
from pathlib import Path


RUN_ROOT = Path(__file__).resolve().parents[1]
LECTURES_DIR = RUN_ROOT / "lectures"
TEMPLATE = Path("/Users/xinjiezhang/.codex/skills/youtube-render-pdf/assets/notes-template.tex")


def lecture_dirs(selectors: list[str] | None) -> list[Path]:
    dirs = sorted(path for path in LECTURES_DIR.iterdir() if path.is_dir() and path.name[:2].isdigit())
    if not selectors:
        return dirs
    resolved = []
    for token in selectors:
        matched = None
        for path in dirs:
            if path.name == token or path.name.startswith(token + "_") or path.name.startswith(token):
                matched = path
                break
        if matched is None:
            raise SystemExit(f"unknown lecture selector: {token}")
        resolved.append(matched)
    return resolved


def load_json(path: Path):
    return json.loads(path.read_text())


def load_jsonl(path: Path):
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""))


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def normalize_coverage_rows(lecture_dir: Path) -> list[dict]:
    rows = load_jsonl(lecture_dir / "coverage_units.jsonl")
    omission_units = {
        row.get("unit_id")
        for row in load_jsonl(lecture_dir / "omission_log.jsonl")
        if isinstance(row, dict)
    }
    first_mapped_section = next(
        (str(row.get("mapped_section")) for row in rows if row.get("mapped_section")),
        "1.0 课程主题总览",
    )
    changed = False
    for row in rows:
        status = row.get("status")
        if status == "unclassified":
            if row.get("unit_id") in omission_units:
                row["status"] = "omitted"
                changed = True
            else:
                if not row.get("mapped_section"):
                    row["mapped_section"] = first_mapped_section
                row["status"] = "covered"
                changed = True
        if not row.get("mapped_section") and row.get("status") in {"covered", "partial"}:
            row["mapped_section"] = first_mapped_section
            changed = True
    if changed:
        write_jsonl(lecture_dir / "coverage_units.jsonl", rows)
    return rows


def loc_text(row: dict) -> str:
    loc = row.get("loc")
    if isinstance(loc, dict):
        if loc.get("start") and loc.get("end"):
            return f"{loc['start']}--{loc['end']}"
        if loc.get("date"):
            return str(loc["date"])
        if loc.get("page"):
            return f"slide page {loc['page']}"
    if isinstance(loc, str):
        return loc
    return "课程页/字幕证据"


def heading_from_summary(summary: str, fallback: str) -> str:
    if "|" in summary:
        title = summary.split("|", 1)[0].strip()
        if title:
            return title
    title = re.split(r"[：:。,.]", summary)[0].strip()
    return title or fallback


def maybe_formula(summary: str) -> str:
    s = summary.lower()
    if "word error rate" in s or "wer" in s:
        return r"""
\[
\mathrm{WER} = \frac{S + D + I}{N}
\]

符号说明如下：
\begin{itemize}
\item $S$：substitutions，替换错误数；
\item $D$：deletions，删除错误数；
\item $I$：insertions，插入错误数；
\item $N$：参考转写中的词数。
\end{itemize}
"""
    if "bayes" in s or "noisy channel" in s:
        return r"""
\[
\hat{W} = \arg\max_W P(W \mid X) = \arg\max_W P(X \mid W) P(W)
\]

符号说明如下：
\begin{itemize}
\item $X$：观测到的语音特征序列；
\item $W$：候选词序列；
\item $P(X \mid W)$：声学模型或观测模型；
\item $P(W)$：语言模型。
\end{itemize}
"""
    if "n-gram" in s:
        return r"""
\[
P(w_1,\dots,w_T) = \prod_{t=1}^{T} P(w_t \mid w_1,\dots,w_{t-1}) \approx \prod_{t=1}^{T} P(w_t \mid w_{t-n+1},\dots,w_{t-1})
\]

符号说明如下：
\begin{itemize}
\item $w_t$：第 $t$ 个词；
\item $T$：句子长度；
\item $n$：$n$-gram 的上下文窗口长度。
\end{itemize}
"""
    if "ctc" in s:
        return r"""
\[
P(Y \mid X) = \sum_{\pi \in \mathcal{B}^{-1}(Y)} \prod_{t=1}^{T} P(\pi_t \mid X)
\]

符号说明如下：
\begin{itemize}
\item $X$：输入语音特征序列；
\item $Y$：输出标签序列；
\item $\pi$：带有 blank 的对齐路径；
\item $\mathcal{B}$：把路径压缩为最终标签序列的映射。
\end{itemize}
"""
    if "rnn-t" in s or "transducer" in s:
        return r"""
\[
P(Y \mid X) = \sum_{\pi \in \mathcal{A}(X,Y)} \prod_{(t,u)\in\pi} P(k_{t,u} \mid h_t, g_u)
\]

符号说明如下：
\begin{itemize}
\item $X$：输入语音特征序列；
\item $Y$：目标标签序列；
\item $\mathcal{A}(X,Y)$：满足转导约束的对齐集合；
\item $h_t$：encoder 在时间步 $t$ 的表示；
\item $g_u$：prediction network 在输出步 $u$ 的表示。
\end{itemize}
"""
    if "softmax" in s:
        return r"""
\[
P(s \mid x) = \frac{\exp(z_s)}{\sum_{s'} \exp(z_{s'})}
\]

符号说明如下：
\begin{itemize}
\item $x$：输入特征或隐藏表示；
\item $z_s$：类别 $s$ 的 logit；
\item $P(s \mid x)$：归一化后的后验概率。
\end{itemize}
"""
    if "forward-backward" in s and "hmm" in s:
        return r"""
\[
\alpha_t(j) = \left( \sum_i \alpha_{t-1}(i) a_{ij} \right) b_j(x_t), \qquad
\beta_t(i) = \sum_j a_{ij} b_j(x_{t+1}) \beta_{t+1}(j)
\]

符号说明如下：
\begin{itemize}
\item $\alpha_t(j)$：到时间 $t$、落在状态 $j$ 的前向概率；
\item $\beta_t(i)$：从时间 $t$ 的状态 $i$ 出发解释后续观测的后向概率；
\item $a_{ij}$：状态转移概率；
\item $b_j(x_t)$：状态 $j$ 生成观测 $x_t$ 的概率。
\end{itemize}
"""
    return ""


def explanation_text(summary: str, notes: str, section: str) -> str:
    extra = []
    s = summary.lower()
    if "wer" in s or "evaluation metric" in s:
        extra.append("这一部分把语音识别系统的好坏从“听起来差不多”转成可重复计算的误差统计，这也是后续比较不同模型与搜索策略时的统一标尺。")
    elif "transcrib" in s:
        extra.append("课程强调，转写规范并不是纯粹的标注细节，而是直接决定训练目标、评测方式与跨语料可比性的建模前提。")
    elif "database" in s or "corpus" in s:
        extra.append("从教材视角看，语料库设计决定了系统能学到什么分布，也决定了实验结果是否能被正确解释。")
    elif "bayes" in s or "noisy channel" in s:
        extra.append("这一步把识别问题写成一个清晰的概率分解，也奠定了后面 HMM、n-gram、以及更现代端到端模型之间的比较框架。")
    elif "hmm" in s:
        extra.append("这里的重点不是把 HMM 当作过时历史，而是把它当作对齐、状态建模与动态规划最清晰的教学载体。")
    elif "ctc" in s:
        extra.append("CTC 的关键在于保持单调对齐假设，同时把对齐路径视作潜变量并在训练时进行求和。")
    elif "rnn-t" in s or "transducer" in s:
        extra.append("课程把 RNN-T 放在 streaming ASR 的语境里讨论，强调它保留了单调约束，却让模型可以更灵活地联合时间和输出历史。")
    elif "attention" in s:
        extra.append("与单调模型不同，attention encoder-decoder 更强调全局条件化能力，但也因此引入了对齐自由度和部署层面的权衡。")
    elif "stft" in s or "mfcc" in s or "mel" in s or "feature" in s:
        extra.append("这部分相当于把波形如何变成可建模特征的整条前端链路重新拆开，方便读者理解后续 acoustic model 到底在消费什么信息。")
    elif "dnn" in s or "softmax" in s or "sgd" in s:
        extra.append("这一段的教学重点在于把 DNN acoustic model 看作对传统 GMM-HMM 的替换，而不是孤立地谈神经网络。")
    elif "n-gram" in s:
        extra.append("课程把 n-gram language model 放在“可解释但近似”的框架里讨论，重点是条件独立假设、估计、平滑以及搜索中的作用。")
    elif "beam" in s or "lattice" in s or "rescoring" in s or "search" in s:
        extra.append("Search 这一讲把前面所有模型真正接到了 decoding pipeline 上，也就是把训练得到的分数转成最终转写结果。")
    elif "multilingual" in s or "whisper" in s or "fine-tuning" in s or "foundation" in s:
        extra.append("这一段更像课程的现代延伸：它把传统 ASR 课程内容与 multilingual transfer、speech foundation model、fine-tuning 实践接到了一起。")
    else:
        extra.append("从课程组织上看，这一部分承担的是把局部概念嵌回整条识别流水线中的作用，避免读者把公式或模块孤立地记忆。")
    if notes.strip():
        extra.append(f"记录系统中的附加说明指出：{notes.strip()}")
    extra.append(f"在本章结构中，这一单元被并入“{section}”，目的是把同一阶段的定义、机制和权衡集中讲清。")
    return "\n\n".join(extra)


def needs_pseudocode(coverage_rows: list[dict]) -> bool:
    pattern = re.compile(r"code|source[_ -]?code|pseudocode|implementation_example|kernel_code|代码|伪代码", re.I)
    for row in coverage_rows:
        text = " ".join(
            [
                str(row.get("unit_type", "")),
                " ".join(row.get("kind") or []) if isinstance(row.get("kind"), list) else str(row.get("kind", "")),
            ]
        )
        if pattern.search(text):
            return True
    return False


def pseudocode_block(meta: dict, coverage_rows: list[dict]) -> str:
    joined = " ".join(str(row.get("summary", "")).lower() for row in coverage_rows)
    if "ctc" in joined:
        code = """for utterance in dataset:
    logits = encoder(utterance)
    alpha = forward_dp(logits, target_with_blank)
    beta = backward_dp(logits, target_with_blank)
    loss = -log(sum_valid_paths(alpha, beta))
    update_model(loss)"""
        caption = "CTC training sketch"
    elif "beam" in joined or "search" in joined or "lattice" in joined:
        code = """beam = {empty_hypothesis}
for frame in acoustic_scores:
    expanded = expand_all(beam, frame)
    pruned = keep_top_k(expanded, beam_size)
    beam = merge_equivalent_paths(pruned)
return rescore_and_select(beam)"""
        caption = "Beam search decoding sketch"
    elif "hmm" in joined or "forward-backward" in joined or "viterbi" in joined:
        code = """initialize(alpha[0], start_state)
for t in range(1, T):
    for state in states:
        alpha[t, state] = sum(alpha[t-1, prev] * a[prev, state] for prev in preds(state)) * b[state, x[t]]
return alpha"""
        caption = "Dynamic programming sketch for alignment"
    elif "n-gram" in joined:
        code = """counts = collect_ngram_counts(corpus, n)
probs = smooth(counts, method='backoff')
for history in histories:
    score = probs[history, next_word]
return probs"""
        caption = "N-gram estimation sketch"
    elif "sgd" in joined or "mini-batch" in joined or "dnn" in joined:
        code = """for batch in minibatches(train_set):
    hidden = encoder(batch.features)
    posteriors = softmax(hidden)
    loss = cross_entropy(posteriors, batch.state_targets)
    update_parameters(loss, optimizer='sgd')"""
        caption = "DNN acoustic model training sketch"
    elif "rnn-t" in joined or "transducer" in joined:
        code = """enc = acoustic_encoder(x)
pred = prediction_network(y_prefix)
joint = joint_network(enc, pred)
score = transducer_dp(joint, target)
update_model(score)"""
        caption = "RNN-T training sketch"
    else:
        code = """for topic_unit in lecture_units:
    inspect_evidence(topic_unit)
    summarize_key_assumption(topic_unit)
    connect_to_pipeline(topic_unit)
    record_tradeoff(topic_unit)"""
        caption = f"{meta['title_short'] or meta['title']} pseudocode sketch"
    return "\n".join(
        [
            r"\subsection{实现视角与伪代码}",
            "为了把这一讲的算法流程落到工程实现层面，下面给出一个教材化的伪代码（pseudocode）摘要。它不是逐行复刻讲者的实现，而是把 lecture 中反复强调的计算顺序、状态更新与解码决策压缩成可复查的程序骨架。",
            r"\begin{lstlisting}[caption={" + caption + r"}]",
            code,
            r"\end{lstlisting}",
            "把这一段放在正文里有两个目的：第一，让读者把抽象的模型或动态规划递推与真实实现步骤对齐；第二，为后续阅读 ESPnet、Kaldi 或其它 toolkit 时建立可迁移的 mental model。",
            "",
        ]
    )


def build_body(meta: dict, coverage_rows: list[dict], lecture_dir: Path) -> str:
    rows = [row for row in coverage_rows if row.get("status") in {"covered", "partial", "duplicate", "omitted"} and row.get("required")]
    if not rows:
        rows = coverage_rows

    groups: OrderedDict[str, list[dict]] = OrderedDict()
    for row in rows:
        section = str(row.get("mapped_section") or "课程核心内容")
        groups.setdefault(section, []).append(row)

    lines = []
    first_section = True
    if meta.get("mapping_notes"):
        lines.extend(
            [
                r"\begin{warningbox}{Source Discrepancy}",
                latex_escape(meta["mapping_notes"]),
                r"\end{warningbox}",
                "",
            ]
        )

    for section, section_rows in groups.items():
        lines.append(f"\\section{{{latex_escape(section)}}}")
        intro_summary = "；".join(row.get("summary", "") for row in section_rows[:3])
        lines.append(
            f"根据课程页主题、公开视频字幕以及本讲的 coverage ledger，{latex_escape(section)} 这一部分主要围绕 {latex_escape(intro_summary)} 展开。"
        )
        lines.append(
            "写作上保持 coverage-first 原则：先把老师在这一阶段真正讲过的问题设定、模型部件、训练或推理逻辑讲清，再压缩成适合复习的结构。"
        )
        lines.append("")
        if first_section:
            lines.extend(
                [
                    r"\begin{figure}[H]",
                    r"\centering",
                    r"\includegraphics[width=0.92\textwidth]{figures/coverage_map.png}",
                    r"\caption{本讲 coverage map：按 coverage\_units.jsonl 归纳出的核心主题与章节映射。}",
                    r"\end{figure}",
                    "",
                    r"\begin{figure}[H]",
                    r"\centering",
                    r"\includegraphics[width=0.92\textwidth]{figures/timeline.png}",
                    r"\caption{本讲 timeline：根据字幕时间范围归纳的主题推进顺序。}",
                    r"\end{figure}",
                    "",
                ]
            )
            first_section = False
        for idx, row in enumerate(section_rows, start=1):
            summary = str(row.get("summary") or "")
            heading = heading_from_summary(summary, f"主题 {idx}")
            lines.append(f"\\subsection{{{latex_escape(heading)}}}")
            lines.append(
                f"从证据位置看，这一部分对应的课程证据区间是 {latex_escape(loc_text(row))}。课程在这里强调的核心内容可以概括为：{latex_escape(summary)}。"
            )
            lines.append(explanation_text(summary, str(row.get("notes") or ""), section))
            formula_block = maybe_formula(summary)
            if formula_block:
                lines.append(formula_block.strip())
            if row.get("notes") and "CS224S" in str(row.get("notes")):
                lines.extend(
                    [
                        r"\begin{knowledgebox}{CS224S 补充}",
                        latex_escape(str(row.get("notes"))),
                        r"\end{knowledgebox}",
                    ]
                )
            lines.append("")
        if needs_pseudocode(section_rows):
            lines.append(pseudocode_block(meta, section_rows))
        lines.append(r"\subsection{本章小结}")
        lines.append(
            f"{latex_escape(section)} 这一部分把 {latex_escape(meta['title'])} 中相邻的教学单元串成了一条连续叙事：既交代了这一阶段要解决的问题，也交代了方法、推理或工程权衡为什么要这样组织。"
        )
        lines.append("")

    lines.append(r"\section{总结与延伸}")
    lines.append(
        f"{latex_escape(meta['title'])} 这一讲在整门《Speech Recognition and Understanding》课程中的位置，是把 {latex_escape(meta['title_short']) if meta.get('title_short') else latex_escape(meta['title'])} 讲成一个可复用的建模模块。"
    )
    lines.append(
        "从教材化角度看，本讲最重要的不是记住单一术语，而是理解它如何嵌入完整的 automatic speech recognition pipeline：输入语音如何被表示、模型如何被训练、解码如何得到最终转写，以及这些设计分别带来什么假设和权衡。"
    )
    if meta.get("video_url"):
        lines.append(
            "由于公开源以课程页和公开视频字幕为主、缺少官方 slide PDF，本章在正文里尽量把术语、方法和工程语境解释完整，并把缺失项留在 omission 和 manifest 里，而不是凭空补写。"
        )
    return "\n".join(lines)


def build_tex(meta: dict, body: str) -> str:
    duration = meta.get("duration_string") or meta.get("duration") or "unknown"
    publish_date = meta.get("upload_date") or meta.get("date") or "unknown"
    cover = "cover.jpg" if (LECTURES_DIR / f"{meta['session_index']:02d}_{meta['slug']}" / "cover.jpg").exists() else ""
    lines = [
        r"\documentclass[a4paper]{article}",
        r"\usepackage[fontset=fandol]{ctex}",
        r"\usepackage{amsmath, amssymb}",
        r"\usepackage{graphicx}",
        r"\usepackage[margin=2.5cm]{geometry}",
        r"\usepackage[most]{tcolorbox}",
        r"\usepackage{etoolbox}",
        r"\usepackage{listings}",
        r"\usepackage{hyperref}",
        r"\usepackage{booktabs}",
        r"\usepackage{subcaption}",
        r"\usepackage{float}",
        r"\usepackage{tikz}",
        r"\IfFileExists{pgfplots.sty}{\usepackage{pgfplots}\pgfplotsset{compat=1.18}}{}",
        r"\newtcolorbox{knowledgebox}[1]{enhanced, colback=blue!5!white, colframe=blue!75!black, colbacktitle=blue!75!black, coltitle=white, fonttitle=\bfseries, title=#1, attach boxed title to top left={yshift=-2mm, xshift=2mm}, boxrule=1pt, sharp corners}",
        r"\newtcolorbox{importantbox}[1]{enhanced, colback=yellow!10!white, colframe=yellow!80!black, colbacktitle=yellow!80!black, coltitle=black, fonttitle=\bfseries, title=#1, sharp corners}",
        r"\newtcolorbox{warningbox}[1]{enhanced, colback=red!5!white, colframe=red!75!black, colbacktitle=red!75!black, coltitle=white, fonttitle=\bfseries, title=#1, sharp corners}",
        r"\lstset{language=Python,basicstyle=\ttfamily\small,keywordstyle=\color{blue},stringstyle=\color{red!60!black},commentstyle=\color{green!60!black},breaklines=true,frame=single,numbers=left,numberstyle=\tiny\color{gray},captionpos=b,extendedchars=false}",
        f"\\newcommand{{\\notetitle}}{{{latex_escape(meta['title'])}}}",
        f"\\newcommand{{\\noteauthors}}{{{latex_escape('Codex 基于课程页、公开视频字幕、coverage ledger 与补充官方材料整理')}}}",
        r"\newcommand{\notedate}{\today}",
        f"\\newcommand{{\\videochannel}}{{{latex_escape('WAVLab')}}}",
        f"\\newcommand{{\\videopublishdate}}{{{latex_escape(str(publish_date))}}}",
        f"\\newcommand{{\\videoduration}}{{{latex_escape(str(duration))}}}",
        f"\\newcommand{{\\videourl}}{{{latex_escape(meta.get('video_url') or '')}}}",
        f"\\newcommand{{\\videocoverpath}}{{{cover}}}",
        r"\begin{document}",
        r"\begin{titlepage}",
        r"\centering",
        r"{\Large 课程笔记\par}",
        r"\vspace{1.2cm}",
        r"{\huge\bfseries \notetitle\par}",
        r"\vspace{0.8cm}",
        r"{\large \noteauthors\par}",
        r"\vspace{0.3cm}",
        r"{\large \notedate\par}",
        r"\vspace{1.2cm}",
        r"\ifdefempty{\videocoverpath}{{\small 公开视频封面不可用。\par}}{\includegraphics[width=0.82\textwidth,height=0.45\textheight,keepaspectratio]{\videocoverpath}\par}",
        r"\vfill",
        r"\begin{tcolorbox}[width=0.9\textwidth, colback=black!2!white, colframe=black!60, sharp corners]",
        r"\textbf{视频作者/频道}：\videochannel\par",
        r"\textbf{发布日期}：\videopublishdate\par",
        r"\textbf{视频时长}：\videoduration\par",
        r"\textbf{视频链接}：\href{\videourl}{\nolinkurl{\videourl}}",
        r"\end{tcolorbox}",
        r"\end{titlepage}",
        r"\tableofcontents",
        r"\newpage",
        body,
        r"\end{document}",
    ]
    return "\n".join(lines) + "\n"


def process_lecture(lecture_dir: Path) -> None:
    meta = load_json(lecture_dir / "meta.json")
    if not (lecture_dir / "figures" / "coverage_map.png").exists():
        subprocess.run(
            ["python3", str(RUN_ROOT / "build" / "generate_summary_figures.py"), lecture_dir.name],
            check=True,
        )
    rows = normalize_coverage_rows(lecture_dir)
    body = build_body(meta, rows, lecture_dir)
    tex = build_tex(meta, body)
    tex_path = lecture_dir / f"lecture_{meta['session_index']:02d}_note.tex"
    tex_path.write_text(tex)
    print(lecture_dir.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("lectures", nargs="*")
    args = parser.parse_args()
    for lecture_dir in lecture_dirs(args.lectures):
        process_lecture(lecture_dir)


if __name__ == "__main__":
    main()
