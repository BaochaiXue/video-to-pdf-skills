# video-to-pdf-skills

这个仓库托管 Codex skills、共享 harness、课程 run 和最终教材交付物，用于将视频讲座转换为结构化的中文 LaTeX 讲义和最终 PDF。

| Skill | 平台 | 说明 |
|-------|------|------|
| `youtube-render-pdf` | YouTube | 原始版本，利用 YouTube CC 字幕和章节结构 |
| `bilibili-render-pdf` | Bilibili (B站) | 适配 B 站的字幕缺失、登录高清、分P视频等特点 |
| `subtitle-refine` | 字幕清洗 | 上游新增的字幕清洗 skill，用于检查和整理 SRT |

视频转教材 skill 共享相同的写作规则、配图策略和 LaTeX 模板，但在素材获取阶段有平台特定的差异。

## 当前项目状态

本仓库现在分为四层：

- `skills/`：Codex skill 定义，包括 YouTube、Bilibili 和 subtitle refine。
- `skills/references/` 与 `scripts/video_note_harness/`：coverage-first 教材生成的共享规则、schema、validator 和 evaluator。
- `runs/`：每门课程的 source record、lecture workspace、coverage sidecar、评估结果和合并脚本。
- `deliverable/`：最外层本地交付目录，汇总每个已完工 run 的最终 PDF。

当前最外层 `deliverable/` 已汇总 9 本最终教材 PDF。PDF 文件是本地 handoff copies，其中包含超过 GitHub 单文件限制的大文件，因此默认不纳入 Git 跟踪；索引 README 可跟踪。

| Run | 状态 | 最终 PDF |
|-----|------|----------|
| `runs/cme295_fall2025/` | 已生成，已补 `deliverable/` | `cme295_complete_notes.pdf` |
| `runs/cs224r_spring2025/` | 已生成，已补 `deliverable/` | `cs224r_complete_notes.pdf` |
| `runs/cs231n_spring2025/` | 已交付 | `cs231n_complete_notes.pdf` |
| `runs/cs294_194_196_fall2025_agentic_ai/` | 已交付 | `agentic_ai_complete_notes.pdf` |
| `runs/cs294_194_280_sp25_agents_textbook/` | 已交付 | `cs294_194_280_sp25_agents_textbook_complete_notes.pdf` |
| `runs/cs336_all/` | 已交付 | `cs336_complete_notes.pdf` |
| `runs/mit_res9_009_iap2025_neuroblox_textbook/` | 已交付 | `mit_res9_009_iap2025_neuroblox_textbook_complete_notes.pdf` |
| `runs/s11_751_18_781_fall2023_textbook/` | 已交付 | `speech_recognition_understanding_fall2023_textbook.pdf` |
| `runs/s294_277_fall2024_robots_that_learn_textbook/` | 已交付 | `s294_277_complete_textbook.pdf` |

## ASR Backend Policy

当视频没有可用 CC 字幕时，skill 使用统一的“设备感知 ASR backend”策略：

- `CUDA / NVIDIA`：默认使用 `Qwen3-ASR-1.7B + Qwen3-ForcedAligner-0.6B`
- `Apple Silicon Mac`：默认使用 Whisper backend，优先 `whisper.cpp`，也可使用 MLX / `openai-whisper`
- 流程层统一输出：标准 `SRT` 或 `timestamped segments`（至少包含 `start` / `end` / `text`）

这样上层的讲义写作、关键帧定位和图文对齐逻辑只依赖统一时间戳接口，不依赖某个具体 ASR 实现。

### Bilibili 版的核心差异

- **字幕三级回退**：CC 字幕 → 设备感知 ASR backend → 纯视觉模式（B 站大量视频无 CC 字幕）
- **登录获取高清**：1080P+ 需要 cookies（`yt-dlp --cookies-from-browser chrome`）
- **分P视频处理**：自动检测多 P，询问用户处理范围
- **平台话术过滤**：额外排除"一键三连"、"关注投币"等非教学内容
- **ASR 默认实现**：CUDA 机器优先 Qwen3-ASR；Mac 优先 Whisper 生态实现

### 共同特点

- 以视频真实教学内容为主，而不是只依赖字幕转写
- 优先使用原始视频封面作为首页封面图
- 按教学价值提取关键画面、图表、公式和代码片段
- 生成带 `\section{}` / `\subsection{}` 结构的完整 `.tex`
- 最终必须落到可交付的 PDF

## 仓库结构

```text
.
├── LICENSE
├── README.md
├── deliverable/
│   ├── README.md
│   └── *.pdf
├── demos/
│   └── video-render-pdf/
│       └── *.pdf
├── docs/
│   └── quality/
├── runs/
│   └── <course-run>/
├── scripts/
│   └── video_note_harness/
├── skills/
│   ├── subtitle-refine/
│   ├── references/
│   ├── youtube-render-pdf/
│   │   ├── SKILL.md
│   │   ├── agents/
│   │   │   └── openai.yaml
│   │   └── assets/
│   │       └── notes-template.tex
│   └── bilibili-render-pdf/
│       ├── SKILL.md
│       ├── agents/
│       │   └── openai.yaml
│       └── assets/
│           └── notes-template.tex
└── templates/
    ├── tex/
    └── writing/
```

## 包含内容

- `skills/youtube-render-pdf/SKILL.md`
  YouTube 版 skill 的主说明文件，定义适用场景、工作流、写作规则、配图规则和最终交付要求。
- `skills/bilibili-render-pdf/SKILL.md`
  Bilibili 版 skill 的主说明文件，在 YouTube 版基础上增加了字幕回退、分P处理等平台适配。
- `skills/*/assets/notes-template.tex`
  共享的默认 LaTeX 模板，包含首页封面位、盒子样式、代码块样式和正文占位结构。
- `skills/*/agents/openai.yaml`
  给 agent UI 使用的显示名称、简介和默认提示。
- `skills/subtitle-refine/`
  上游新增的字幕清洗 skill，包含 SRT 检查脚本。
- `skills/references/`
  共享 workflow、coverage schema、figure provenance、evaluator playbook 和 note quality rubric。
- `scripts/video_note_harness/`
  跨课程复用的 bootstrap、evaluation、validation 和 note gardening 工具。
- `docs/quality/`
  教材质量原则和已知失败模式。
- `templates/`
  上游新增的通用 TeX 与 writing 模板。
- `demos/video-render-pdf/`
  样例 PDF 输出目录，当前沿用历史目录名 `video-render-pdf`。
- `runs/`
  课程级 record system。每个 run 应保留 source manifests、coverage sidecars、lecture notes、evaluator/validator 记录和最终 deliverable。
- `deliverable/`
  最外层本地 handoff 目录，汇总所有已完工 run 的最终 PDF。

## 使用方式

如果你想在本地 Codex 环境中使用这些 skill，可以把对应目录放到你的技能目录中：

```bash
mkdir -p ~/.codex/skills

# shared references required by both skills
cp -R skills/references ~/.codex/skills/

# YouTube 版
cp -R skills/youtube-render-pdf ~/.codex/skills/

# Bilibili 版
cp -R skills/bilibili-render-pdf ~/.codex/skills/
```

然后在 Codex 中使用对应 skill 处理视频链接，请求生成讲义 `.tex` 和最终 PDF。

## YouTube Canonical Pipeline

当前 YouTube 侧的推荐实现以 `runs/cme295_fall2025/` 为准：

- 每讲独立 workspace，直接从单讲 source bundle 生成 `lecture_XX_note.tex/.pdf`
- 主输入层为 `transcript.jsonl`、`slides.jsonl`、`segments.jsonl`
- `transcript.txt` 和 `official.txt` 仅作为 debug artifacts 保留
- `build/validate_youtube_note.py` 负责 coverage 与 figure manifest 的硬校验

`runs/cs336_all/` 保留为 legacy/reference-only 示例，不再承载新的 YouTube v1 能力，也不再作为单讲生成的 canonical 路线。

## 外部依赖

| 工具 | 用途 |
|------|------|
| `yt-dlp` | 两个 skill 都需要，用于获取 metadata、字幕、封面和视频 |
| `ffmpeg` | 两个 skill 都需要，用于抽音频、切片和抽帧 |
| `xelatex` (TeX Live + CTeX) | 两个 skill 都需要，用于渲染最终 PDF |
| `magick` (ImageMagick) | 两个 skill 都需要，用于图片裁剪和预处理 |
| `qwen-asr` + `Qwen3-ASR-1.7B` + `Qwen3-ForcedAligner-0.6B` | CUDA 机器上的默认 ASR fallback |
| `whisper.cpp` 或 MLX / `openai-whisper` | Apple Silicon Mac 上的默认 ASR fallback |

说明：

- `bilibili-render-pdf` 通常更依赖 ASR fallback，因为 B 站视频缺失 CC 字幕更常见。
- `youtube-render-pdf` 优先使用平台字幕；只有在缺少合适字幕时才需要启用同一套 backend policy。
- 无论底层使用 Qwen 还是 Whisper，最终都应标准化为 `SRT` 或 `timestamped segments`，供后续抽帧和写作阶段复用。

此外，运行 skill 的 coding agent 必须具备一定的读图能力，否则很难选择关键帧，很难做到图文align（即至少是一个还不错的 vlm model，ps. MiniMax 2.7 只是一个纯文本模型）。

## 适用场景

- 技术课程笔记整理
- YouTube / Bilibili 教学视频转 LaTeX 讲义
- 需要封面图、关键帧和总结章节的高质量课程文档生成

## Subagents 的触发

- Codex 中对 `spawn_agent` 的触发限制较严格："Only use spawn_agent if and only if the user explicitly asks for sub-agents, delegation, or parallel agent work." 因此需要在 query 中显式要求，才会触发 subagents。
- 这两个 skill 现在会在长视频、playlist、course mode、分P 等大工作量场景下主动建议你这样写，因为否则 agent 只能单线程做完整覆盖。

```
$youtube-render-pdf https://www.youtube.com/watch?v=vXb2QYOUzl4 请 spawn 多个 subagents 并行执行，隔离上下文，避免 master agent 的“上下文焦虑”，形成一个完整全面的 PDF：
  - 1 个 outline agent：先定全局目录、术语、符号表、章节边界等
  - 5 个 writer agents：各自直接写成完整章节草稿，落盘成 section_*.tex
  - 1 个 figure agent：单独负责抽帧、筛图、crop、脚本生成新的示意图、图注和时间脚注等
  - 1 个 consistency agent：检查重复定义、前后术语不一致、章节衔接断裂
```


## License

仓库保留了根目录下原有的 `LICENSE` 文件。使用、分发或二次修改时，请以该许可证为准。
