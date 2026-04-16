#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import subprocess
import urllib.request
from pathlib import Path


RUN_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RUN_ROOT.parents[1]

COURSE_ID = "cmu-s11-751-18-781-fall-2023"
COURSE_TITLE = "CMU S11-751/18-781: Speech Recognition and Understanding (Fall 2023)"
COURSE_PAGE_URL = "https://www.wavlab.org/activities/2023/11751-2023f/"
CMU_COURSE_INFO_URL = "https://courses.ece.cmu.edu/18781"
YOUTUBE_CHANNEL_URL = "https://www.youtube.com/channel/UCLrv4TqqVoyAM3f-dV1Mkjw"
YOUTUBE_CHANNEL_VIDEOS_URL = YOUTUBE_CHANNEL_URL + "/videos"
PLAYLIST_URL = "https://www.youtube.com/playlist?list=PLfVqr2l0FG-tW8d5ZSz-_tCgQed_F1ndb"

CS224S_HOME_URL = "https://web.stanford.edu/class/cs224s/"
CS224S_SYLLABUS_URL = "https://web.stanford.edu/class/cs224s/semesters/2025-spring/syllabus"
CS224S_FAQ_URL = "https://web.stanford.edu/class/cs224s/faq"
CS224S_ASSIGNMENTS_URL = "https://web.stanford.edu/class/cs224s/semesters/2025-spring/assignments/"
CS224S_PROJECT_URL = "https://web.stanford.edu/class/cs224s/semesters/2025-spring/project"

LECTURES = [
    {
        "session_index": 1,
        "date": "2023-08-28",
        "kind": "lecture",
        "title": "Course Overview",
        "title_short": "Course Overview",
        "slug": "course_overview",
        "topics": ["Course explanation and introduction"],
        "public_lecture_number": None,
        "video_id": None,
        "video_url": None,
        "video_title": None,
        "slide_url": None,
        "mapping_notes": "Official WAVLab course page lists the session; no public WAVLab video was confirmed during bootstrap.",
    },
    {
        "session_index": 2,
        "date": "2023-08-30",
        "kind": "lecture",
        "title": "Introduction of Speech Recognition",
        "title_short": "Introduction of Speech Recognition",
        "slug": "introduction_of_speech_recognition",
        "topics": ["Evaluation metric", "How to transcribe speech", "Databases"],
        "public_lecture_number": 2,
        "video_id": "thEKvvFwMvQ",
        "video_url": "https://www.youtube.com/watch?v=thEKvvFwMvQ",
        "video_title": "[Fall 2023] Speech Recognition and Understanding (Lecture2: Introduction of Speech Recognition)",
        "slide_url": None,
        "mapping_notes": "",
    },
    {
        "session_index": 3,
        "date": "2023-09-06",
        "kind": "lecture",
        "title": "Speech Recognition Formulations",
        "title_short": "Speech Recognition Formulations",
        "slug": "speech_recognition_formulations",
        "topics": ["Probabilistic rules", "From Bayes decision theory to HMM + n-gram, CTC, RNN-T, and attention"],
        "public_lecture_number": 3,
        "video_id": "Gjc0mSTsoms",
        "video_url": "https://www.youtube.com/watch?v=Gjc0mSTsoms",
        "video_title": "[Fall 2023] Speech Recognition and Understanding (Lecture 3: Speech Recognition Formulation)",
        "slide_url": None,
        "mapping_notes": "",
    },
    {
        "session_index": 4,
        "date": "2023-09-11",
        "kind": "lecture",
        "title": "Feature Extraction",
        "title_short": "Feature Extraction",
        "slug": "feature_extraction",
        "topics": ["Basic pipeline", "Some advances in feature extractions"],
        "public_lecture_number": 4,
        "video_id": "Z5v_hj__CWU",
        "video_url": "https://www.youtube.com/watch?v=Z5v_hj__CWU",
        "video_title": "[Fall 2023] Speech Recognition and Understanding (Lecture 4: Feature extraction)",
        "slide_url": None,
        "mapping_notes": "",
    },
    {
        "session_index": 5,
        "date": "2023-09-13",
        "kind": "lecture",
        "title": "Acoustic Model Overview",
        "title_short": "Acoustic Model Overview",
        "slug": "acoustic_model_overview",
        "topics": ["Acoustic model overview"],
        "public_lecture_number": None,
        "video_id": None,
        "video_url": None,
        "video_title": None,
        "slide_url": None,
        "mapping_notes": "The official syllabus lists this topic, but no corresponding public WAVLab recording was confirmed during bootstrap.",
    },
    {
        "session_index": 6,
        "date": "2023-09-18",
        "kind": "lecture",
        "title": "Alignment Problems",
        "title_short": "Alignment Problems",
        "slug": "alignment_problems",
        "topics": ["3 state left-to-right HMM", "CTC", "Transducer"],
        "public_lecture_number": 5,
        "video_id": "cnd25kvveZk",
        "video_url": "https://www.youtube.com/watch?v=cnd25kvveZk",
        "video_title": "[Fall 2023] Speech Recognition and Understanding (Lecture 5: Alignments)",
        "slide_url": None,
        "mapping_notes": "The public video numbering is offset relative to the official syllabus row count.",
    },
    {
        "session_index": 7,
        "date": "2023-09-20",
        "kind": "lecture",
        "title": "K-means, GMM, EM Algorithm",
        "title_short": "K-means, GMM, EM Algorithm",
        "slug": "kmeans_gmm_em_algorithm",
        "topics": ["K-means", "GMM", "EM algorithm"],
        "public_lecture_number": 7,
        "video_id": "-1jTd8zM6r4",
        "video_url": "https://www.youtube.com/watch?v=-1jTd8zM6r4",
        "video_title": "[Fall 2023] Speech Recognition and Understanding (Lecture 7: Hidden Markov Models)",
        "slide_url": None,
        "mapping_notes": "The official course page lists K-means/GMM/EM on 2023-09-20, while the public recording title says Hidden Markov Models. Treat this as a source discrepancy until transcript coverage resolves it.",
    },
    {
        "session_index": 8,
        "date": "2023-09-25",
        "kind": "lecture",
        "title": "Forward-Backward Algorithm for HMM",
        "title_short": "Forward-Backward for HMM I",
        "slug": "forward_backward_hmm_i",
        "topics": ["Forward-backward algorithm for HMM"],
        "public_lecture_number": 8,
        "video_id": "6_oNjh0nf-E",
        "video_url": "https://www.youtube.com/watch?v=6_oNjh0nf-E",
        "video_title": "[Fall 2023] Speech Recognition and Understanding (Lecture 8: Hidden Markov Models part II)",
        "slide_url": None,
        "mapping_notes": "Public video title uses Hidden Markov Models part II, while the syllabus row emphasizes the forward-backward algorithm.",
    },
    {
        "session_index": 9,
        "date": "2023-09-27",
        "kind": "lecture",
        "title": "Forward-Backward Algorithm for HMM",
        "title_short": "Forward-Backward for HMM II",
        "slug": "forward_backward_hmm_ii",
        "topics": ["Forward-backward algorithm for HMM"],
        "public_lecture_number": 9,
        "video_id": "ct_ZEdYi12Q",
        "video_url": "https://www.youtube.com/watch?v=ct_ZEdYi12Q",
        "video_title": "[Fall 2023] Speech Recognition and Understanding (Lecture 9: Hidden Markov Models part III)",
        "slide_url": None,
        "mapping_notes": "Public video title uses Hidden Markov Models part III, while the official syllabus keeps the forward-backward naming.",
    },
    {
        "session_index": 10,
        "date": "2023-10-02",
        "kind": "lecture",
        "title": "Forward-Backward Algorithm for CTC and Viterbi Algorithm",
        "title_short": "CTC Forward-Backward and Viterbi",
        "slug": "forward_backward_ctc_viterbi",
        "topics": ["Forward-backward algorithm for CTC", "Viterbi algorithm"],
        "public_lecture_number": 10,
        "video_id": "kSIndDl35X8",
        "video_url": "https://www.youtube.com/watch?v=kSIndDl35X8",
        "video_title": "[Fall 2023] Speech Recognition and Understanding (Lecture 10: Forward-backward algorithm for CTC)",
        "slide_url": None,
        "mapping_notes": "",
    },
    {
        "session_index": 11,
        "date": "2023-10-02",
        "kind": "lecture",
        "title": "N-gram Language Models",
        "title_short": "N-gram Language Models",
        "slug": "ngram_language_models",
        "topics": ["N-gram language models"],
        "public_lecture_number": 11,
        "video_id": "8_Y3g7pM88Y",
        "video_url": "https://www.youtube.com/watch?v=8_Y3g7pM88Y",
        "video_title": "[Fall 2023] Speech Recognition and Understanding (Lecture 11: N-gram Language Models)",
        "slide_url": None,
        "mapping_notes": "",
    },
    {
        "session_index": 12,
        "date": "2023-10-11",
        "kind": "lecture",
        "title": "Search",
        "title_short": "Search",
        "slug": "search",
        "topics": ["Time-synchronous beam search", "Label-synchronous beam search", "N-best and lattice", "Rescoring"],
        "public_lecture_number": 13,
        "video_id": "rRGgO8vZEW8",
        "video_url": "https://www.youtube.com/watch?v=rRGgO8vZEW8",
        "video_title": "[Fall 2023] Speech Recognition and Understanding (Lecture 13: Search)",
        "slide_url": None,
        "mapping_notes": "Public video numbering appears to skip a non-public lecture slot around the midterm week.",
    },
    {
        "session_index": 13,
        "date": "2023-10-23",
        "kind": "lecture",
        "title": "ESPnet Hands-on Tutorial I",
        "title_short": "ESPnet Tutorial I",
        "slug": "espnet_hands_on_i",
        "topics": ["Introduction of toolkit", "How to make a new recipe"],
        "public_lecture_number": None,
        "video_id": None,
        "video_url": None,
        "video_title": None,
        "slide_url": None,
        "mapping_notes": "The official syllabus lists the session, but no public WAVLab recording was confirmed during bootstrap.",
    },
    {
        "session_index": 14,
        "date": "2023-10-25",
        "kind": "lecture",
        "title": "ESPnet Hands-on Tutorial II",
        "title_short": "ESPnet Tutorial II",
        "slug": "espnet_hands_on_ii",
        "topics": ["How to make a new task"],
        "public_lecture_number": None,
        "video_id": None,
        "video_url": None,
        "video_title": None,
        "slide_url": None,
        "mapping_notes": "The official syllabus lists the session, but no public WAVLab recording was confirmed during bootstrap.",
    },
    {
        "session_index": 15,
        "date": "2023-10-30",
        "kind": "lecture",
        "title": "Deep Neural Network for Acoustic Modeling",
        "title_short": "DNN for Acoustic Modeling",
        "slug": "dnn_for_acoustic_modeling",
        "topics": ["Deep neural network for acoustic modeling"],
        "public_lecture_number": 16,
        "video_id": "HnaLmkym0Ec",
        "video_url": "https://www.youtube.com/watch?v=HnaLmkym0Ec",
        "video_title": "[Fall 2023] Speech Recognition and Understanding (Lecture 16: DNN for Acoustic Modeling)",
        "slide_url": None,
        "mapping_notes": "",
    },
    {
        "session_index": 16,
        "date": "2023-11-01",
        "kind": "lecture",
        "title": "Neural Network Language Model",
        "title_short": "Neural Network Language Model",
        "slug": "neural_network_language_model",
        "topics": ["Neural network language model"],
        "public_lecture_number": 17,
        "video_id": "JmDeT8X9Izg",
        "video_url": "https://www.youtube.com/watch?v=JmDeT8X9Izg",
        "video_title": "[Fall 2023] Speech Recognition and Understanding (Lecture 17: Neural Network Language Model)",
        "slide_url": None,
        "mapping_notes": "",
    },
    {
        "session_index": 17,
        "date": "2023-11-06",
        "kind": "lecture",
        "title": "End-to-End ASR: Attention",
        "title_short": "E2E ASR: Attention",
        "slug": "end_to_end_asr_attention",
        "topics": ["End-to-End ASR: Attention"],
        "public_lecture_number": 18,
        "video_id": "1FsamWmRO7Q",
        "video_url": "https://www.youtube.com/watch?v=1FsamWmRO7Q",
        "video_title": "[Fall 2023] Speech Recognition and Understanding (Lecture 18: End-to-End ASR: Attention)",
        "slide_url": None,
        "mapping_notes": "",
    },
    {
        "session_index": 18,
        "date": "2023-11-08",
        "kind": "lecture",
        "title": "End-to-End ASR: CTC",
        "title_short": "E2E ASR: CTC",
        "slug": "end_to_end_asr_ctc",
        "topics": ["End-to-End ASR: CTC"],
        "public_lecture_number": 19,
        "video_id": "00GHpnRTAQE",
        "video_url": "https://www.youtube.com/watch?v=00GHpnRTAQE",
        "video_title": "[Fall 2023] Speech Recognition and Understanding (Lecture 19: End-to-End ASR: CTC)",
        "slide_url": None,
        "mapping_notes": "",
    },
    {
        "session_index": 19,
        "date": "2023-11-13",
        "kind": "lecture",
        "title": "End-to-End ASR: RNN-T",
        "title_short": "E2E ASR: RNN-T",
        "slug": "end_to_end_asr_rnnt",
        "topics": ["End-to-End ASR: RNN-T"],
        "public_lecture_number": 20,
        "video_id": "BQBOu9BOFpc",
        "video_url": "https://www.youtube.com/watch?v=BQBOu9BOFpc",
        "video_title": "[Fall 2023] Speech Recognition and Understanding (Lecture 20: End-to-End ASR: RNN Transducer)",
        "slide_url": None,
        "mapping_notes": "",
    },
    {
        "session_index": 20,
        "date": "2023-11-15",
        "kind": "lecture",
        "title": "Advanced Topics on End-to-End ASR I",
        "title_short": "Advanced E2E ASR I",
        "slug": "advanced_topics_on_end_to_end_asr_i",
        "topics": ["Advanced topics on end-to-end ASR I"],
        "public_lecture_number": 21,
        "video_id": "08BIoO2n-TM",
        "video_url": "https://www.youtube.com/watch?v=08BIoO2n-TM",
        "video_title": "[Fall 2023] Speech Recognition and Understanding (Lecture 21: Advanced topics on end-to-end ASR)",
        "slide_url": None,
        "mapping_notes": "",
    },
    {
        "session_index": 21,
        "date": "2023-11-20",
        "kind": "lecture",
        "title": "Advanced Topics on End-to-End ASR II",
        "title_short": "Advanced E2E ASR II",
        "slug": "advanced_topics_on_end_to_end_asr_ii",
        "topics": ["Advanced topics on end-to-end ASR II"],
        "public_lecture_number": 22,
        "video_id": "MtwolOGs0_A",
        "video_url": "https://www.youtube.com/watch?v=MtwolOGs0_A",
        "video_title": "[Fall 2023] Speech Recognition and Understanding (Lecture 22: Advanced topics on end-to-end ASR II)",
        "slide_url": None,
        "mapping_notes": "",
    },
    {
        "session_index": 22,
        "date": "2023-11-27",
        "kind": "guest_lecture",
        "title": "Guest Lecture I",
        "title_short": "Guest Lecture I",
        "slug": "guest_lecture_i",
        "topics": ["Guest Lecture"],
        "public_lecture_number": None,
        "video_id": None,
        "video_url": None,
        "video_title": None,
        "slide_url": None,
        "mapping_notes": "The official syllabus lists a guest lecture, but no public official recording was confirmed during bootstrap.",
    },
    {
        "session_index": 23,
        "date": "2023-11-29",
        "kind": "guest_lecture",
        "title": "Guest Lecture II",
        "title_short": "Guest Lecture II",
        "slug": "guest_lecture_ii",
        "topics": ["Guest Lecture"],
        "public_lecture_number": None,
        "video_id": None,
        "video_url": None,
        "video_title": None,
        "slide_url": None,
        "mapping_notes": "The official syllabus lists a guest lecture, but no public official recording was confirmed during bootstrap.",
    },
]

NON_LECTURE_SESSIONS = [
    {"date": "2023-10-09", "kind": "midterm", "title": "Midterm Exam"},
    {"date": "2023-12-04", "kind": "project_event", "title": "Project Event"},
    {"date": "2023-12-06", "kind": "project_event", "title": "Project Event"},
]

CS224S_SUPPLEMENT = {
    "course_id": "stanford-cs224s-spring-2025",
    "title": "CS224S: Spoken Language Processing (Spring 2025)",
    "home_url": CS224S_HOME_URL,
    "pages": [
        {"source_id": "cs224s_home", "source_type": "course_homepage", "origin_url": CS224S_HOME_URL, "local_path": "pages/index.html"},
        {"source_id": "cs224s_syllabus", "source_type": "course_syllabus", "origin_url": CS224S_SYLLABUS_URL, "local_path": "pages/syllabus.html"},
        {"source_id": "cs224s_faq", "source_type": "course_faq", "origin_url": CS224S_FAQ_URL, "local_path": "pages/faq.html"},
        {"source_id": "cs224s_assignments", "source_type": "assignments_index", "origin_url": CS224S_ASSIGNMENTS_URL, "local_path": "pages/assignments.html"},
        {"source_id": "cs224s_project", "source_type": "project_page", "origin_url": CS224S_PROJECT_URL, "local_path": "pages/project.html"},
    ],
    "topic_groups": [
        {
            "topic": "tts",
            "label": "Text-to-Speech",
            "resources": [
                {"source_id": "cs224s_lec03_slide", "source_type": "slide_pdf", "origin_url": "https://web.stanford.edu/class/cs224s/semesters/2025-spring/lecture-slides/224s.25.lec3.pdf", "local_path": "slides/224s.25.lec3.pdf"},
                {"source_id": "cs224s_lec04_slide", "source_type": "slide_pdf", "origin_url": "https://web.stanford.edu/class/cs224s/semesters/2025-spring/lecture-slides/224s.25.lec4.pdf", "local_path": "slides/224s.25.lec4.pdf"},
                {"source_id": "cs224s_lec05_slide", "source_type": "slide_pdf", "origin_url": "https://web.stanford.edu/class/cs224s/semesters/2025-spring/lecture-slides/224s.25.lec5.pdf", "local_path": "slides/224s.25.lec5.pdf"},
                {"source_id": "cs224s_lec06_slide", "source_type": "slide_pdf", "origin_url": "https://web.stanford.edu/class/cs224s/semesters/2025-spring/lecture-slides/224s.25.lec6.pdf", "local_path": "slides/224s.25.lec6.pdf"},
            ],
        },
        {
            "topic": "asr",
            "label": "Automatic Speech Recognition",
            "resources": [
                {"source_id": "cs224s_lec09_slide", "source_type": "slide_pdf", "origin_url": "https://web.stanford.edu/class/cs224s/semesters/2025-spring/lecture-slides/224s.25.lec9.pdf", "local_path": "slides/224s.25.lec9.pdf"},
                {"source_id": "cs224s_lec10_slide", "source_type": "slide_pdf", "origin_url": "https://web.stanford.edu/class/cs224s/semesters/2025-spring/lecture-slides/224s.25.lec10.pdf", "local_path": "slides/224s.25.lec10.pdf"},
                {"source_id": "cs224s_lec11_slide", "source_type": "slide_pdf", "origin_url": "https://web.stanford.edu/class/cs224s/semesters/2025-spring/lecture-slides/224s.25.lec11.pdf", "local_path": "slides/224s.25.lec11.pdf"},
                {"source_id": "cs224s_assignment_a3", "source_type": "assignment_page", "origin_url": "https://web.stanford.edu/class/cs224s/semesters/2025-spring/assignments/a3", "local_path": "pages/assignment_a3.html"},
            ],
        },
        {
            "topic": "foundation_models",
            "label": "Speech Foundation Models and Multilingual ASR",
            "resources": [
                {"source_id": "cs224s_lec13_slide", "source_type": "slide_pdf", "origin_url": "https://web.stanford.edu/class/cs224s/semesters/2025-spring/lecture-slides/224s.25.lec13.pdf", "local_path": "slides/224s.25.lec13.pdf"},
                {"source_id": "cs224s_lec14_slide", "source_type": "slide_pdf", "origin_url": "https://web.stanford.edu/class/cs224s/semesters/2025-spring/lecture-slides/224s.25.lec14.pdf", "local_path": "slides/224s.25.lec14.pdf"},
                {"source_id": "cs224s_lec15_slide", "source_type": "slide_pdf", "origin_url": "https://web.stanford.edu/class/cs224s/semesters/2025-spring/lecture-slides/224s.25.lec15.shared.pdf", "local_path": "slides/224s.25.lec15.shared.pdf"},
            ],
        },
        {
            "topic": "dialog",
            "label": "Spoken Dialog and LLM-Based Dialog",
            "resources": [
                {"source_id": "cs224s_lec07_slide", "source_type": "slide_pdf", "origin_url": "https://web.stanford.edu/class/cs224s/semesters/2025-spring/lecture-slides/224s.25.lec7.pdf", "local_path": "slides/224s.25.lec7.pdf"},
                {"source_id": "cs224s_lec08_slide", "source_type": "slide_pdf", "origin_url": "https://web.stanford.edu/class/cs224s/semesters/2025-spring/lecture-slides/224s.25.lec8.pdf", "local_path": "slides/224s.25.lec8.pdf"},
                {"source_id": "cs224s_lec16_syllabus_section", "source_type": "syllabus_section", "origin_url": "https://web.stanford.edu/class/cs224s/semesters/2025-spring/syllabus#lecture-16-wed-52125", "local_path": "pages/syllabus.html"},
                {"source_id": "cs224s_lec18_syllabus_section", "source_type": "syllabus_section", "origin_url": "https://web.stanford.edu/class/cs224s/semesters/2025-spring/syllabus#lecture-18-mon-6225", "local_path": "pages/syllabus.html"},
                {"source_id": "cs224s_assignment_a1", "source_type": "assignment_page", "origin_url": "https://web.stanford.edu/class/cs224s/semesters/2025-spring/assignments/a1", "local_path": "pages/assignment_a1.html"},
                {"source_id": "cs224s_assignment_a2", "source_type": "assignment_page", "origin_url": "https://web.stanford.edu/class/cs224s/semesters/2025-spring/assignments/a2", "local_path": "pages/assignment_a2.html"},
            ],
        },
    ],
    "video_access_note": "Official CS224S Spring 2025 materials state that lecture recordings are available only to registered students via Canvas.",
}


def ensure_dirs() -> None:
    for dirname in [
        "build",
        "book",
        "lectures",
        "materials/slides",
        "meta",
        "raw",
        "supplement/cs224s_spring2025/pages",
        "supplement/cs224s_spring2025/slides",
        "text",
    ]:
        (RUN_ROOT / dirname).mkdir(parents=True, exist_ok=True)


def rel(path: Path | None) -> str | None:
    if path is None:
        return None
    return str(path.relative_to(RUN_ROOT))


def log_download_failure(url: str, dest: Path, error: str) -> None:
    path = RUN_ROOT / "meta" / "download_failures.jsonl"
    row = {
        "url": url,
        "dest": rel(dest),
        "error": error,
    }
    with path.open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def download(url: str | None, dest: Path) -> bool:
    if not url:
        return False
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as exc:
        log_download_failure(url, dest, repr(exc))
        return False
    return True


def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def clear_dir(path: Path) -> None:
    if not path.exists():
        return
    for child in sorted(path.iterdir()):
        if child.is_dir():
            clear_dir(child)
            child.rmdir()
        else:
            child.unlink()


def raw_dir(item: dict) -> Path:
    video_id = item.get("video_id") or "missing"
    return RUN_ROOT / "raw" / f"{item['session_index']:02d}_{video_id}"


def text_dir(item: dict) -> Path:
    return RUN_ROOT / "text" / f"{item['session_index']:02d}_{item['slug']}"


def lecture_dir(item: dict) -> Path:
    return RUN_ROOT / "lectures" / f"{item['session_index']:02d}_{item['slug']}"


def lecture_slide_pdf(item: dict) -> Path | None:
    if not item.get("slide_url"):
        return None
    return RUN_ROOT / "materials" / "slides" / f"{item['session_index']:02d}_{item['slug']}.pdf"


def best_subtitle_path(item: dict) -> Path | None:
    if not item.get("video_url"):
        return None
    candidates = sorted(raw_dir(item).glob("*.srt"))
    if not candidates:
        return None
    scored: list[tuple[int, str, Path]] = []
    for path in candidates:
        name = path.name
        score = 100
        if ".en-US.srt" in name:
            score = 10
        elif ".en.srt" in name:
            score = 20
        elif ".en-GB.srt" in name:
            score = 30
        elif ".en-orig.srt" in name:
            score = 40
        scored.append((score, name, path))
    scored.sort()
    return scored[0][2]


def best_cover_path(item: dict) -> Path | None:
    if not item.get("video_url"):
        return None
    candidates = sorted(raw_dir(item).glob("*.jpg"))
    return candidates[0] if candidates else None


def srt_to_text(srt_path: Path) -> str:
    text = srt_path.read_text(errors="ignore")
    blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n"))
    lines: list[str] = []
    for block in blocks:
        raw_lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(raw_lines) < 2:
            continue
        maybe_ts = raw_lines[1] if raw_lines[0].isdigit() else raw_lines[0]
        if "-->" not in maybe_ts:
            continue
        payload = raw_lines[2:] if raw_lines[0].isdigit() else raw_lines[1:]
        payload = [re.sub(r"<[^>]+>", "", line).strip() for line in payload]
        payload = [line for line in payload if line]
        if payload:
            lines.append(f"[{maybe_ts}] {' '.join(payload)}")
    return "\n".join(lines).strip() + ("\n" if lines else "")


def extract_slide_text(pdf_path: Path, out_path: Path) -> None:
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    run(["pdftotext", str(pdf_path), str(out_path)])


def render_slide_pages(pdf_path: Path, out_dir: Path) -> None:
    if out_dir.exists() and any(out_dir.glob("page-*.png")):
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / "page"
    run(["pdftoppm", "-png", str(pdf_path), str(prefix)])


def ensure_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(os.path.relpath(src, dst.parent))


def download_course_pages() -> None:
    download(COURSE_PAGE_URL, RUN_ROOT / "meta" / "course_page.html")
    download(CMU_COURSE_INFO_URL, RUN_ROOT / "meta" / "cmu_course_info.html")
    download(CS224S_HOME_URL, RUN_ROOT / "supplement" / "cs224s_spring2025" / "pages" / "index.html")
    download(CS224S_SYLLABUS_URL, RUN_ROOT / "supplement" / "cs224s_spring2025" / "pages" / "syllabus.html")
    download(CS224S_FAQ_URL, RUN_ROOT / "supplement" / "cs224s_spring2025" / "pages" / "faq.html")
    download(CS224S_ASSIGNMENTS_URL, RUN_ROOT / "supplement" / "cs224s_spring2025" / "pages" / "assignments.html")
    download(CS224S_PROJECT_URL, RUN_ROOT / "supplement" / "cs224s_spring2025" / "pages" / "project.html")
    download("https://web.stanford.edu/class/cs224s/semesters/2025-spring/assignments/a1", RUN_ROOT / "supplement" / "cs224s_spring2025" / "pages" / "assignment_a1.html")
    download("https://web.stanford.edu/class/cs224s/semesters/2025-spring/assignments/a2", RUN_ROOT / "supplement" / "cs224s_spring2025" / "pages" / "assignment_a2.html")
    download("https://web.stanford.edu/class/cs224s/semesters/2025-spring/assignments/a3", RUN_ROOT / "supplement" / "cs224s_spring2025" / "pages" / "assignment_a3.html")
    for topic_group in CS224S_SUPPLEMENT["topic_groups"]:
        for resource in topic_group["resources"]:
            if resource["source_type"] == "slide_pdf":
                download(resource["origin_url"], RUN_ROOT / "supplement" / "cs224s_spring2025" / resource["local_path"])


def fetch_video_assets(item: dict) -> None:
    if not item.get("video_url"):
        return
    rdir = raw_dir(item)
    rdir.mkdir(parents=True, exist_ok=True)
    target_prefix = rdir / f"{item['session_index']:02d}_{item['video_id']}"
    info_json = target_prefix.with_suffix(".info.json")
    if info_json.exists():
        try:
            info = json.loads(info_json.read_text())
        except json.JSONDecodeError:
            info = {}
        if info.get("id") == item["video_id"] and list(rdir.glob("*.jpg")) and list(rdir.glob("*.srt")):
            return
        if info.get("id") not in {None, item["video_id"]}:
            clear_dir(rdir)

    base_cmd = [
        "yt-dlp",
        "--no-playlist",
        "--skip-download",
        "-o",
        str(target_prefix) + ".%(ext)s",
    ]
    run(
        base_cmd
        + [
            "--write-info-json",
            "--write-thumbnail",
            "--convert-thumbnails",
            "jpg",
            item["video_url"],
        ]
    )
    run(
        base_cmd
        + [
            "--write-subs",
            "--sub-langs",
            "en-US,en,en-GB",
            "--sub-format",
            "srt/vtt/best",
            "--convert-subs",
            "srt",
            item["video_url"],
        ],
        check=False,
    )
    if not list(rdir.glob("*.srt")):
        run(
            base_cmd
            + [
                "--write-auto-subs",
                "--sub-langs",
                "en-US,en,en-GB",
                "--sub-format",
                "srt/vtt/best",
                "--convert-subs",
                "srt",
                item["video_url"],
            ],
            check=False,
        )


def fetch_playlist_metadata() -> None:
    out = RUN_ROOT / "meta" / "playlist_full.json"
    if out.exists() and out.stat().st_size > 0:
        return
    output = subprocess.check_output(
        ["yt-dlp", "--flat-playlist", "--dump-single-json", PLAYLIST_URL],
        text=True,
    )
    out.write_text(output)


def write_course_page_excerpt(item: dict) -> Path:
    tdir = text_dir(item)
    tdir.mkdir(parents=True, exist_ok=True)
    path = tdir / "course_page_excerpt.txt"
    lines = [
        f"Date: {item['date']}",
        f"Title: {item['title']}",
        f"Kind: {item['kind']}",
        "Topics:",
    ]
    lines.extend([f"- {topic}" for topic in item["topics"]])
    if item.get("video_title"):
        lines.append(f"Public video title: {item['video_title']}")
    if item.get("mapping_notes"):
        lines.append(f"Mapping notes: {item['mapping_notes']}")
    path.write_text("\n".join(lines) + "\n")
    return path


def write_text_bundle(item: dict) -> None:
    tdir = text_dir(item)
    tdir.mkdir(parents=True, exist_ok=True)
    write_course_page_excerpt(item)
    subtitle = best_subtitle_path(item)
    transcript_path = tdir / "transcript.txt"
    if subtitle and not transcript_path.exists():
        transcript_path.write_text(srt_to_text(subtitle))
    slide_pdf = lecture_slide_pdf(item)
    official_text = tdir / "official.txt"
    if slide_pdf and slide_pdf.exists():
        extract_slide_text(slide_pdf, official_text)


def placeholder_transcript_rows(item: dict) -> list[dict]:
    text = " | ".join(item["topics"])
    return [
        {
            "unit_id": f"schedule_{item['session_index']:02d}_0001",
            "source_type": "course_page_schedule_entry",
            "source_id": "course_schedule_json",
            "loc": {
                "start": f"{item['date']} schedule entry",
                "end": f"{item['date']} schedule entry",
                "date": item["date"],
                "session_index": item["session_index"],
            },
            "text": f"{item['title']}: {text}",
            "required": True,
        }
    ]


def placeholder_slide_rows(item: dict) -> list[dict]:
    return [
        {
            "unit_id": f"schedule_slide_{item['session_index']:02d}_0001",
            "source_type": "course_page_schedule_entry",
            "source_id": "course_schedule_json",
            "loc": {"date": item["date"], "session_index": item["session_index"]},
            "text": f"{item['title']} | {' | '.join(item['topics'])}",
            "asset_path": "course_page_excerpt.txt",
            "required": True,
        }
    ]


def source_entry(
    source_id: str,
    source_type: str,
    local_path: Path | None,
    required: bool,
    origin_url: str | None = None,
    status: str | None = None,
    notes: str = "",
) -> dict:
    final_status = status or ("available" if local_path and local_path.exists() else "missing")
    return {
        "source_id": source_id,
        "source_type": source_type,
        "origin_url": origin_url,
        "local_path": rel(local_path) if local_path and local_path.exists() else None,
        "required_for_coverage": required,
        "status": final_status,
        "notes": notes,
    }


def write_source_manifest(item: dict, ldir: Path) -> None:
    subtitle = best_subtitle_path(item)
    cover = best_cover_path(item)
    slide_pdf = lecture_slide_pdf(item)
    manifest = {
        "course_id": COURSE_ID,
        "course_mode": True,
        "lecture_id": f"{item['session_index']:02d}",
        "lecture_slug": ldir.name,
        "title": item["title"],
        "origin_url": item.get("video_url"),
        "course_page_url": COURSE_PAGE_URL,
        "supplement_manifest": "supplement/cs224s_spring2025/source_manifest.json",
        "sources": [
            source_entry(
                "lecture_meta",
                "lecture_metadata",
                ldir / "meta.json",
                True,
                origin_url=item.get("video_url") or COURSE_PAGE_URL,
                notes="Normalized lecture metadata for harness-managed textbook generation.",
            ),
            source_entry(
                "course_page_html",
                "course_page_html",
                RUN_ROOT / "meta" / "course_page.html",
                True,
                origin_url=COURSE_PAGE_URL,
                notes="Official WAVLab course page.",
            ),
            source_entry(
                "course_page_excerpt",
                "course_page_schedule_excerpt",
                text_dir(item) / "course_page_excerpt.txt",
                True,
                origin_url=COURSE_PAGE_URL,
                notes="Locally normalized excerpt for the official syllabus row.",
            ),
            source_entry(
                "raw_info_json",
                "platform_metadata",
                next(iter(sorted(raw_dir(item).glob("*.info.json"))), None),
                bool(item.get("video_url")),
                origin_url=item.get("video_url"),
                notes="yt-dlp metadata dump for the public recording.",
            ),
            source_entry(
                "cover_jpg",
                "cover_image",
                cover,
                bool(item.get("video_url")),
                notes="Highest-resolution public YouTube thumbnail fetched during bootstrap.",
            ),
            source_entry(
                "subtitle_srt",
                "platform_subtitle",
                subtitle,
                bool(item.get("video_url")),
                notes="Preferred local subtitle track. Auto-generated captions are used when manual captions are absent.",
            ),
            source_entry(
                "transcript_txt",
                "debug_transcript_text",
                text_dir(item) / "transcript.txt",
                False,
                notes="Debug transcript text derived from subtitles.",
            ),
            source_entry(
                "slides_pdf",
                "official_slide_pdf",
                slide_pdf,
                False,
                origin_url=item.get("slide_url"),
                notes="Official slide deck if later discovered or added.",
            ),
            source_entry(
                "official_txt",
                "debug_slide_text",
                text_dir(item) / "official.txt",
                False,
                notes="pdftotext extraction for the official slide deck when available.",
            ),
            source_entry(
                "cs224s_supplement_manifest",
                "supplement_manifest",
                RUN_ROOT / "supplement" / "cs224s_spring2025" / "source_manifest.json",
                False,
                origin_url=CS224S_HOME_URL,
                notes="Supplemental official public materials from CS224S Spring 2025.",
            ),
        ],
    }
    (ldir / "source_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


def write_lecture_dir(item: dict) -> None:
    ldir = lecture_dir(item)
    ldir.mkdir(parents=True, exist_ok=True)
    cover = best_cover_path(item)
    subtitle = best_subtitle_path(item)
    slide_pdf = lecture_slide_pdf(item)
    course_page_excerpt = text_dir(item) / "course_page_excerpt.txt"
    transcript_txt = text_dir(item) / "transcript.txt"
    official_txt = text_dir(item) / "official.txt"

    meta = {
        **item,
        "playlist_index": item["session_index"],
        "course_id": COURSE_ID,
        "course_title": COURSE_TITLE,
        "course_page_url": COURSE_PAGE_URL,
        "youtube_channel_url": YOUTUBE_CHANNEL_URL,
        "thumbnail": rel(cover),
        "subtitle": rel(subtitle),
        "material": rel(slide_pdf),
        "transcript_text": rel(transcript_txt) if transcript_txt.exists() else None,
        "official_text": rel(official_txt) if official_txt.exists() else None,
        "course_page_excerpt": rel(course_page_excerpt),
        "slide_pages_dir": rel(ldir / "pdf_pages"),
        "supplement_manifest": "supplement/cs224s_spring2025/source_manifest.json",
        "course_mode": True,
        "segmentation_required": True,
    }
    (ldir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")

    if cover and cover.exists():
        ensure_symlink(cover, ldir / "cover.jpg")
    if subtitle and subtitle.exists():
        ensure_symlink(subtitle, ldir / "subtitle.srt")
    if transcript_txt.exists():
        ensure_symlink(transcript_txt, ldir / "transcript.txt")
    if official_txt.exists():
        ensure_symlink(official_txt, ldir / "official.txt")
    if course_page_excerpt.exists():
        ensure_symlink(course_page_excerpt, ldir / "course_page_excerpt.txt")
    if slide_pdf and slide_pdf.exists():
        ensure_symlink(slide_pdf, ldir / "slides.pdf")
        render_slide_pages(slide_pdf, ldir / "pdf_pages")

    transcript_jsonl = ldir / "transcript.jsonl"
    if not subtitle and not transcript_jsonl.exists():
        payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in placeholder_transcript_rows(item))
        transcript_jsonl.write_text(payload + "\n")

    slides_jsonl = ldir / "slides.jsonl"
    if (not slide_pdf or not slide_pdf.exists()) and not slides_jsonl.exists():
        payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in placeholder_slide_rows(item))
        slides_jsonl.write_text(payload + "\n")

    coverage_path = ldir / "coverage_units.jsonl"
    if not coverage_path.exists():
        coverage_rows = [
            {
                "unit_id": f"{item['session_index']:02d}-u01",
                "source_type": "course_page_schedule_entry",
                "source_id": "course_page_excerpt",
                "loc": item["date"],
                "kind": ["course_schedule_topic"],
                "summary": f"{item['title']} | {' | '.join(item['topics'])}",
                "required": True,
                "status": "unclassified",
                "mapped_section": None,
                "figure_ids": [],
                "notes": item.get("mapping_notes", ""),
                "unit_type": "course_schedule_topic",
            }
        ]
        coverage_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in coverage_rows) + "\n")

    omission_path = ldir / "omission_log.jsonl"
    if not omission_path.exists():
        omission_path.write_text("")

    if not (ldir / "figure_manifest.json").exists():
        (ldir / "figure_manifest.json").write_text("[]\n")

    write_source_manifest(item, ldir)

    lines = [
        f"# {item['title']}",
        "",
        f"- Date: `{item['date']}`",
        f"- Kind: `{item['kind']}`",
        f"- Official course page: [course_page_excerpt.txt](course_page_excerpt.txt)",
        f"- Public video: [YouTube]({item['video_url']})" if item.get("video_url") else "- Public video: unavailable",
        f"- Public video title: `{item['video_title']}`" if item.get("video_title") else "- Public video title: unavailable",
        f"- Public video lecture number: `{item['public_lecture_number']}`" if item.get("public_lecture_number") else "- Public video lecture number: unavailable",
        f"- Cover: [cover.jpg](cover.jpg)" if (ldir / "cover.jpg").exists() else "- Cover: unavailable",
        f"- Subtitle: [subtitle.srt](subtitle.srt)" if (ldir / "subtitle.srt").exists() else "- Subtitle: unavailable",
        f"- Transcript: [transcript.txt](transcript.txt)" if (ldir / "transcript.txt").exists() else "- Transcript: unavailable",
        f"- Slides: [slides.pdf](slides.pdf)" if (ldir / "slides.pdf").exists() else "- Slides: unavailable",
        f"- Mapping notes: {item['mapping_notes']}" if item.get("mapping_notes") else "- Mapping notes: none",
        "",
        "## Topics",
    ]
    lines.extend([f"- {topic}" for topic in item["topics"]])
    lines.extend(
        [
            "",
            "## Writing requirements",
            "",
            "- Write in Chinese.",
            "- Cover official course-page evidence even when no video or slide is public.",
            "- Mark any CS224S-derived additions as supplementation or extension.",
            "- If material is inaccessible, keep explicit blocking and omission records.",
        ]
    )
    (ldir / "README.md").write_text("\n".join(lines) + "\n")


def write_course_schedule_json() -> None:
    payload = {
        "course_id": COURSE_ID,
        "title": COURSE_TITLE,
        "course_page_url": COURSE_PAGE_URL,
        "cmu_course_info_url": CMU_COURSE_INFO_URL,
        "youtube_channel_url": YOUTUBE_CHANNEL_URL,
        "playlist_url": PLAYLIST_URL,
        "lecture_count": len(LECTURES),
        "lectures": LECTURES,
        "non_lecture_sessions": NON_LECTURE_SESSIONS,
    }
    (RUN_ROOT / "meta" / "course_schedule.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def write_cs224s_supplement_manifest() -> None:
    root = RUN_ROOT / "supplement" / "cs224s_spring2025"
    manifest = {
        **CS224S_SUPPLEMENT,
        "pages": [
            {
                **page,
                "local_path": page["local_path"] if (root / page["local_path"]).exists() else None,
            }
            for page in CS224S_SUPPLEMENT["pages"]
        ],
        "topic_groups": [
            {
                **topic_group,
                "resources": [
                    {
                        **resource,
                        "local_path": resource["local_path"] if (root / resource["local_path"]).exists() else None,
                    }
                    for resource in topic_group["resources"]
                ],
            }
            for topic_group in CS224S_SUPPLEMENT["topic_groups"]
        ],
    }
    (root / "source_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


def write_public_video_catalog() -> None:
    catalog: list[dict] = []
    for item in LECTURES:
        info_path = next(iter(sorted(raw_dir(item).glob("*.info.json"))), None)
        if not info_path:
            continue
        try:
            info = json.loads(info_path.read_text())
        except json.JSONDecodeError:
            continue
        catalog.append(
            {
                "session_index": item["session_index"],
                "official_title": item["title"],
                "public_lecture_number": item.get("public_lecture_number"),
                "video_id": info.get("id"),
                "video_url": item.get("video_url"),
                "video_title": info.get("title"),
                "channel": info.get("channel"),
                "upload_date": info.get("upload_date"),
                "duration": info.get("duration"),
                "duration_string": info.get("duration_string"),
                "subtitle_languages": sorted((info.get("subtitles") or {}).keys()),
                "automatic_caption_languages": sorted((info.get("automatic_captions") or {}).keys()),
                "playlist_url": PLAYLIST_URL,
            }
        )
    (RUN_ROOT / "meta" / "public_video_catalog.json").write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n")


def write_course_manifest_seed() -> None:
    manifest = {
        "course_id": COURSE_ID,
        "title": COURSE_TITLE,
        "playlist_origin_url": PLAYLIST_URL,
        "course_page_url": COURSE_PAGE_URL,
        "course_mode": True,
        "lecture_count": len(LECTURES),
        "supplement_manifest": "supplement/cs224s_spring2025/source_manifest.json",
        "lectures": [
            {
                "lecture_id": f"{item['session_index']:02d}",
                "lecture_slug": f"{item['session_index']:02d}_{item['slug']}",
                "title": item["title"],
                "date": item["date"],
                "video_url": item.get("video_url"),
                "slide_url": item.get("slide_url"),
                "kind": item["kind"],
                "public_lecture_number": item.get("public_lecture_number"),
            }
            for item in LECTURES
        ],
    }
    (RUN_ROOT / "build" / "course_manifest_seed.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


def write_course_plan() -> None:
    plan = {
        "course_id": COURSE_ID,
        "title": COURSE_TITLE,
        "workflow": [
            "bootstrap source inventory",
            "build structured evidence",
            "segment lecture",
            "write coverage ledger",
            "draft lecture note",
            "evaluate and repair",
            "merge deliverable lectures into textbook",
        ],
        "delivery_policy": {
            "require_evaluator_pass": True,
            "require_validator_pass": True,
            "allow_blocked_lectures_with_reason": True,
            "blocked_lectures_must_be_excluded_from_final_book": True,
        },
        "supplement_policy": {
            "supplement_course_id": "stanford-cs224s-spring-2025",
            "allowed_use": ["tts", "asr", "foundation_models", "dialog"],
            "must_label_supplementation": True,
        },
    }
    (RUN_ROOT / "meta" / "course_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")


def write_lectures_index() -> None:
    lines = ["# S11-751/18-781 Lecture Folders", ""]
    for item in LECTURES:
        slug = f"{item['session_index']:02d}_{item['slug']}"
        lines.append(f"- [{item['session_index']:02d} {item['title']}](./{slug}/README.md)")
    (RUN_ROOT / "lectures" / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ensure_dirs()
    download_course_pages()
    fetch_playlist_metadata()
    write_course_schedule_json()
    write_course_plan()
    for item in LECTURES:
        slide_pdf = lecture_slide_pdf(item)
        if slide_pdf:
            download(item["slide_url"], slide_pdf)
        fetch_video_assets(item)
        write_text_bundle(item)
        write_lecture_dir(item)
    write_cs224s_supplement_manifest()
    write_public_video_catalog()
    write_course_manifest_seed()
    write_lectures_index()
    run(
        [
            "python3",
            str(REPO_ROOT / "scripts" / "video_note_harness" / "bootstrap_harness.py"),
            "--run-root",
            str(RUN_ROOT),
        ]
    )
    print(f"bootstrapped={len(LECTURES)}")
    print(RUN_ROOT)


if __name__ == "__main__":
    main()
