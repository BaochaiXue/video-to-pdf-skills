#!/usr/bin/env python3
from __future__ import annotations

import html
import json
import re
import shutil
import subprocess
import sys
import urllib.parse
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.video_note_harness.common import (
    build_segments,
    build_slide_units,
    build_transcript_units,
    extract_slide_pages,
    write_json,
    write_jsonl,
)


RUN_ROOT = Path(__file__).resolve().parents[1]
COURSE_ID = "ucb-s294-277-robots-that-learn-fall-2024"
COURSE_TITLE = "UC Berkeley CS/S294-277: Robots That Learn (Fall 2024)"
COURSE_PAGE_URL = "https://robots-that-learn.github.io/fall2024"
PLAYLIST_URL = "https://youtube.com/playlist?list=PLPaC96j0xdLcYLTSoSk9PO1Yg-1udJd-S&feature=shared"
COURSE_NOTES_URL = "https://drive.google.com/file/d/1UJ2yMrHnMR04xJ8kAecsWd9dhGicFahb/view?usp=sharing"
BACKUP_NOTES_URL = "https://drive.google.com/file/d/1xVJPowbvqQWjAKmL-_9oviynx-_F3Cm5/view?usp=sharing"
MAX_GOOGLE_DRIVE_BYTES = 300 * 1024 * 1024


def resource_url(path: str) -> str:
    return urllib.parse.urljoin(COURSE_PAGE_URL, path)


LECTURES: list[dict[str, Any]] = [
    {
        "chapter_index": 1,
        "schedule_id": "1A",
        "date": "2024-09-09",
        "title": "Lecture 1A: Introduction",
        "title_short": "Introduction",
        "slug": "introduction",
        "youtube_playlist_index": None,
        "video_id": None,
        "video_url": None,
        "slide_urls": [
            "https://docs.google.com/presentation/d/1UKZNyoHFu8mWRt0XFhaX-5VzRanN-cjR/edit?usp=sharing&ouid=100618406782889124957&rtpof=true&sd=true"
        ],
        "backup_video_urls": [],
        "topics": [
            "Course framing and logistics",
            "Why robot learning is hard",
            "Embodied intelligence motivation",
            "Course roadmap",
        ],
        "readings": [],
    },
    {
        "chapter_index": 2,
        "schedule_id": "1B",
        "date": "2024-09-09",
        "title": "Lecture 1B: Biomechanics of Walking and Running",
        "title_short": "Biomechanics of Walking and Running",
        "slug": "biomechanics_of_walking_and_running",
        "youtube_playlist_index": None,
        "video_id": None,
        "video_url": None,
        "slide_urls": [
            "https://docs.google.com/presentation/d/1f394PmuISnU3F_I7eluYVfl1ImPhnsZp/edit?usp=sharing&ouid=100618406782889124957&rtpof=true&sd=true"
        ],
        "backup_video_urls": [],
        "topics": [
            "Biomechanics of locomotion",
            "Walking and running dynamics",
            "Biological inspiration for robotics",
            "Neuromechanics",
        ],
        "readings": [
            {
                "text": "T. K. Uchida and S. L. Delp. Biomechanics of movement: the science of sports, robotics, and rehabilitation. MIT Press, 2021.",
                "links": [],
            },
            {
                "text": "P. Ramdya and A. J. Ijspeert. The neuromechanics of animal locomotion: From biology to robotics and back. Science Robotics, 2023.",
                "links": [resource_url("/resources/Locomotion-biology-robotics-scirobotics.pdf")],
            },
        ],
    },
    {
        "chapter_index": 3,
        "schedule_id": "2A",
        "date": "2024-09-16",
        "title": "Lecture 2A: Robot Mechanisms - Kinematics and Dynamics",
        "title_short": "Robot Kinematics and Dynamics",
        "slug": "robot_kinematics_and_dynamics",
        "youtube_playlist_index": 1,
        "video_id": "_pBgzfzOT_8",
        "video_url": "https://www.youtube.com/watch?v=_pBgzfzOT_8",
        "slide_urls": [
            "https://docs.google.com/presentation/d/1Qy1wLVB8vfIeROXJg6NoSIredIjrc8Eu/edit?usp=sharing&ouid=100618406782889124957&rtpof=true&sd=true"
        ],
        "backup_video_urls": [
            "https://drive.google.com/file/d/1qtep5QVViDW7rvdXWt7DLDgbIRbYVoop/view?usp=sharing"
        ],
        "topics": [
            "Rigid body motions",
            "Exponential coordinates",
            "Twists and screws",
            "Product of exponentials",
            "Robot kinematics and dynamics",
        ],
        "readings": [
            {
                "text": "Recommended Lynch-Park Chapter 3 lectures on exponential coordinates and twists.",
                "links": ["https://www.youtube.com/watch?v=29LhXWjn7Pc&list=PLggLP4f-rq02vX0OQQ5vrCxbJrzamYDfx&index=10"],
            }
        ],
    },
    {
        "chapter_index": 4,
        "schedule_id": "2B",
        "date": "2024-09-16",
        "title": "Lecture 2B: The Human Hand and Dexterous Object Manipulation",
        "title_short": "Human Hand and Dexterous Manipulation",
        "slug": "human_hand_and_dexterous_object_manipulation",
        "youtube_playlist_index": None,
        "video_id": None,
        "video_url": None,
        "slide_urls": [],
        "backup_video_urls": [],
        "topics": [
            "Human hand anatomy",
            "Dexterous object manipulation",
            "Hand function",
            "Manipulation primitives",
        ],
        "readings": [
            {
                "text": "Video on human hand anatomy.",
                "links": ["https://www.youtube.com/watch?v=-y69D76RdMs"],
            }
        ],
    },
    {
        "chapter_index": 5,
        "schedule_id": "3AB",
        "date": "2024-09-23",
        "title": "Lecture 3: Robot Hands; Proprioception and Tactile Perception",
        "title_short": "Robot Hands, Proprioception, and Touch",
        "slug": "robot_hands_proprioception_and_touch",
        "youtube_playlist_index": 2,
        "video_id": "RvVLwi1bZ00",
        "video_url": "https://www.youtube.com/watch?v=RvVLwi1bZ00",
        "slide_urls": [
            "https://docs.google.com/presentation/d/13oX9ucOYo4WkWkm9LX7lGhQKiZnyoHZV/edit?usp=sharing&ouid=100618406782889124957&rtpof=true&sd=true",
            "https://docs.google.com/presentation/d/1RNP3zXqSS9H71bPs6zg-PV4iFkSwhrn4/edit?usp=sharing&ouid=100618406782889124957&rtpof=true&sd=true",
        ],
        "backup_video_urls": [
            "https://drive.google.com/file/d/12V3xa6jonkklXauQCIbwml4EQufg8E0_/view?usp=sharing"
        ],
        "topics": [
            "Robot hand design",
            "A century of robotic hands",
            "Proprioception",
            "Tactile perception",
            "Sensing for manipulation",
        ],
        "readings": [
            {
                "text": "Cristina Piazza et al. A century of robotic hands.",
                "links": [resource_url("/resources/century-of-robotic-hands.pdf")],
            },
            {
                "text": "LEAP Hand and LEAP Hand v2.",
                "links": ["https://leaphand.com/", "https://openreview.net/forum?id=eQomRzRZEP"],
            },
            {
                "text": "L. A. Jones. Human hand function.",
                "links": [resource_url("/resources/Human_Hand_Function.pdf")],
            },
            {
                "text": "Esther P. Gardner. Touch.",
                "links": [resource_url("/resources/Gardner-ELS2010.pdf")],
            },
        ],
    },
    {
        "chapter_index": 6,
        "schedule_id": "4A",
        "date": "2024-09-30",
        "title": "Lecture 4A: Vision for Action",
        "title_short": "Vision for Action",
        "slug": "vision_for_action",
        "youtube_playlist_index": 3,
        "video_id": "iUB0vRmTPVE",
        "video_url": "https://www.youtube.com/watch?v=iUB0vRmTPVE",
        "slide_urls": [
            "https://drive.google.com/file/d/1Ly9pzqWH_R6SB9N6ZqCpiGK5TrYT0KnR/view?usp=sharing"
        ],
        "backup_video_urls": [
            "https://drive.google.com/file/d/1UwiBsTV5JqMENDkKWr0DlVFz3tcjgluW/view?usp=sharing"
        ],
        "topics": [
            "Vision in control loops",
            "Eye movements",
            "Perception for action",
            "Activities of daily living",
        ],
        "readings": [
            {
                "text": "Land et al. The roles of vision and eye movements in the control of activities of daily living.",
                "links": [resource_url("/resources/teamaking-land-mennie-rusted-1999.pdf")],
            }
        ],
    },
    {
        "chapter_index": 7,
        "schedule_id": "4B",
        "date": "2024-09-30",
        "title": "Lecture 4B: The Developmental Perspective on Motor Control",
        "title_short": "Developmental Perspective on Motor Control",
        "slug": "developmental_perspective_on_motor_control",
        "youtube_playlist_index": 4,
        "video_id": "g86ewFu8uXQ",
        "video_url": "https://www.youtube.com/watch?v=g86ewFu8uXQ",
        "slide_urls": [
            "https://drive.google.com/file/d/1_6jJp2tpgMV9Kobwbgp2jkpU9iS0c_sZ/view?usp=sharing"
        ],
        "backup_video_urls": [
            "https://drive.google.com/file/d/1R5TzzT34Em6ToH0-Kcd2jDnuxZh4Fs2e/view?usp=sharing"
        ],
        "topics": [
            "Embodied cognition",
            "Motor development",
            "Learning through interaction",
            "Cross-modal supervision",
        ],
        "readings": [
            {
                "text": "Loquercio, Kumar, Malik. Learning visual locomotion with cross-modal supervision.",
                "links": [resource_url("/resources/ICRA23_Learning_to_see_by_walking_blind.pdf")],
            },
            {
                "text": "Smith and Gasser. The Development of Embodied Cognition: Six Lessons from Babies.",
                "links": [resource_url("/resources/6_lessons.pdf")],
            },
        ],
    },
    {
        "chapter_index": 8,
        "schedule_id": "5A",
        "date": "2024-10-07",
        "title": "Lecture 5A: Robot Dynamics, Control, and Motion Planning",
        "title_short": "Robot Dynamics, Control, and Motion Planning",
        "slug": "robot_dynamics_control_and_motion_planning",
        "youtube_playlist_index": 5,
        "video_id": "p8ZALhlrBrI",
        "video_url": "https://www.youtube.com/watch?v=p8ZALhlrBrI",
        "slide_urls": [
            "https://drive.google.com/file/d/1zuBwmd1igHun76r41DAyx2BOw3qmmp9P/view?usp=sharing"
        ],
        "backup_video_urls": [
            "https://drive.google.com/file/d/1Lg_r7QsSusHigRSUO1e2zCRGOUsTmQvC/view?usp=sharing"
        ],
        "topics": [
            "Robot dynamics",
            "Control theory",
            "Trajectory planning",
            "Motion planning",
            "Internal models",
        ],
        "readings": [
            {
                "text": "Kawato. Internal models for motor control and trajectory planning.",
                "links": [resource_url("/resources/Kawato-internal-models.pdf")],
            },
            {
                "text": "Flanagan, Bowman, Johansson. Control strategies in object manipulation tasks.",
                "links": [resource_url("/resources/FlanaganBowmanJohansson06.pdf")],
            },
        ],
    },
    {
        "chapter_index": 9,
        "schedule_id": "5B",
        "date": "2024-10-07",
        "title": "Lecture 5B: Computational Neuroscience Perspective on Prediction and Control",
        "title_short": "Prediction and Control from Computational Neuroscience",
        "slug": "prediction_and_control_from_computational_neuroscience",
        "youtube_playlist_index": 6,
        "video_id": "-L13vtGzWYA",
        "video_url": "https://www.youtube.com/watch?v=-L13vtGzWYA",
        "slide_urls": [
            "https://drive.google.com/file/d/1Pp8oCOdSTGlImExUWYk4AVVyOJwR1KOH/view?usp=sharing"
        ],
        "backup_video_urls": [
            "https://drive.google.com/file/d/1fofpZDHeFhAp3blPq54sZD1RGyNSZRRW/view?usp=sharing"
        ],
        "topics": [
            "Prediction in motor control",
            "Forward and inverse models",
            "Feedback and feedforward control",
            "Neuroscience perspective on action",
        ],
        "readings": [
            {
                "text": "Kawato. Internal models for motor control and trajectory planning.",
                "links": [resource_url("/resources/Kawato-internal-models.pdf")],
            },
            {
                "text": "Flanagan, Bowman, Johansson. Control strategies in object manipulation tasks.",
                "links": [resource_url("/resources/FlanaganBowmanJohansson06.pdf")],
            },
        ],
    },
    {
        "chapter_index": 10,
        "schedule_id": "6A",
        "date": "2024-10-14",
        "title": "Lecture 6A: Reinforcement Learning",
        "title_short": "Reinforcement Learning Part A",
        "slug": "reinforcement_learning_part_a",
        "youtube_playlist_index": 7,
        "video_id": "PRMjicOLOrk",
        "video_url": "https://www.youtube.com/watch?v=PRMjicOLOrk",
        "slide_urls": [
            "https://docs.google.com/presentation/d/1UnrXmJvnByo8epKVNVYzVRehwxumnfwh/edit?usp=sharing&ouid=100618406782889124957&rtpof=true&sd=true"
        ],
        "backup_video_urls": [
            "https://drive.google.com/file/d/16yps-U3OyD1I2YWZVHYZwNc9LceEx26V/view?usp=sharing"
        ],
        "topics": [
            "Markov decision processes",
            "Value functions and policies",
            "Policy optimization",
            "Robot learning with RL",
        ],
        "readings": [
            {
                "text": "Learning to Walk via Deep Reinforcement Learning. RSS 2019.",
                "links": [],
            },
            {
                "text": "Learning Dexterous In-Hand Manipulation. IJRR 2019.",
                "links": [],
            },
        ],
    },
    {
        "chapter_index": 11,
        "schedule_id": "6B",
        "date": "2024-10-14",
        "title": "Lecture 6B: Reinforcement Learning",
        "title_short": "Reinforcement Learning Part B",
        "slug": "reinforcement_learning_part_b",
        "youtube_playlist_index": 8,
        "video_id": "pvTYmRXLixo",
        "video_url": "https://www.youtube.com/watch?v=pvTYmRXLixo",
        "slide_urls": [
            "https://docs.google.com/presentation/d/1UnrXmJvnByo8epKVNVYzVRehwxumnfwh/edit?usp=sharing&ouid=100618406782889124957&rtpof=true&sd=true"
        ],
        "backup_video_urls": [
            "https://drive.google.com/file/d/1B0OFwxpp71xdy_0l96lC7-oVn-27Mwwg/view?usp=sharing"
        ],
        "topics": [
            "Policy gradients",
            "Actor-critic thinking",
            "Robot RL case studies",
            "Learning to walk and manipulate",
        ],
        "readings": [
            {
                "text": "Learning to Walk via Deep Reinforcement Learning. RSS 2019.",
                "links": [],
            },
            {
                "text": "Learning Dexterous In-Hand Manipulation. IJRR 2019.",
                "links": [],
            },
        ],
    },
    {
        "chapter_index": 12,
        "schedule_id": "7AB",
        "date": "2024-10-21",
        "title": "Lecture 7: Behavior Cloning",
        "title_short": "Behavior Cloning",
        "slug": "behavior_cloning",
        "youtube_playlist_index": 9,
        "video_id": "UC-hdat5YqU",
        "video_url": "https://www.youtube.com/watch?v=UC-hdat5YqU",
        "slide_urls": [
            "https://docs.google.com/presentation/d/1JoCCHD6VXlSPsoYgHrXbrdpnoW4j0_wM/edit?usp=sharing&ouid=100618406782889124957&rtpof=true&sd=true"
        ],
        "backup_video_urls": [
            "https://drive.google.com/file/d/1UfsexpwcaTMvXIo_c7grZsJyTFk-exrs/view?usp=sharing"
        ],
        "topics": [
            "Supervised policy learning",
            "Covariate shift",
            "Dataset aggregation intuition",
            "Behavior cloning for robotics",
        ],
        "readings": [
            {
                "text": "Diffusion Policy.",
                "links": [
                    "https://diffusion-policy.cs.columbia.edu/",
                    resource_url("/resources/diffusion_policy_2023.pdf"),
                ],
            }
        ],
    },
    {
        "chapter_index": 13,
        "schedule_id": "8AB",
        "date": "2024-10-28",
        "title": "Lecture 8: Visual Imitation",
        "title_short": "Visual Imitation",
        "slug": "visual_imitation",
        "youtube_playlist_index": 10,
        "video_id": "TmQJe2npA34",
        "video_url": "https://www.youtube.com/watch?v=TmQJe2npA34",
        "slide_urls": [
            "https://drive.google.com/file/d/1hZgefKjJ2ljLC3DNnJ7wtoo68K4N9G7N/view?usp=sharing",
            "https://drive.google.com/file/d/1QYUqOcxGc7XkCX0kWoblHVrPG31bq6vV/view?usp=sharing",
        ],
        "backup_video_urls": [
            "https://drive.google.com/file/d/13tjIM89D61HAK5s-r1kcNV7hDtmleqJq/view?usp=sharing"
        ],
        "topics": [
            "Visual imitation learning",
            "Internet video supervision",
            "Hands as probes",
            "Video-based robot teaching",
        ],
        "readings": [
            {
                "text": "Shaw et al. Learning dexterity from human hand motion in internet videos.",
                "links": [resource_url("/resources/shaw-et-al-2024-learning-dexterity-from-human-hand-motion-in-internet-videos.pdf")],
            },
            {
                "text": "Goyal et al. Human hands as probes for interactive object understanding.",
                "links": [resource_url("/resources/Goyal_Human_Hands_As_Probes_for_Interactive_Object_Understanding_CVPR_2022_paper.pdf")],
            },
            {
                "text": "Kumar, Gupta, Malik. Learning navigation subroutines from egocentric videos.",
                "links": [resource_url("/resources/ashish_kumar19.pdf")],
            },
        ],
    },
    {
        "chapter_index": 14,
        "schedule_id": "9AB",
        "date": "2024-11-04",
        "title": "Lecture 9: Case Studies in Locomotion",
        "title_short": "Case Studies in Locomotion",
        "slug": "case_studies_in_locomotion",
        "youtube_playlist_index": 11,
        "video_id": "98uo2OuXDYc",
        "video_url": "https://www.youtube.com/watch?v=98uo2OuXDYc",
        "slide_urls": [
            "https://drive.google.com/file/d/1-HrfGShe2l8HUHUNvcdQ11rQ_WDlmPnV/view?usp=sharing",
            "https://drive.google.com/file/d/1rOHWD1HHP5b55kmV7SjTEnbI4gnNADQE/view?usp=sharing",
        ],
        "backup_video_urls": [
            "https://drive.google.com/file/d/1sLe0eCY6vYoRmMLBDPS5KU0yo3Arbd-e/view?usp=sharing"
        ],
        "topics": [
            "Legged locomotion",
            "BD-MPC",
            "ZMP and humanoids",
            "Locomotion case studies",
        ],
        "readings": [
            {
                "text": "Scott Kuindersma talk on BD MPC.",
                "links": ["https://youtu.be/mlTLxpKdHfA?feature=shared"],
            },
            {
                "text": "Robin Deit RSS talk on BD MPC.",
                "links": ["https://www.youtube.com/watch?v=aQi6QxMKxQM"],
            },
            {
                "text": "Russ Tedrake lecture on humanoids covering ZMP.",
                "links": ["https://www.youtube.com/watch?v=cRu4EqBswbk"],
            },
        ],
    },
    {
        "chapter_index": 15,
        "schedule_id": "10AB",
        "date": "2024-11-18",
        "title": "Lecture 10: Case Studies in Navigation",
        "title_short": "Case Studies in Navigation",
        "slug": "case_studies_in_navigation",
        "youtube_playlist_index": 12,
        "video_id": "HgK-86Hr1Wo",
        "video_url": "https://www.youtube.com/watch?v=HgK-86Hr1Wo",
        "slide_urls": [
            "https://drive.google.com/file/d/1Jr9EFvF-2zj2ecvb-z8_hPKk7J-g0UQt/view?usp=sharing",
            "https://drive.google.com/file/d/1OGi_IKK1EOKl50d-xjXetofiXPCTF4c-/view?usp=sharing",
        ],
        "backup_video_urls": [
            "https://drive.google.com/file/d/14UEuvn4osdItFpBxMMLYrxNJoxk8mCRd/view?usp=sharing"
        ],
        "topics": [
            "Navigation policies",
            "Goal-conditioned navigation",
            "Traversability estimation",
            "Egocentric video for navigation",
        ],
        "readings": [
            {
                "text": "Chang et al. Goat: Go to any thing.",
                "links": [resource_url("/resources/59_goat_go_to_any_thing.pdf")],
            }
        ],
    },
    {
        "chapter_index": 16,
        "schedule_id": "11A",
        "date": "2024-11-25",
        "title": "Lecture 11A: Case Studies in Dexterous Manipulation",
        "title_short": "Dexterous Manipulation Part A",
        "slug": "dexterous_manipulation_part_a",
        "youtube_playlist_index": 13,
        "video_id": "DtQ78SK8yC8",
        "video_url": "https://www.youtube.com/watch?v=DtQ78SK8yC8",
        "slide_urls": [
            "https://docs.google.com/presentation/d/12MNCNT46vh17VtX4UeTdGE8uAEdGLQme/edit?usp=sharing&ouid=100618406782889124957&rtpof=true&sd=true"
        ],
        "backup_video_urls": [
            "https://drive.google.com/file/d/1XDS9uYatX8ovMkgqsQHpYF9oFoq1kcTC/view?usp=sharing"
        ],
        "topics": [
            "Dexterous manipulation",
            "Bimanual manipulation",
            "Low-cost hardware",
            "Robot dexterity",
        ],
        "readings": [
            {
                "text": "Zhao et al. Learning fine-grained bimanual manipulation with low-cost hardware.",
                "links": [resource_url("/resources/Aloha23.pdf")],
            },
            {
                "text": "Zhao et al. Aloha unleashed: A simple recipe for robot dexterity.",
                "links": [resource_url("/resources/447_ALOHA_Unleashed_A_Simple_R.pdf")],
            },
            {
                "text": "Dalal et al. Local Policies Enable Zero-shot Long-horizon Manipulation.",
                "links": [resource_url("/resources/Dalal24.pdf")],
            },
        ],
    },
    {
        "chapter_index": 17,
        "schedule_id": "11B",
        "date": "2024-11-25",
        "title": "Lecture 11B: Case Studies in Dexterous Manipulation",
        "title_short": "Dexterous Manipulation Part B",
        "slug": "dexterous_manipulation_part_b",
        "youtube_playlist_index": 14,
        "video_id": "mkI7WGgj5xA",
        "video_url": "https://www.youtube.com/watch?v=mkI7WGgj5xA",
        "slide_urls": [
            "https://docs.google.com/presentation/d/12MNCNT46vh17VtX4UeTdGE8uAEdGLQme/edit?usp=sharing&ouid=100618406782889124957&rtpof=true&sd=true"
        ],
        "backup_video_urls": [
            "https://drive.google.com/file/d/10RC7-EtBiYy0w6AB5V-lYyhARdlAO7Co/view?usp=sharing"
        ],
        "topics": [
            "Dexterous manipulation case studies",
            "ALOHA-style systems",
            "Long-horizon manipulation",
            "Policy composition",
        ],
        "readings": [
            {
                "text": "Zhao et al. Learning fine-grained bimanual manipulation with low-cost hardware.",
                "links": [resource_url("/resources/Aloha23.pdf")],
            },
            {
                "text": "Zhao et al. Aloha unleashed: A simple recipe for robot dexterity.",
                "links": [resource_url("/resources/447_ALOHA_Unleashed_A_Simple_R.pdf")],
            },
            {
                "text": "Dalal et al. Local Policies Enable Zero-shot Long-horizon Manipulation.",
                "links": [resource_url("/resources/Dalal24.pdf")],
            },
        ],
    },
    {
        "chapter_index": 18,
        "schedule_id": "12AB",
        "date": "2024-12-02",
        "title": "Lecture 12: Long Horizon Planning and the Role of Language",
        "title_short": "Long Horizon Planning and Language",
        "slug": "long_horizon_planning_and_language",
        "youtube_playlist_index": 15,
        "video_id": "Ty_XNoXlQ5c",
        "video_url": "https://www.youtube.com/watch?v=Ty_XNoXlQ5c",
        "slide_urls": [
            "https://drive.google.com/file/d/1WLnSnCiSo6QerOUwozLF1BO_vfhzT40j/view?usp=sharing"
        ],
        "backup_video_urls": [
            "https://drive.google.com/file/d/1KyoA2AXyxPoA9NUh_OOpVDFZtcCrGo1x/view?usp=sharing"
        ],
        "topics": [
            "Long-horizon planning",
            "Language-conditioned control",
            "Task decomposition",
            "Planning abstractions",
        ],
        "readings": [],
    },
]


def run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd or RUN_ROOT,
        check=check,
        text=True,
        capture_output=capture_output,
    )


def rel_repo(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def ensure_dirs() -> None:
    for dirname in ["build", "lectures", "materials/readings", "materials/shared", "meta", "raw", "text"]:
        (RUN_ROOT / dirname).mkdir(parents=True, exist_ok=True)


def slugify(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    return lowered or "topic"


def normalize_text(text: str, *, limit: int = 240) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    return compact[:limit]


def copy_if_exists(src: Path | None, dst: Path) -> Path | None:
    if src is None or not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def mime_type(path: Path) -> str:
    proc = run(["file", "--mime-type", "-b", str(path)], capture_output=True)
    return proc.stdout.strip()


def content_length(url: str) -> int | None:
    proc = run(["curl", "-sIL", url], capture_output=True, check=False)
    values = re.findall(r"(?im)^content-length:\s*(\d+)\s*$", proc.stdout)
    if not values:
        return None
    return int(values[-1])


def extract_google_file_id(url: str) -> str | None:
    patterns = [
        r"/d/([A-Za-z0-9_-]+)",
        r"[?&]id=([A-Za-z0-9_-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def best_subtitle_path(raw_dir: Path) -> Path | None:
    candidates = sorted(raw_dir.glob("*.srt"))
    if not candidates:
        return None
    preferred: list[tuple[int, str, Path]] = []
    for path in candidates:
        name = path.name
        score = 100
        if ".en.srt" in name:
            score = 10
        elif ".en-orig.srt" in name:
            score = 20
        elif ".en-US.srt" in name or ".en-GB.srt" in name:
            score = 30
        elif ".en." in name:
            score = 40
        preferred.append((score, name, path))
    preferred.sort()
    return preferred[0][2]


def ensure_cover_jpg(raw_dir: Path, stem: str) -> Path | None:
    jpg_path = raw_dir / f"{stem}.jpg"
    if jpg_path.exists():
        return jpg_path
    candidates = list(raw_dir.glob(f"{stem}.*"))
    for path in candidates:
        if path.suffix.lower() in {".webp", ".png", ".jpeg", ".jpg"}:
            run(["magick", str(path), str(jpg_path)], check=False)
            if jpg_path.exists():
                return jpg_path
    return None


def download_to(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    run(["curl", "-L", "--fail", url, "-o", str(dest)], check=True)


def download_google_drive_asset(url: str, dest_pdf: Path, *, max_bytes: int = MAX_GOOGLE_DRIVE_BYTES) -> dict[str, Any]:
    file_id = extract_google_file_id(url)
    if not file_id:
        return {"status": "missing", "local_path": None, "notes": "Could not extract Google file id."}

    initial_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    temp_dir = dest_pdf.parent / ".tmp_downloads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    first_pass = temp_dir / f"{file_id}.first"

    def finalize(result: dict[str, Any]) -> dict[str, Any]:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return result

    run(["curl", "-L", initial_url, "-o", str(first_pass)], check=True)

    initial_mime = mime_type(first_pass)
    file_name = f"{file_id}.bin"
    payload_path = first_pass

    if initial_mime.startswith("text/html"):
        html_text = first_pass.read_text(errors="ignore")
        if "file is too large to open" in html_text.lower():
            return finalize(
                {
                "status": "skipped_too_large",
                "local_path": None,
                "notes": "Google export endpoint reports the presentation is too large to open directly.",
                }
            )
        fields = dict(re.findall(r'name="([^"]+)" value="([^"]*)"', html_text))
        if not fields:
            return finalize({"status": "missing", "local_path": None, "notes": "Could not parse Google Drive confirmation page."})
        name_match = re.search(r'<a href="/open\?id=[^"]+">([^<]+)</a>', html_text)
        if name_match:
            file_name = html.unescape(name_match.group(1)).strip()
        confirm_url = "https://drive.usercontent.google.com/download?" + urllib.parse.urlencode(fields)
        size = content_length(confirm_url)
        if size is not None and size > max_bytes:
            return finalize(
                {
                "status": "skipped_too_large",
                "local_path": None,
                "notes": f"Drive asset size {size} bytes exceeds bootstrap limit {max_bytes}.",
                }
            )
        payload_path = temp_dir / file_name
        run(["curl", "-L", confirm_url, "-o", str(payload_path)], check=True)

    final_mime = mime_type(payload_path)
    if final_mime == "application/pdf":
        dest_pdf.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(payload_path), str(dest_pdf))
        return finalize({"status": "available", "local_path": rel_repo(dest_pdf), "notes": "Downloaded PDF from Google Drive."})

    suffix = payload_path.suffix.lower()
    if final_mime == "application/zip" and suffix == ".zip":
        archive_dest = dest_pdf.with_suffix(".zip")
        archive_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(payload_path), str(archive_dest))
        return finalize(
            {
            "status": "available",
            "local_path": rel_repo(archive_dest),
            "notes": "Downloaded Google Drive archive without conversion.",
            }
        )

    office_suffixes = {".ppt", ".pptx", ".odp", ".potx", ".ppsx"}
    if suffix in office_suffixes or final_mime in {"application/zip", "application/vnd.openxmlformats-officedocument.presentationml.presentation"}:
        converted = temp_dir / (payload_path.stem + ".pdf")
        run(
            ["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(temp_dir), str(payload_path)],
            check=True,
        )
        if converted.exists():
            dest_pdf.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(converted), str(dest_pdf))
            return finalize(
                {
                "status": "available",
                "local_path": rel_repo(dest_pdf),
                "notes": f"Downloaded Google Drive presentation ({payload_path.name}) and converted to PDF.",
                }
            )

    return finalize(
        {
        "status": "missing",
        "local_path": None,
        "notes": f"Unsupported Google asset type {final_mime} ({payload_path.name}).",
        }
    )


def download_youtube_bundle(lecture: dict[str, Any]) -> dict[str, Any]:
    if not lecture.get("video_url") or not lecture.get("video_id"):
        return {"raw_dir": None, "subtitle": None, "cover": None, "info_json": None}

    raw_dir = RUN_ROOT / "raw" / f"{lecture['chapter_index']:02d}_{lecture['video_id']}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{lecture['chapter_index']:02d}_{lecture['video_id']}"
    base = raw_dir / stem

    info_json = raw_dir / f"{stem}.info.json"
    if not info_json.exists():
        run(
            [
                "yt-dlp",
                "--skip-download",
                "--write-info-json",
                "--write-thumbnail",
                "--write-subs",
                "--write-auto-subs",
                "--sub-langs",
                "en.*,en",
                "--convert-subs",
                "srt",
                "-o",
                str(base) + ".%(ext)s",
                lecture["video_url"],
            ],
            check=True,
        )

    cover = ensure_cover_jpg(raw_dir, stem)
    subtitle = best_subtitle_path(raw_dir)
    return {
        "raw_dir": raw_dir,
        "subtitle": subtitle,
        "cover": cover,
        "info_json": info_json if info_json.exists() else None,
    }


def merge_slide_pdfs(paths: list[Path], dest: Path) -> Path | None:
    available = [path for path in paths if path.exists()]
    if not available:
        return None
    if len(available) == 1:
        shutil.copy2(available[0], dest)
        return dest
    run(["pdfunite", *[str(path) for path in available], str(dest)], check=True)
    return dest


def seed_coverage_units(
    lecture: dict[str, Any],
    transcript_rows: list[dict[str, Any]],
    slide_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for idx, topic in enumerate(lecture.get("topics") or [], start=1):
        rows.append(
            {
                "unit_id": f"topic_{idx:02d}",
                "source_type": "lecture_topic_seed",
                "source_id": "lecture_meta",
                "loc": {"topic_index": idx},
                "kind": [slugify(topic)],
                "summary": topic,
                "required": True,
                "status": "unclassified",
                "mapped_section": None,
                "figure_ids": [],
                "notes": "Topic seed created during source acquisition to preserve must-cover scope.",
            }
        )

    def infer_kinds(text: str, source_type: str) -> list[str]:
        kinds = ["slide_page" if source_type == "slide_page" else "subtitle_span"]
        lowered = text.lower()
        keyword_map = {
            "formula": ["equation", "bellman", "loss", "objective", "matrix", "gradient", "twist", "zmp", "mpc"],
            "code": ["code", "algorithm", "pseudocode", "implementation", "python"],
            "example": ["example", "case study", "application", "demo"],
            "figure": ["figure", "diagram", "plot", "chart", "image"],
        }
        for kind, terms in keyword_map.items():
            if any(term in lowered for term in terms):
                kinds.append(kind)
        return kinds

    for row in transcript_rows + slide_rows:
        text = row.get("text", "")
        rows.append(
            {
                "unit_id": row["unit_id"],
                "source_type": row["source_type"],
                "source_id": row["source_id"],
                "loc": row["loc"],
                "kind": infer_kinds(text, row["source_type"]),
                "summary": normalize_text(text),
                "required": row.get("required", True),
                "status": "unclassified",
                "mapped_section": None,
                "figure_ids": [],
                "notes": "",
            }
        )
    return rows


def prepare_debug_texts(transcript_rows: list[dict[str, Any]], slide_pages: list[str], lecture_dir: Path) -> None:
    transcript_text = "\n".join(row["text"] for row in transcript_rows)
    official_text = "\n\n".join(page for page in slide_pages if page.strip())
    write_text(lecture_dir / "transcript.txt", transcript_text + ("\n" if transcript_text else ""))
    write_text(lecture_dir / "official.txt", official_text + ("\n" if official_text else ""))


def lecture_reading_dir(lecture: dict[str, Any]) -> Path:
    return RUN_ROOT / "materials" / "readings" / f"{lecture['chapter_index']:02d}_{lecture['slug']}"


def download_readings(lecture: dict[str, Any]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    reading_dir = lecture_reading_dir(lecture)
    reading_dir.mkdir(parents=True, exist_ok=True)

    for idx, reading in enumerate(lecture.get("readings") or [], start=1):
        links = reading.get("links") or []
        if not links:
            sources.append(
                {
                    "source_id": f"reading_{idx:02d}",
                    "source_type": "official_reading_reference",
                    "origin_url": None,
                    "local_path": None,
                    "required_for_coverage": True,
                    "status": "remote_only",
                    "notes": reading["text"],
                }
            )
            continue

        for link_idx, url in enumerate(links, start=1):
            parsed = urllib.parse.urlparse(url)
            basename = Path(parsed.path).name or f"reading_{idx:02d}_{link_idx:02d}"
            dest = reading_dir / basename
            status = "remote_only"
            local_path = None
            notes = reading["text"]

            if url.startswith(COURSE_PAGE_URL) and basename.lower().endswith(".pdf"):
                try:
                    if not dest.exists():
                        download_to(url, dest)
                    status = "available"
                    local_path = rel_repo(dest)
                except subprocess.CalledProcessError as exc:
                    status = "missing"
                    notes = f"{reading['text']} Download failed: {exc}"
            elif basename.lower().endswith(".pdf"):
                try:
                    if not dest.exists():
                        download_to(url, dest)
                    status = "available"
                    local_path = rel_repo(dest)
                except subprocess.CalledProcessError:
                    status = "remote_only"
                    notes = f"{reading['text']} Remote PDF retained as URL after download failure."

            sources.append(
                {
                    "source_id": f"reading_{idx:02d}_{link_idx:02d}",
                    "source_type": "official_reading_reference",
                    "origin_url": url,
                    "local_path": local_path,
                    "required_for_coverage": True,
                    "status": status,
                    "notes": notes,
                }
            )
    return sources


def shared_asset_sources(shared: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    mapping = [
        ("course_page_html", "course_page_snapshot", COURSE_PAGE_URL, RUN_ROOT / "meta" / "course_page.html", True, "Raw HTML snapshot of the official course page."),
        ("playlist_flat_json", "playlist_metadata", PLAYLIST_URL, RUN_ROOT / "meta" / "playlist_flat.json", True, "yt-dlp flat playlist metadata snapshot."),
        ("course_schedule_json", "course_schedule_manifest", None, RUN_ROOT / "meta" / "course_schedule.json", True, "Structured course schedule extracted for the harness."),
        ("course_readings_json", "course_readings_manifest", None, RUN_ROOT / "meta" / "course_readings.json", True, "Structured official reading list extracted for the harness."),
    ]
    for source_id, source_type, origin_url, path, required, notes in mapping:
        rows.append(
            {
                "source_id": source_id,
                "source_type": source_type,
                "origin_url": origin_url,
                "local_path": rel_repo(path) if path.exists() else None,
                "required_for_coverage": required,
                "status": "available" if path.exists() else "missing",
                "notes": notes,
            }
        )

    rows.append(
        {
            "source_id": "course_notes_pdf",
            "source_type": "shared_course_notes_pdf",
            "origin_url": COURSE_NOTES_URL,
            "local_path": shared["course_notes"]["local_path"],
            "required_for_coverage": True,
            "status": shared["course_notes"]["status"],
            "notes": shared["course_notes"]["notes"],
        }
    )
    rows.append(
        {
            "source_id": "backup_materials_pdf",
            "source_type": "shared_backup_materials_pdf",
            "origin_url": BACKUP_NOTES_URL,
            "local_path": shared["backup_notes"]["local_path"],
            "required_for_coverage": False,
            "status": shared["backup_notes"]["status"],
            "notes": shared["backup_notes"]["notes"],
        }
    )
    return rows


def build_source_manifest(
    lecture: dict[str, Any],
    lecture_dir: Path,
    shared: dict[str, dict[str, Any]],
    youtube_bundle: dict[str, Any],
    slide_sources: list[dict[str, Any]],
    reading_sources: list[dict[str, Any]],
) -> dict[str, Any]:
    sources: list[dict[str, Any]] = [
        {
            "source_id": "lecture_meta",
            "source_type": "lecture_metadata",
            "origin_url": lecture.get("video_url") or COURSE_PAGE_URL,
            "local_path": rel_repo(lecture_dir / "meta.json"),
            "required_for_coverage": True,
            "status": "available",
            "notes": "Normalized lecture metadata for harness-managed note generation.",
        },
        *shared_asset_sources(shared),
    ]

    for url_idx, url in enumerate(lecture.get("slide_urls") or [], start=1):
        matching = next((row for row in slide_sources if row["source_id"] == f"slide_source_{url_idx:02d}"), None)
        if matching is not None:
            sources.append(matching)

    if (lecture_dir / "slides.pdf").exists():
        sources.append(
            {
                "source_id": "slides_pdf",
                "source_type": "official_slide_pdf",
                "origin_url": lecture["slide_urls"][0] if lecture.get("slide_urls") else None,
                "local_path": rel_repo(lecture_dir / "slides.pdf"),
                "required_for_coverage": True,
                "status": "available",
                "notes": "Merged per-lecture slide deck used by the harness.",
            }
        )

    if youtube_bundle["cover"] is not None:
        sources.append(
            {
                "source_id": "cover_jpg",
                "source_type": "cover_image",
                "origin_url": None,
                "local_path": rel_repo(lecture_dir / "cover.jpg"),
                "required_for_coverage": True,
                "status": "available",
                "notes": "",
            }
        )
    if youtube_bundle["subtitle"] is not None:
        sources.append(
            {
                "source_id": "subtitle_srt",
                "source_type": "platform_subtitle",
                "origin_url": None,
                "local_path": rel_repo(lecture_dir / "subtitle.srt"),
                "required_for_coverage": True,
                "status": "available",
                "notes": "",
            }
        )
    if youtube_bundle["info_json"] is not None:
        sources.append(
            {
                "source_id": "raw_info_json",
                "source_type": "platform_metadata",
                "origin_url": lecture.get("video_url"),
                "local_path": rel_repo(youtube_bundle["info_json"]),
                "required_for_coverage": True,
                "status": "available",
                "notes": "Original yt-dlp metadata dump for the video.",
            }
        )

    sources.extend(
        [
            {
                "source_id": "transcript_txt",
                "source_type": "normalized_transcript",
                "origin_url": None,
                "local_path": rel_repo(lecture_dir / "transcript.txt"),
                "required_for_coverage": True,
                "status": "available" if (lecture_dir / "transcript.txt").exists() else "missing",
                "notes": "",
            },
            {
                "source_id": "official_txt",
                "source_type": "slide_text_extract",
                "origin_url": None,
                "local_path": rel_repo(lecture_dir / "official.txt"),
                "required_for_coverage": True,
                "status": "available" if (lecture_dir / "official.txt").exists() else "missing",
                "notes": "",
            },
        ]
    )

    for idx, url in enumerate(lecture.get("backup_video_urls") or [], start=1):
        sources.append(
            {
                "source_id": f"backup_video_{idx:02d}",
                "source_type": "backup_video_link",
                "origin_url": url,
                "local_path": None,
                "required_for_coverage": False,
                "status": "remote_only",
                "notes": "Backup Google Drive lecture video listed on the official course page.",
            }
        )

    sources.extend(reading_sources)
    return {
        "course_id": COURSE_ID,
        "course_mode": True,
        "lecture_id": f"{lecture['chapter_index']:02d}",
        "lecture_slug": lecture_dir.name,
        "title": lecture["title"],
        "origin_url": lecture.get("video_url") or COURSE_PAGE_URL,
        "slide_origin_url": lecture["slide_urls"][0] if lecture.get("slide_urls") else None,
        "sources": sources,
    }


def build_meta(
    lecture: dict[str, Any],
    lecture_dir: Path,
    youtube_bundle: dict[str, Any],
) -> dict[str, Any]:
    return {
        "playlist_index": lecture["chapter_index"],
        "schedule_id": lecture["schedule_id"],
        "date": lecture["date"],
        "kind": "lecture",
        "title": lecture["title"],
        "title_short": lecture["title_short"],
        "slug": lecture["slug"],
        "video_id": lecture.get("video_id"),
        "video_url": lecture.get("video_url"),
        "youtube_playlist_index": lecture.get("youtube_playlist_index"),
        "course_id": COURSE_ID,
        "playlist_url": PLAYLIST_URL,
        "course_page_url": COURSE_PAGE_URL,
        "course_notes_url": COURSE_NOTES_URL,
        "backup_notes_url": BACKUP_NOTES_URL,
        "thumbnail": rel_repo(youtube_bundle["cover"]) if youtube_bundle["cover"] else None,
        "subtitle": rel_repo(youtube_bundle["subtitle"]) if youtube_bundle["subtitle"] else None,
        "material": rel_repo(lecture_dir / "slides.pdf") if (lecture_dir / "slides.pdf").exists() else None,
        "transcript_text": rel_repo(lecture_dir / "transcript.txt"),
        "official_text": rel_repo(lecture_dir / "official.txt"),
        "slide_pages_dir": rel_repo(lecture_dir / "pdf_pages") if (lecture_dir / "pdf_pages").exists() else None,
        "course_mode": True,
        "segmentation_required": True,
        "topics": lecture.get("topics") or [],
    }


def write_lecture_workspace(lecture: dict[str, Any], shared: dict[str, dict[str, Any]]) -> dict[str, Any]:
    lecture_dir = RUN_ROOT / "lectures" / f"{lecture['chapter_index']:02d}_{lecture['slug']}"
    lecture_dir.mkdir(parents=True, exist_ok=True)

    youtube_bundle = download_youtube_bundle(lecture)
    if youtube_bundle["cover"] is not None:
        copy_if_exists(youtube_bundle["cover"], lecture_dir / "cover.jpg")
    if youtube_bundle["subtitle"] is not None:
        copy_if_exists(youtube_bundle["subtitle"], lecture_dir / "subtitle.srt")

    downloaded_slide_paths: list[Path] = []
    slide_source_rows: list[dict[str, Any]] = []
    slide_material_dir = RUN_ROOT / "materials" / "slides" / f"{lecture['chapter_index']:02d}_{lecture['slug']}"
    slide_material_dir.mkdir(parents=True, exist_ok=True)
    for idx, slide_url in enumerate(lecture.get("slide_urls") or [], start=1):
        slide_dest = slide_material_dir / f"source_{idx:02d}.pdf"
        result = download_google_drive_asset(slide_url, slide_dest)
        if result["status"] == "available" and slide_dest.exists():
            downloaded_slide_paths.append(slide_dest)
        slide_source_rows.append(
            {
                "source_id": f"slide_source_{idx:02d}",
                "source_type": "official_slide_link",
                "origin_url": slide_url,
                "local_path": result["local_path"],
                "required_for_coverage": True,
                "status": result["status"],
                "notes": result["notes"],
            }
        )

    merged_slides = merge_slide_pdfs(downloaded_slide_paths, lecture_dir / "slides.pdf")

    transcript_rows = build_transcript_units(lecture_dir / "subtitle.srt") if (lecture_dir / "subtitle.srt").exists() else []
    slide_pages = extract_slide_pages(merged_slides) if merged_slides and merged_slides.exists() else []
    slide_rows = build_slide_units(slide_pages, lecture_dir) if slide_pages else []

    if transcript_rows or slide_rows:
        segments = build_segments(lecture.get("topics") or [lecture["title_short"]], transcript_rows, slide_rows)
    else:
        segments = [
            {
                "segment_id": "seg_01",
                "start": None,
                "end": None,
                "source_unit_ids": [],
                "target_section_hint": lecture["title_short"],
            }
        ]

    write_jsonl(lecture_dir / "transcript.jsonl", transcript_rows)
    write_jsonl(lecture_dir / "slides.jsonl", slide_rows)
    write_jsonl(lecture_dir / "segments.jsonl", segments)
    write_jsonl(lecture_dir / "coverage_units.jsonl", seed_coverage_units(lecture, transcript_rows, slide_rows))
    write_jsonl(lecture_dir / "omission_log.jsonl", [])
    write_json(lecture_dir / "figure_manifest.json", [])
    prepare_debug_texts(transcript_rows, slide_pages, lecture_dir)

    reading_sources = download_readings(lecture)
    meta = build_meta(lecture, lecture_dir, youtube_bundle)
    write_json(lecture_dir / "meta.json", meta)
    write_json(
        lecture_dir / "source_manifest.json",
        build_source_manifest(lecture, lecture_dir, shared, youtube_bundle, slide_source_rows, reading_sources),
    )
    write_text(
        lecture_dir / "README.md",
        "\n".join(
            [
                f"# {lecture['title']}",
                "",
                f"- schedule id: `{lecture['schedule_id']}`",
                f"- date: `{lecture['date']}`",
                f"- video: `{lecture.get('video_url') or 'missing'}`",
                f"- slide sources: `{len(lecture.get('slide_urls') or [])}`",
                f"- readings: `{len(lecture.get('readings') or [])}`",
                "",
                "This workspace was bootstrapped by `build/bootstrap_course.py` and must be refined by planner / coverage / writer / evaluator passes.",
                "",
            ]
        ),
    )

    return {
        "lecture_dir": lecture_dir.name,
        "schedule_id": lecture["schedule_id"],
        "video_url": lecture.get("video_url"),
        "has_subtitle": bool(transcript_rows),
        "has_slides_pdf": bool(merged_slides and merged_slides.exists()),
        "slide_source_statuses": {row["source_id"]: row["status"] for row in slide_source_rows},
        "reading_count": len(reading_sources),
    }


def download_shared_assets() -> dict[str, dict[str, Any]]:
    shared_dir = RUN_ROOT / "materials" / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)
    course_notes = download_google_drive_asset(COURSE_NOTES_URL, shared_dir / "course_notes.pdf")
    backup_notes = download_google_drive_asset(BACKUP_NOTES_URL, shared_dir / "backup_all_slides_and_notes.pdf")
    return {
        "course_notes": course_notes,
        "backup_notes": backup_notes,
    }


def write_course_metadata(shared: dict[str, dict[str, Any]]) -> None:
    download_to(COURSE_PAGE_URL, RUN_ROOT / "meta" / "course_page.html")
    playlist_json = run(["yt-dlp", "--flat-playlist", "-J", PLAYLIST_URL], capture_output=True).stdout
    write_text(RUN_ROOT / "meta" / "playlist_flat.json", playlist_json)

    schedule_rows = []
    reading_rows = []
    for lecture in LECTURES:
        schedule_rows.append(
            {
                "lecture_id": f"{lecture['chapter_index']:02d}",
                "schedule_id": lecture["schedule_id"],
                "date": lecture["date"],
                "title": lecture["title"],
                "video_url": lecture.get("video_url"),
                "slide_urls": lecture.get("slide_urls") or [],
                "backup_video_urls": lecture.get("backup_video_urls") or [],
            }
        )
        reading_rows.append(
            {
                "lecture_id": f"{lecture['chapter_index']:02d}",
                "schedule_id": lecture["schedule_id"],
                "title": lecture["title"],
                "readings": lecture.get("readings") or [],
            }
        )

    write_json(RUN_ROOT / "meta" / "course_schedule.json", schedule_rows)
    write_json(RUN_ROOT / "meta" / "course_readings.json", reading_rows)
    write_json(
        RUN_ROOT / "build" / "course_manifest_seed.json",
        {
            "course_id": COURSE_ID,
            "title": COURSE_TITLE,
            "playlist_origin_url": PLAYLIST_URL,
            "course_page_url": COURSE_PAGE_URL,
            "course_notes_url": COURSE_NOTES_URL,
            "backup_notes_url": BACKUP_NOTES_URL,
            "course_mode": True,
            "lecture_count": len(LECTURES),
            "shared_assets": {
                "course_notes": shared["course_notes"],
                "backup_notes": shared["backup_notes"],
            },
            "lectures": [
                {
                    "lecture_id": f"{lecture['chapter_index']:02d}",
                    "schedule_id": lecture["schedule_id"],
                    "lecture_slug": f"{lecture['chapter_index']:02d}_{lecture['slug']}",
                    "title": lecture["title"],
                    "date": lecture["date"],
                    "video_url": lecture.get("video_url"),
                    "slide_urls": lecture.get("slide_urls") or [],
                }
                for lecture in LECTURES
            ],
        },
    )


def main() -> None:
    ensure_dirs()
    shared = download_shared_assets()
    write_course_metadata(shared)

    acquisition_report = []
    for lecture in LECTURES:
        acquisition_report.append(write_lecture_workspace(lecture, shared))

    write_json(RUN_ROOT / "meta" / "source_acquisition_report.json", acquisition_report)

    run(
        [
            "python3",
            str(REPO_ROOT / "scripts" / "video_note_harness" / "bootstrap_harness.py"),
            "--run-root",
            str(RUN_ROOT),
        ]
    )
    run(["python3", str(RUN_ROOT / "build" / "build_course_manifest.py")])
    print(RUN_ROOT)


if __name__ == "__main__":
    main()
