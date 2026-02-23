# SMPLX Visualizer for Inter-X Dataset

Inter-X 데이터셋의 두 사람 인터랙션 모션을 시각화하고, 신체 부착 카메라 뷰를 추출하는 도구입니다.

## Features

- **SMPLX 메시 시각화** — NPZ 파라미터로부터 full body mesh 렌더링
- **스켈레톤 시각화** — NPY Optitrack 관절 데이터 기반 skeleton 렌더링
- **신체 부착 카메라** — 각 사람(P1, P2)별 head, left/right hand 카메라 + 공유 top view
- **인터랙티브 뷰어** — 시퀀스 간 UP/DOWN 키로 탐색, GUI에서 카메라 전환
- **Headless 추출** — 카메라 뷰별 프레임 이미지 또는 MP4 비디오 일괄 추출

## Project Structure

```
visualizer/
├── src/
│   ├── interx_loader.py    # Inter-X 데이터 로더 (NPZ/NPY)
│   ├── body_cameras.py     # 신체 부착 카메라 시스템
│   └── gl_setup.py         # Linux GL 라이브러리 호환성 패치
├── scripts/
│   ├── visualize.py        # 인터랙티브 뷰어
│   └── extract_views.py    # Headless 카메라 뷰 추출
├── aitvconfig.yaml         # aitviewer 설정
└── pyproject.toml
```

## Prerequisites

- Python 3.10
- CUDA-compatible GPU (권장)
- [uv](https://docs.astral.sh/uv/) 패키지 매니저

## Installation

### 1. Clone

```bash
git clone <repo-url> visualizer
cd visualizer
```

### 2. Dependencies 설치

```bash
uv sync
```

### 3. SMPLX 모델 다운로드

[SMPL-X 공식 사이트](https://smpl-x.is.tue.mpg.de/)에서 모델 파일을 다운로드하고 `smplx/` 디렉토리에 배치합니다.

```
smplx/
├── SMPLX_NEUTRAL.npz
├── SMPLX_MALE.npz
├── SMPLX_FEMALE.npz
├── SMPLX_NEUTRAL.pkl
├── SMPLX_MALE.pkl
└── SMPLX_FEMALE.pkl
```

### 4. Inter-X 데이터셋

[Inter-X](https://github.com/liangxuy/Inter-X) 데이터셋을 `datasets/interx/` 에 배치합니다.

```
datasets/interx/
├── motions/
│   └── G001T000A000R000/
│       ├── P1.npz (또는 P1.npy)
│       └── P2.npz (또는 P2.npy)
└── texts/
    └── G001T000A000R000.txt
```

## Usage

### Interactive Viewer

```bash
# 특정 시퀀스 시각화
uv run python scripts/visualize.py --data_dir datasets/interx --sequence G001T000A000R000

# 전체 시퀀스 (UP/DOWN 키로 탐색)
uv run python scripts/visualize.py --data_dir datasets/interx
```

### Headless Camera View Extraction

```bash
# 단일 시퀀스 전체 카메라 뷰 추출
uv run python scripts/extract_views.py --data_dir datasets/interx --sequence G001T000A000R000

# 특정 카메라만 추출
uv run python scripts/extract_views.py --data_dir datasets/interx --sequence G001T000A000R000 \
    --cameras P1_head_cam P2_head_cam top_cam

# 비디오로 추출
uv run python scripts/extract_views.py --data_dir datasets/interx --output_format video
```

**Available cameras:** `P1_head_cam`, `P1_right_hand_cam`, `P1_left_hand_cam`, `P2_head_cam`, `P2_right_hand_cam`, `P2_left_hand_cam`, `top_cam`

## Camera System

| Camera | Description |
|--------|-------------|
| `P{1,2}_head_cam` | 두 눈 사이에서 정면 방향을 바라보는 1인칭 시점 |
| `P{1,2}_right_hand_cam` | 오른쪽 손목에서 팔뚝 방향으로 바라보는 시점 |
| `P{1,2}_left_hand_cam` | 왼쪽 손목에서 팔뚝 방향으로 바라보는 시점 |
| `top_cam` | 고정된 bird's-eye view (두 사람 모두 포함) |

## Dependencies

- [aitviewer](https://github.com/Angledsugar/aitviewer) — ETH aitviewer 수정 포크 (moderngl-window 3.x / imgui 2.0.0 호환 패치)

## License

- **aitviewer**: MIT License (ETH Zurich)
- **Inter-X dataset**: CC-BY-NC-SA 4.0 (비상업적 용도만 허용)
