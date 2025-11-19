# Datasets

## ValidDataset
- **용도**: 평가(validation/test) 데이터셋
- **모드**
  - **Self (`mode="self"`)**
    - 하나의 비디오 내에서 frame 시퀀스를 추출  
    - `sample_n_frames=2` → 랜덤하게 2프레임 선택 (src, tar)  
    - `sample_n_frames>2` → 비디오를 `sample_n_frames` 길이의 구간으로 나누고 시작 프레임 단위로 클립 구성  
  - **Cross (`mode="cross"`)**
    - 단순히 비디오 목록만 들고 있음 (`self.videos`)  
    - 실제 clip 추출 시에는 반드시 `start_idx`를 외부에서 넘겨줘야 함  
    - 일반적으로 `PairedDataset`이 (랜덤/CSV 기반) `start_idx`를 정해주고 이를 `ValidDataset`에 전달하여 시퀀스를 잘라옴  
- **입력**:  
  - `root_dir/test/` 폴더 구조 (각 video마다 frame이 저장된 디렉토리)  
  - `pairs_list`: (옵션) CSV 경로 (cross 모드에서 PairedDataset이 활용)  
- **출력(dict)**:
  - `src_img`: 소스 이미지 (H,W,3) → contrast normalization 후 반환  
  - `tar_gt`: 타겟 시퀀스 (N,H,W,3) or 단일 프레임 (H,W,3)  
  - `name`: `"video_name#start_idx"` 형태의 식별자 

### Processing Flow
1. **Load frames**
   - self 모드 → 각 비디오별로 frame list 확보  
     - `sample_n_frames=2` → 시퀀스에서 랜덤 두 프레임 선택  
     - `sample_n_frames>2` → 고정 길이(`sample_n_frames`) 구간 단위로 분할  
   - cross 모드 → 비디오 이름 목록만 저장 (`self.videos`), 실제 clip 추출은 `start_idx` 필요  

2. **Sample indices**
   - self 모드 → `frame_sequences`에서 미리 정의된 구간/프레임 가져오기  
   - cross 모드 → `PairedDataset`이 정해준 `start_idx`부터 `sample_n_frames`만큼 연속 추출  

3. **Load images**
   - 선택된 프레임 읽기(OpenCV → RGB)  
   - contrast normalization 적용  
   - Source: 단일 프레임  
   - Target: 프레임 시퀀스 or 단일 프레임 


## PairedDataset
- **용도**: cross 모드에서 source–driving 비디오 쌍 생성
- **입력**:
  - `ValidDataset(mode="cross")`
  - `pairs_list` (CSV, 선택)  
- **동작**:
  - CSV 있으면: 지정된 (source, driving, idx) 사용  
  - CSV 없으면: 서로 다른 비디오 랜덤 페어링, start_idx 랜덤  
- **출력**:
  - ValidDataset 결과를 `driving_`, `source_` prefix로 묶어 반환


## sample_subset
- **용도**: `ValidDataset`에서 일부 클립만 뽑아 효율적인 평가 수행  
- **동작 방식**:  
  - 각 ID마다 무작위 비디오 하나 선택 후 최대 `clips_per_video` 클립 추출  
  - 이렇게 모든 ID에서 최소 1개 클립 확보  
  - 부족하면 전체 풀에서 랜덤 추가해 최대 `total_clips`까지 채움  
- **출력**: `(video_name, start_idx)` 리스트  