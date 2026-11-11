# VAD 경계 마진과 STT 성능 실험

VAD(Voice Activity Detection)로 잘라낸 음성 구간 앞뒤에 **마진(여유 구간)을 얼마나 붙이느냐**에
따라 STT 인식 성능과 지연이 어떻게 달라지는지 측정한 실험 코드입니다.

핵심 질문은 이것입니다.

> 정확도 이득은 주로 **앞쪽(leading) 마진**에서 오고, 지연 비용은 **뒤쪽(trailing) 마진**에서
> 생긴다면, 앞뒤를 다르게 준 **비대칭 마진**으로 대칭 마진과 비슷한 정확도를 더 낮은 지연에 얻을 수
> 있는가? 그리고 이 관계가 낭독체가 아닌 실환경 음성에서도 유지되는가?

## 실험 구성

### 데이터셋

| 데이터셋 | 성격 | 비고 |
|---|---|---|
| LibriSpeech (dev/test, clean/other) | 낭독체, 깨끗한 음성 | MFA 강제정렬로 단어 경계 생성 |
| TED-LIUM 3 (dev/test) | 강연체 | talk 전체 오디오 + STM, talk 단위 MFA 정렬 |
| AMI (IHM / SDM) | 회의체, 자유발화 | 같은 발화를 근접 마이크(IHM)와 원거리 마이크(SDM)로 비교 |

AMI IHM/SDM은 같은 발화·같은 주석이라 **마이크 조건만 바뀐 대조**가 됩니다.
양(+)의 마진을 실제로 붙일 수 있도록 모든 클립은 발화 앞뒤 컨텍스트를 포함해 잘라냅니다.

### 모델

- **ASR**: wav2vec2-base-960h (CTC), Emformer RNN-T (스트리밍 Transducer), HuBERT-large (CTC),
  SpeechBrain Transformer (attention enc-dec), NeMo FastConformer hybrid (CTC + RNN-T, cache-aware 스트리밍)
- **VAD**: Silero, WebRTC, pyannote `segmentation-3.0`

### 실험 목록

| 실험 | 내용 | 주요 스크립트 |
|---|---|---|
| 1D 마진 스윕 | 강제정렬 경계를 기준으로 한쪽 마진만 -100 ~ +500 ms 변화 | `scripts/run_experiment_pipeline.sh step1` |
| VAD 기반 디코딩 | 기본 VAD 대비 마진 정책을 적용한 VAD 디코딩 비교 | `scripts/run_experiment_pipeline.sh step2` |
| `speech_pad_ms` 대조군 | 마진 정책이 단순히 Silero 패딩을 키우는 것과 다른지 확인 | `scripts/run_vad_decode.py` |
| 2D 마진 그리드 | leading × trailing 조합 (50 ms 간격) | `scripts/launch/run_silero_grid50.sh`, `scripts/run_w2v_vad_grid.py` |
| VAD 3종 비교 | Silero / WebRTC / pyannote × 마진 정책 | `scripts/run_vad_decode.py --vad ...` |
| 컨텍스트 여유(headroom) 대조 | 클립 컨텍스트 길이를 제한했을 때 마진 효과 변화 | `scripts/launch/run_cap_*.sh` |
| 디코더/우측 컨텍스트 분리 | FastConformer hybrid로 디코더 종류와 look-ahead를 분리 | `scripts/run_nemo_hybrid_grid.py` |
| 스트리밍 지연 측정 | 실제 청크 스트리밍에서 발화 종료 후 최종 결과까지의 지연 | `scripts/run_nemo_streaming_latency.py` |
| RTF 측정 | VAD·ASR 백본별 실시간 계수 | `scripts/bench_rtf.py` |
| 메타데이터 분절 기준선 | VAD·마진 없이 코퍼스 제공 분절로 디코딩 | `scripts/run_metaseg_*.py`, `scripts/summarize_metaseg.py` |
| 통합 실험 (mv2) | dev에서 최적 마진 선택 → eval에 그대로 적용, S/D/I 오류 분해 | `scripts/mv2/` |

### 지연 정의

```
알고리즘 지연 = min_silence_duration_ms + trailing_pad_ms + VAD lookahead
```

leading 마진은 이미 버퍼에 들어온 과거 오디오를 더 쓰는 것이라 지연 식에 포함되지 않습니다.
실측 지연은 스트리밍 모델에서 발화 종료 시점 대비 최종 결과가 확정되는 시점으로 측정합니다.

### 평가 지표

평균 CER, 중앙값 CER, 상한 CER(min(CER, 1.0)), CER > 100% 발화 비율, 첫/끝 단어 오류,
치환/삭제/삽입(S/D/I) 분해, 부트스트랩 신뢰구간.

## 저장소 구조

```text
conf/             실험 설정 (base.yaml이 기본, 나머지는 split/GPU별 변형)
src/              핵심 코드 (ASR 백본, VAD, 마진 정책, 지표, 데이터 로더)
scripts/          실행 스크립트
  data/           데이터 다운로드·전처리·MFA 정렬
  launch/         여러 조건을 순회하는 실험 런처
  mv2/            통합 실험 파이프라인 (s1 → s5, s7)
```

`data/`, `artifacts/`, `exp/`, `logs/`는 실행 중에 생성되며 저장소에는 포함하지 않습니다.

## 환경 설정

```bash
conda create -n vad-margin python=3.10
conda activate vad-margin
pip install -r requirements.txt
```

- **MFA** (강제정렬)는 별도 환경에 설치합니다.
  ```bash
  conda create -n mfa-aligner -c conda-forge montreal-forced-aligner pyyaml
  conda activate mfa-aligner
  mfa model download dictionary english_us_mfa
  mfa model download acoustic english_mfa
  ```
- **NeMo** (FastConformer 실험)도 의존성 충돌을 피하려고 별도 환경을 씁니다.
  ```bash
  conda create -n vad-margin-nemo python=3.10
  conda activate vad-margin-nemo
  pip install "nemo_toolkit[asr]"
  ```
- 여러 환경을 오가는 스크립트는 아래 환경 변수로 인터프리터 경로를 받습니다.
  ```bash
  export PY_TORCH=$(conda run -n vad-margin which python)
  export PY_NEMO=$(conda run -n vad-margin-nemo which python)
  export MFA_BIN=$(conda run -n mfa-aligner bash -c 'dirname $(which mfa)')
  ```
- pyannote, TED-LIUM 3 등 gated 리소스는 Hugging Face 토큰이 필요합니다.
  `.env.example`을 `.env`로 복사해 `HF_TOKEN`을 채우세요.

## 실행 예시

```bash
# 데이터 준비
bash scripts/data/download_librispeech.sh
python scripts/data/run_mfa_librispeech.py --jobs 8     # mfa-aligner 환경
python scripts/data/prepare_ami.py
python scripts/data/prepare_tedlium.py

# 백본 동작 확인
python scripts/smoke_test_backbones.py

# 1D 마진 스윕 + VAD 기반 디코딩
bash scripts/run_experiment_pipeline.sh all_steps librispeech
bash scripts/run_experiment_pipeline.sh step1 ami --model wav2vec2-base-960h --language none

# VAD 종류·마진 정책 지정
python scripts/run_vad_decode.py --manifest artifacts/manifests/test.jsonl \
  --vad webrtc --leading-pad-ms 400 --trailing-pad-ms 100 \
  --model wav2vec2-base-960h --language none

# 2D 그리드 (split 하나씩)
bash scripts/launch/run_silero_grid50.sh test
```

결과는 `exp/<실험명>/` 아래에 `<dataset>__<asr_tag>` 단위로 저장되어, 같은 데이터를 여러 백본으로
돌려도 서로 덮어쓰지 않습니다.

## License

MIT — [LICENSE](LICENSE)
