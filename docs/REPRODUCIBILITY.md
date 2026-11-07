# 실행과 재현 범위

## 데이터 없이 확인 가능한 범위

Python 3.12 이상에서 프로젝트 루트 기준:

```bash
python tools/verify_evidence.py
python -m unittest discover -s portfolio_tests -v
```

이 경로는 표준 라이브러리만 사용한다. 공개한 집계의 RMSE/SSE 관계, 최종 유효 가중치, 점수 감소율, 소스 manifest를 검사한다. 원본500장·OOF 행·test 정답을 공개하지 않았으므로 점수 자체를 독립적으로 재현하는 검증은 아니다.

## 공개된 연구 소스

`tools/blend_predictions.py`는 세 구성 요소의 예측이 이미 있을 때 고정50:25:25 혼합을 수행하는 독립 도구다. ID 순서·중복·유한값을 확인하고 기존 출력파일을 덮어쓰지 않는다. 모델 학습이나 제출 업로드를 수행하지 않는다.

```bash
python tools/blend_predictions.py --scattering s.csv --large l.csv --radio c.csv --output new_blend.csv
```

차트는 공개 집계에서 만들었다. matplotlib이 있는 환경에서`python tools/render_figures.py`로 다시 생성할 수 있다.

`scripts/`와`tests/`는 종료 시점의 연구 코드 사본이다. 공개를 위해 학습 알고리즘이나 검증 규칙을 바꾸지 않았으며, 파일별SHA256은`evidence/source_manifest.json`에 기록했다. 서로 다른 실험의 환경과 자산이 필요하므로 모든 스크립트를 한 환경에서 즉시 실행할 수 있다고 보장하지 않는다.

대표적인 합성 테스트는 원래 실험 환경에서 다음 명령으로 검증했다.

```powershell
python -m unittest tests.test_cradio_fisher tests.test_large_radio_d4 -v
```

13개 테스트: Fisher 수식, 학습 문맥 분리, alpha 선택, 25%만 교체하는 혼합, 기존 특징 일치, 캐시 변조 거부, 재개 시 중복 적합 방지. 원래 환경에는 PyTorch, torchvision, NumPy, SciPy, scikit-learn, Pillow, timm 등이 설치돼 있다. `requirements-*.txt`는 용도별 기록이며, 이들을 무조건 하나로 합쳐 설치하는 경로가 아니다. TabPFN 코드 패키지 버전과 실제 v2 checkpoint 이름도 구분해야 한다.

## 전체 모델 재실행에 필요한 비공개/외부 자산

1. 공식 경로에서 받은 `open/train`, `open/test`, 학습 정답 및 제출 양식.
2. 각 모델의 원래 사용 조건에 따라 받은 checkpoint와 지정 revision의 구현.
3. 초기 benchmark부터 이어지는 학습 특징·내부 문맥 예측·검증 결과 캐시.
4. 호환되는CUDA/PyTorch 환경과RTX5060급GPU. 원래Python3.12, torch2.10.0+cu128, torchvision0.25.0+cu128 환경을 사용했다.

기존 실험 파일은 위 자산의 경로·소스·환경을 해시로 검사한다. 공개 저장소를 clone한 직후 최종 runner만 실행하면 누락된 입력 때문에 멈추는 것이 정상이며, 이 검사를 끄고 기존 점수를 재현했다고 주장하면 안 된다. **공개 패키지는 학습 전체를 자동 복원하는 turnkey 배포판이 아니다.**

로컬 원본 환경에서 최종 제출 경로는 아래와 같다. 이미 있는 검증된 결과를 재사용하며 완료된 결과의 덮어쓰기를 요구하지 않는다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_final_large_radio_d4.ps1 -Device cuda
```

최종 제출은`reports/final_large_radio_d4_blend/submission/submission.csv`, SHA256은`27041a3c35ba5014e0166dfedef16efbb1d570f807f275db3e79f8018a713104`이다. Public 점수는 플랫폼에서 참가자가 별도로 보고한 값이다.

## 포함하지 않은 파일

원본 이미지·정답표·개별 OOF 예측·test 예측CSV·구조 주석·모델 가중치·가상환경·캐시·개인 Notion 문서는 포함하지 않는다. 공개 근거는 집계 통계와 출처 파일의 상대경로/SHA256으로 제한했다. 제3자 모델의 라이선스와 데이터 조건은 [출처 문서](ASSETS.md)를 참조한다.
