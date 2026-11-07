# 데이터·모델 출처와 공개 범위

## 데이터

[DAKER 합성 미세구조 기반 재료 물성 예측 AI 경진대회](https://daker.ai/public/hackathons/synthetic-microstructure-property-prediction-ai)의256×256 합성 회색조 영상과 경도 대용값을 사용했다. 실제 측정 경도 데이터가 아니다. 이 저장소는 대회 데이터 재배포 권한을 주장하지 않으며 원본·정답·제출 예측을 포함하지 않는다.

## 주요 외부 모델/라이브러리

| 자산 | 공식 출처 | 이 저장소의 취급 |
|---|---|---|
| DINOv2 | [Meta 공식 저장소](https://github.com/facebookresearch/dinov2) | 연구용 로더·설정만 포함, checkpoint 제외 |
| C-RADIOv4-SO400M | [NVIDIA 모델 카드](https://huggingface.co/nvidia/C-RADIOv4-SO400M) | revision/hash lock 포함, 가중치와 외부 모델 소스 제외 |
| TabPFN-v2 regression | [Prior Labs 모델 카드](https://huggingface.co/Prior-Labs/TabPFN-v2-reg) | 호출·대조 코드만 포함, checkpoint 제외 |
| ResNet18 | [torchvision](https://docs.pytorch.org/vision/stable/models/resnet.html) | 라이브러리 사용, 가중치 제외 |
| Scattering | [Kymatio](https://www.kymat.io/) | 특징 추출 래퍼와 설치 기록 포함 |

모델 가중치의 권리는 각 배포자의 조건을 따른다. C-RADIO 모델 카드는NVIDIA Open Model License, TabPFN-v2 모델 카드는Prior Labs의별도조건을 안내한다. 이 문서는 사용 허가를 대신하지 않으며 필요한 자산은 공식 출처에서 조건을 확인한 뒤 받아야 한다.

## 포함된 제3자 소스

과거 DRUNet 대조용[scripts/vendor/drunet](../scripts/vendor/drunet)의2개 소스 파일은 원본MIT LICENSE와PROVENANCE.txt를 함께 보존했다. 그 파일의권리는 동봉 라이선스를 따른다. 나머지 공개 코드에 모델·데이터 전체의 재배포 권리를 포괄적으로 부여하는 라이선스는 추가하지 않았다.

## 프로젝트 작성 과정

Codex를 실험 설계·코드 작성·결과 분석·문서화에 활용한 개인 프로젝트다. 실행·제출은 참가자가 수행했다. AI 생성 내용도 결과 파일·수치 검증·대조 실험을 근거로 채택 여부를 판단했다.
