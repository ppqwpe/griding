"""Griding — 머리 위에 격자를 덮어 시댄스 실사 얼굴 검열을 피하는 도구

모듈 구성
  config     경로 · 상수
  geometry   Rect (박스 계산)
  settings   GridSettings (격자 설정 검증)
  patterns   Pattern 과 모양별 하위 클래스 · PatternRegistry
  renderer   GridRenderer (격자 불투명도 맵 · 합성)
  detector   HeadDetector (YOLO11 head ONNX)
  tracker    Track · Tracker (프레임 사이 박스 잇기)
  media      MediaSource (VideoSource / ImageSource) · FFmpegWriter
  composer   Composer (overlay · 3칸 비교 · 밀착 시트 그림)
  outputs    Output 과 결과물별 하위 클래스 · OutputCatalog
  pipeline   FaceGridPipeline · JobSummary
  store      ResultStore (결과 폴더 · verification.json) · UploadStore
  jobs       Job · JobQueue (한 번에 하나씩 처리)
  gallery    Gallery (설정 예시 · 모양 · 가리기 예시 그림)
  server     FaceGridApp (웹 서버 + 화면)
"""
__version__ = "1.0.0"
