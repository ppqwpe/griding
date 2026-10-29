# Griding

실사 플레이트 속 얼굴을 격자나 가림으로 덮어서, 시댄스에 넣을 수 있는 영상으로 바꿔 주는 Windows 앱입니다.
ComfyUI 없이 `Griding.bat` 하나로 켜지고, 모든 처리는 그 PC 안에서 끝납니다.

![Griding](로고/griding_logo_dark.png)

## 쓰는 사람

배포 zip(`Griding_배포.zip`)을 풀고 `Griding.bat`을 두 번 누르면 됩니다. 그래픽카드 때문에 멈추면 `Griding_CPU.bat`으로 켜세요.
처음 켤 때 바탕화면에 아이콘이 달린 바로가기가 생깁니다. 쓰는 법은 [사용법.txt](사용법.txt)에 있습니다.

## 고치는 사람

이 저장소에는 파이썬 실행 파일과 라이브러리(`app/python`, `app/lib`, 약 460MB)가 없습니다. 너무 커서 깃에 넣지 않았습니다.
처음 한 번만 배포 zip에서 그 두 폴더를 이 저장소의 같은 자리로 복사하면 바로 돌아갑니다.

```
git clone <이 저장소 주소> Griding
(배포 zip 의 Griding\app\python, Griding\app\lib 를 Griding\app\ 아래로 복사)
Griding\Griding.bat
```

고친 뒤에는 테스트를 돌립니다. 155개가 모두 통과해야 합니다.

```
app\python\python.exe app\tests\run_tests.py
```

새 배포 zip은 이렇게 만듭니다. 저장소 바깥에 `Griding_배포.zip`이 생깁니다.

```
app\python\python.exe app\tools\build_zip.py
```

### 어디에 무엇이 있나

| 위치 | 내용 |
|---|---|
| `app/facegrid/` | 앱 본체 (파이썬 패키지 이름은 예전 이름 `facegrid` 그대로) |
| `app/ui.html` | 브라우저 화면 하나 (HTML · CSS · JS) |
| `app/tests/` | 자동 테스트 |
| `app/model/` | 머리 검출 모델 2개, 표정 인식 모델, 예시 캐릭터 그림 |
| `app/tools/` | 로고 · 예시 캐릭터 얼굴 점 · 배포 zip 만드는 스크립트 |
| `로고/` | 따로 쓰는 로고 PNG (아이콘 크기별, 글자 포함 가로형) |

로고를 고치려면 `app/tools/make_logo.py`를, 예시 캐릭터 얼굴 점을 고치려면 `app/tools/make_character_face.py`를 고쳐서 다시 실행합니다.

### 작업 흐름

`main` 브랜치는 늘 배포할 수 있는 상태로 둡니다. 고칠 일이 생기면 브랜치를 따로 만들고, 테스트가 통과하면 `main`에 합칩니다.
배포할 때는 `build_zip.py`로 만든 zip을 GitHub Releases에 버전 번호(v1.1 등)와 함께 올립니다. 그러면 받는 사람은 그 페이지에서 내려받기만 하면 됩니다.

## 쓰는 모델

| 단계 | 모델 | 라이선스 |
|---|---|---|
| 머리 찾기 | YOLO11s · YOLO11m head (deepghs/real_head_detection, ONNX) | AGPL-3.0 |
| 표정 찾기 | Google MediaPipe Face Landmarker (얼굴 점 478개) | Apache-2.0 |
| 읽기 · 저장 | FFmpeg 7.1 (gyan.dev essentials 빌드, imageio-ffmpeg 0.6.0 에 포함) | [확인 필요: 이 빌드의 라이선스] |

머리 검출 모델이 AGPL-3.0이라, 앱을 밖으로 나눠 줄 때는 그 조건을 따라야 합니다.
