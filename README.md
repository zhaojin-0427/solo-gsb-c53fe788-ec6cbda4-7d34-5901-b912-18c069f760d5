# 包围曝光合成 HTTP API

用 Python + FastAPI + OpenCV + NumPy 实现的包围曝光（exposure bracketing）合成服务：
接收 3～9 张**同尺寸 16 位线性 RGB PNG**（Base64），先依据**未剪切的静态背景**
做整数平移配准（允许各轴 ±8 px，移动的高对比度前景**不会**牵引位移），再按
曝光时间归一化为辐亮度，逐像素判定运动并融合。运动区域只取参考帧（无重影），
无法恢复的像素标为未知（NaN）。

- `POST /merges`：合成
- `GET /health`：健康检查
- 自动生成的交互式文档：`GET /docs`（Swagger UI）

---

## 1. 一键启动（Docker Compose）

需要 Docker（含 Compose 插件）。

```bash
docker compose up --build
```

服务监听 `http://localhost:8000`。健康检查：

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

停止：`Ctrl+C`，或在另一终端执行 `docker compose down`。

### 不用 Docker（本地直接运行）

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## 2. 请求

`POST /merges`，`Content-Type: application/json`，请求体：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `frames` | 数组，长度 3～9 | 每项 `{"image_base64": "..."}`，Base64 编码的 16 位线性 RGB PNG，所有帧同尺寸 |
| `exposure_seconds` | 数字数组 | 每张帧的曝光秒数，与 `frames` 一一对应；必须为**有限正数** |
| `reference_index` | 整数 | 参考帧序号（从 0 开始），`0 ≤ reference_index ≤ 8` |
| `tolerance` | 数字 | 辐亮度容差，**有限非负数**；任一通道差异超过它即判定为运动 |

### 语义约定

- **无效观测**：某像素任一通道数字值为 `0` 或 `65535`（黑剪切/饱和）时，该观测无效，不参与配准与融合。
- **辐亮度归一化**：有效像素 `radiance = digital / 65535 / exposure_seconds`（float32）。
- **配准**：每帧对参考帧估计整数平移 `(dy, dx)`，`|dy|,|dx| ≤ 8`；
  含义：帧 i 的源像素 `(y, x)` 对应参考坐标 `(y + dy, x + dx)`。
  评分仅使用双方都未剪切、且有纹理的背景块，并用分块中值抑制少数被运动
  前景/高光占据的块——因此移动的物体不会影响位移结果。背景无法可靠或唯一
  配准时返回 422。
- **运动判定**：对齐后，某像素至少有两个有效观测，且**任意两帧任意通道**
  辐亮度之差 `max(|ri - rj|) > tolerance` 时，该像素为运动区。
- **融合**：
  - 静态像素：对全部有效观测取均值；
  - 运动像素：**只取参考帧**的归一化辐亮度（彻底避免移动物体重影）。
- **未知像素**（输出 TIFF 中为 `NaN`，并在 unknown 掩膜中标 255）：
  - 运动区但参考帧该观测无效；
  - 静态区但所有观测无效（含全部剪切，或帧平移后该位置越界）。

---

## 3. 响应

`200 OK`，JSON：

| 字段 | 说明 |
| --- | --- |
| `radiance_tiff_base64` | Base64 编码的 **32 位浮点 RGB TIFF**（按扫描行 RGB 顺序，Deflate 压缩）；未知像素为 `NaN` |
| `motion_mask_base64` | Base64 8 位灰度 PNG：运动像素 `255`，其余 `0` |
| `unknown_mask_base64` | Base64 8 位灰度 PNG：未知像素 `255`，其余 `0`（可与运动掩膜重叠） |
| `shifts` | 每帧 `[dy, dx]`，参考帧为 `[0, 0]` |
| `width` / `height` | 输出尺寸（与输入一致） |
| `reference_index` | 回显参考帧序号 |

错误（非法输入或背景无法配准）返回 `422`，形如 `{"error": "...", "detail": ...}`。

---

## 4. 实际请求 / 响应示例

仓库内置一个与实现完全一致的合成示例（48×64，3 张 16 位 PNG；曝光
`[0.25, 1.0, 2.0]` 秒；参考帧 `1`；容差 `0.05`）：
场景是带纹理的静态背景 + 一块移动的高对比度矩形 + 一处长曝光会饱和的高光，
帧间还有整数平移。完整报文见 [`examples/request.json`](examples/request.json)
和 [`examples/response.json`](examples/response.json)。

请求（Base64 串已截断）：

```json
{
  "frames": [
    {"image_base64": "iVBORw0KGgoAAAANSUhEUgAAAEAAAAAwEAIAAAB+...RU5ErkJggg=="},
    {"image_base64": "iVBORw0KGgoAAAANSUhEUgAAAEAAAAAwEAIAAAB+...RU5ErkJggg=="},
    {"image_base64": "iVBORw0KGgoAAAANSUhEUgAAAEAAAAAwEAIAAAB+...RU5ErkJggg=="}
  ],
  "exposure_seconds": [0.25, 1.0, 2.0],
  "reference_index": 1,
  "tolerance": 0.05
}
```

### 用 curl 调用

```bash
curl -s -X POST http://localhost:8000/merges \
  -H 'Content-Type: application/json' \
  --data @examples/request.json | python3 -m json.tool
```

### 实际响应（Base64 串已截断，数值为服务真实输出）

```json
{
    "height": 48,
    "motion_mask_base64": "iVBORw0KGgoAAAANSUhEUgAAAEAAAAAw...",
    "radiance_tiff_base64": "SUkqAC4pAAB4nNWZXWgUVxTHz2ygvtmH...",
    "reference_index": 1,
    "shifts": [[3, -5], [0, 0], [-4, 6]],
    "unknown_mask_base64": "iVBORw0KGgoAAAANSUhEUgAAAEAAAAAw...",
    "width": 64
}
```

解码该响应可验证（OpenCV / NumPy）：

- 位移恢复为 `[[3,-5],[0,0],[-4,6]]`——正是生成数据时注入的平移，
  且未被移动矩形牵引；
- TIFF 为 `float32`、形状 `(48, 64, 3)`；未知像素为 `NaN`；
- 运动掩膜 311 个像素（移动矩形的新、旧位置），未知掩膜 7 个像素
  （参考帧饱和高光且其它帧也未覆盖的位置）；
- 静态点 `(42, 8)` 三帧融合辐亮度约 `[0.4260, 0.2740, 0.4000]`；
  运动点 `(25, 46)` 仅取参考帧，约 `[0.3750, 0.3880, 0.4210]`，无重影。

下面的脚本复现上述请求并解码、保存三个产物（运行前先启动服务）：

```python
import base64, json, urllib.request
import cv2, numpy as np

payload = json.load(open("examples/request.json"))
req = urllib.request.Request(
    "http://localhost:8000/merges",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
resp = json.load(urllib.request.urlopen(req))

def decode_png(b64):
    return cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8),
                        cv2.IMREAD_UNCHANGED)

radiance = cv2.imdecode(
    np.frombuffer(base64.b64decode(resp["radiance_tiff_base64"]), np.uint8),
    cv2.IMREAD_UNCHANGED,
)  # float32 HxWx3, 未知像素是 NaN
motion = decode_png(resp["motion_mask_base64"])     # uint8, 0/255
unknown = decode_png(resp["unknown_mask_base64"])   # uint8, 0/255

print(resp["shifts"], radiance.dtype, radiance.shape)
```

### 用你自己的 16 位 PNG 构造请求

```python
import base64, json, urllib.request

def b64_png(path):
    return base64.b64encode(open(path, "rb").read()).decode()

payload = {
    "frames": [
        {"image_base64": b64_png("short.png")},
        {"image_base64": b64_png("mid.png")},
        {"image_base64": b64_png("long.png")},
    ],
    "exposure_seconds": [0.25, 1.0, 2.0],
    "reference_index": 1,
    "tolerance": 0.05,
}
req = urllib.request.Request(
    "http://localhost:8000/merges",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
print(urllib.request.urlopen(req).status)
```

### 错误示例

```bash
# 曝光秒数为 0（非有限正数）
curl -s -X POST http://localhost:8000/merges -H 'Content-Type: application/json' \
  --data '{"frames":[...3 张...],"exposure_seconds":[0,1,2],
           "reference_index":1,"tolerance":0.05}'
# HTTP/1.1 422
# {"error":"请求参数不合法","detail":[...]}

# 某帧与其余帧毫不相干，静态背景无法配准
# HTTP/1.1 422
# {"error":"registration failed: no shift aligns enough unsheared static
#  background (moving occlusion or frame mismatch too large)"}
```

其它会触发 422 的情况：帧数不在 3～9、帧尺寸不一致、非 PNG / 非 16 位、
Base64 非法、`tolerance` 为负数或 NaN/Infinity、`reference_index` 越界、
背景纹理不足或配准不唯一。

---

## 5. 项目结构与测试

```
app/
  main.py          FastAPI 路由、错误映射、NaN 宽容的 JSON 解析
  schemas.py       Pydantic 请求/响应模型与输入校验
  merge.py         配准调度、运动判定、辐亮度融合、未知区标注
  registration.py  ±8 px 整数平移的抗前景背景配准
  imaging.py       16 位 PNG 解码、有效性掩膜、float32 TIFF/掩膜编码
  errors.py        MergeError
tests/             合成场景的端到端（含真实 HTTP）测试
examples/          真实请求/响应报文与解码后的产物
```

运行测试（本地 Python 环境）：

```bash
pip install -r requirements.txt pytest httpx
pytest -q
```

测试覆盖：真位移恢复（±8 边界、9 帧、大面积移动前景不牵引）、静态融合
辐亮度精度、运动掩膜、运动区只取参考帧、剪切高光从其它帧恢复、
全无效/越界标未知、配准失败与各类非法输入返回 422、TIFF/PNG 往返解码。
