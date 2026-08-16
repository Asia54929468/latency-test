"""
亚像素坐标算法
实际测试过程中发现匹配到的坐标与实际存在一定偏差
因而丢弃该版本
"""

import math
import time
from pathlib import Path

import cv2
import numpy as np
import pyautogui


# ============================================================
# 参数配置区
# ============================================================

# 拖拽起始坐标
START = (251, 132)

# 拖拽终点坐标
END = (900, 132)

# 截图区域左上角坐标
REGION_TOP_LEFT = (330, 100)

# 截图区域右下角坐标
REGION_BOTTOM_RIGHT = (1000, 200)

# 模板图片路径
TEMPLATE_PATH = "template.jpeg"

# 截图结果保存目录
OUTPUT_DIR = "output"

# 每轮正向拖拽截图文件名前缀
OUTPUT_PREFIX = "match_result"

# 重复次数
REPEAT_N = 10

# 从起点到终点的 3/5 位置进行截图
CAPTURE_RATIO = 3.0 / 5.0

# 正向或反向拖拽持续时间，单位：秒
DRAG_DURATION = 1.0

# 鼠标位置更新频率，单位：Hz
UPDATE_HZ = 180.0

# 是否启用缓入缓出
SMOOTH = True

# 按下左键后等待时间
PRESS_DELAY = 0.15

# 到达终点后，释放左键前等待时间
RELEASE_DELAY = 0.10

# 模板匹配阈值
MATCH_THRESHOLD = 0.35

# PyAutoGUI 参数
pyautogui.PAUSE = 0
pyautogui.MINIMUM_DURATION = 0
pyautogui.FAILSAFE = True


# ============================================================
# 基础函数
# ============================================================

def ease_in_out(t: float) -> float:
    """
    缓入缓出曲线。

    t 的取值范围为 0~1。
    """
    return 0.5 - 0.5 * math.cos(math.pi * t)


def read_image(image_path: str) -> np.ndarray:
    """
    读取图片，兼容中文路径。
    """
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"图片不存在：{path}")

    image_data = np.fromfile(
        str(path),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        image_data,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise RuntimeError(f"无法读取图片：{path}")

    return image


def save_image(
    image: np.ndarray,
    output_path: str
) -> None:
    """
    保存图片，兼容中文路径。
    """
    output = Path(output_path)

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    extension = output.suffix.lower()

    if not extension:
        extension = ".png"
        output = output.with_suffix(extension)

    success, encoded_image = cv2.imencode(
        extension,
        image
    )

    if not success:
        raise RuntimeError(
            f"图片编码失败：{output}"
        )

    encoded_image.tofile(str(output))


def refine_match_position(
    match_result: np.ndarray,
    x: int,
    y: int
) -> tuple[float, float]:
    """
    使用匹配峰值周围的像素进行亚像素位置估计。

    返回模板左上角在截图中的浮点坐标。
    """
    result_height, result_width = match_result.shape[:2]

    # 位于边界时无法使用左右、上下邻域
    if (
        x <= 0
        or x >= result_width - 1
        or y <= 0
        or y >= result_height - 1
    ):
        return float(x), float(y)

    center = float(match_result[y, x])

    left_value = float(
        match_result[y, x - 1]
    )

    right_value = float(
        match_result[y, x + 1]
    )

    top_value = float(
        match_result[y - 1, x]
    )

    bottom_value = float(
        match_result[y + 1, x]
    )

    # 横向抛物线插值
    denominator_x = (
        left_value
        - 2.0 * center
        + right_value
    )

    if abs(denominator_x) > 1e-12:
        offset_x = (
            0.5
            * (left_value - right_value)
            / denominator_x
        )
    else:
        offset_x = 0.0

    # 纵向抛物线插值
    denominator_y = (
        top_value
        - 2.0 * center
        + bottom_value
    )

    if abs(denominator_y) > 1e-12:
        offset_y = (
            0.5
            * (top_value - bottom_value)
            / denominator_y
        )
    else:
        offset_y = 0.0

    # 限制插值偏移范围，避免异常值
    offset_x = float(
        np.clip(offset_x, -0.5, 0.5)
    )

    offset_y = float(
        np.clip(offset_y, -0.5, 0.5)
    )

    refined_x = float(x) + offset_x
    refined_y = float(y) + offset_y

    return refined_x, refined_y


# ============================================================
# 截图和模板匹配
# ============================================================

def screenshot_and_match(
    region_top_left: tuple[int, int],
    region_bottom_right: tuple[int, int],
    template_path: str,
    output_path: str,
    match_threshold: float
) -> dict | None:
    """
    截取区域并匹配唯一图标。

    返回结果示例：

    {
        "confidence": 0.95,
        "top_left_in_screenshot": (123.21, 45.36),
        "center_in_screenshot": (155.21, 77.36),
        "top_left_on_screen": (223.21, 145.36),
        "center_on_screen": (255.21, 177.36)
    }
    """
    if not 0.0 <= match_threshold <= 1.0:
        raise ValueError(
            "MATCH_THRESHOLD 必须在 0~1 之间"
        )

    left, top = region_top_left
    right, bottom = region_bottom_right

    if right <= left or bottom <= top:
        raise ValueError(
            "截图区域右下角必须位于左上角右下方"
        )

    region_width = right - left
    region_height = bottom - top

    # 截图
    screenshot_pil = pyautogui.screenshot(
        region=(
            left,
            top,
            region_width,
            region_height
        )
    )

    # PIL RGB 转 OpenCV BGR
    screenshot_rgb = np.array(
        screenshot_pil
    )

    screenshot_bgr = cv2.cvtColor(
        screenshot_rgb,
        cv2.COLOR_RGB2BGR
    )

    template = read_image(template_path)

    template_height, template_width = (
        template.shape[:2]
    )

    if (
        template_width > region_width
        or template_height > region_height
    ):
        raise ValueError(
            "模板尺寸不能大于截图区域："
            f"模板={template_width}x{template_height}，"
            f"区域={region_width}x{region_height}"
        )

    # 模板匹配
    match_result = cv2.matchTemplate(
        screenshot_bgr,
        template,
        cv2.TM_CCOEFF_NORMED
    )

    # 只取全局最佳匹配位置
    _, max_value, _, max_location = cv2.minMaxLoc(
        match_result
    )

    confidence = float(max_value)

    # 无论匹配成功与否，都先保存截图
    annotated_image = screenshot_bgr.copy()

    if confidence < match_threshold:
        save_image(
            annotated_image,
            output_path
        )

        print(
            f"未找到匹配图标，"
            f"置信度={confidence:.4f}，"
            f"截图已保存：{output_path}"
        )

        return None

    integer_x, integer_y = max_location

    # 亚像素细化模板左上角坐标
    local_x, local_y = refine_match_position(
        match_result=match_result,
        x=int(integer_x),
        y=int(integer_y)
    )

    # 模板中心，保留浮点精度
    local_center_x = (
        local_x
        + float(template_width) / 2.0
    )

    local_center_y = (
        local_y
        + float(template_height) / 2.0
    )

    # 转换为屏幕绝对坐标
    screen_left = (
        float(left) + local_x
    )

    screen_top = (
        float(top) + local_y
    )

    screen_center_x = (
        float(left)
        + local_center_x
    )

    screen_center_y = (
        float(top)
        + local_center_y
    )

    # ========================================================
    # 仅用于绘图的整数坐标
    # 不影响上面的浮点坐标
    # ========================================================

    draw_left = round(local_x)
    draw_top = round(local_y)

    draw_right = round(
        local_x
        + float(template_width)
        - 1.0
    )

    draw_bottom = round(
        local_y
        + float(template_height)
        - 1.0
    )

    draw_center_x = round(
        local_center_x
    )

    draw_center_y = round(
        local_center_y
    )

    # 绘制匹配框
    cv2.rectangle(
        annotated_image,
        (draw_left, draw_top),
        (draw_right, draw_bottom),
        (0, 255, 0),
        2
    )

    # 绘制中心点
    cv2.circle(
        annotated_image,
        (draw_center_x, draw_center_y),
        4,
        (0, 0, 255),
        -1
    )

    # 图片中的文字标注
    label = (
        f"score={confidence:.4f} "
        f"center=("
        f"{screen_center_x:.2f},"
        f"{screen_center_y:.2f})"
    )

    text_x = max(draw_left, 0)
    text_y = max(
        round(local_y - 8.0),
        20
    )

    cv2.putText(
        annotated_image,
        label,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
        cv2.LINE_AA
    )

    # 保存标注后的截图
    save_image(
        annotated_image,
        output_path
    )

    print(
        f"截图结果已保存：{output_path}"
    )

    print(
        f"匹配左上角屏幕坐标："
        f"({screen_left:.2f}, {screen_top:.2f})"
    )

    print(
        f"匹配图标中心坐标："
        f"({screen_center_x:.2f}, "
        f"{screen_center_y:.2f})"
    )

    print(
        f"匹配置信度：{confidence:.4f}"
    )

    return {
        "confidence": confidence,

        # 原始整数匹配位置
        "integer_top_left_in_screenshot": (
            int(integer_x),
            int(integer_y)
        ),

        # 亚像素左上角，截图内坐标
        "top_left_in_screenshot": (
            float(local_x),
            float(local_y)
        ),

        # 亚像素中心，截图内坐标
        "center_in_screenshot": (
            float(local_center_x),
            float(local_center_y)
        ),

        # 亚像素左上角，屏幕绝对坐标
        "top_left_on_screen": (
            float(screen_left),
            float(screen_top)
        ),

        # 亚像素中心，屏幕绝对坐标
        "center_on_screen": (
            float(screen_center_x),
            float(screen_center_y)
        ),

        # 模板尺寸
        "template_size": (
            int(template_width),
            int(template_height)
        )
    }


# ============================================================
# 拖拽函数
# ============================================================

def drag_path(
    start: tuple[int, int],
    end: tuple[int, int],
    duration: float,
    update_hz: float,
    smooth: bool,
    press_delay: float,
    release_delay: float,
    capture_ratio: float | None = None,
    capture_callback=None
) -> dict | None:
    """
    按住左键，从 start 拖拽到 end。

    当 capture_ratio 和 capture_callback 不为空时，
    在路径指定比例位置执行截图匹配。
    """
    if duration < 0:
        raise ValueError(
            "duration 不能小于 0"
        )

    if update_hz <= 0:
        raise ValueError(
            "update_hz 必须大于 0"
        )

    pyautogui.moveTo(*start)

    mouse_pressed = False
    capture_result = None
    capture_completed = False

    try:
        pyautogui.mouseDown(
            button="left"
        )

        mouse_pressed = True

        if press_delay > 0:
            time.sleep(press_delay)

        if duration == 0:
            if (
                capture_ratio is not None
                and capture_callback is not None
            ):
                capture_result = capture_callback()
                capture_completed = True

            pyautogui.moveTo(*end)

        else:
            interval = 1.0 / update_hz

            begin_time = time.perf_counter()
            next_update_time = begin_time

            while True:
                now = time.perf_counter()

                progress = (
                    now - begin_time
                ) / duration

                if progress >= 1.0:
                    break

                movement_ratio = (
                    ease_in_out(progress)
                    if smooth
                    else progress
                )

                # 到达指定比例时执行截图
                if (
                    not capture_completed
                    and capture_ratio is not None
                    and capture_callback is not None
                    and movement_ratio >= capture_ratio
                ):
                    capture_begin_time = (
                        time.perf_counter()
                    )

                    capture_result = (
                        capture_callback()
                    )

                    capture_completed = True

                    capture_elapsed = (
                        time.perf_counter()
                        - capture_begin_time
                    )

                    # 排除截图和匹配耗时，
                    # 保持拖拽过程时间控制稳定
                    begin_time += capture_elapsed
                    next_update_time += capture_elapsed

                current_x_float = (
                    float(start[0])
                    + (
                        float(end[0])
                        - float(start[0])
                    )
                    * movement_ratio
                )

                current_y_float = (
                    float(start[1])
                    + (
                        float(end[1])
                        - float(start[1])
                    )
                    * movement_ratio
                )

                # 操作系统鼠标坐标通常为整数像素
                pyautogui.moveTo(
                    round(current_x_float),
                    round(current_y_float)
                )

                next_update_time += interval

                sleep_time = (
                    next_update_time
                    - time.perf_counter()
                )

                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    next_update_time = (
                        time.perf_counter()
                    )

            # 防止时间过短或频率过低导致未触发截图
            if (
                not capture_completed
                and capture_ratio is not None
                and capture_callback is not None
            ):
                capture_result = capture_callback()

            # 确保准确到达终点
            pyautogui.moveTo(*end)

        if release_delay > 0:
            time.sleep(release_delay)

    finally:
        # 确保异常时也释放左键
        if mouse_pressed:
            pyautogui.mouseUp(
                button="left"
            )

    return capture_result


# ============================================================
# 单轮：正向拖拽、截图、反向拖回
# ============================================================

def execute_one_round(
    round_index: int
) -> dict | None:
    """
    执行一轮完整操作：

    1. START -> END，途中截图匹配
    2. END -> START，拖回原位置
    """
    output_path = Path(OUTPUT_DIR) / (
        f"{OUTPUT_PREFIX}_{round_index:03d}.png"
    )

    start_x, start_y = START
    end_x, end_y = END

    # 截图触发点的理论浮点坐标
    capture_x_float = (
        float(start_x)
        + (
            float(end_x)
            - float(start_x)
        )
        * CAPTURE_RATIO
    )

    capture_y_float = (
        float(start_y)
        + (
            float(end_y)
            - float(start_y)
        )
        * CAPTURE_RATIO
    )

    # 鼠标最终只能移动到整数屏幕坐标
    capture_point = (
        round(capture_x_float),
        round(capture_y_float)
    )

    print()
    print("=" * 60)
    print(f"开始第 {round_index}/{REPEAT_N} 轮")
    print(
        f"正向拖拽：{START} -> {END}"
    )
    print(
        "截图理论位置："
        f"({capture_x_float:.2f}, "
        f"{capture_y_float:.2f})"
    )

    def capture_callback():
        """
        截图回调函数。
        此时左键保持按下状态。
        """
        # 确保光标位于起点至终点 3/5 处
        pyautogui.moveTo(*capture_point)

        return screenshot_and_match(
            region_top_left=REGION_TOP_LEFT,
            region_bottom_right=REGION_BOTTOM_RIGHT,
            template_path=TEMPLATE_PATH,
            output_path=str(output_path),
            match_threshold=MATCH_THRESHOLD
        )

    # 正向拖拽，并在 3/5 处截图
    match_result = drag_path(
        start=START,
        end=END,
        duration=DRAG_DURATION,
        update_hz=UPDATE_HZ,
        smooth=SMOOTH,
        press_delay=PRESS_DELAY,
        release_delay=RELEASE_DELAY,
        capture_ratio=CAPTURE_RATIO,
        capture_callback=capture_callback
    )
    time.sleep(1)

    # 从终点拖回起始位置
    drag_path(
        start=END,
        end=START,
        duration=DRAG_DURATION,
        update_hz=UPDATE_HZ,
        smooth=SMOOTH,
        press_delay=PRESS_DELAY,
        release_delay=RELEASE_DELAY,
        capture_ratio=None,
        capture_callback=None
    )
    time.sleep(1)

    return match_result


# ============================================================
# 主程序
# ============================================================

def main():
    if REPEAT_N <= 0:
        raise ValueError(
            "REPEAT_N 必须大于 0"
        )

    template_path = Path(TEMPLATE_PATH)

    if not template_path.exists():
        raise FileNotFoundError(
            f"模板图片不存在：{template_path}"
        )

    print("程序将在 3 秒后开始执行")
    print("请在此期间切换到目标窗口")
    time.sleep(3)

    successful_results: list[dict] = []

    for round_index in range(1, REPEAT_N + 1):
        result = execute_one_round(
            round_index=round_index
        )

        if result is not None:
            successful_results.append(result)

    print()
    print("=" * 60)
    print("所有操作执行完成")

    if not successful_results:
        print("没有成功匹配到图标，无法计算平均中心坐标")
        return

    # 取每轮匹配到的屏幕绝对中心坐标
    center_array = np.array(
        [
            result["center_on_screen"]
            for result in successful_results
        ],
        dtype=np.float64
    )

    # 对 X、Y 分别求平均值
    average_center_x = float(
        np.mean(center_array[:, 0])
    )

    average_center_y = float(
        np.mean(center_array[:, 1])
    )

    print(
        f"成功匹配次数："
        f"{len(successful_results)}/{REPEAT_N}"
    )

    print(
        "平均图标中心坐标："
        f"({average_center_x:.2f}, "
        f"{average_center_y:.2f})"
    )

    print(
        "平均图标中心坐标原始浮点值："
        f"({average_center_x!r}, "
        f"{average_center_y!r})"
    )


if __name__ == "__main__":
    main()