import math
import time
from pathlib import Path

import cv2
import mss
import numpy as np
import pyautogui


# ============================================================
# 参数配置区
# ============================================================

# 拖拽起始、终点坐标
START = (1960, 32)
END = (2500, 32)

# 截图区域左上、右下角坐标
REGION_TOP_LEFT = (2100, 0)
REGION_BOTTOM_RIGHT = (2600, 100)

# 提前准备好的模板图片
TEMPLATE_PATH = "template.jpeg"

# 截图结果保存目录及文件名前缀
OUTPUT_DIR = "output"
OUTPUT_PREFIX = "match_result"

# 重复次数
REPEAT_N = 5

# 在起点到终点的 3/5 位置截图
CAPTURE_RATIO = 3.0 / 5.0

# 正向、反向拖拽持续时间，单位：秒
DRAG_DURATION = 1.0

# 鼠标位置更新频率，单位：Hz
UPDATE_HZ = 120.0

# 是否使用缓入缓出效果
SMOOTH = True

# 按下左键后的等待时间
PRESS_DELAY = 0.15

# 到达目标点后的释放延迟
RELEASE_DELAY = 0.10

# 模板匹配最低置信度
MATCH_THRESHOLD = 0.75

# 开始前等待时间
START_DELAY = 3.0


# PyAutoGUI 设置
pyautogui.PAUSE = 0
pyautogui.MINIMUM_DURATION = 0
pyautogui.FAILSAFE = True


# ============================================================
# 基础函数
# ============================================================

def ease_in_out(t: float) -> float:
    """
    缓入缓出曲线。

    t 的范围为 0~1。
    """
    return 0.5 - 0.5 * math.cos(math.pi * t)


def read_image(image_path: str) -> np.ndarray:
    """
    读取图片，兼容中文路径。
    """
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(
            f"模板图片不存在：{path}"
        )

    image_data = np.fromfile(
        str(path),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        image_data,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise RuntimeError(
            f"无法读取模板图片：{path}"
        )

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


# ============================================================
# 截图函数
# ============================================================

def capture_screen_region(
    region_top_left: tuple[int, int],
    region_bottom_right: tuple[int, int]
) -> np.ndarray:
    """
    使用 mss 截取屏幕区域。

    返回 BGR 格式的 OpenCV 图片。
    """
    left, top = region_top_left
    right, bottom = region_bottom_right

    if right <= left or bottom <= top:
        raise ValueError(
            "截图区域右下角必须位于左上角的右下方"
        )

    width = right - left
    height = bottom - top

    monitor = {
        "left": left,
        "top": top,
        "width": width,
        "height": height
    }

    # mss 比 pyautogui.screenshot() 通常更快
    with mss.mss() as screen:
        screenshot = screen.grab(monitor)

    # mss 返回 BGRA，转换为 BGR
    screenshot_bgr = np.asarray(
        screenshot,
        dtype=np.uint8
    )[:, :, :3].copy()

    return screenshot_bgr


# ============================================================
# 整数坐标模板匹配
# ============================================================

def match_single_icon(
    screenshot_bgr: np.ndarray,
    region_top_left: tuple[int, int],
    template_path: str,
    output_path: str,
    match_threshold: float
) -> dict | None:
    """
    在截图中匹配唯一图标。

    使用纯整数坐标：
    - matchTemplate 得到整数左上角位置；
    - 中心坐标使用整数除法；
    - 不使用亚像素插值。
    """
    template = read_image(template_path)

    screenshot_height, screenshot_width = (
        screenshot_bgr.shape[:2]
    )

    template_height, template_width = (
        template.shape[:2]
    )

    if (
        template_width > screenshot_width
        or template_height > screenshot_height
    ):
        raise ValueError(
            "模板尺寸不能大于截图区域："
            f"模板={template_width}x{template_height}，"
            f"截图={screenshot_width}x{screenshot_height}"
        )

    # 模板匹配
    match_result = cv2.matchTemplate(
        screenshot_bgr,
        template,
        cv2.TM_CCOEFF_NORMED
    )

    # 取得全局最佳匹配位置
    _, max_value, _, max_location = (
        cv2.minMaxLoc(match_result)
    )

    confidence = float(max_value)

    annotated_image = screenshot_bgr.copy()

    if confidence < match_threshold:
        save_image(
            annotated_image,
            output_path
        )

        print(
            f"未找到匹配图标，"
            f"置信度={confidence:.4f}"
        )

        print(
            f"原始截图已保存：{output_path}"
        )

        return None

    # 纯整数匹配位置
    local_x, local_y = max_location

    local_x = int(local_x)
    local_y = int(local_y)

    # 使用整数中心坐标
    local_center_x = (
        local_x + template_width // 2
    )

    local_center_y = (
        local_y + template_height // 2
    )

    region_left, region_top = region_top_left

    # 转换为屏幕绝对整数坐标
    screen_left = region_left + local_x
    screen_top = region_top + local_y

    screen_center_x = (
        region_left + local_center_x
    )

    screen_center_y = (
        region_top + local_center_y
    )

    # 绘制匹配框
    draw_right = (
        local_x + template_width - 1
    )

    draw_bottom = (
        local_y + template_height - 1
    )

    cv2.rectangle(
        annotated_image,
        (local_x, local_y),
        (draw_right, draw_bottom),
        (0, 255, 0),
        2
    )

    # 绘制中心点
    cv2.circle(
        annotated_image,
        (local_center_x, local_center_y),
        4,
        (0, 0, 255),
        -1
    )

    # 图片上的文字标注
    label = (
        f"score={confidence:.4f} "
        f"center=("
        f"{screen_center_x},"
        f"{screen_center_y})"
    )

    text_x = max(local_x, 0)
    text_y = max(local_y - 8, 20)

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

    # 保存标注后的图片
    save_image(
        annotated_image,
        output_path
    )

    print(
        f"匹配图标左上角坐标："
        f"({screen_left}, {screen_top})"
    )

    print(
        f"匹配图标中心坐标："
        f"({screen_center_x}, "
        f"{screen_center_y})"
    )

    print(
        f"匹配置信度：{confidence:.4f}"
    )

    return {
        "confidence": confidence,

        # 截图区域内的整数左上角坐标
        "top_left_in_screenshot": (
            local_x,
            local_y
        ),

        # 截图区域内的整数中心坐标
        "center_in_screenshot": (
            local_center_x,
            local_center_y
        ),

        # 屏幕绝对整数左上角坐标
        "top_left_on_screen": (
            screen_left,
            screen_top
        ),

        # 屏幕绝对整数中心坐标
        "center_on_screen": (
            screen_center_x,
            screen_center_y
        ),

        "template_size": (
            template_width,
            template_height
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
) -> np.ndarray | None:
    """
    按住鼠标左键从 start 拖拽到 end。

    如果设置 capture_ratio 和 capture_callback，
    则在指定路径比例位置进行截图。

    注意：
    截图回调中只截图，不执行模板匹配，
    以减少拖拽过程中的停顿。
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
    screenshot_completed = False
    screenshot_data = None

    try:
        # 按住左键
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
                screenshot_data = (
                    capture_callback()
                )

                screenshot_completed = True

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

                # 到达指定路径比例时截图
                if (
                    not screenshot_completed
                    and capture_ratio is not None
                    and capture_callback is not None
                    and movement_ratio >= capture_ratio
                ):
                    screenshot_data = (
                        capture_callback()
                    )

                    screenshot_completed = True

                current_x = round(
                    start[0]
                    + (end[0] - start[0])
                    * movement_ratio
                )

                current_y = round(
                    start[1]
                    + (end[1] - start[1])
                    * movement_ratio
                )

                pyautogui.moveTo(
                    current_x,
                    current_y
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

            # 低频率或持续时间过短时补充截图
            if (
                not screenshot_completed
                and capture_ratio is not None
                and capture_callback is not None
            ):
                screenshot_data = (
                    capture_callback()
                )

            # 确保最终到达终点
            pyautogui.moveTo(*end)

        if release_delay > 0:
            time.sleep(release_delay)

    finally:
        # 异常时也释放左键
        if mouse_pressed:
            pyautogui.mouseUp(
                button="left"
            )

    return screenshot_data


# ============================================================
# 单轮操作
# ============================================================

def execute_one_round(
    round_index: int
) -> dict | None:
    """
    单轮操作流程：

    1. START -> END
    2. 在正向路径 3/5 处截图
    3. 释放左键
    4. 对刚才的截图进行模板匹配
    5. END -> START
    """
    output_path = Path(OUTPUT_DIR) / (
        f"{OUTPUT_PREFIX}_{round_index:03d}.png"
    )

    start_x, start_y = START
    end_x, end_y = END

    # 计算理论截图位置
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

    # 实际鼠标只能移动到整数坐标
    capture_point = (
        round(capture_x_float),
        round(capture_y_float)
    )

    print()
    print("=" * 60)
    print(
        f"开始第 {round_index}/{REPEAT_N} 轮"
    )

    print(
        "截图理论位置："
        f"({capture_x_float:.2f}, "
        f"{capture_y_float:.2f})"
    )


    def capture_callback():
        """
        截图回调。

        此时左键保持按下状态。
        这里只进行快速截图，不进行模板匹配。
        """
        pyautogui.moveTo(*capture_point)

        screenshot = capture_screen_region(
            REGION_TOP_LEFT,
            REGION_BOTTOM_RIGHT
        )

        return screenshot

    # 正向拖拽，同时在 3/5 处截图
    screenshot_data = drag_path(
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

    # 反向拖拽，将图标从终点拖回起始位置
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

    # 拖拽全部完成后再进行模板匹配
    if screenshot_data is None:
        print("本轮未获得截图")
        return None

    result = match_single_icon(
        screenshot_bgr=screenshot_data,
        region_top_left=REGION_TOP_LEFT,
        template_path=TEMPLATE_PATH,
        output_path=str(output_path),
        match_threshold=MATCH_THRESHOLD
    )

    if result is None:
        return None

    # 取得匹配到的图标中心坐标
    matched_center_x, matched_center_y = (
        result["center_on_screen"]
    )

    # 计算匹配中心坐标与截图触发坐标的差值
    dx = matched_center_x - capture_point[0]
    dy = matched_center_y - capture_point[1]

    # 保存到当前轮次结果中
    result["capture_point"] = capture_point
    result["dx"] = float(dx)
    result["dy"] = float(dy)

    print(
        f"第 {round_index} 轮坐标差值："
        f"dx={dx:.2f}, dy={dy:.2f}"
    )

    time.sleep(1)
    return result


# ============================================================
# 主程序
# ============================================================

def main():
    if REPEAT_N <= 0:
        raise ValueError(
            "REPEAT_N 必须大于 0"
        )

    if not 0.0 <= CAPTURE_RATIO <= 1.0:
        raise ValueError(
            "CAPTURE_RATIO 必须在 0~1 之间"
        )

    if not Path(TEMPLATE_PATH).exists():
        raise FileNotFoundError(
            f"模板文件不存在：{TEMPLATE_PATH}"
        )

    print(
        f"程序将在 {START_DELAY} 秒后开始"
    )

    print("请在等待期间切换到目标窗口")
    time.sleep(START_DELAY)

    results: list[dict] = []

    for round_index in range(1, REPEAT_N + 1):
        result = execute_one_round(
            round_index=round_index
        )

        if result is not None:
            results.append(result)

    print()
    print("=" * 60)
    print("全部操作执行完成")

    if not results:
        print(
            "没有成功匹配到图标，"
            "无法计算平均中心坐标"
        )
        return

    # 取每一轮的屏幕绝对中心坐标
    center_array = np.array(
        [
            result["center_on_screen"]
            for result in results
        ],
        dtype=np.float64
    )

    # 提取每轮 dx、dy
    delta_array = np.array(
        [
            (result["dx"], result["dy"])
            for result in results
        ],
        dtype=np.float64
    )

    # X、Y 分别计算平均值
    average_x = float(
        np.mean(center_array[:, 0])
    )

    average_y = float(
        np.mean(center_array[:, 1])
    )

    average_dx = float(
        np.mean(delta_array[:, 0])
    )

    average_dy = float(
        np.mean(delta_array[:, 1])
    )

    print(
        f"成功匹配次数："
        f"{len(results)}/{REPEAT_N}"
    )

    print(
        f"平均图标中心坐标："
        f"({average_x:.2f}, {average_y:.2f})"
    )

    print(
        f"平均坐标差值："
        f"dx={average_dx:.2f}, "
        f"dy={average_dy:.2f}"
    )


if __name__ == "__main__":
    main()