import ctypes
from ctypes import wintypes


# 必须在调用任何窗口、显示器、鼠标相关 API 前设置 DPI 感知
def enable_dpi_awareness():
    user32 = ctypes.windll.user32

    # Windows 10 1703+：Per-Monitor DPI Aware V2
    try:
        set_dpi_context = user32.SetProcessDpiAwarenessContext
        set_dpi_context.argtypes = [ctypes.c_void_p]
        set_dpi_context.restype = wintypes.BOOL

        dpi_awareness_context_per_monitor_aware_v2 = ctypes.c_void_p(-4)

        if set_dpi_context(
            dpi_awareness_context_per_monitor_aware_v2
        ):
            return "Per-Monitor DPI Aware V2"
    except (AttributeError, OSError):
        pass

    # Windows 8.1+ 回退方案
    try:
        shcore = ctypes.windll.shcore
        set_process_dpi_awareness = shcore.SetProcessDpiAwareness
        set_process_dpi_awareness.argtypes = [ctypes.c_int]
        set_process_dpi_awareness.restype = ctypes.c_long

        process_per_monitor_dpi_aware = 2
        result = set_process_dpi_awareness(
            process_per_monitor_dpi_aware
        )

        if result in (0, -2147024891):
            return "Per-Monitor DPI Aware"
    except (AttributeError, OSError):
        pass

    # Windows Vista+ 回退方案，只能感知系统 DPI
    try:
        if user32.SetProcessDPIAware():
            return "System DPI Aware"
    except (AttributeError, OSError):
        pass

    return "DPI awareness may already be configured"


DPI_MODE = enable_dpi_awareness()

# DPI 设置完成后再导入键盘监听库
import keyboard  # noqa: E402


user32 = ctypes.windll.user32


class POINT(ctypes.Structure):
    _fields_ = [
        ("x", wintypes.LONG),
        ("y", wintypes.LONG),
    ]


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", wintypes.LONG),
        ("top", wintypes.LONG),
        ("right", wintypes.LONG),
        ("bottom", wintypes.LONG),
    ]


class MONITORINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("rcMonitor", RECT),
        ("rcWork", RECT),
        ("dwFlags", wintypes.DWORD),
    ]


MONITOR_DEFAULTTONEAREST = 2
MDT_EFFECTIVE_DPI = 0


def get_monitor_dpi(monitor_handle):
    """获取显示器的有效 DPI；96 DPI 对应 100% 缩放。"""
    try:
        shcore = ctypes.windll.shcore
        get_dpi_for_monitor = shcore.GetDpiForMonitor

        get_dpi_for_monitor.argtypes = [
            wintypes.HMONITOR,
            ctypes.c_int,
            ctypes.POINTER(wintypes.UINT),
            ctypes.POINTER(wintypes.UINT),
        ]
        get_dpi_for_monitor.restype = ctypes.c_long

        dpi_x = wintypes.UINT()
        dpi_y = wintypes.UINT()

        result = get_dpi_for_monitor(
            monitor_handle,
            MDT_EFFECTIVE_DPI,
            ctypes.byref(dpi_x),
            ctypes.byref(dpi_y),
        )

        if result == 0:
            return dpi_x.value, dpi_y.value
    except (AttributeError, OSError):
        pass

    # 旧版 Windows 回退为 96 DPI
    return 96, 96


def get_cursor_position():
    point = POINT()

    if not user32.GetCursorPos(ctypes.byref(point)):
        raise ctypes.WinError()

    # 获取鼠标所在显示器；鼠标不在有效区域时选择最近显示器
    monitor_handle = user32.MonitorFromPoint(
        point,
        MONITOR_DEFAULTTONEAREST,
    )

    if not monitor_handle:
        raise ctypes.WinError()

    monitor_info = MONITORINFO()
    monitor_info.cbSize = ctypes.sizeof(MONITORINFO)

    if not user32.GetMonitorInfoW(
        monitor_handle,
        ctypes.byref(monitor_info),
    ):
        raise ctypes.WinError()

    monitor_rect = monitor_info.rcMonitor
    dpi_x, dpi_y = get_monitor_dpi(monitor_handle)

    absolute_x = point.x
    absolute_y = point.y

    # 物理坐标，相对于当前显示器左上角
    relative_x = absolute_x - monitor_rect.left
    relative_y = absolute_y - monitor_rect.top

    # 转换为以 96 DPI 为基准的逻辑坐标
    logical_relative_x = round(relative_x * 96 / dpi_x)
    logical_relative_y = round(relative_y * 96 / dpi_y)

    return {
        "absolute_physical": (absolute_x, absolute_y),
        "relative_physical": (relative_x, relative_y),
        "relative_logical": (
            logical_relative_x,
            logical_relative_y,
        ),
        "monitor_bounds": (
            monitor_rect.left,
            monitor_rect.top,
            monitor_rect.right,
            monitor_rect.bottom,
        ),
        "monitor_size": (
            monitor_rect.right - monitor_rect.left,
            monitor_rect.bottom - monitor_rect.top,
        ),
        "dpi": (dpi_x, dpi_y),
        "scale": (dpi_x / 96, dpi_y / 96),
    }


def print_cursor_position():
    try:
        position = get_cursor_position()

        scale_x, scale_y = position["scale"]

        print()
        print(f"绝对物理坐标：{position['absolute_physical']}")
        print(f"当前屏幕相对物理坐标：{position['relative_physical']}")
        print(f"当前屏幕相对逻辑坐标：{position['relative_logical']}")
        print(f"显示器范围：{position['monitor_bounds']}")
        print(f"显示器物理尺寸：{position['monitor_size']}")
        print(f"显示器 DPI：{position['dpi']}")
        print(
            "显示器缩放："
            f"{scale_x * 100:.0f}% x {scale_y * 100:.0f}%"
        )
        print("-" * 45)
    except OSError as error:
        print(f"获取坐标失败：{error}")


def main():
    print(f"DPI 感知模式：{DPI_MODE}")
    print("按 Enter 获取坐标，按 Esc 退出")

    keyboard.on_release_key(
        "enter",
        lambda _: print_cursor_position(),
    )

    keyboard.wait("esc")
    keyboard.unhook_all()

    print("程序已退出")


if __name__ == "__main__":
    main()
