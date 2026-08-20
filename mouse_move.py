import pyautogui
from screeninfo import get_monitors

def relative_to_absolute(monitor_index, coord: tuple[int, int], normalized=False):
    """
    将屏幕相对坐标转换为虚拟桌面绝对坐标 (x1, y1)
    """
    rel_x, rel_y = coord  # 解包元组

    monitors = get_monitors()

    if monitor_index < 1 or monitor_index > len(monitors):
        raise ValueError(
            f"屏幕索引 {monitor_index} 无效，当前共 {len(monitors)} 块屏幕")

    m = monitors[monitor_index - 1]  # 列表从0开始，索引减1

    print(m.x, m.y)
    print(m.width, m.height)
    if normalized:
        x1 = m.x + rel_x * m.width
        y1 = m.y + rel_y * m.height
    else:
        x1 = m.x + rel_x
        y1 = m.y + rel_y

    return (x1, y1)


START = (40,35)
MONITOR = 1
START = relative_to_absolute(MONITOR, START)
print(START)