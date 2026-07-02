"""
论文统一配色方案 — 基于科研绘图配色参考/123.jpg

红-海军蓝互补色系 (Nature/Science 期刊常用风格)
"""

# 方法配色 (4种) — 按视觉区分度从冷到暖
METHOD_COLORS = {
    "ResNet":  "#154760",  # 深海军蓝
    "SCN":     "#6b92a5",  # 钢蓝
    "RUL":     "#c46b6b",  # 玫瑰红
    "MHAN":    "#bf1a24",  # 亮红
}

METHOD_COLORS_LIST = ["#154760", "#6b92a5", "#c46b6b", "#bf1a24"]

# 7类表情配色 (从蓝到红渐变)
EMOTION_COLORS = [
    "#154760",  # 0 angry    — 深海军蓝
    "#2c6e85",  # 1 disgust  — 深青蓝
    "#5a8fa3",  # 2 fear     — 钢蓝
    "#8ab0bf",  # 3 happy    — 浅钢蓝
    "#c99595",  # 4 sad      — 灰玫红
    "#c46b6b",  # 5 surprise — 玫瑰红
    "#bf1a24",  # 6 neutral  — 亮红
]

EMOTION_COLORS_MAP = {
    0: "#154760", 1: "#2c6e85", 2: "#5a8fa3",
    3: "#8ab0bf", 4: "#c99595", 5: "#c46b6b", 6: "#bf1a24",
}

# 域内/跨域对比色
IN_DOMAIN_COLOR = "#154760"
CROSS_DOMAIN_COLOR = "#bf1a24"

# 背景/辅助
BACKGROUND = "#fefefd"
GRID_COLOR = "#d9e5ea"
