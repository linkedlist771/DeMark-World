import cv2
import numpy as np

from src.demark_world.configs import WATER_MARK_TEMPLATE_IMAGE_PATH

tmpl = cv2.imread(WATER_MARK_TEMPLATE_IMAGE_PATH)
tmpl_gray = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)
h_tmpl, w_tmpl = tmpl_gray.shape


def detect_watermark(
    img: np.array,
    region_fraction: float = 0.25,
    threshold: float = 0.5,
    debug=False,  # 添加调试参数
):
    """检测图像中的水印"""
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h_img, w_img = img_gray.shape

    # 临时：先搜索全图，确认能检测到
    search_region = img_gray
    res = cv2.matchTemplate(search_region, tmpl_gray, cv2.TM_CCOEFF_NORMED)

    if debug:
        print(f"匹配结果范围: {res.min():.3f} ~ {res.max():.3f}")
        max_val = res.max()
        max_loc = np.unravel_index(res.argmax(), res.shape)
        print(f"最佳匹配位置: {max_loc}, 置信度: {max_val:.3f}")

    locs = np.where(res >= threshold)

    mask_full = np.zeros((h_img, w_img), dtype=np.uint8)
    detections = []

    for x, y in zip(*locs[::-1]):
        detections.append((x, y, w_tmpl, h_tmpl))
        mask_full[y : y + h_tmpl, x : x + w_tmpl] = 255

    kernel = np.ones((3, 3), np.uint8)
    mask_full = cv2.dilate(mask_full, kernel, iterations=1)

    return mask_full, detections


def get_bounding_box(detections, w_tmpl, h_tmpl):
    """
    计算所有检测的总边界框
    """
    if not detections:
        return (0, 0, 0, 0)

    # 检测格式: (x, y, w, h)
    if len(detections[0]) == 4:
        min_x = min(x for x, y, w, h in detections)
        min_y = min(y for x, y, w, h in detections)
        max_x = max(x + w for x, y, w, h in detections)
        max_y = max(y + h for x, y, w, h in detections)
    # 兼容旧格式: (x, y)
    else:
        min_x = min(x for x, y in detections)
        min_y = min(y for x, y in detections)
        max_x = max(x for x, y in detections) + w_tmpl
        max_y = max(y for x, y in detections) + h_tmpl

    return (min_x, min_y, max_x, max_y)
