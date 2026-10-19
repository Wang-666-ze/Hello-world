"""
Find text regions by analyzing bright pixel x-ranges in each y-band.
"""
import cv2
import numpy as np

SRC = '/workspace/.uploads/a4a71375-601a-4389-b7c7-8b30e2d25f48_005CgOGzgy1i3a57g2sd8j31hc0u0qj6.jpg'

img = cv2.imread(SRC)
H, W = img.shape[:2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# For each y-band, find x-extent of bright pixels (>180 — catches antialiased text)
def find_x_extent(y1, y2, thresh=180, min_cnt=3):
    xs = []
    for y in range(y1, y2):
        row = gray[y]
        idx = np.where(row > thresh)[0]
        if len(idx) >= min_cnt:
            xs.extend(idx.tolist())
    if not xs:
        return None, None
    return min(xs), max(xs)

# Define bands to check
bands = [
    ('7月15日', 600, 770),
    ('正式上线', 770, 870),
    ('2025官方重置版+7.12-7.13 BML', 855, 970),
    ('credits row', 1000, 1070),
    ('logo', 10, 130),
]

for name, y1, y2 in bands:
    x_min, x_max = find_x_extent(y1, y2)
    if x_min is not None:
        print(f'{name}: y={y1}-{y2} -> x={x_min}-{x_max}')
    else:
        print(f'{name}: not found')

# Also find x-extent of bright pixels in the left half (calligraphy mask)
print()
print('Left calligraphy extent per row band:')
for y in range(40, 880, 50):
    x_min, x_max = find_x_extent(y, y+50)
    if x_min is not None:
        print(f'  y={y}-{y+50} -> x={x_min}-{x_max}')
