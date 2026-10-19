"""
Find exact text positions by detecting bright/colored pixels against background.
"""
import cv2
import numpy as np

SRC = '/workspace/.uploads/a4a71375-601a-4389-b7c7-8b30e2d25f48_005CgOGzgy1i3a57g2sd8j31hc0u0qj6.jpg'

img = cv2.imread(SRC)
H, W = img.shape[:2]
print(f'Size: {W}x{H}')

# Look at horizontal slices and find rows with text-like patterns (high contrast + bright pixels)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# For each row, find the row's maximum brightness and contrast
rows_white = []
for y in range(H):
    row = gray[y]
    max_v = row.max()
    if max_v > 230:
        rows_white.append((y, max_v))

# Group consecutive y-rows that have white pixels
groups = []
cur = None
for y, v in rows_white:
    if cur is None or y - cur[-1][0] > 3:
        if cur:
            groups.append(cur)
        cur = [(y, v)]
    else:
        cur.append((y, v))
if cur:
    groups.append(cur)

print('White-text-row groups:')
for g in groups:
    ys = [x[0] for x in g]
    print(f'  y={min(ys)}-{max(ys)} ({len(g)} rows, max_val={max(x[1] for x in g)})')

# Also for each row count white-ish pixels
print()
print('Rows with >20 white pixels:')
for y in range(0, H, 1):
    row = gray[y]
    cnt = int(np.sum(row > 230))
    if cnt > 20:
        # Show only every 20th row
        if y % 5 == 0:
            print(f'  y={y}: white_cnt={cnt}')
