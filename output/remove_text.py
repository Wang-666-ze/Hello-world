"""
Remove all text using larger, more accurate mask regions.
Strategy: Cover text generously, apply multi-pass inpainting.
"""
import cv2
import numpy as np

SRC = '/workspace/.uploads/a4a71375-601a-4389-b7c7-8b30e2d25f48_005CgOGzgy1i3a57g2sd8j31hc0u0qj6.jpg'
OUT = '/workspace/output/cleaned.png'
MASK_DEBUG = '/workspace/output/mask_debug.png'

img = cv2.imread(SRC)
H, W = img.shape[:2]
print(f'Image size: {W}x{H}')

mask = np.zeros((H, W), dtype=np.uint8)

# More accurate, generous rectangles covering all text
regions = [
    # Top-right logo 洛天依 + LUO TIANYI
    (1640, 10, 1910, 135),

    # Big calligraphic 权御天下 on left - whole column
    (10, 60, 545, 830),

    # Right side 7月15日 large white text
    (1340, 590, 1900, 770),

    # Right side 正式上线 (small, with diamond markers)
    (1380, 750, 1900, 880),

    # Bottom band: 2025官方重置版 (left) + 7.12-7.13 BML舞台首唱 (right)
    # These are on a continuous dark/colored band — cover the whole band for clean removal
    (10, 880, 1910, 985),

    # Bottom credits row (8 items) - cover entire bottom area
    (0, 995, 1920, 1080),
]

for (x1, y1, x2, y2) in regions:
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)

# Aggressive dilation to fully cover strokes & antialiasing
kernel = np.ones((7, 7), np.uint8)
mask_d = cv2.dilate(mask, kernel, iterations=3)
# Slight Gaussian blur on mask edge to help inpainting blend
mask_d = cv2.GaussianBlur(mask_d, (15, 15), 0)
_, mask_d = cv2.threshold(mask_d, 127, 255, cv2.THRESH_BINARY)

cv2.imwrite(MASK_DEBUG, mask_d)

# First inpainting pass (larger radius for big regions like calligraphy)
pass1 = cv2.inpaint(img, mask_d, 15, cv2.INPAINT_TELEA)

# Second pass with NS algorithm
pass2 = cv2.inpaint(pass1, mask_d, 8, cv2.INPAINT_NS)

# Third pass with Telea for any remaining artifacts
result = cv2.inpaint(pass2, mask_d, 5, cv2.INPAINT_TELEA)

cv2.imwrite(OUT, result)
print(f'Saved cleaned image to {OUT}')

# Print region coverage statistics
print(f'Mask total area: {int((mask_d > 0).sum())} pixels ({100 * (mask_d > 0).sum() / (H*W):.1f}%)')
