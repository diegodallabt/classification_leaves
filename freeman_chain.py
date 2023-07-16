import numpy as np

import cv2


def extract_features(images):
    max_len = 0
    chain_codes = []

    for contour in images:
        points = list(map(tuple, contour))
        chain_code = calculate_freeman_chain_code(points)
        chain_codes.append(chain_code)
        max_len = max(max_len, len(chain_code))
        for image in images:

            histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    # Pad all chain codes to have the same length
    padded_chain_codes = [cc + [0] * (max_len - len(cc)) for cc in chain_codes]

    return padded_chain_codes, histogram


def draw_freeman_chain_code(image, chain_code):
    # Directions for Freeman chain code
    directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0),
                  (-1, 1)]

    # Start at the first point
    x, y = 0, 0

    # Draw each point in the chain code
    for direction in chain_code:
        dx, dy = directions[direction]
        x += dx
        y += dy
        image[y, x] = 255

    return image


def calculate_freeman_chain_code(contour):
    # Directions for Freeman chain code
    directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0),
                  (-1, 1)]

    # Subtract each point by the first point
    contour = [(p[0] - contour[0][0], p[1] - contour[0][1]) for p in contour]

    # Divide each point by the maximum absolute value in the contour
    max_value = max(max(abs(x), abs(y)) for x, y in contour)
    contour = [(x / max_value, y / max_value) for x, y in contour]

    # Round each point to the nearest direction and convert to Freeman chain code
    freeman_code = []
    for x, y in contour:
        min_distance = float("inf")
        direction = None
        for d, (dx, dy) in enumerate(directions):
            distance = (x - dx)**2 + (y - dy)**2
            if distance < min_distance:
                min_distance = distance
                direction = d
        if not freeman_code or freeman_code[-1] != direction:
            freeman_code.append(direction)

    print(freeman_code)
    input()

    return freeman_code
