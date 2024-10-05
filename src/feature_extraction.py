"""
This module contains the functions used to extract features from the
handwritten digits.
"""

import numpy as np
from numpy import ndarray


BLACK = 0
WHITE = 1


def density(image: ndarray) -> float:
    """
    `Feature 1`

    Returns the average greyscale value of the given `image`.
    """

    np_density = np.mean(image)

    assert 0 <= np_density <= 255

    return np_density


def horizontal_symmetry(image: ndarray) -> float:
    """
    `Feature 2`

    Returns the degree of symmetry between the image and its reflection;
    the line of reflection is along the y-axis on the right side of the image.
    - Notes:
        - The measure of symmetry is defined as the average greyscale value
        of the image obtained by the bitwise XOR of each pixel with its
        corresponding vertically reflected image.
        - Thus, if I is the image, let I' be the image whose j-th column is the
        (28 - j)-th column of I.
        - Then, the measure of symmetry is the density of I XOR I'.
        - Uses numpy.bitwise_xor() to perforn I XOR I'.
        - Used numpy.fliplr() to get the left-right reflection of image, I.
    """

    degree_of_symmetry = density(np.bitwise_xor(image, np.fliplr(image)))

    assert 0.0 <= degree_of_symmetry <= 1.0

    return degree_of_symmetry


def horizontal_intersections(image: ndarray) -> tuple[int, float]:
    """
    `Features 3 & 4`

    Returns the (maximum, average) number of (left-right) horizontal intersections
    in the image.
    """

    return vertical_intersections(np.flip(np.rot90(image)))


def vertical_intersections(image: ndarray) -> tuple[int, float]:
    """
    `Features 5 & 6`

    Returns the (maximum, average) number of (top down) vertical intersections
    in the image.
    """

    num_rows, num_cols = image.shape

    counts = []
    for col in range(num_cols):
        count, prev = 0, 0
        for row in range(num_rows):
            current = image[row][col]
            if prev == BLACK and current == WHITE:
                count += 1
            prev = current
        counts.append(count)

    maximum = max(counts)
    average = np.mean(counts)

    assert 0 <= maximum <= num_cols
    assert 0 <= average <= num_cols

    return maximum, average


def number_of_loops(image: ndarray) -> int:
    """
    `Feature 7`

    Returns the number of loops in the given image via BFS flood fill.

    Black (background) pixels are considered unvisited.
    White pixels are considered visited.
    The number of loops is thus the number of black sections that must be
    filled to turn the entire image white.
    """

    image = image.copy()
    num_rows, num_cols = len(image), len(image[0])
    # if np.all(image == WHITE):
    #     return 0

    def is_valid(x: int, y: int) -> bool:
        return (0 <= x < num_rows) and (0 <= y < num_cols)

    def bfs(x: int, y: int):
        queue = [[x, y]]

        directions = [[1, -1], [1, 0], [1, 1],
                      [0, -1], [0, 1],
                      [-1, -1], [-1, 0], [-1, 1]]

        while len(queue) > 0:
            x, y = queue.pop(0)
            image[x][y] = WHITE

            for direction in directions:
                nx, ny = x+direction[0], y+direction[1]

                if is_valid(nx, ny) and image[nx][ny] == BLACK:
                    queue.append([nx, ny])
                    image[nx][ny] = WHITE

    count = 0
    for row in range(num_rows):
        for col in range(num_cols):
            if image[row][col] == BLACK:
                bfs(row, col)
                count += 1

    return count - 1  # subtract 1 to account for the background.


def horizontally_split_symmetry(image: ndarray) -> float:
    """
    `Features 8`

    Returns the degree of symmetry between the left and right halves of the image.
    """

    left, right, *remaining = np.hsplit(image, 2)

    assert len(remaining) == 0

    return density(image=np.bitwise_xor(left, right))


def vertically_split_symmetry(image: ndarray) -> float:
    """
    `Feature 9`

    Returns the degree of symmetry between the top and bottom halves of the image.
    """

    top, bottom, *remaining = np.vsplit(image, 2)

    assert len(remaining) == 0

    return density(image=np.bitwise_xor(top, bottom))


def get_image_features(image: ndarray, as_dict: bool = False) -> list:
    """
    Returns the feature values for the given `image`.

    Features include:
        - Density of the image.
        - Degree of (horizontal) symmetry of the image.
        - Maximum number of horizontal intersections.
        - Average number of horizontal intersections.
        - Maximum number of vertical intersections.
        - Average number of vertical intersections.
        - Number of loops in the image.
        - Degree of (horizontal) symmetry between the left and right halves
        of the image.
        - Degree of (vertical) symmetry between the top and bottom halves
        of the image.
    """

    ft_1 = density(image)
    ft_2 = horizontal_symmetry(image)
    ft_3, ft_4 = horizontal_intersections(image)  # maximum, average
    ft_5, ft_6 = vertical_intersections(image)  # maximum, average
    ft_7 = number_of_loops(image)
    ft_8 = horizontally_split_symmetry(image)
    ft_9 = vertically_split_symmetry(image)
    # ft_10 = vertical_symmetry(image)

    if as_dict:  # for debugging
        features = {
            '1': ('Density', ft_1),
            '2': ('Horizontal Symmetry', ft_2),
            '3': ('Max Vertical Intersections', ft_3),
            '4': ('Avg Vertical Intersections', ft_4),
            '5': ('Max Horizontal Intersections', ft_5),
            '6': ('Avg Horizontal Intersections', ft_6),
            '7': ('Number of Loops', ft_7),
            '8': ('Vertically Split Symmetry', ft_8),
            '9': ('Horizontally Split Symmetry', ft_9),
        }
        return features

    features = [ft_1, ft_2, ft_3, ft_4, ft_5, ft_6, ft_7, ft_8, ft_9]

    return features
