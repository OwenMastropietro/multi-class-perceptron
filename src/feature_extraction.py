'''
This module contains the functions used to extract features from the
handwritten digits.
'''

import numpy as np
from numpy import ndarray


# pylint: disable=invalid-name
# pylint: disable=unused-variable


##########################################################################
#                                                                        #
# START: Feature Extraction                                              #
#                                                                        #
##########################################################################


# Feature 1
def density(image: ndarray) -> float:
    '''`Feature 1` returns the average greyscale value of the given `image`.'''

    assert isinstance(image, ndarray)

    np_density = np.mean(image)

    assert isinstance(np_density, float)
    assert 0 <= np_density <= 255

    return np_density


# Feature 2
def horizontal_symmetry(image: ndarray) -> float:
    '''
    `Feature 2` returns the degree of symmetry (a float between 0 and 1)
    of the given `image` and its reflected counterpart where the line of
    reflection is along the y-axis on the right side of the image.
    - Notes:
        - The measure of symmetry is defined as the average greyscale value
        of the image obtained by the bitwise XOR of each pixel with its
        corresponding vertically reflected image.
        - Thus, if I is the image, let I' be the image whose j-th column is the
        (28 - j)-th column of I.
        - Then, the measure of symmetry is the density of I XOR I'.
        - Uses numpy.bitwise_xor() to perforn I XOR I'.
        - Used numpy.fliplr() to get the left-right reflection of image, I.
    '''

    assert isinstance(image, ndarray)

    degree_of_symmetry = density(np.bitwise_xor(image, np.fliplr(image)))

    assert isinstance(degree_of_symmetry, float)
    assert 0.0 <= degree_of_symmetry <= 1.0

    return degree_of_symmetry


# Features 3 & 4
def horizontal_intersections(image: ndarray) -> tuple[int, float]:
    '''
    `Features 3 & 4` return the `maximum` and `average` number of
    horizontal intersections in the given `image`.
    '''

    assert isinstance(image, ndarray)

    return vertical_intersections(np.flip(np.rot90(image)))


# Features 5 & 6
def vertical_intersections(image: ndarray) -> tuple[int, float]:
    '''
    `Features 5 & 6` return the `maximum` and `average` number of
    'vertical intersections', respectively, in the given `image` by
    scanning each column of the `image` from top to bottom,
    counting the number of times we 'enter' a white pixel from a black pixel.
    '''

    assert isinstance(image, ndarray)

    NUM_ROWS, NUM_COLS = image.shape
    BLACK, WHITE = 0, 1

    intersections_per_vertical_search = []
    for col in range(NUM_COLS):
        num_intersections, prev = 0, 0

        for row in range(NUM_ROWS):
            current = image[row][col]

            if prev == BLACK and current == WHITE:
                num_intersections += 1
            prev = current

        intersections_per_vertical_search.append(num_intersections)

    maximum = max(intersections_per_vertical_search)
    average = np.mean(intersections_per_vertical_search)

    assert isinstance(maximum, int)
    assert isinstance(average, float)
    assert 0 <= maximum <= NUM_COLS
    assert 0 <= average <= NUM_COLS

    return maximum, average


# Feature 7
def number_of_loops(image: ndarray) -> int:
    '''
    `Feature 7` returns the number of loops in the given `image` by
    performing a breadth-first search on the image. The black pixels are
    considered unvisited and the white pixels are considered visited.
    The number of loops is the number of black sections that must be flood
    filled to turn the entire image white.
    '''

    assert isinstance(image, ndarray)

    BLACK, WHITE = 0, 1
    NUM_ROWS, NUM_COLS = len(image), len(image[0])

    # if np.all(image == WHITE):
    #     return 0

    def is_valid(x: int, y: int) -> bool:
        return (0 <= x < NUM_ROWS) and (0 <= y < NUM_COLS)

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

    search_count = 0
    for row in range(NUM_ROWS):
        for col in range(NUM_COLS):
            if image[row][col] == BLACK:
                bfs(row, col)
                search_count += 1

    assert isinstance(search_count, int)

    return search_count - 1


# Feature 8
def horizontally_split_symmetry(image: ndarray) -> float:
    '''
    `Features 8` returns the degree of symmetry (a float between 0 and 1)
    between the left and right halves of the given `image`.
    '''

    left, right, *remaining = np.hsplit(image, 2)

    assert len(remaining) == 0

    return density(image=np.bitwise_xor(left, right))


# Feature 9
def vertically_split_symmetry(image: ndarray) -> float:
    '''
    `Feature 9` returns the degree of symmetry (a float between 0 and 1)
    between the top and bottom halves of the given `image`.
    '''

    top, bottom, *remaining = np.vsplit(image, 2)

    assert len(remaining) == 0

    return density(image=np.bitwise_xor(top, bottom))


def get_image_features(image: ndarray, as_dict: bool = False) -> list:
    '''
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
    '''

    assert isinstance(image, ndarray)

    ft_1 = density(image)
    ft_2 = horizontal_symmetry(image)
    ft_3 = horizontal_intersections(image)[0]  # maximum intersections.
    ft_4 = horizontal_intersections(image)[1]  # average intersections.
    ft_5 = vertical_intersections(image)[0]  # maximum intersections.
    ft_6 = vertical_intersections(image)[1]  # average intersections.
    ft_7 = number_of_loops(image)
    ft_8 = horizontally_split_symmetry(image)
    ft_9 = vertically_split_symmetry(image)
    # ft_10 = vertical_symmetry(bl_wh_image)

    if as_dict:
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
        assert isinstance(features, dict)
        return features

    features = [ft_1, ft_2, ft_3, ft_4, ft_5, ft_6, ft_7, ft_8, ft_9]

    assert isinstance(features, list)
    assert len(features) == 9
    assert all(isinstance(feature, (int, float)) for feature in features)

    return features


##########################################################################
#                                                                        #
# END: Feature Extraction                                                #
#                                                                        #
##########################################################################
