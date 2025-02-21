from PIL import Image

from postgrad import Level
from postgrad.drawing import Tilesheet
from postgrad.geometry import Direction


def draw_graph():
    tilesheet = Tilesheet.load()
    level = Level.create(tilesheet)
    graph = level.faculty_graph

    image = level.image
    half = level.tilesize // 2
    for tile in graph.tiles:
        row, column = tile
        r = row * level.tilesize + half
        c = column * level.tilesize + half
        if tile == level.warp_left:
            image[r-1:r+2, :c+2, :] = [0, 0, 255, 255]
            continue

        if tile == level.warp_right:
            image[r-1:r+2, c-1:, :] = [0, 0, 255, 255]
            continue

        for d in graph.edges[tile]:
            match d:
                case Direction.UP:
                    r0 = r - level.tilesize - 1
                    c0 = c - 1
                    r1 = r + 1
                    c1 = c + 1
                case Direction.DOWN:
                    r0 = r - 1
                    c0 = c - 1
                    r1 = r + level.tilesize + 1
                    c1 = c + 1
                case Direction.LEFT:
                    r0 = r - 1
                    c0 = c - level.tilesize - 1
                    r1 = r + 1
                    c1 = c + 1
                case Direction.RIGHT:
                    r0 = r - 1
                    c0 = c - 1
                    r1 = r + 1
                    c1 = c + level.tilesize + 1
            image[r0:r1, c0:c1, :] = [0, 255, 0, 255]

    for tile in graph.tiles:
        row, column = tile
        r = row * level.tilesize + half
        c = column * level.tilesize + half
        r0 = r - 3
        r1 = r0 + 6
        c0 = c - 3
        c1 = c0 + 6
        image[r0:r1, c0:c1, :] = [255, 0, 0, 255]

    Image.fromarray(image).save("level_graph.png")


if __name__ == "__main__":
    draw_graph()
