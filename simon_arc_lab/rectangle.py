class Rectangle:
    @staticmethod
    def empty():
        return Rectangle(0, 0, 0, 0)

    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def __eq__(self, other):
        if not isinstance(other, Rectangle):
            return NotImplemented
        return (self.x == other.x and
                self.y == other.y and
                self.width == other.width and
                self.height == other.height)

    def __repr__(self):
        return f"Rectangle(x={self.x}, y={self.y}, width={self.width}, height={self.height})"
