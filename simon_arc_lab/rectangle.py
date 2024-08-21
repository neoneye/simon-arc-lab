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

    def is_empty(self) -> bool:
        return self.width <= 0 or self.height <= 0
    
    def is_not_empty(self) -> bool:
        return not self.is_empty()
    
    def intersection(self, other: 'Rectangle') -> 'Rectangle':
        """
        Find the intersection of two rectangles.
        """
        x0 = max(self.x, other.x)
        y0 = max(self.y, other.y)
        x1 = min(self.x + self.width, other.x + other.width)
        y1 = min(self.y + self.height, other.y + other.height)
        if x0 >= x1 or y0 >= y1:
            return Rectangle.empty()
        return Rectangle(x0, y0, x1 - x0, y1 - y0)
