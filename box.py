import numpy as np

"""
Code implementing Box (hyperrectangle/k-cell) class.
# Example usage
>>> box1 = Box(center=[0, 0, 0], radius=[1, 1, 1])
>>> print(box1)
Box(center=[0.0, 0.0, 0.0], radius=[1.0, 1.0, 1.0])
Bounds: [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
"""

class Box:
    def __init__(self, center=[0, 0], radius=[1, 1]):
        if len(center) != len(radius):
            raise ValueError("Center and radius must have the same dimension.")

        self.center = np.array(center, dtype=float)
        self.radius = np.array(radius, dtype=float)

        if np.any(self.radius <= 0):
            raise ValueError("Radius must be positive in every component.")

    def __repr__(self):
        lo = self.center - self.radius
        hi = self.center + self.radius
        return f"Box(center={self.center.tolist()}, radius={self.radius.tolist()})\n" \
               f"Bounds: {[(lo[i], hi[i]) for i in range(len(lo))]}"

    def __eq__(self, other):
        return np.all(self.center == other.center) and np.all(self.radius == other.radius)

    def __add__(self, other):
        """Overloading operator +

        Allows adding a list, ndarray, or another Box to the Box's center.
        >>> print(Box(center=[0, 0, 0], radius=[1, 1, 1]) + [1, 1, 1])  # Adding a list
        Box(center=[1.0, 1.0, 1.0], radius=[1.0, 1.0, 1.0])
        Bounds: [(0.0, 2.0), (0.0, 2.0), (0.0, 2.0)]

        >>> print(Box(center=[0, 0, 0], radius=[1, 1, 1]) + np.array([1, 1, 1]))  # Adding an ndarray
        Box(center=[1.0, 1.0, 1.0], radius=[1.0, 1.0, 1.0])
        Bounds: [(0.0, 2.0), (0.0, 2.0), (0.0, 2.0)]

        >>> print(Box(center=[0, 0, 0], radius=[1, 1, 1]) + Box(center=[1, 1, 1], radius=[1, 1, 1]))  # Adding another Box
        Box(center=[1.0, 1.0, 1.0], radius=[1.0, 1.0, 1.0])
        Bounds: [(0.0, 2.0), (0.0, 2.0), (0.0, 2.0)]

        Other types will raise TypeError.
        >>> Box(center=[0, 0, 0], radius=[1, 1, 1]) + "invalid"  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        TypeError: Can only add list, ndarray, or another Box to a Box.
        """
        if not isinstance(other, (list, np.ndarray, Box)):
            raise TypeError("Can only add list, ndarray, or another Box to a Box.")
        if isinstance(other, Box):
            other = other.center
        if len(other) != len(self.center):
            raise ValueError("Centers must have the same dimension.")
        new_center = self.center + np.array(other, dtype=float)
        return Box(new_center, self.radius)
    
    def __sub__(self, other):
        if not isinstance(other, (list, np.ndarray, Box)):
            raise TypeError("Can only subtract list, ndarray, or another Box to a Box.")
        if isinstance(other, Box):
            other = other.center
        if len(other) != len(self.center):
            raise ValueError("Centers must have the same dimension.")
        new_center = self.center - np.array(other, dtype=float)
        return Box(new_center, self.radius)

    def __mul__(self, other):
        if not isinstance(other, (int, float, list, np.ndarray, Box)):
            raise TypeError("Can only multiply Box by a scalar, a list, or another Box.")
        if isinstance(other, Box):
            if len(other.center) != len(self.center):
                raise ValueError("Boxes must have the same dimension.")
            other = other.radius
        if isinstance(other, (int, float)):
            other = np.array([other] * len(self.center), dtype=float)
        new_radius = self.radius * np.array(other, dtype=float)
        return Box(self.center, new_radius)

    def __truediv__(self, other):
        if not isinstance(other, (int, float, list, np.ndarray, Box)):
            raise TypeError("Can only divide Box by a scalar, a list, or another Box.")
        if isinstance(other, Box):
            if len(other.center) != len(self.center):
                raise ValueError("Boxes must have the same dimension.")
            other = other.radius
        if isinstance(other, (int, float)):
            other = np.array([other] * len(self.center), dtype=float)
        new_radius = self.radius / np.array(other, dtype=float)
        return Box(self.center, new_radius)

    def volume(self):
        return np.prod(2 * self.radius)

    def contains(self, point):
        if len(point) != len(self.center):
            raise ValueError("Point must have the same dimension as the box.")
        return np.all(self.center - self.radius <= np.array(point)) and np.all(np.array(point) < self.center + self.radius)

    def intersect(self, other):
        lo = np.maximum(self.center - self.radius, other.center - other.radius)
        hi = np.minimum(self.center + self.radius, other.center + other.radius)
        if np.any(lo >= hi):
            return None
        return Box((hi + lo) / 2, (hi - lo) / 2)

    def vertices(self):
        n = len(self.center)
        for idx in np.ndindex(*(2,) * n):
            offset = np.array(idx) * 2 - 1  # from (0,1) to (-1,1)
            yield self.center + offset * self.radius

    def subdivide(self, dim):
        new_radius = self.radius.copy()
        new_radius[dim] /= 2
        center1 = self.center.copy()
        center2 = self.center.copy()
        center1[dim] -= new_radius[dim]
        center2[dim] += new_radius[dim]
        return Box(center1, new_radius), Box(center2, new_radius)

    def rescale(self, point):
        """Scale a point from [-1, 1]^N to the box."""
        return self.center + point * self.radius
    

if __name__ == '__main__':
    import doctest
    doctest.testmod()