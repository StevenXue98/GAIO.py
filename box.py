import numpy as np

"""
Code implementing Box (hyperrectangle/k-cell) class.
Supports most mathematical set operations in default Python set syntax.
# Example usage
>>> box1 = Box(center=[0, 0, 0], radius=[1, 1, 1])
>>> print(box1)
Box(center=[0.0, 0.0, 0.0], radius=[1.0, 1.0, 1.0])
Bounds: [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
"""

class Box:
    def __init__(self, center=[0, 0, 0], radius=[1, 1, 1]):
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
        if not isinstance(other, Box):
            raise TypeError("Can only compare with another Box.")
        if len(other.center) != len(self.center):
            raise ValueError("Boxes must have the same dimension.")
        return np.all(self.center == other.center) and np.all(self.radius == other.radius)

    def __ne__(self, other):
        """Overloading operator != to check if two Boxes are not equal."""
        return not self.__eq__(other)
    
    def __le__(self, other):
        """Overloading operator <= as subset
        A <= B (A⊆B): A is a subset of B"""
        if not isinstance(other, Box):
            raise TypeError("Can only compare with another Box.")
        if len(other.center) != len(self.center):
            raise ValueError("Boxes must have the same dimension.")
        return np.all(self.center - self.radius >= other.center - other.radius) and \
               np.all(self.center + self.radius <= other.center + other.radius)
    
    def __lt__(self, other):
        """Overloading operator < as proper subset
        A < B (A⊂B): A is a proper subset of B"""
        if not isinstance(other, Box):
            raise TypeError("Can only compare with another Box.")
        if len(other.center) != len(self.center):
            raise ValueError("Boxes must have the same dimension.")
        return np.all(self.center - self.radius > other.center - other.radius) and \
               np.all(self.center + self.radius < other.center + other.radius)

    def __ge__(self, other):
        """Overloading operator >= as superset
        A >= B (A⊇B): A is a superset of B"""
        if not isinstance(other, Box):
            raise TypeError("Can only compare with another Box.")
        if len(other.center) != len(self.center):
            raise ValueError("Boxes must have the same dimension.")
        return np.all(self.center - self.radius <= other.center - other.radius) and \
               np.all(self.center + self.radius >= other.center + other.radius)

    def __gt__(self, other):
        """Overloading operator > as proper superset
        A > B (A⊃B): A is a proper superset of B"""
        if not isinstance(other, Box):
            raise TypeError("Can only compare with another Box.")
        if len(other.center) != len(self.center):
            raise ValueError("Boxes must have the same dimension.")
        return np.all(self.center - self.radius < other.center - other.radius) and \
               np.all(self.center + self.radius > other.center + other.radius)

    def __add__(self, other):
        """Overloading operator + to shift the Box's center.

        Allows adding a list or ndarray to the Box's center.
        >>> print(Box(center=[0, 0, 0], radius=[1, 1, 1]) + [1, 1, 1])  # Adding a list
        Box(center=[1.0, 1.0, 1.0], radius=[1.0, 1.0, 1.0])
        Bounds: [(0.0, 2.0), (0.0, 2.0), (0.0, 2.0)]

        >>> print(Box(center=[0, 0, 0], radius=[1, 1, 1]) + np.array([1, 1, 1]))  # Adding an ndarray
        Box(center=[1.0, 1.0, 1.0], radius=[1.0, 1.0, 1.0])
        Bounds: [(0.0, 2.0), (0.0, 2.0), (0.0, 2.0)]

        Other types will raise TypeError.
        >>> Box(center=[0, 0, 0], radius=[1, 1, 1]) + "invalid"  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        TypeError: Can only add list, ndarray, or another Box to a Box.
        """
        if not isinstance(other, (list, np.ndarray)):
            raise TypeError("Can only add list or ndarray to a Box.")
        if len(other) != len(self.center):
            raise ValueError("Centers must have the same dimension.")
        new_center = self.center + np.array(other, dtype=float)
        return Box(new_center, self.radius)
    
    def __sub__(self, other):
        """
        Overloading operator - as set minus.
        A - B (A\B): difference between Boxes A and B.
        """
        if not isinstance(other, Box):
            raise TypeError("Can only subtract another Box from a Box.")
        if len(other.center) != len(self.center):
            raise ValueError("Centers must have the same dimension.")
        lo = self.center - self.radius
        hi = self.center + self.radius
        other_lo = other.center - other.radius
        other_hi = other.center + other.radius
        if np.any(lo >= other_lo & hi <= other_hi):
            raise ValueError("Set difference is empty set.")
        if np.any(lo < other_lo & hi > other_hi):
            raise ValueError("Cannot split Box A by Box B.")
        mask = lo >= other_lo
        new_lo = np.array([0] * len(lo), dtype=float)
        new_hi = np.array([0] * len(hi), dtype=float)
        new_lo[mask] = other_hi[mask]
        new_hi[mask] = hi[mask]
        new_lo[~mask] = lo[~mask]
        new_hi[~mask] = other_lo[~mask]
        new_center = (new_lo + new_hi) / 2
        new_radius = (new_hi - new_lo) / 2
        return Box(new_center, new_radius)

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

    def contains(self, point):
        if not isinstance(point, (list, np.ndarray)):
            raise TypeError("Point must be a list or ndarray.")
        if len(point) != len(self.center):
            raise ValueError("Point must have the same dimension as the box.")
        point = np.array(point, dtype=float)
        return np.all(self.center - self.radius <= point) and np.all(point <= self.center + self.radius)

    def __contains__(self, point):
        """Overloading operator 'in' to check if a point is inside the Box."""
        return self.contains(point)

    def intersect(self, other):
        """Calculate the intersection of two Boxes."""
        if not isinstance(other, Box):
            raise TypeError("Can only intersect with another Box.")
        if len(other.center) != len(self.center):
            raise ValueError("Boxes must have the same dimension.")
        lo = np.maximum(self.center - self.radius, other.center - other.radius)
        hi = np.minimum(self.center + self.radius, other.center + other.radius)
        if np.any(lo >= hi):
            raise ValueError("Intersection is empty set.")
        return Box((hi + lo) / 2, (hi - lo) / 2)

    def __and__(self, other):
        """Overloading operator & to calculate intersection of two Boxes."""
        return self.intersect(other)

    def union(self, other):
        """Calculate the union of two Boxes."""
        if not isinstance(other, Box):
            raise TypeError("Can only union with another Box.")
        if len(other.center) != len(self.center):
            raise ValueError("Boxes must have the same dimension.")
        if self.isdisjoint(other):
            raise ValueError("Union is not defined for disjoint Boxes.")
        lo = np.minimum(self.center - self.radius, other.center - other.radius)
        hi = np.maximum(self.center + self.radius, other.center + other.radius)
        return Box((hi + lo) / 2, (hi - lo) / 2)

    def __or__(self, other):
        """Overloading operator | to calculate union of two Boxes."""
        return self.union(other)

    def isdisjoint(self, other):
        """Check if two Boxes are disjoint."""
        if not isinstance(other, Box):
            raise TypeError("Can only check disjoint with another Box.")
        if len(other.center) != len(self.center):
            raise ValueError("Boxes must have the same dimension.")
        lo = self.center - self.radius
        hi = self.center + self.radius
        other_lo = other.center - other.radius
        other_hi = other.center + other.radius
        return np.any(hi <= other_lo) or np.any(lo >= other_hi)

    def volume(self):
        return np.prod(2 * self.radius)
    
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