import numpy as np

class Box:
    def __init__(self, center, radius):
        if len(center) != len(radius):
            raise ValueError("Center and radius must have the same dimension.")

        self.center = np.array(center, dtype=float)
        self.radius = np.array(radius, dtype=float)

        if np.any(self.radius <= 0):
            raise ValueError("Radius must be positive in every component.")

    def __eq__(self, other):
        return np.all(self.center == other.center) and np.all(self.radius == other.radius)

    def __repr__(self):
        lo = self.center - self.radius
        hi = self.center + self.radius
        return f"Box(center={self.center.tolist()}, radius={self.radius.tolist()})\n" \
               f"Bounds: {[(lo[i], hi[i]) for i in range(len(lo))]}"

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
    # Example usage
    box1 = Box(center=[0, 0, 0], radius=[1, 1, 1])
    print(box1)

    point = np.array([0.5, 0.5, 0.5])
    print("Contains point:", box1.contains(point))

    box2 = Box(center=np.array([1, 1, 1]), radius=np.array([1, 1, 1]))
    intersection = box1.intersect(box2)
    print("Intersection:", intersection)

    print("Volume:", box1.volume())
    for vertex in box1.vertices():
        print("Vertex:", vertex)