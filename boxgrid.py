import numpy as np
from box import Box

class BoxGrid:
    def __init__(self, domain, left, scale, dims):
        self.domain = domain  # Box object
        self.left = np.array(left)
        self.scale = np.array(scale)
        self.dims = np.array(dims)

        if len(domain.center) != len(left):
            raise ValueError("Center and left must have the same dimension.")
        if len(domain.center) != len(scale):
            raise ValueError("Center and scale must have the same dimension.")
        if len(domain.center) != len(dims):
            raise ValueError("Center and dims must have the same dimension.")

    @classmethod
    def from_dims(cls, domain, dims):
        dims = np.array(dims)
        left = domain.center - domain.radius
        scale = dims / (2 * domain.radius)
        return cls(domain, left, scale, dims)

    @classmethod
    def from_domain(cls, domain):
        dims = np.ones_like(domain.center, dtype=int)
        return cls.from_dims(domain, dims)

    def __eq__(self, other):
        return (np.array_equal(self.domain.center, other.domain.center) and
                np.array_equal(self.domain.radius, other.domain.radius) and
                np.array_equal(self.dims, other.dims))

    def __repr__(self):
        return f"{' x '.join(map(str, self.dims))} - element BoxGrid"

    def ndims(self):
        return len(self.dims)

    def size(self):
        return self.dims

    def length(self):
        return np.prod(self.dims)

    def center(self):
        return self.domain.center

    def radius(self):
        return self.domain.radius

    def keys(self):
        # Generating Cartesian indices
        for idx in np.ndindex(*self.dims):
            yield idx

    def subdivide(self, dim):
        new_dims = self.dims.copy()
        new_dims[dim] *= 2
        new_scale = self.scale.copy()
        new_scale[dim] *= 2
        return BoxGrid(self.domain, self.left, new_scale, new_dims)

    def marginal(self, dim):
        center = np.delete(self.domain.center, dim)
        radius = np.delete(self.domain.radius, dim)
        dims = np.delete(self.dims, dim)
        return BoxGrid(Box(center, radius), self.left, self.scale, dims)

    def key_to_box(self, key):
        if not self.check_bounds(key):
            raise IndexError(f"Index {key} is out of bounds for BoxGrid.")
        radius = self.domain.radius / self.dims
        center = self.left + radius + 2 * radius * (np.array(key) - 1)
        return Box(center, radius)

    def check_bounds(self, key):
        return np.all(1 <= np.array(key)) and np.all(np.array(key) <= self.dims)

    def point_to_key(self, point):
        if not self.point_in_domain(point):
            return None
        x = (np.array(point) - self.left) * self.scale
        key = np.floor(x).astype(int) + 1
        if not self.check_bounds(key):
            key = np.clip(key, 1, self.dims)
        return key

    def bounded_point_to_key(self, point):
        # Adjust point to stay within bounds
        center, radius = self.domain.center, self.domain.radius
        small_bound = 1 / (2 * self.scale)
        left = center - radius + small_bound
        right = center + radius - small_bound
        point = np.where(np.isnan(point), np.inf, point)
        point = np.minimum(np.maximum(point, left), right)
        return self.point_to_key(point)

    def point_in_domain(self, point):
        lower_bound = self.domain.center - self.domain.radius
        upper_bound = self.domain.center + self.domain.radius
        return np.all(lower_bound <= np.array(point)) and np.all(np.array(point) < upper_bound)

    def point_to_box(self, point):
        key = self.point_to_key(point)
        if key is None:
            return None
        return self.key_to_box(key)
    

if __name__ == '__main__':
    # Usage example
    # Create a Box
    domain = Box(center=[0, 0], radius=[1, 1])

    # Create a BoxGrid from a domain with dimensions (4, 4)
    grid = BoxGrid.from_dims(domain, dims=(4, 4))

    # Get the size of the grid
    print(grid.size())  # Output: (4, 4)

    # Subdivide along the first dimension
    new_grid = grid.subdivide(dim=0)
    print(new_grid.size())  # Output: (8, 4)

    # Map a point to its corresponding box
    box = grid.point_to_box([0.5, 0.5])
    print(box.center, box.radius)