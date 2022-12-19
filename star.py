from dataclasses import dataclass
from constants import G
from utils import get_distance
import numpy as np

@dataclass
class Star():
    position: np.ndarray
    velocity: np.ndarray
    mass: float

    def __init__(self, simulation_size=100, velocity_coef=5, max_mass=5) -> 'Star':
        self.position = np.random.normal(0, simulation_size, 3)
        self.velocity = np.random.normal(0, velocity_coef, 3)
        self.mass = np.random.normal(0.1, max_mass, 1)[0]

    def __str__(self) -> str:
        return f"Star at position <{self.position[0]}, {self.position[1]}, {self.position[2]}> with velocity <{self.velocity[0]}, {self.velocity[1]}, {self.velocity[2]}>m/s and mass of {self.mass}kg"

    def calculate_attraction(self, other: 'Star') -> np.ndarray:
        return G*(other.mass*(self.position-other.position)/get_distance(self.position, other.position))

    def tick_velocity(self, other_stars: list['Star'], time_step = 0.01):
        for attraction in map(self.calculate_attraction, other_stars):
            self.velocity = self.velocity + attraction*time_step

    def tick_position(self, time_step = 0.01):
        self.position = self.position + self.velocity*time_step

