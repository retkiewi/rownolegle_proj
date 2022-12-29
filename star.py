from dataclasses import dataclass
from constants import G
from utils import get_distance
import numpy as np

TIME_STEP = 0.005

@dataclass
class Star():
    position: np.ndarray
    velocity: np.ndarray
    mass: float

    def __init__(self, simulation_size=200, velocity_coef=0.1, max_mass=20000) -> 'Star':
        self.position = np.random.normal(0, simulation_size, 3)
        self.velocity = np.random.normal(0, velocity_coef, 3)
        # self.velocity = np.zeros(3)
        self.mass = abs(np.random.normal(max_mass, max_mass/2, 1)[0])

    def __str__(self) -> str:
        return f"Star at position <{self.position[0]}, {self.position[1]}, {self.position[2]}> with velocity <{self.velocity[0]}, {self.velocity[1]}, {self.velocity[2]}>m/s and mass of {self.mass}kg"

    def calculate_attraction(self, other: 'Star') -> np.ndarray:
        r = get_distance(self.position, other.position)
        if r <= 00.1:
            return 0
        attraction = G*other.mass*(other.position-self.position)/r
        return attraction

    def tick_velocity(self, other_stars: list['Star'], time_step = TIME_STEP):
        for attraction in map(self.calculate_attraction, other_stars):
            self.velocity = self.velocity + attraction*time_step

    def tick_position(self, time_step = TIME_STEP):
        self.position = self.position + self.velocity*time_step

