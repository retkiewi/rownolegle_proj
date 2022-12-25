from star import Star


def run_simulation(n_stars, iters):
    stars = [Star() for _ in range(n_stars)]
    
    result = []

    for _ in range(iters):
        for star in stars:
            star.tick_velocity(stars)
        for star in stars:
            star.tick_position()
        result.append([(star.position, int(star.mass**(1/3))) for star in stars])

    return result