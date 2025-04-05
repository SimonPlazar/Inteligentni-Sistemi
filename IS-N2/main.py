import pygame
import random
import math
import numpy as np
from pygame import Vector2

# Constants
WIDTH, HEIGHT = 800, 600
BACKGROUND_COLOR = (20, 20, 20)
MAX_SPEED = 2.0
MAX_FORCE = 0.1
PERCEPTION_RADIUS = 50
ALIGNMENT_WEIGHT = 1.0
COHESION_WEIGHT = 1.0
SEPARATION_WEIGHT = 1.2
BOUNDARY_WEIGHT = 1.0
SEEK_WEIGHT = 1.0


class Boid:
    def __init__(self, x, y, flock_id=0):
        self.position = Vector2(x, y)
        self.velocity = Vector2(random.uniform(-1, 1), random.uniform(-1, 1))
        self.velocity.scale_to_length(random.uniform(2, MAX_SPEED))
        self.acceleration = Vector2(0, 0)
        self.max_speed = MAX_SPEED
        self.max_force = MAX_FORCE
        self.flock_id = flock_id
        self.size = 6
        self.vision_angle = 270  # in degrees

    def update(self):
        # randomness to break circular patterns
        if random.random() < 0.02:
            random_jitter = Vector2(random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1))
            self.velocity += random_jitter

        self.velocity += self.acceleration
        if self.velocity.length() > self.max_speed:
            self.velocity.scale_to_length(self.max_speed)
        self.position += self.velocity
        self.acceleration *= 0

    def apply_force(self, force):
        self.acceleration += force

    def seek(self, target):
        desired = target - self.position
        if desired.length() > 0:
            desired.scale_to_length(self.max_speed)

        steer = desired - self.velocity
        if steer.length() > self.max_force:
            steer.scale_to_length(self.max_force)

        return steer

    def arrive(self, target, slowing_radius=100):
        desired = target - self.position
        distance = desired.length()

        if distance > 0:
            if distance < slowing_radius:
                # Scale speed based on distance
                speed = self.max_speed * (distance / slowing_radius)
            else:
                speed = self.max_speed

            desired.scale_to_length(speed)
            steer = desired - self.velocity
            if steer.length() > self.max_force:
                steer.scale_to_length(self.max_force)
            return steer
        else:
            return Vector2(0, 0)

    def is_in_vision(self, other):
        # Check if other boid is within vision angle
        if self.position == other.position:
            return False

        heading = self.velocity.normalize() if self.velocity.length() > 0 else Vector2(1, 0)
        direction = (other.position - self.position).normalize()
        angle = math.degrees(math.acos(heading.dot(direction)))

        return angle <= self.vision_angle / 2

    def align(self, boids):
        steering = Vector2(0, 0)
        total = 0

        for boid in boids:
            if boid != self and boid.flock_id == self.flock_id:
                dist = self.position.distance_to(boid.position)
                if dist < PERCEPTION_RADIUS and self.is_in_vision(boid):
                    steering += boid.velocity
                    total += 1

        if total > 0:
            steering /= total
            steering.scale_to_length(self.max_speed)
            steering -= self.velocity
            if steering.length() > self.max_force:
                steering.scale_to_length(self.max_force)

        return steering

    def cohesion(self, boids):
        steering = Vector2(0, 0)
        total = 0

        for boid in boids:
            if boid != self and boid.flock_id == self.flock_id:
                dist = self.position.distance_to(boid.position)
                if dist < PERCEPTION_RADIUS and self.is_in_vision(boid):
                    steering += boid.position
                    total += 1

        if total > 0:
            steering /= total
            return self.seek(steering)

        return steering

    def separation(self, boids):
        steering = Vector2(0, 0)
        total = 0
        min_separation = 1

        for boid in boids:
            if boid != self:
                dist = self.position.distance_to(boid.position)
                if 0 < dist < PERCEPTION_RADIUS * 0.7 and self.is_in_vision(boid):
                    # Stronger force for closer boids
                    diff = self.position - boid.position

                    # Apply greater force when too close
                    if dist < min_separation:
                        diff *= 3.0  # Much stronger separation when too close
                    else:
                        diff /= dist  # Weight by distance

                    steering += diff
                    total += 1

        if total > 0:
            steering /= total
            if steering.length() > 0:
                steering.scale_to_length(self.max_speed)
                steering -= self.velocity
                if steering.length() > self.max_force:
                    steering.scale_to_length(self.max_force * 2.0)  # Allow stronger separation force

        return steering

    def boundary_behavior(self):
        margin = 25
        next_pos = self.position + self.velocity

        # prevent going outside the screen
        if next_pos.x < margin:
            self.position.x = margin
            self.velocity.x = abs(self.velocity.x) * 0.8  # Bounce with reduced velocity
        elif next_pos.x > WIDTH - margin:
            self.position.x = WIDTH - margin
            self.velocity.x = -abs(self.velocity.x) * 0.8  # Bounce with reduced velocity

        if next_pos.y < margin:
            self.position.y = margin
            self.velocity.y = abs(self.velocity.y) * 0.8  # Bounce with reduced velocity
        elif next_pos.y > HEIGHT - margin:
            self.position.y = HEIGHT - margin
            self.velocity.y = -abs(self.velocity.y) * 0.8  # Bounce with reduced velocity

        # return a steering force for smoother approach to boundaries
        desired = None
        outer_margin = 50

        if self.position.x < outer_margin:
            desired = Vector2(self.max_speed, self.velocity.y)
        elif self.position.x > WIDTH - outer_margin:
            desired = Vector2(-self.max_speed, self.velocity.y)

        if self.position.y < outer_margin:
            desired = Vector2(self.velocity.x, self.max_speed)
        elif self.position.y > HEIGHT - outer_margin:
            desired = Vector2(self.velocity.x, -self.max_speed)

        if desired:
            desired = desired.normalize() * self.max_speed
            steer = desired - self.velocity
            if steer.length() > self.max_force:
                steer.scale_to_length(self.max_force)
            return steer

        return Vector2(0, 0)

    def check_obstacle_collision(self, obstacles):
        next_pos = self.position + self.velocity

        for obstacle in obstacles:
            dist_next = next_pos.distance_to(obstacle.position)

            # If the next position would be inside the obstacle
            if dist_next < obstacle.radius:
                # Get vector from obstacle to boid
                away_vector = self.position - obstacle.position
                if away_vector.length() > 0:
                    away_vector.normalize_ip()

                    # Move position to be just outside the obstacle
                    correction = (obstacle.radius - dist_next + 1)
                    self.position += away_vector * correction

                    # Reflect velocity (bounce)
                    # Calculate reflection vector
                    normal = away_vector
                    reflection = self.velocity.reflect(normal)
                    self.velocity = reflection * 0.8  # Reduce speed after bouncing
                    return True
        return False

    def flock(self, boids, obstacles=None, target=None):
        # First check if we would collide with any obstacles
        if obstacles:
            collided = self.check_obstacle_collision(obstacles)
            if collided:
                return  # Skip other behaviors if we just collided

        # Apply flocking behaviors
        alignment = self.align(boids) * ALIGNMENT_WEIGHT
        cohesion = self.cohesion(boids) * COHESION_WEIGHT
        separation = self.separation(boids) * SEPARATION_WEIGHT
        boundary = self.boundary_behavior() * BOUNDARY_WEIGHT

        self.apply_force(alignment)
        self.apply_force(cohesion)
        self.apply_force(separation)
        self.apply_force(boundary)

        # Add some randomness to avoid circles
        if random.random() < 0.01:  # 1% chance each frame
            random_force = Vector2(random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5))
            self.apply_force(random_force)

        # Optional seek behavior if target is provided
        if target:
            seek = self.seek(target) * SEEK_WEIGHT
            self.apply_force(seek)

        # Avoid obstacles - stronger prevention
        if obstacles:
            for obstacle in obstacles:
                dist = self.position.distance_to(obstacle.position)
                if dist < obstacle.radius + 40:  # Increased perception radius for obstacles
                    avoid = self.position - obstacle.position
                    if avoid.length() > 0:
                        avoid.scale_to_length(self.max_force * 3.0 * (1.0 - dist / (obstacle.radius + 40)))
                        self.apply_force(avoid)

    def draw(self, screen):
        # Draw triangle in the direction of movement
        angle = math.atan2(self.velocity.y, self.velocity.x)

        # Triangle vertices - fix the syntax error
        points = [
            (self.position.x + self.size * math.cos(angle),
             self.position.y + self.size * math.sin(angle)),
            (self.position.x + self.size * math.cos(angle + 2.5),
             self.position.y + self.size * math.sin(angle + 2.5)),
            (self.position.x + self.size * math.cos(angle - 2.5),
             self.position.y + self.size * math.sin(angle - 2.5))
        ]

        # Get color based on flock ID
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        color = colors[self.flock_id % len(colors)]

        pygame.draw.polygon(screen, color, points)


class Obstacle:
    def __init__(self, x, y, radius):
        self.position = Vector2(x, y)
        self.radius = radius

    def draw(self, screen):
        pygame.draw.circle(screen, (150, 150, 150), (int(self.position.x), int(self.position.y)), self.radius)


if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Boids Simulation")
    clock = pygame.time.Clock()

    # Create two flocks of boids
    boids = []
    for _ in range(30):
        boids.append(Boid(random.randint(0, WIDTH), random.randint(0, HEIGHT), 0))
    for _ in range(30):
        boids.append(Boid(random.randint(0, WIDTH), random.randint(0, HEIGHT), 1))

    # Create some obstacles
    obstacles = [
        Obstacle(WIDTH / 2, HEIGHT / 2, 50),
        Obstacle(WIDTH / 4, HEIGHT / 4, 30),
        Obstacle(3 * WIDTH / 4, 3 * HEIGHT / 4, 40)
    ]

    target = None
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Set target on mouse click
                target = Vector2(pygame.mouse.get_pos())
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    target = None

        screen.fill(BACKGROUND_COLOR)

        # Update and draw obstacles
        for obstacle in obstacles:
            obstacle.draw(screen)

        # Update and draw boids
        for boid in boids:
            boid.flock(boids, obstacles, target)
            boid.update()
            boid.draw(screen)

        # Draw target if it exists
        if target:
            pygame.draw.circle(screen, (255, 255, 0), (int(target.x), int(target.y)), 10, 2)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
