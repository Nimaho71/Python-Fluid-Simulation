import unittest
import math

class TestSPHFunctions(unittest.TestCase):
    def test_smoothing_kernel(self):
        radius = 5
        
        # Case 1: Distance is greater than or equal to the radius
        self.assertEqual(smoothing_kernel(radius, 5), 0)
        self.assertEqual(smoothing_kernel(radius, 6), 0)
        
        # Case 2: Distance is less than the radius
        distance = 3
        expected_influence = (radius**2 - distance**2)**3
        self.assertAlmostEqual(smoothing_kernel(radius, distance), expected_influence)

    def test_smoothing_kernel_derivative(self):
        radius = 5
        
        # Case 1: Distance greater than or equal to the radius
        self.assertEqual(smoothing_kernel_derivative(radius, 5), 0)
        self.assertEqual(smoothing_kernel_derivative(radius, 6), 0)

        # Case 2: Distance less than the radius
        distance = 3
        f = radius**2 - distance**2
        expected_derivative = (-24 / (math.pi * radius**8)) * distance * (f ** 2)
        self.assertAlmostEqual(smoothing_kernel_derivative(radius, distance), expected_derivative)

    def test_calculate_density(self):
        # Mock particle positions (create some simple cases for testing)
        class MockParticle:
            def __init__(self, x, y):
                self.x = x
                self.y = y
        
        particles = [MockParticle(0, 0), MockParticle(1, 0), MockParticle(0, 1), MockParticle(1, 1)]
        target_particle = MockParticle(0, 0)
        radius = 5
        
        # Manually calculate expected density (assuming all particles are inside the smoothing radius)
        expected_density = 0
        for particle in particles:
            distance = math.hypot(particle.x - target_particle.x, particle.y - target_particle.y)
            influence = smoothing_kernel(radius, distance)
            expected_density += 1 * influence  # assuming mass = 1 for simplicity
        
        # Normalize by kernel volume
        kernel_volume = (math.pi * radius**8) / 4
        expected_density /= kernel_volume

        # Test if the calculated density matches the expected result
        self.assertAlmostEqual(calculate_density(particles, target_particle, radius), expected_density)

if __name__ == "__main__":
    unittest.main()
