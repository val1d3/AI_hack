import unittest
from models.detection.factory_detector import FactoryPeopleDetector

class TestDetection(unittest.TestCase):
    def test_hog_init(self):
        detector = FactoryPeopleDetector('hog')
        self.assertIsNotNone(detector.hog)

if __name__ == '__main__':
    unittest.main()