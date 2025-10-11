import unittest
from tools.compare_sethares import canonical_sethares_total_roughness, CanonicalConfig, dyad_fundamentals


class TestSetharesCanonical(unittest.TestCase):
    def test_minor_second_more_dissonant_than_octave(self):
        cfg = CanonicalConfig(base_freq=440.0, n_harmonics=10, decay=0.8)
        # 1 semitono vs 12 (octava)
        r1 = canonical_sethares_total_roughness(dyad_fundamentals(cfg.base_freq, 1), cfg)
        r12 = canonical_sethares_total_roughness(dyad_fundamentals(cfg.base_freq, 12), cfg)
        self.assertGreater(r1, r12)

    def test_minor_second_more_dissonant_than_perfect_fifth(self):
        cfg = CanonicalConfig(base_freq=500.0, n_harmonics=10, decay=0.8)
        r1 = canonical_sethares_total_roughness(dyad_fundamentals(cfg.base_freq, 1), cfg)
        r7 = canonical_sethares_total_roughness(dyad_fundamentals(cfg.base_freq, 7), cfg)
        self.assertGreater(r1, r7)


if __name__ == "__main__":
    unittest.main()

