import unittest
import numpy as np

from pre_process import Acorde, ModeloSetharesVec


# Importa la versión no vectorial si está disponible para comparar.
try:
    from pre_process import ModeloSetharesObsoleto as ModeloSetharesScalar
except Exception:  # pragma: no cover
    ModeloSetharesScalar = None


class TestSetharesVectorConsistency(unittest.TestCase):
    def test_dyad_total_roughness_close(self):
        """
        En un caso simple (díada), la versión vectorial debería alinear en rugosidad total
        con la versión escalar (si está disponible), dentro de una tolerancia numérica.
        """
        if ModeloSetharesScalar is None:
            self.skipTest("ModeloSethares escalar no disponible para comparación")

        cfg = {"base_freq": 440.0, "n_armonicos": 6, "decaimiento": 0.88}
        vec = ModeloSetharesVec(config=cfg)
        sca = ModeloSetharesScalar(config=cfg)

        # Díada: quinta justa (7 semitonos)
        a = Acorde(name="dyad_7", intervals=[7])
        v_vec, t_vec = vec.calcular(a)
        v_sca, t_sca = sca.calcular(a)

        # Comparar rugosidad total con tolerancia relativa
        rel_diff = abs(t_vec - t_sca) / max(abs(t_sca), 1e-12)
        self.assertLess(rel_diff, 1e-6, f"Diferencia relativa alta: {rel_diff}")

        # El bin activo (para 7 semitonos) debería dominar en ambos
        self.assertEqual(np.argmax(v_vec), np.argmax(v_sca))


if __name__ == "__main__":
    unittest.main()

