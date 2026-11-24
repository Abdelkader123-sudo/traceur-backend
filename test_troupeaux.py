import unittest

from DBSCAN import analyser_troupeaux


PARAMETRES = {
    "eps_troupeau_m": 30,
    "min_animaux_troupeau": 5,
    "distance_alerte_m": 120,
}


class TroupeauxTests(unittest.TestCase):
    def test_detecte_des_groupes_de_deux_animaux_ou_plus(self):
        pas_lat = 5 / 111_320
        positions = [
            {"animal_id": "a1", "lat": 36.75, "lon": 3.04},
            {"animal_id": "a2", "lat": 36.75 + pas_lat, "lon": 3.04},
            {"animal_id": "b1", "lat": 36.76, "lon": 3.04},
            {"animal_id": "b2", "lat": 36.76 + pas_lat, "lon": 3.04},
            {"animal_id": "b3", "lat": 36.76 + 2 * pas_lat, "lon": 3.04},
        ]

        principal, autres, _ = analyser_troupeaux(positions, PARAMETRES)

        self.assertEqual(principal["nb_animaux"], 3)
        self.assertEqual(len(autres), 1)
        self.assertEqual(autres[0]["type"], "troupeau")
        self.assertEqual(autres[0]["zone"]["nb_animaux"], 2)

    def test_elargit_le_rayon_si_le_reglage_ne_detecte_aucun_groupe(self):
        pas_lat = 50 / 111_320
        positions = [
            {"animal_id": f"a{index}", "lat": 36.75 + index * pas_lat, "lon": 3.04}
            for index in range(5)
        ]

        principal, autres, _ = analyser_troupeaux(positions, PARAMETRES)

        self.assertEqual(principal["nb_animaux"], 5)
        self.assertEqual(autres, [])


if __name__ == "__main__":
    unittest.main()
