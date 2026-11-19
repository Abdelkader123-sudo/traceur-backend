import base64
import os
import unittest
from io import BytesIO
from unittest.mock import patch

os.environ.setdefault("MQTT_USER", "animal-photo-test")
os.environ.setdefault("MQTT_PASSWORD", "animal-photo-test")
os.environ.setdefault("MONGO_URI", "mongodb://127.0.0.1:27017")
os.environ.setdefault(
    "JWT_SECRET",
    "animal-photo-tests-secret-with-at-least-32-bytes",
)

import main
from fastapi import HTTPException


class CollectionMemoire:
    def __init__(self):
        self.documents = {}

    def find_one(self, filtre, projection=None):
        document = self.documents.get((filtre["client_id"], filtre["animal_id"]))
        return dict(document) if document else None

    def find(self, filtre, projection=None):
        return [
            dict(document)
            for (client_id, _), document in self.documents.items()
            if client_id == filtre["client_id"]
        ]

    def update_one(self, filtre, modifications, upsert=False):
        cle = (filtre["client_id"], filtre["animal_id"])
        document = self.documents.setdefault(
            cle,
            dict(modifications.get("$setOnInsert", {})),
        )
        document.update(modifications.get("$set", {}))


class GridFSMemoire:
    def __init__(self):
        self.fichiers = {}
        self.prochain_id = 0

    def upload_from_stream(self, nom, source, metadata=None):
        self.prochain_id += 1
        self.fichiers[self.prochain_id] = source.read()
        return self.prochain_id

    def delete(self, image_id):
        self.fichiers.pop(image_id, None)

    def open_download_stream(self, image_id):
        contenu = self.fichiers.get(image_id)
        if contenu is None:
            from gridfs.errors import NoFile

            raise NoFile("photo introuvable")
        return BytesIO(contenu)


class PhotosAnimauxTests(unittest.TestCase):
    def setUp(self):
        self.collection = CollectionMemoire()
        self.gridfs = GridFSMemoire()
        self.patch_collection = patch.object(main, "animaux", self.collection)
        self.patch_gridfs = patch.object(main, "photos_animaux", self.gridfs)
        self.patch_collection.start()
        self.patch_gridfs.start()
        self.addCleanup(self.patch_collection.stop)
        self.addCleanup(self.patch_gridfs.stop)

    def test_photo_est_partagee_et_limitee_au_client(self):
        client = {"client_id": "client-a", "numero_client": "A"}
        autre_client = {"client_id": "client-b", "numero_client": "B"}
        contenu = b"\x89PNG\r\n\x1a\nphoto"
        photo = main.PhotoAnimal(
            type_contenu="image/png",
            contenu_base64=base64.b64encode(contenu).decode(),
        )

        main.enregistrer_photo_animal("animal-1", photo, u=client)

        self.assertEqual(
            main.lire_photo_animal("animal-1", u=client).body,
            contenu,
        )
        profil = main.enregistrer_animal(
            "animal-1",
            main.InfoAnimal(type="Vache"),
            u=client,
        )
        self.assertTrue(profil["photo_disponible"])
        self.assertTrue(main.lire_animaux(u=client)[0]["photo_disponible"])
        with self.assertRaises(HTTPException) as erreur:
            main.lire_photo_animal("animal-1", u=autre_client)
        self.assertEqual(erreur.exception.status_code, 404)

    def test_refuse_une_signature_incoherente(self):
        photo = main.PhotoAnimal(
            type_contenu="image/png",
            contenu_base64=base64.b64encode(b"pas un PNG").decode(),
        )

        with self.assertRaises(HTTPException) as erreur:
            main.decoder_photo_animal(photo)

        self.assertEqual(erreur.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
