import json
import logging
import os
from threading import Lock

import firebase_admin
from firebase_admin import credentials, messaging
from firebase_admin.exceptions import FirebaseError

_log = logging.getLogger("notifications_fcm")
_nom_application = "traceur_notifications"
_verrou = Lock()


def fcm_configuree() -> bool:
    return bool(os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON"))


def fcm_prete() -> bool:
    if not fcm_configuree():
        return False
    try:
        _application_firebase()
        return True
    except (FirebaseError, RuntimeError, TypeError, ValueError) as erreur:
        _log.exception("Configuration FCM invalide : %s", erreur)
        return False


def _application_firebase():
    try:
        return firebase_admin.get_app(_nom_application)
    except ValueError:
        with _verrou:
            try:
                return firebase_admin.get_app(_nom_application)
            except ValueError:
                contenu = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
                if not contenu:
                    raise RuntimeError(
                        "FIREBASE_SERVICE_ACCOUNT_JSON n’est pas configurée."
                    )
                compte = json.loads(contenu)
                return firebase_admin.initialize_app(
                    credentials.Certificate(compte),
                    name=_nom_application,
                )


def envoyer_notifications(
    identifiants_installation: list[str],
    titre: str,
    texte: str,
    donnees: dict[str, str],
) -> list[str]:
    if not identifiants_installation:
        return []
    try:
        application = _application_firebase()
    except (FirebaseError, RuntimeError, TypeError, ValueError) as erreur:
        _log.exception("Configuration FCM invalide : %s", erreur)
        return []

    identifiants_invalides = []
    for identifiant in identifiants_installation:
        try:
            messaging.send(
                messaging.Message(
                    fid=identifiant,
                    data={
                        "title": titre,
                        "body": texte,
                        **donnees,
                    },
                    android=messaging.AndroidConfig(priority="high"),
                ),
                app=application,
            )
        except messaging.UnregisteredError:
            identifiants_invalides.append(identifiant)
        except FirebaseError:
            _log.exception("Échec d’envoi d’une notification FCM.")
    return identifiants_invalides
