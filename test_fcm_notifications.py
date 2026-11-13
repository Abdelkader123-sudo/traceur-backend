import unittest
from unittest.mock import patch

from fcm_notifications import envoyer_notifications


class NotificationsFcmTests(unittest.TestCase):
    @patch("fcm_notifications.messaging.send")
    @patch("fcm_notifications._application_firebase", return_value=object())
    def test_envoie_un_message_generique_sans_contenu_du_chat(
        self,
        _application,
        envoyer,
    ):
        envoyer_notifications(
            ["installation-de-test"],
            "Nouveau message",
            "Vous avez reçu un nouveau message.",
            {"type": "nouveau_message", "conversation_id": "conversation-1"},
        )

        message = envoyer.call_args.args[0]
        self.assertEqual(
            message.data,
            {
                "title": "Nouveau message",
                "body": "Vous avez reçu un nouveau message.",
                "type": "nouveau_message",
                "conversation_id": "conversation-1",
            },
        )
        self.assertEqual(message.fid, "installation-de-test")
        self.assertNotIn("texte", message.data)


if __name__ == "__main__":
    unittest.main()
