from datetime import datetime, timezone
from typing import Optional

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from pymongo.errors import DuplicateKeyError

from auth import creer_dependance_courant


class ConversationDirecte(BaseModel):
    destinataire_id: str = Field(min_length=1)


class NouveauGroupe(BaseModel):
    titre: str = Field(min_length=1, max_length=80)
    membres: list[str] = Field(min_length=1, max_length=100)


class NouveauMessage(BaseModel):
    texte: str = Field(min_length=1, max_length=4000)


class NouveauxMembres(BaseModel):
    membres: list[str] = Field(min_length=1, max_length=100)


def _object_id(valeur: str) -> ObjectId:
    if not ObjectId.is_valid(valeur):
        raise HTTPException(status_code=400, detail="Identifiant utilisateur invalide.")
    return ObjectId(valeur)


def _utilisateur_public(utilisateur: dict) -> dict:
    return {
        "id": str(utilisateur["_id"]),
        "nom": utilisateur.get("nom", ""),
        "prenom": utilisateur.get("prenom", ""),
        "role": utilisateur.get("role", "utilisateur"),
    }


def _message_public(message: Optional[dict]) -> Optional[dict]:
    if message is None:
        return None
    return {
        "id": str(message["_id"]),
        "conversation_id": str(message["conversation_id"]),
        "sender_id": str(message["sender_id"]),
        "texte": message["texte"],
        "cree_le": message["cree_le"],
    }


def creer_routeur_messagerie(db):
    users = db["utilisateurs"]
    conversations = db["conversations"]
    messages = db["messages"]
    conversations.create_index(
        "direct_key",
        unique=True,
        partialFilterExpression={"type": "direct"},
    )
    conversations.create_index([("membres", 1), ("updated_at", -1)])
    messages.create_index([("conversation_id", 1), ("cree_le", -1)])

    router = APIRouter(prefix="/messagerie", tags=["messagerie"])
    courant = creer_dependance_courant(users)

    def exiger_conversation(conversation_id: str, utilisateur: dict) -> dict:
        if not ObjectId.is_valid(conversation_id):
            raise HTTPException(status_code=404, detail="Conversation introuvable.")
        conversation = conversations.find_one({
            "_id": ObjectId(conversation_id),
            "membres": utilisateur["_id"],
            "client_id": utilisateur["client_id"],
        })
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation introuvable.")
        return conversation

    def conversation_publique(conversation: dict, utilisateur: dict) -> dict:
        membres = list(users.find({
            "_id": {"$in": conversation["membres"]},
            "client_id": utilisateur["client_id"],
        }))
        participants = [_utilisateur_public(membre) for membre in membres]
        titre = conversation.get("titre", "")
        if conversation["type"] == "direct":
            autre = next(
                (participant for participant in participants
                 if participant["id"] != str(utilisateur["_id"])),
                participants[0] if participants else None,
            )
            titre = (
                " ".join(
                    part for part in (autre.get("prenom"), autre.get("nom")) if part
                )
                if autre
                else ""
            )
        return {
            "id": str(conversation["_id"]),
            "type": conversation["type"],
            "titre": titre,
            "membres": participants,
            "dernier_message": _message_public(conversation.get("dernier_message")),
            "cree_le": conversation["cree_le"],
            "updated_at": conversation["updated_at"],
        }

    @router.get("/utilisateurs")
    def lister_interlocuteurs(utilisateur: dict = Depends(courant)):
        return [
            _utilisateur_public(autre)
            for autre in users.find({
                "_id": {"$ne": utilisateur["_id"]},
                "client_id": utilisateur["client_id"],
            })
        ]

    @router.get("/conversations")
    def lister_conversations(utilisateur: dict = Depends(courant)):
        documents = conversations.find({
            "membres": utilisateur["_id"],
            "client_id": utilisateur["client_id"],
        }).sort(
            "updated_at", -1
        )
        return [
            conversation_publique(document, utilisateur) for document in documents
        ]

    @router.post("/conversations/direct", status_code=201)
    def ouvrir_directe(
        corps: ConversationDirecte,
        utilisateur: dict = Depends(courant),
    ):
        destinataire_id = _object_id(corps.destinataire_id)
        if destinataire_id == utilisateur["_id"]:
            raise HTTPException(status_code=400, detail="Vous ne pouvez pas vous écrire.")
        if not users.find_one({
            "_id": destinataire_id,
            "client_id": utilisateur["client_id"],
        }):
            raise HTTPException(status_code=404, detail="Utilisateur introuvable.")

        membres = sorted(
            (utilisateur["_id"], destinataire_id), key=str
        )
        cle = ":".join(str(membre) for membre in membres)
        conversation = conversations.find_one({"direct_key": cle})
        if conversation is None:
            maintenant = datetime.now(timezone.utc)
            document = {
                "type": "direct",
                "direct_key": cle,
                "membres": membres,
                "created_by": utilisateur["_id"],
                "cree_le": maintenant,
                "updated_at": maintenant,
                "dernier_message": None,
                "client_id": utilisateur["client_id"],
                "numero_client": utilisateur["numero_client"],
            }
            try:
                document["_id"] = conversations.insert_one(document).inserted_id
                conversation = document
            except DuplicateKeyError:
                conversation = conversations.find_one({
                    "direct_key": cle,
                    "client_id": utilisateur["client_id"],
                })
                if conversation is None:
                    raise
        return conversation_publique(conversation, utilisateur)

    @router.post("/conversations/groupe", status_code=201)
    def creer_groupe(
        corps: NouveauGroupe,
        utilisateur: dict = Depends(courant),
    ):
        ids = {_object_id(identifiant) for identifiant in corps.membres}
        ids.add(utilisateur["_id"])
        membres = sorted(ids, key=str)
        trouves = list(users.find({
            "_id": {"$in": membres},
            "client_id": utilisateur["client_id"],
        }))
        if len(trouves) != len(membres):
            raise HTTPException(status_code=404, detail="Un membre est introuvable.")

        titre = corps.titre.strip()
        if not titre:
            raise HTTPException(status_code=422, detail="Le nom du groupe est obligatoire.")

        maintenant = datetime.now(timezone.utc)
        document = {
            "type": "groupe",
            "titre": titre,
            "membres": membres,
            "created_by": utilisateur["_id"],
            "cree_le": maintenant,
            "updated_at": maintenant,
            "dernier_message": None,
            "client_id": utilisateur["client_id"],
            "numero_client": utilisateur["numero_client"],
        }
        document["_id"] = conversations.insert_one(document).inserted_id
        return conversation_publique(document, utilisateur)

    @router.get("/conversations/{conversation_id}/messages")
    def lister_messages(
        conversation_id: str,
        limite: int = Query(default=100, ge=1, le=200),
        utilisateur: dict = Depends(courant),
    ):
        conversation = exiger_conversation(conversation_id, utilisateur)
        documents = list(
            messages.find({
                "conversation_id": conversation["_id"],
                "client_id": utilisateur["client_id"],
            })
            .sort("cree_le", -1)
            .limit(limite)
        )
        return [_message_public(document) for document in reversed(documents)]

    @router.post("/conversations/{conversation_id}/messages", status_code=201)
    def envoyer_message(
        conversation_id: str,
        corps: NouveauMessage,
        utilisateur: dict = Depends(courant),
    ):
        conversation = exiger_conversation(conversation_id, utilisateur)
        texte = corps.texte.strip()
        if not texte:
            raise HTTPException(status_code=422, detail="Le message ne peut pas être vide.")

        maintenant = datetime.now(timezone.utc)
        document = {
            "conversation_id": conversation["_id"],
            "sender_id": utilisateur["_id"],
            "texte": texte,
            "cree_le": maintenant,
            "client_id": utilisateur["client_id"],
            "numero_client": utilisateur["numero_client"],
        }
        document["_id"] = messages.insert_one(document).inserted_id
        conversations.update_one(
            {"_id": conversation["_id"], "client_id": utilisateur["client_id"]},
            {"$set": {
                "dernier_message": document,
                "updated_at": maintenant,
            }},
        )
        return _message_public(document)

    @router.post("/conversations/{conversation_id}/membres", status_code=200)
    def ajouter_membres(
        conversation_id: str,
        corps: NouveauxMembres,
        utilisateur: dict = Depends(courant),
    ):
        conversation = exiger_conversation(conversation_id, utilisateur)
        if conversation["type"] != "groupe":
            raise HTTPException(status_code=400, detail="Cette conversation n'est pas un groupe.")
        ids = {_object_id(identifiant) for identifiant in corps.membres}
        nouveaux = ids.difference(conversation["membres"])
        if not nouveaux:
            return conversation_publique(conversation, utilisateur)
        trouves = list(users.find({
            "_id": {"$in": list(nouveaux)},
            "client_id": utilisateur["client_id"],
        }))
        if len(trouves) != len(nouveaux):
            raise HTTPException(status_code=404, detail="Un membre est introuvable.")
        conversations.update_one(
            {"_id": conversation["_id"], "client_id": utilisateur["client_id"]},
            {"$addToSet": {"membres": {"$each": list(nouveaux)}}},
        )
        conversation["membres"] = list(set(conversation["membres"]).union(nouveaux))
        return conversation_publique(conversation, utilisateur)

    return router
