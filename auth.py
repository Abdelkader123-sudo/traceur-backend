"""Authentification et gestion des utilisateurs (FastAPI + MongoDB).

Installation :  pip install bcrypt pyjwt
Variable d'environnement obligatoire : JWT_SECRET (longue chaîne aléatoire)

Dans main.py :
    from auth import creer_routeur
    app.include_router(creer_routeur(db))   # db = base MongoDB déjà utilisée
"""
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

import bcrypt
import jwt
from bson import ObjectId
from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field
from pymongo.errors import DuplicateKeyError

SECRET = os.environ["JWT_SECRET"]
DUREE_H = int(os.getenv("JWT_DUREE_H", "12"))


class Connexion(BaseModel):
    nom: str = Field(min_length=1, max_length=60)
    mot_de_passe: str = Field(min_length=1, max_length=72)


class PremierInscrit(BaseModel):
    nom: str = Field(min_length=2, max_length=60)
    mot_de_passe: str = Field(min_length=8, max_length=72)


class NouvelUtilisateur(BaseModel):
    nom: str = Field(min_length=1, max_length=60)
    prenom: str = Field(min_length=1, max_length=60)
    age: int = Field(ge=0, le=120)
    residence: str = Field(min_length=1, max_length=120)
    telephone: str = Field(min_length=6, max_length=20)
    cin: str = Field(pattern=r"^\d{8}$")  # le CIN sert de mot de passe
    role: Literal["utilisateur", "admin"] = "utilisateur"


def _cle(texte: str) -> str:
    return re.sub(r"\s+", " ", texte.strip()).lower()


def _hacher(mdp: str) -> str:
    return bcrypt.hashpw(mdp.encode(), bcrypt.gensalt()).decode()


def _verifier(mdp: str, empreinte: str) -> bool:
    try:
        return bcrypt.checkpw(mdp.encode(), empreinte.encode())
    except ValueError:
        return False


def _public(u: dict) -> dict:
    return {
        "id": str(u["_id"]),
        "nom": u["nom"],
        "prenom": u.get("prenom", ""),
        "age": u.get("age"),
        "residence": u.get("residence", ""),
        "telephone": u.get("telephone", ""),
        "role": u["role"],
    }


def creer_dependance_courant(users):
    def courant(authorization: Optional[str] = Header(None)) -> dict:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(401, "Connexion requise.")
        try:
            data = jwt.decode(authorization[7:], SECRET, algorithms=["HS256"])
            u = users.find_one({"_id": ObjectId(data["sub"])})
        except Exception:
            raise HTTPException(401, "Session invalide ou expirée.")
        if not u:
            raise HTTPException(401, "Compte introuvable.")
        return u

    return courant


def creer_routeur(db) -> APIRouter:
    users = db["utilisateurs"]
    users.create_index([("nom_cle", 1), ("prenom_cle", 1)], unique=True)
    # Un seul superadmin possible, même si deux inscriptions arrivent en même temps
    users.create_index(
        "est_superadmin",
        unique=True,
        partialFilterExpression={"est_superadmin": True},
    )

    router = APIRouter(tags=["auth"])
    mdp_factice = _hacher("factice-pour-temps-constant")
    courant = creer_dependance_courant(users)

    def superadmin(u: dict = Depends(courant)) -> dict:
        if u["role"] != "superadmin":
            raise HTTPException(403, "Réservé au super administrateur.")
        return u

    def _reponse_session(u: dict) -> dict:
        exp = datetime.now(timezone.utc) + timedelta(hours=DUREE_H)
        token = jwt.encode(
            {"sub": str(u["_id"]), "role": u["role"], "exp": exp},
            SECRET,
            algorithm="HS256",
        )
        return {"token": token, "utilisateur": _public(u)}

    # --- État : combien d'utilisateurs ? ------------------------------------
    @router.get("/auth/etat")
    def etat():
        # False  -> aucun utilisateur : la page affiche l'inscription
        # True   -> au moins un utilisateur : la page affiche la connexion
        return {"initialise": users.count_documents({}, limit=1) > 0}

    # --- Inscription du tout premier utilisateur (superadmin) ---------------
    @router.post("/auth/inscription", status_code=201)
    def inscription(corps: PremierInscrit):
        if users.count_documents({}, limit=1) > 0:
            raise HTTPException(409, "Inscription fermée : utilisez la connexion.")
        doc = {
            "nom": corps.nom.strip(),
            "prenom": "",
            "nom_cle": _cle(corps.nom),
            "prenom_cle": "",
            "role": "superadmin",
            "est_superadmin": True,
            "mot_de_passe": _hacher(corps.mot_de_passe),
            "cree_le": datetime.now(timezone.utc),
        }
        try:
            doc["_id"] = users.insert_one(doc).inserted_id
        except DuplicateKeyError:
            raise HTTPException(409, "Inscription fermée : utilisez la connexion.")
        return _reponse_session(doc)

    # --- Connexion ----------------------------------------------------------
    @router.post("/auth/connexion")
    def connexion(corps: Connexion):
        candidats = list(users.find({"nom_cle": _cle(corps.nom)}))
        if not candidats:
            _verifier(corps.mot_de_passe, mdp_factice)  # évite de révéler si le nom existe
        for u in candidats:
            if _verifier(corps.mot_de_passe, u["mot_de_passe"]):
                return _reponse_session(u)
        raise HTTPException(401, "Nom ou mot de passe incorrect.")

    @router.get("/auth/moi")
    def moi(u: dict = Depends(courant)):
        return _public(u)

    # --- Gestion des utilisateurs (superadmin) ------------------------------
    @router.get("/utilisateurs")
    def lister(_: dict = Depends(superadmin)):
        liste = users.find({"role": {"$ne": "superadmin"}}).sort(
            [("nom_cle", 1), ("prenom_cle", 1)]
        )
        return [_public(u) for u in liste]

    @router.post("/utilisateurs", status_code=201)
    def ajouter(corps: NouvelUtilisateur, _: dict = Depends(superadmin)):
        doc = {
            "nom": corps.nom.strip(),
            "prenom": corps.prenom.strip(),
            "nom_cle": _cle(corps.nom),
            "prenom_cle": _cle(corps.prenom),
            "age": corps.age,
            "residence": corps.residence.strip(),
            "telephone": corps.telephone.strip(),
            "role": corps.role,
            "mot_de_passe": _hacher(corps.cin),  # CIN = mot de passe, jamais stocké en clair
            "cree_le": datetime.now(timezone.utc),
        }
        try:
            doc["_id"] = users.insert_one(doc).inserted_id
        except DuplicateKeyError:
            raise HTTPException(409, "Un utilisateur avec ce nom et ce prénom existe déjà.")
        return _public(doc)

    @router.delete("/utilisateurs/{user_id}")
    def supprimer(user_id: str, _: dict = Depends(superadmin)):
        try:
            oid = ObjectId(user_id)
        except Exception:
            raise HTTPException(400, "Identifiant invalide.")
        res = users.delete_one({"_id": oid, "role": {"$ne": "superadmin"}})
        if res.deleted_count == 0:
            raise HTTPException(404, "Utilisateur introuvable.")
        return {"supprime": True}

    return router
