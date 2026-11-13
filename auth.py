"""Authentification et isolation des comptes clients (FastAPI + MongoDB)."""
import hashlib
import base64
import io
import os
import re
import secrets
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

import bcrypt
import jwt
from bson import ObjectId
import qrcode
from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError
from qrcode.image.svg import SvgPathImage

SECRET = os.environ["JWT_SECRET"]
DUREE_H = int(os.getenv("JWT_DUREE_H", "12"))


class Connexion(BaseModel):
    numero_client: Optional[str] = Field(
        None, min_length=8, max_length=8, pattern=r"^\d{8}$"
    )
    nom: str = Field(min_length=1, max_length=60)
    mot_de_passe: str = Field(min_length=1, max_length=72)


class NouveauClient(BaseModel):
    code_activation: str = Field(min_length=8, max_length=128)
    nom: str = Field(min_length=2, max_length=60)
    mot_de_passe: str = Field(min_length=8, max_length=72)
    ferme: str = Field(min_length=1, max_length=80)


class NouvelUtilisateur(BaseModel):
    nom: str = Field(min_length=1, max_length=60)
    prenom: str = Field(min_length=1, max_length=60)
    age: int = Field(ge=0, le=120)
    residence: str = Field(min_length=1, max_length=120)
    telephone: str = Field(min_length=6, max_length=20)
    cin: str = Field(pattern=r"^\d{8}$")
    role: Literal["utilisateur", "admin"] = "utilisateur"


class NouveauCodeActivation(BaseModel):
    duree_jours: int = Field(default=30, ge=1, le=365)


def _cle(texte: str) -> str:
    return re.sub(r"\s+", " ", texte.strip()).lower()


def _hacher(mdp: str) -> str:
    return bcrypt.hashpw(mdp.encode(), bcrypt.gensalt()).decode()


def _verifier(mdp: str, empreinte: str) -> bool:
    try:
        return bcrypt.checkpw(mdp.encode(), empreinte.encode())
    except ValueError:
        return False


def _hachage_code(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()


def _numero_client(clients) -> str:
    for _ in range(10):
        numero = f"{secrets.randbelow(100_000_000):08d}"
        if not clients.find_one({"numero_client": numero}, {"_id": 1}):
            return numero
    raise HTTPException(503, "Impossible de générer un numéro client. Réessayez.")


def _public(u: dict) -> dict:
    return {
        "id": str(u["_id"]),
        "nom": u["nom"],
        "prenom": u.get("prenom", ""),
        "age": u.get("age"),
        "residence": u.get("residence", ""),
        "telephone": u.get("telephone", ""),
        "role": u["role"],
        "numero_client": u["numero_client"],
        "biometrie_activee": bool(u.get("appareils_biometriques")),
    }


def _migrer_client_existant(db, users, clients):
    admin = users.find_one(
        {"role": "superadmin", "client_id": {"$exists": False}}
    )
    if not admin:
        return

    client_id = ObjectId()
    numero = _numero_client(clients)
    maintenant = datetime.now(timezone.utc)
    clients.insert_one({
        "_id": client_id,
        "numero_client": numero,
        "nom": admin.get("nom", "Client existant"),
        "cree_le": maintenant,
    })
    users.update_many(
        {"client_id": {"$exists": False}},
        {"$set": {"client_id": client_id, "numero_client": numero}},
    )
    for nom in (
        "positions", "alerts", "etat_animaux", "animaux", "pastures",
        "parametres", "mesures_activite", "lstm_alertes_silencieuses",
        "conversations", "messages",
    ):
        collection = db[nom]
        valeurs = {"client_id": client_id, "numero_client": numero}
        if nom == "animaux":
            valeurs["ferme"] = "Ferme principale"
        collection.update_many(
            {"client_id": {"$exists": False}},
            {"$set": valeurs},
        )


def creer_dependance_courant(users):
    def courant(authorization: Optional[str] = Header(None)) -> dict:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(401, "Connexion requise.")
        try:
            data = jwt.decode(authorization[7:], SECRET, algorithms=["HS256"])
            u = users.find_one({"_id": ObjectId(data["sub"])})
        except (jwt.InvalidTokenError, KeyError, TypeError, ValueError):
            raise HTTPException(401, "Session invalide ou expirée.")
        if not u:
            raise HTTPException(401, "Compte introuvable.")
        if not u.get("client_id") or not u.get("numero_client"):
            raise HTTPException(403, "Le compte client doit être migré.")
        if data.get("client_id") and data["client_id"] != str(u["client_id"]):
            raise HTTPException(401, "Session invalide ou expirée.")
        return u

    return courant


def creer_routeur(db) -> APIRouter:
    users = db["utilisateurs"]
    clients = db["clients"]
    codes = db["codes_activation"]

    _migrer_client_existant(db, users, clients)

    clients.create_index("numero_client", unique=True)
    codes.create_index("code_hash", unique=True)
    codes.create_index("expire_le", expireAfterSeconds=0)

    indexes = users.index_information()
    for nom in ("nom_cle_1_prenom_cle_1", "est_superadmin_1"):
        if nom in indexes:
            users.drop_index(nom)
    users.create_index(
        [("client_id", 1), ("nom_cle", 1), ("prenom_cle", 1)],
        unique=True,
    )
    users.create_index(
        [("client_id", 1), ("est_superadmin", 1)],
        unique=True,
        partialFilterExpression={"est_superadmin": True},
    )
    users.create_index([("numero_client", 1), ("nom_cle", 1)])

    router = APIRouter(tags=["auth"])
    mdp_factice = _hacher("factice-pour-temps-constant")
    courant = creer_dependance_courant(users)

    def superadmin(u: dict = Depends(courant)) -> dict:
        if u["role"] != "superadmin":
            raise HTTPException(403, "Réservé au super administrateur.")
        return u

    def _reponse_session(u: dict) -> dict:
        exp = datetime.now(timezone.utc) + timedelta(hours=DUREE_H)
        jwt_token = jwt.encode(
            {
                "sub": str(u["_id"]),
                "role": u["role"],
                "client_id": str(u["client_id"]),
                "exp": exp,
            },
            SECRET,
            algorithm="HS256",
        )
        return {
            "token": jwt_token,
            "utilisateur": _public(u),
            "verification_biometrique_requise": bool(
                u.get("appareils_biometriques")
            ),
        }

    @router.post("/auth/codes-activation", status_code=201)
    def creer_code_activation(
        corps: NouveauCodeActivation,
        cle: Optional[str] = Header(None, alias="X-Enrollment-Admin-Key"),
    ):
        cle_admin = os.getenv("CLIENT_ENROLLMENT_ADMIN_KEY")
        if not cle_admin:
            raise HTTPException(503, "Création des codes d’activation non configurée.")
        if not secrets.compare_digest(cle or "", cle_admin):
            raise HTTPException(403, "Clé d’administration invalide.")

        code = secrets.token_urlsafe(24)
        qr = qrcode.QRCode(
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=8,
            border=4,
        )
        qr.add_data(code)
        qr.make(fit=True)
        image = qr.make_image(image_factory=SvgPathImage)
        contenu = io.BytesIO()
        image.save(contenu)
        codes.insert_one({
            "code_hash": _hachage_code(code),
            "cree_le": datetime.now(timezone.utc),
            "expire_le": datetime.now(timezone.utc) + timedelta(days=corps.duree_jours),
        })
        return {
            "code_activation": code,
            "expire_dans_jours": corps.duree_jours,
            "qr_svg_base64": base64.b64encode(contenu.getvalue()).decode("ascii"),
        }

    @router.get("/admin/codes-activation", response_class=HTMLResponse)
    def page_codes_activation():
        return HTMLResponse(
            content="""<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>QR d’inscription — Traceur</title>
<style>
body{font:16px system-ui,sans-serif;max-width:560px;margin:40px auto;padding:0 18px;color:#18352c}
main{padding:24px;border:1px solid #dfe8e2;border-radius:20px}
label{display:block;margin:16px 0 6px;font-weight:600}
input,button{box-sizing:border-box;width:100%;padding:12px;border:1px solid #bdcbc2;border-radius:10px;font:inherit}
button{margin-top:20px;background:#23845b;color:white;border:0;font-weight:700}
#result{text-align:center;margin-top:24px;overflow-wrap:anywhere}
#qr{width:min(100%,320px);background:white;padding:12px}
[hidden]{display:none!important}.error{color:#a3212a}
</style></head><body><main>
<h1>Créer un QR d’inscription</h1>
<p>Le code est à usage unique et expire après la durée choisie. Ne partagez le QR qu’avec le nouveau client.</p>
<form id="form">
<label for="key">Clé d’administration</label>
<input id="key" type="password" autocomplete="off" required>
<label for="days">Expiration (jours)</label>
<input id="days" type="number" min="1" max="365" value="1" required>
<button id="submit" type="submit">Générer le QR</button>
</form>
<section id="result" hidden><img id="qr" alt="QR code d’inscription">
<p id="expiry"></p><p id="code"></p></section>
<p id="error" class="error" role="alert"></p>
</main><script>
const form=document.getElementById('form'),key=document.getElementById('key');
const days=document.getElementById('days'),button=document.getElementById('submit');
const result=document.getElementById('result'),error=document.getElementById('error');
form.addEventListener('submit',async event=>{
 event.preventDefault(); error.textContent=''; result.hidden=true; button.disabled=true;
 try {
  const response=await fetch('/auth/codes-activation',{method:'POST',
   headers:{'Content-Type':'application/json','X-Enrollment-Admin-Key':key.value},
   body:JSON.stringify({duree_jours:Number(days.value)})});
  const data=await response.json();
  if(!response.ok) throw new Error(data.detail||'Impossible de générer le QR.');
  document.getElementById('qr').src='data:image/svg+xml;base64,'+data.qr_svg_base64;
  document.getElementById('expiry').textContent='Valide '+data.expire_dans_jours+' jour(s), une seule utilisation.';
  document.getElementById('code').textContent='Code de secours : '+data.code_activation;
  result.hidden=false; key.value='';
 } catch (e) { error.textContent=e.message; }
 finally { button.disabled=false; }
});
</script></body></html>""",
            headers={
                "Cache-Control": "no-store",
                "Content-Security-Policy": (
                    "default-src 'none'; script-src 'unsafe-inline'; "
                    "style-src 'unsafe-inline'; connect-src 'self'; img-src data:; "
                    "form-action 'none'; base-uri 'none'"
                ),
                "X-Content-Type-Options": "nosniff",
            },
        )

    @router.post("/auth/nouveau-client", status_code=201)
    def nouveau_client(corps: NouveauClient):
        maintenant = datetime.now(timezone.utc)
        code = codes.find_one_and_update(
            {
                "code_hash": _hachage_code(corps.code_activation),
                "expire_le": {"$gt": maintenant},
                "utilise_le": {"$exists": False},
            },
            {"$set": {"utilise_le": maintenant}},
            return_document=ReturnDocument.BEFORE,
        )
        if not code:
            raise HTTPException(403, "Code d’activation invalide, expiré ou déjà utilisé.")

        client_id = ObjectId()
        numero = _numero_client(clients)
        client = {
            "_id": client_id,
            "numero_client": numero,
            "nom": corps.nom.strip(),
            "ferme_principale": corps.ferme.strip(),
            "cree_le": maintenant,
        }
        utilisateur = {
            "nom": corps.nom.strip(),
            "prenom": "",
            "nom_cle": _cle(corps.nom),
            "prenom_cle": "",
            "role": "superadmin",
            "est_superadmin": True,
            "client_id": client_id,
            "numero_client": numero,
            "mot_de_passe": _hacher(corps.mot_de_passe),
            "cree_le": maintenant,
        }
        try:
            clients.insert_one(client)
            utilisateur["_id"] = users.insert_one(utilisateur).inserted_id
        except DuplicateKeyError:
            clients.delete_one({"_id": client_id})
            raise HTTPException(409, "Un compte avec ce nom existe déjà pour ce client.")
        return _reponse_session(utilisateur)

    @router.post("/auth/connexion")
    def connexion(corps: Connexion):
        filtre = {"nom_cle": _cle(corps.nom)}
        if corps.numero_client:
            filtre["numero_client"] = corps.numero_client.strip()
        candidats = list(users.find(filtre))
        if not candidats:
            _verifier(corps.mot_de_passe, mdp_factice)
        valides = [
            u for u in candidats
            if _verifier(corps.mot_de_passe, u["mot_de_passe"])
        ]
        if len(valides) == 1:
            return _reponse_session(valides[0])
        if len(valides) > 1:
            raise HTTPException(409, "Saisissez votre numéro client pour choisir votre espace.")
        raise HTTPException(401, "Numéro client, nom ou mot de passe incorrect.")

    @router.get("/auth/moi")
    def moi(u: dict = Depends(courant)):
        return _public(u)

    @router.get("/utilisateurs")
    def lister(u: dict = Depends(superadmin)):
        liste = users.find({
            "client_id": u["client_id"],
            "role": {"$ne": "superadmin"},
        }).sort([("nom_cle", 1), ("prenom_cle", 1)])
        return [_public(utilisateur) for utilisateur in liste]

    @router.post("/utilisateurs", status_code=201)
    def ajouter(corps: NouvelUtilisateur, u: dict = Depends(superadmin)):
        doc = {
            "nom": corps.nom.strip(),
            "prenom": corps.prenom.strip(),
            "nom_cle": _cle(corps.nom),
            "prenom_cle": _cle(corps.prenom),
            "age": corps.age,
            "residence": corps.residence.strip(),
            "telephone": corps.telephone.strip(),
            "role": corps.role,
            "client_id": u["client_id"],
            "numero_client": u["numero_client"],
            "mot_de_passe": _hacher(corps.cin),
            "cree_le": datetime.now(timezone.utc),
        }
        try:
            doc["_id"] = users.insert_one(doc).inserted_id
        except DuplicateKeyError:
            raise HTTPException(409, "Un utilisateur avec ce nom et ce prénom existe déjà.")
        return _public(doc)

    @router.delete("/utilisateurs/{user_id}")
    def supprimer(user_id: str, u: dict = Depends(superadmin)):
        try:
            oid = ObjectId(user_id)
        except (TypeError, ValueError):
            raise HTTPException(400, "Identifiant invalide.")
        res = users.delete_one({
            "_id": oid,
            "client_id": u["client_id"],
            "role": {"$ne": "superadmin"},
        })
        if res.deleted_count == 0:
            raise HTTPException(404, "Utilisateur introuvable.")
        return {"supprime": True}

    return router
