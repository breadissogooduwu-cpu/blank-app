import streamlit as st
import torch
import tempfile
import os
import cv2
import numpy as np

# ---------------------------------------------------------
# 1. Initialisation de l'état
# ---------------------------------------------------------
if "model_paths" not in st.session_state:
    st.session_state["model_paths"] = {}   # {nom_fichier: chemin_local}
if "current_model" not in st.session_state:
    st.session_state["current_model"] = None
if "current_model_name" not in st.session_state:
    st.session_state["current_model_name"] = None


# ---------------------------------------------------------
# 2. Fonctions modèle (TorchScript)
# ---------------------------------------------------------

def load_torchscript_model(model_path, device="cpu"):
    """
    Charge un modèle TorchScript (.pth) directement
    sans avoir besoin d'une classe Python.
    """
    model = torch.jit.load(model_path, map_location=device)
    model.to(device)
    model.eval()
    return model


def run_inference_on_frame(model, frame, device="cpu"):
    """
    Applique le modèle sur UN frame et renvoie le frame annoté.

    ⚠️ Version générique :
       - convertit en RGB
       - resize en 224x224
       - normalise en [0,1]
       - appelle le modèle
       - écrit juste "Inference OK" (à personnaliser selon les outputs)
    """

    # BGR -> RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # resize vers 224x224 (à adapter si besoin)
    img_resized = cv2.resize(rgb, (224, 224))

    # HWC -> CHW -> batch
    tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0)  # (1,3,224,224)
    tensor = tensor.to(device) / 255.0

    with torch.no_grad():
        outputs = model(tensor)

    # Ici tu peux interpréter outputs (classification, détection, etc.)
    # Pour l'instant, on fait juste une overlay texte
    annotated = frame.copy()
    cv2.putText(
        annotated,
        "Inference OK",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return annotated


# ---------------------------------------------------------
# 3. Sidebar : upload & sélection de modèles
# ---------------------------------------------------------
st.sidebar.title("🧠 Modèles (.pth)")

uploaded_models = st.sidebar.file_uploader(
    "Uploader un ou plusieurs modèles TorchScript (.pth)",
    type=["pth", "pt"],
    accept_multiple_files=True,
)

models_dir = os.path.join(tempfile.gettempdir(), "streamlit_models")
os.makedirs(models_dir, exist_ok=True)

# Sauvegarder les modèles uploadés dans un répertoire temporaire
if uploaded_models:
    for f in uploaded_models:
        if f.name not in st.session_state["model_paths"]:
            save_path = os.path.join(models_dir, f.name)
            with open(save_path, "wb") as out:
                out.write(f.getbuffer())
            st.session_state["model_paths"][f.name] = save_path
            st.sidebar.success(f"Modèle ajouté : {f.name}")

if len(st.session_state["model_paths"]) == 0:
    st.sidebar.info("Aucun modèle uploadé pour l'instant.")
    selected_model_name = None
else:
    selected_model_name = st.sidebar.selectbox(
        "Sélectionner un modèle",
        options=list(st.session_state["model_paths"].keys()),
        index=0,
    )

# Device (tu peux ajouter "cuda" si tu es sur une machine avec GPU)
device = st.sidebar.selectbox("Device", ["cpu"], index=0)

# Bouton pour charger le modèle sélectionné
if selected_model_name is not None:
    if st.sidebar.button("Charger ce modèle"):
        path = st.session_state["model_paths"][selected_model_name]
        try:
            st.session_state["current_model"] = load_torchscript_model(path, device=device)
            st.session_state["current_model_name"] = selected_model_name
            st.sidebar.success(f"Modèle courant : {selected_model_name}")
        except Exception as e:
            st.sidebar.error(f"Erreur au chargement du modèle : {e}")


# ---------------------------------------------------------
# 4. Zone principale : vidéo + inference
# ---------------------------------------------------------
st.title("🎥 Tester un modèle TorchScript sur une vidéo")

if st.session_state["current_model"] is None:
    st.warning("Aucun modèle chargé. Merci d'en sélectionner un dans la barre latérale.")
else:
    st.success(f"Modèle courant : {st.session_state['current_model_name']}")

video_file = st.file_uploader(
    "Uploader une vidéo",
    type=["mp4", "mov", "avi", "mkv"],
)

video_path = None

if video_file is not None:
    # sauvegarde temporaire de la vidéo
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_file.read())
        video_path = tmp.name

    st.subheader("Vidéo originale")
    st.video(video_path)

# Lancer l'inférence sur la vidéo
if video_path is not None and st.session_state["current_model"] is not None:
    if st.button("Lancer l'inférence sur la vidéo"):
        st.info("Traitement de la vidéo en cours...")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Impossible de lire la vidéo.")
        else:
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # fichier vidéo de sortie
            out_fd, out_path = tempfile.mkstemp(suffix=".mp4")
            os.close(out_fd)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

            progress = st.progress(0)
            current_model = st.session_state["current_model"]

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                annotated_frame = run_inference_on_frame(
                    current_model,
                    frame,
                    device=device
                )

                writer.write(annotated_frame)

                frame_idx += 1
                if frame_count > 0:
                    progress.progress(min(frame_idx / frame_count, 1.0))

            cap.release()
            writer.release()

            st.success("Inférence terminée ✅")
            st.subheader("Vidéo annotée")
            st.video(out_path)
